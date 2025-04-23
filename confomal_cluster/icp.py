from __future__ import division
from collections import defaultdict
from functools import partial

import abc
from tqdm import tqdm
import numpy as np
import sklearn.base
from sklearn.base import BaseEstimator
import torch
import torch.autograd.functional as F
# from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .utils import compute_coverage, default_loss
import sys

class RegressionErrFunc(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(RegressionErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):
        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):
        pass


class AbsErrorErrFunc(RegressionErrFunc):
    def __init__(self):
        super(AbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        err = np.abs(prediction - y)
        if err.ndim > 1:
            err = np.linalg.norm(err, ord=np.inf, axis=1)
        return err

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


class BaseScorer(sklearn.base.BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseScorer, self).__init__()

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def score(self, x, y=None):
        pass

    @abc.abstractmethod
    def score_batch(self, dataloader):
        pass


class BaseModelNc(BaseScorer):
    def __init__(self, model, err_func, normalizer=None, beta=1e-6):
        super(BaseModelNc, self).__init__()
        self.err_func = err_func
        self.model = model
        self.normalizer = normalizer
        self.beta = beta

        if (self.normalizer is not None and hasattr(self.normalizer, 'base_model')):
            self.normalizer.base_model = self.model

        self.last_x, self.last_y = None, None
        self.last_prediction = None
        self.clean = False

    def fit(self, x, y):
        self.model.fit(x, y)
        if self.normalizer is not None:
            self.normalizer.fit(x, y)
        self.clean = False

    def score(self, x, y=None):
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if prediction.ndim > 1:
            ret_val = self.err_func.apply(prediction, y)
        else:
            ret_val = self.err_func.apply(prediction, y) / norm

        return ret_val

    def score_batch(self, dataloader):
        ret_val = []
        for x, _, y in tqdm(dataloader):
            prediction = self.model.predict(x)
            if self.normalizer is not None:
                norm = self.normalizer.score(x) + self.beta
            else:
                norm = np.ones(len(x))

            if prediction.ndim > 1:
                batch_ret_val = self.err_func.apply(prediction, y.detach().cpu().numpy())
            else:
                batch_ret_val = self.err_func.apply(prediction, y.detach().cpu().numpy()) / norm
            ret_val.append(batch_ret_val)
        ret_val = np.concatenate(ret_val, axis=0)
        return ret_val


class RegressorNc(BaseModelNc):
    def __init__(self, model, err_func=AbsErrorErrFunc(), normalizer=None, beta=1e-6):
        super(RegressorNc, self).__init__(model, err_func, normalizer, beta)

    def predict(self, x, nc, significance=None):
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((x.shape[0], self.model.model.out_shape, 2))
            err_dist = self.err_func.apply_inverse(nc, significance)  # (2, y_dim)
            err_dist = np.stack([err_dist] * n_test)  # (B, 2, y_dim)
            if prediction.ndim > 1:  # CQR
                intervals[..., 0] = prediction - err_dist[:, 0]
                intervals[..., 1] = prediction + err_dist[:, 1]
            else:  # regular conformal prediction
                err_dist *= norm[:, None, None]
                intervals[..., 0] = prediction[:, None] - err_dist[:, 0]
                intervals[..., 1] = prediction[:, None] + err_dist[:, 1]

            return intervals
        else:  # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals


class FeatErrorErrFunc(RegressionErrFunc):
    def __init__(self, feat_norm):
        super(FeatErrorErrFunc, self).__init__()
        self.feat_norm = feat_norm

    def apply(self, prediction, z):
        ret = (prediction - z).norm(p=self.feat_norm, dim=1)
        return ret

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])
    
    # def apply(self, prediction, z):
    #     ret = torch.nn.functional.cosine_similarity(prediction, z, dim=1)
    #     ret = 1 - ret
    #     return ret


class FeatRegressorNc(BaseModelNc):
    def __init__(self, model,
                 # err_func=FeatErrorErrFunc(),
                 inv_lr, inv_step, criterion=default_loss, feat_norm=np.inf, certification_method=0, cert_optimizer='sgd',
                 normalizer=None, beta=1e-6, g_out_process=None):
        if feat_norm in ["inf", np.inf, float('inf')]:
            self.feat_norm = np.inf
        elif (type(feat_norm) == int or float):
            self.feat_norm = feat_norm
        else:
            raise NotImplementedError
        err_func = FeatErrorErrFunc(feat_norm=self.feat_norm)

        super(FeatRegressorNc, self).__init__(model, err_func, normalizer, beta)
        self.criterion = criterion
        self.inv_lr = inv_lr
        self.inv_step = inv_step
        self.certification_method = certification_method
        self.cmethod = ['IBP', 'IBP+backward', 'backward', 'CROWN-Optimized'][self.certification_method]
        print(f"Use {self.cmethod} method for certification")

        self.cert_optimizer = cert_optimizer
        # the function to post process the output of g, because FCN needs interpolate and reshape
        self.g_out_process = g_out_process

    def inv_g(self, z0, y, step=None, record_each_step=False):
        z = z0.detach().clone()
        z = z.detach()
        z.requires_grad_()
        if self.cert_optimizer == "sgd":
            optimizer = torch.optim.SGD([z], lr=self.inv_lr)
        elif self.cert_optimizer == "adam":
            optimizer = torch.optim.Adam([z], lr=self.inv_lr)

        self.model.model.eval()
        each_step_z = []
        for _ in range(step):
            pred = self.model.model.g(z)
            if self.g_out_process is not None:
                pred = self.g_out_process(pred)

            loss = self.criterion(pred.squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if record_each_step:
                each_step_z.append(z.detach().cpu().clone())

        if record_each_step:
            return each_step_z
        else:
            return z.detach().cpu()

    def inv_g_Geodesic_Distance_single_data(self, x, onehot_label, step=None, num_copies=16, noise_std=0.1):
        """
        x:一个img
        onehot_label:一个onehot_label
        """
        if step == None:
            step = self.inv_step
        
        x, onehot_label = x.to(self.model.device), onehot_label.to(self.model.device)
# 分类
        onehot_label = onehot_label.repeat(num_copies)

        x = torch.unsqueeze(x, dim=0)
        z0 = self.model.model.encoder(x)
            
        z = z0.detach().clone().squeeze()
        
        z_argument_batch = [z]
        for i in range(num_copies-1):
            noise = torch.randn_like(z) * noise_std
            z_argument_batch.append(z + noise)
        
        z_argument_batch = torch.stack(z_argument_batch)
        z_argument_batch.requires_grad_()
        
        # 初始化优化器
        if self.cert_optimizer == "sgd":
            optimizer = torch.optim.SGD([z_argument_batch], lr=self.inv_lr)
        elif self.cert_optimizer == "adam":
            optimizer = torch.optim.Adam([z_argument_batch], lr=self.inv_lr)
        
        self.model.model.eval()
        each_step_distance = []
        for _ in range(step):
            # 深度复制z_argument_batch，确保在更新前保留其值
            z_clone = z_argument_batch.detach().clone()
            
            # 前向传播-----回归
            # pred = self.model.model.g(z_argument_batch)
            # print(pred.shape)
            # print(onehot_label.shape)
            # sys.exit()
            # loss = self.criterion(pred.squeeze(), onehot_label)
            # 前向传播-----分类
            preds = self.model.model.g(z_argument_batch)
            # print(preds.shape)
            # print(onehot_label.shape)
            loss = self.criterion(preds, onehot_label)
            # sys.exit()
            
            # 优化步骤
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算更新前后的距离（L1距离或L2距离均可）
            # distance = torch.mean(torch.abs(z_clone - z_argument_batch))  # L1范数
            distance = torch.mean(torch.norm(z_clone - z_argument_batch, p=2, dim=1))  # L2 norm across copies
            each_step_distance.append(distance)
            # print(f"{_} step done ... ")
        
        # 计算每个label中各步的平均距离
        Geodesic_Distance = torch.mean(torch.stack(each_step_distance))

        return Geodesic_Distance
        

    def get_each_step_err_dist(self, x, y, z_pred, steps):
        each_step_z_true = self.inv_g(z_pred, y, step=steps, record_each_step=True)  #each_step_z_true shape: List, (200*103*2048) (steps, len(x), 2048)
        # print(type(each_step_z_true))
        # print(len(each_step_z_true),   each_step_z_true[0].shape)
        # import sys
        # sys.exit()
        if self.normalizer is not None:
            raise NotImplementedError
        else:
            norm = np.ones(len(x))

        err_dist_list = []
        for i, step_z_true in enumerate(each_step_z_true):
            err_dist = self.err_func.apply(z_pred.detach().cpu(), step_z_true.detach().cpu()).numpy() / norm
            err_dist_list.append(err_dist)
        return err_dist_list

    def coverage_loose(self, x, y, z_pred, steps, val_significance):
        z_pred_detach = z_pred.detach().clone()

        idx = torch.randperm(len(z_pred_detach))
        n_val = int(np.floor(len(z_pred_detach) / 5))
        val_idx, cal_idx = idx[:n_val], idx[n_val:]

        cal_x, val_x = x[cal_idx], x[val_idx]
        cal_y, val_y = y[cal_idx], y[val_idx]
        cal_z_pred, val_z_pred = z_pred_detach[cal_idx], z_pred_detach[val_idx]

        cal_score_list = self.get_each_step_err_dist(cal_x, cal_y, cal_z_pred, steps=steps)

        val_coverage_list = []
        for i, step_cal_score in enumerate(cal_score_list):
            val_predictions = self.predict(x=val_x.detach().cpu().numpy(), nc=step_cal_score,
                                           significance=val_significance)
            val_y_lower, val_y_upper = val_predictions[..., 0], val_predictions[..., 1]
            val_coverage, _ = compute_coverage(val_y.detach().cpu().numpy(), val_y_lower, val_y_upper, val_significance,
                                               name="{}-th step's validation".format(i), verbose=False)
            val_coverage_list.append(val_coverage)
        return val_coverage_list, len(val_x)

    def coverage_tight(self, x, y, z_pred, steps, val_significance):
        z_pred_detach = z_pred.detach().clone()

        idx = torch.randperm(len(z_pred_detach))
        n_val = int(np.floor(len(z_pred_detach) / 5))
        val_idx, cal_idx = idx[:n_val], idx[n_val:]
        
        cal_x, val_x = x[cal_idx], x[val_idx]
        cal_y, val_y = y[cal_idx], y[val_idx]
        cal_z_pred, val_z_pred = z_pred_detach[cal_idx], z_pred_detach[val_idx]

        cal_score_list = self.get_each_step_err_dist(cal_x, cal_y, cal_z_pred, steps=steps)  #cal_score_list [[103], []], shape: (200*103)
        val_score_list = self.get_each_step_err_dist(val_x, val_y, val_z_pred, steps=steps)  #val_score_list [[25], []], shape: (200*25)
        # print(len(cal_score_list), cal_score_list[0].shape)
        # import sys
        # sys.exit()
        val_coverage_list = []
        for i, (cal_score, val_score) in enumerate(zip(cal_score_list, val_score_list)):
            err_dist_threshold = self.err_func.apply_inverse(nc=cal_score, significance=val_significance)[0][0]
            val_coverage = np.sum(val_score < err_dist_threshold) * 100 / len(val_score)
            # print(val_score < err_dist_threshold)
            # sys.exit()
            val_coverage_list.append(val_coverage)
        return val_coverage_list, len(val_x)     #val_coverage_list shape: (steps,)

    def find_best_step_num(self, x, y, z_pred):
        max_inv_steps = 200
        val_significance = 0.1

        # each_step_val_coverage, val_num = self.coverage_loose(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)
        each_step_val_coverage, val_num = self.coverage_tight(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)

        tolerance = 3
        count = 0
        final_coverage, best_step = None, None
        for i, val_coverage in enumerate(each_step_val_coverage):
            # print("{}-th step's validation coverage is {}".format(i, val_coverage))
            if val_coverage > (1 - val_significance) * 100 and final_coverage is None:
                count += 1
                if count == tolerance:
                    final_coverage = val_coverage
                    best_step = i
            elif val_coverage <= (1 - val_significance) * 100 and count > 0:
                count = 0

        if final_coverage is None or best_step is None:
            raise ValueError(
                "does not find a good step to make the coverage higher than {}".format(1 - val_significance))
        print("The best inv_step is {}, which gets {} coverage on val set".format(best_step + 1, final_coverage))
        return best_step + 1

    def find_best_step_num_batch_sony(self, dataloader):
        '''
        简易的寻找最好的寻找代理特征层的step
        '''
        max_inv_steps = 200
        val_significance = 0.1

        print("begin to find the best step number")
        all_x, all_y = [], []
        for x, _, y in dataloader:
            all_x.append(x)
            all_y.append(y)
        all_x = [x.unsqueeze(0) for x in all_x]
        all_y = [y.unsqueeze(0) for y in all_y]
        all_x = torch.cat(all_x, dim=0).to(self.model.device)
        all_y = torch.cat(all_y, dim=0).to(self.model.device)
        self.model.model.eval()
        z_pred = self.model.model.encoder(all_x)
        batch_each_step_val_coverage, val_num = self.coverage_tight(all_x, all_y, z_pred, steps=max_inv_steps, val_significance=val_significance)  # length: max_inv_steps
        return  batch_each_step_val_coverage    

    def find_best_step_num_batch(self, dataloader):
        max_inv_steps = 200
        val_significance = 0.1

        accumulate_val_coverage = np.zeros(max_inv_steps)
        accumulate_val_num = 0
        print("begin to find the best step number")
        batch_each_step_val_coverage_list = []
        for x, _, y in tqdm(dataloader):
            x, y = x.to(self.model.device), y.to(self.model.device)
            z_pred = self.model.model.encoder(x)
            # batch_each_step_val_coverage, val_num = self.coverage_loose(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)  # length: max_inv_steps
            batch_each_step_val_coverage, val_num = self.coverage_tight(x, y, z_pred, steps=max_inv_steps, val_significance=val_significance)  # length: max_inv_steps
            accumulate_val_coverage += np.array(batch_each_step_val_coverage) * val_num
            accumulate_val_num += val_num
            
            batch_each_step_val_coverage_list.append(np.unique(batch_each_step_val_coverage))
            # print(batch_each_step_val_coverage.shape)
            # print(batch_each_step_val_coverage)
            # print(len(batch_each_step_val_coverage))
            # print(err_dist_threshold_list)
            # print(len(err_dist_threshold_list))
        # print("*" * 100)
        # print(len(batch_each_step_val_coverage_list))
        # print(batch_each_step_val_coverage_list)
        # sys.exit()
        
        each_step_val_coverage = accumulate_val_coverage / accumulate_val_num
        # print("*" * 100)
        # print(len(each_step_val_coverage))
        # print(each_step_val_coverage[:10])
        # print(np.unique(each_step_val_coverage))
        # sys.exit()
        tolerance = 1
        count = 0
        final_coverage, best_step = None, None
        for i, val_coverage in enumerate(each_step_val_coverage):
            # print("{}-th step's validation tight coverage is {}".format(i, val_coverage))
            if val_coverage > (1 - val_significance - 0.6) * 100 and final_coverage is None:
                count += 1
                if count == tolerance:
                    final_coverage = val_coverage
                    best_step = i
            elif val_coverage <= (1 - val_significance - 0.6) * 100 and count > 0:
                count = 0

        if final_coverage is None or best_step is None:
            print("Final coverage at each step:", np.max(each_step_val_coverage))
            raise ValueError(
                "does not find a good step to make the coverage higher than {}".format(1 - val_significance))
        print("The best inv_step is {}, which gets {} coverage on val set".format(best_step + 1, final_coverage))
        return best_step + 1

    def score(self, x, y=None):  # overwrite BaseModelNc.score()
        self.model.model.eval()
        n_test = x.shape[0]
        x, y = torch.from_numpy(x).to(self.model.device), torch.from_numpy(y).to(self.model.device)
        z_pred = self.model.model.encoder(x)

        if self.inv_step is None:
            self.inv_step = self.find_best_step_num(x, y, z_pred)

        z_true = self.inv_g(z_pred, y, step=self.inv_step)

        if self.normalizer is not None:
            raise NotImplementedError
        else:
            norm = np.ones(n_test)

        ret_val = self.err_func.apply(z_pred.detach().cpu(), z_true.detach().cpu())  # || z_pred - z_true ||
        ret_val = ret_val.numpy() / norm
        return ret_val

    def score_batch(self, dataloader):
        self.model.model.eval()
        if self.inv_step is None:
            self.inv_step = self.find_best_step_num_batch(dataloader)

        print('calculating score:')
        ret_val = []
        for x, _, y in tqdm(dataloader):
            x, y = x.to(self.model.device), y.to(self.model.device)

            if self.normalizer is not None:
                raise NotImplementedError
            else:
                norm = np.ones(len(x))

            z_pred = self.model.model.encoder(x)
            z_true = self.inv_g(z_pred, y, step=self.inv_step)
            batch_ret_val = self.err_func.apply(z_pred.detach().cpu(), z_true.detach().cpu())
            batch_ret_val = batch_ret_val.detach().cpu().numpy() / norm
            ret_val.append(batch_ret_val)
        ret_val = np.concatenate(ret_val, axis=0)
        return ret_val
    

# 单一线程:calibration
    # def score_Geodesic_Distance(self, x, onehot_label):  # overwrite BaseModelNc.score()
    #     """
    #     校准阶段，针对一个class中的每个数据，计算其对应onehot_label的score
    #     Args:
    #         x : imgs
    #         onehot_label : onehot_labels
    #     Returns:
    #         _type_: _description_
    #     """
        
    #     self.model.model.eval()
    #     n_test = x.shape[0]
    #     x, onehot_label = torch.from_numpy(x).to(self.model.device), torch.from_numpy(onehot_label).to(self.model.device)
    #     scores_all = []
    #     for i in range(n_test):
    #         # 欧式距离
    #         # score = self.inv_g_Geodesic_Distance_single_data(x[i], onehot_label[i], 
    #         #                                                  step=self.inv_step, num_copies=16, noise_std=0.1)
    #         # scores_all.append(score.item())
            
    #         # 测地距离
    #         score = self.compute_feature_space_geodesic(x[i], onehot_label[i], step=self.inv_step, lr=self.inv_lr, alpha=0.5)
    #         scores_all.append(score)
        
    #     return scores_all
    
# 多线程:calibration
    # def score_Geodesic_Distance(self, x, onehot_label):
    #     """
    #     校准阶段，并行计算每个数据对应onehot_label的测地距离得分

    #     Args:
    #         x : imgs
    #         onehot_label : onehot_labels
    #     Returns:
    #         测地距离分数列表
    #     """

    #     self.model.model.eval()
    #     n_test = x.shape[0]
    #     x, onehot_label = torch.from_numpy(x).to(self.model.device), torch.from_numpy(onehot_label).to(self.model.device)

    #     # 使用线程池并行计算测地距离
    #     from concurrent.futures import ThreadPoolExecutor, as_completed
    #     import numpy as np

    #     # 确定要使用的线程数和每个线程处理的数据块大小
    #     num_threads = min(10, n_test)  # 可调整线程数，根据系统资源情况
    #     chunk_size = int(np.ceil(n_test / num_threads))

    #     def process_chunk(start_idx, end_idx):
    #         # 处理一个数据块
    #         chunk_scores = []
    #         for i in range(start_idx, min(end_idx, n_test)):
    #             # 测地距离计算
    #             score = self.compute_feature_space_geodesic(
    #                 x[i], onehot_label[i], 
    #                 step=self.inv_step, lr=self.inv_lr, alpha=0.5
    #             )
    #             chunk_scores.append((i, score))
    #         return chunk_scores

    #     # 使用线程池并行处理数据块
    #     scores_all = [None] * n_test  # 预分配结果列表

    #     with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #         # 提交所有任务
    #         futures = []
    #         for i in range(0, n_test, chunk_size):
    #             futures.append(executor.submit(process_chunk, i, i + chunk_size))

    #         # 收集结果
    #         for future in as_completed(futures):
    #             for idx, score in future.result():
    #                 scores_all[idx] = score

    #     return scores_all


# 多线程并行
    def score_Geodesic_Distance(self, x, onehot_label):
        """
        校准阶段，并行计算每个数据对应onehot_label的测地距离得分，使用CUDA流实现真正的GPU并行

        Args:
            x : imgs
            onehot_label : onehot_labels
        Returns:
            测地距离分数列表
        """
        self.model.model.eval()
        n_test = x.shape[0]
        x, onehot_label = torch.from_numpy(x).to(self.model.device), torch.from_numpy(onehot_label).to(self.model.device)

        # 使用线程池并行计算测地距离
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import numpy as np

        # 修正了线程数计算，确保有多个线程
        num_threads = min(13, n_test)  # 可根据系统资源调整，8只是一个示例值
        # print(f"num_threads: {num_threads}")
        chunk_size = int(np.ceil(n_test / num_threads))

        # 为每个线程创建一个CUDA流
        streams = [torch.cuda.Stream() for _ in range(num_threads)]

        # 预分配结果列表
        scores_all = [None] * n_test

        def process_chunk(start_idx, end_idx, stream_idx):
            # 获取分配给该线程的CUDA流
            stream = streams[stream_idx]
            chunk_scores = []

            # 在指定的CUDA流上执行操作
            with torch.cuda.stream(stream):
                for i in range(start_idx, min(end_idx, n_test)):
                    # 测地距离计算
                    score = self.compute_feature_space_geodesic(
                        x[i], onehot_label[i], 
                        step=self.inv_step, lr=self.inv_lr, alpha=0.5
                    )
                    chunk_scores.append((i, score))

            return chunk_scores

        # 使用线程池并行处理数据块
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务，为每个任务分配一个流索引
            futures = []
            for i in range(num_threads):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_test)
                futures.append(executor.submit(process_chunk, start_idx, end_idx, i))

            # 收集结果
            for future in as_completed(futures):
                for idx, score in future.result():
                    scores_all[idx] = score

        # 同步所有CUDA流，确保所有计算完成
        torch.cuda.synchronize()

        return scores_all


# 单线程  inference
    # def score_all_labels_Geodesic_Distance(self, x, onehot_labels):  # overwrite BaseModelNc.score()
    #     """
    #     测试阶段，针对每一个数据，计算其对所有可能onehot_labels的scores
        
    #     Args:
    #         x : imgs, shape:(batch, imgs)
    #         onehot_labels : onehot_labels, shape(args.num_classes, args.num_classes)
    #     Returns:
    #         batch_scores: shape:(batch, args.num_classes)
    #     """
    #     self.model.model.eval()
        
    #     x  = x.to(self.model.device)
    #     n_test = x.shape[0]
    #     batch_scores = []
        
    #     for i in range(n_test):
    #         one_scores = []
    #         for onehot_label in onehot_labels:
    #             # 欧式距离
    #             # score = self.inv_g_Geodesic_Distance_single_data(x[i], onehot_label, step=self.inv_step, 
    #             #                                                  num_copies=16, noise_std=0.1)
    #             # one_scores.append(score.item())

    #             # 测地距离
    #             score = self.compute_feature_space_geodesic(x[i], onehot_label, step=self.inv_step, lr=self.inv_lr, alpha=0.5)
    #             one_scores.append(score)

    #         batch_scores.append(one_scores)
    #     return batch_scores


# 多线程并行  inference
    def score_all_labels_Geodesic_Distance(self, x, onehot_labels):  # overwrite BaseModelNc.score()
        """
        测试阶段，针对每一个数据，计算其对所有可能onehot_labels的scores
        使用CUDA流和线程池实现真正的并行处理

        Args:
            x : imgs, shape:(batch, imgs)
            onehot_labels : onehot_labels, shape(args.num_classes, args.num_classes)
        Returns:
            batch_scores: shape:(batch, args.num_classes)
        """
        self.model.model.eval()

        x = x.to(self.model.device)
        n_test = x.shape[0]
        num_classes = onehot_labels.shape[0]

        # 将onehot_labels转移到GPU
        onehot_labels = onehot_labels.to(self.model.device)

        # 导入并行处理模块
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import numpy as np

        # 确定线程数 - 根据图像数量决定
        num_threads = min(13, n_test)  # 限制最大线程数为30
        img_per_thread = int(np.ceil(n_test / num_threads))

        # 为每个线程创建CUDA流
        streams = [torch.cuda.Stream() for _ in range(num_threads)]

        # 预分配结果数组
        batch_scores = [[None for _ in range(num_classes)] for _ in range(n_test)]

        def process_chunk(start_idx, end_idx, stream_idx):
            # 获取分配给该线程的CUDA流
            stream = streams[stream_idx]
            chunk_results = []

            # 在指定的CUDA流上执行操作
            with torch.cuda.stream(stream):
                for i in range(start_idx, min(end_idx, n_test)):
                    img_scores = []
                    for class_idx, onehot_label in enumerate(onehot_labels):
                        # 测地距离计算
                        score = self.compute_feature_space_geodesic(
                            x[i], onehot_label, 
                            step=self.inv_step, lr=self.inv_lr, alpha=0.5
                        )
                        img_scores.append(score)
                    chunk_results.append((i, img_scores))

            return chunk_results

        # 使用线程池并行处理数据块
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务，为每个任务分配一个流索引
            futures = []
            for i in range(num_threads):
                start_idx = i * img_per_thread
                end_idx = min((i + 1) * img_per_thread, n_test)
                futures.append(executor.submit(process_chunk, start_idx, end_idx, i))

            # 收集结果
            for future in as_completed(futures):
                for idx, scores in future.result():
                    batch_scores[idx] = scores

        # 同步所有CUDA流，确保所有计算完成
        torch.cuda.synchronize()

        return batch_scores



# ture geodesti distance
# 测地距离

    def compute_feature_space_geodesic(self, x, onehot_label, step=100, lr=0.01, alpha=0.5):
        """
        计算原始特征向量与优化后的"代理特征向量"间的测地距离

        参数:
        x: 输入图像
        onehot_label: 目标类别的独热编码
        step: 优化迭代次数
        lr: 学习率
        alpha: 测地线离散化插值参数，控制度量估计点在路径上的位置，alpha=0表示前一点，alpha=1表示当前点，alpha=0.5表示中点

        返回:
        原始特征向量和代理特征向量之间的测地距离
        """
        self.model.model.eval()
        x = x.to(self.model.device)
        onehot_label = onehot_label.to(self.model.device)

        # 1. 获取原始特征向量
        with torch.no_grad():
            x_input = x.unsqueeze(0)
            original_z = self.model.model.encoder(x_input).squeeze()

        # 2. 创建可优化的代理特征向量
        proxy_z = original_z.detach().clone().requires_grad_(True)


        # 3. 设置优化器
        if self.cert_optimizer == "sgd":
            optimizer = torch.optim.SGD([proxy_z], lr=lr)
        elif self.cert_optimizer == "adam":
            optimizer = torch.optim.Adam([proxy_z], lr=lr)

                # 初始化阶段
        try:
            model_g = torch.jit.script(self.model.model.g)
        except:
            model_g = self.model.model.g

        # 4. 存储测地线路径上的点
        geodesic_path = [original_z.clone()]
        # 4.2 修改开销
        geodesic_path = torch.zeros((step + 1, original_z.shape[0]), device=self.model.device)
        geodesic_path[0] = original_z.clone()

        # 5. 优化过程 - 找到最优的代理特征向量
        for i in range(step):
            # 前向传播
            # pred = self.model.model.g(proxy_z.unsqueeze(0))
            
            # C++ JIT
            pred = model_g(proxy_z.unsqueeze(0))
            
            loss = self.criterion(pred, onehot_label.unsqueeze(0))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 存储当前点以构建测地线路径
            # if (i+1) % (step // 10) == 0 or i == 0:  # 存储10个关键点以节省内存
            #     geodesic_path.append(proxy_z.detach().clone())
            
            # 4 存储所有点
            # geodesic_path.append(proxy_z.detach().clone())
            # 4.1 修改开销
            geodesic_path[i+1] = proxy_z.detach()

            # 打印优化进度
            # if (i+1) % (step // 5) == 0:
            #     print(f"优化步骤 {i+1}/{step}, 损失: {loss.item():.6f}")

        # 6. 计算测地距离 - 沿路径积分
        # total_distance = 0.0

        # for i in range(1, len(geodesic_path)):
        #     prev_z = geodesic_path[i-1]
        #     curr_z = geodesic_path[i]

        #     # 计算欧式位移向量
        #     displacement = curr_z - prev_z

        #     # 估计该点的局部黎曼度量
        #         # 原始
        #     # local_metric = self.estimate_local_riemann_metric(
        #     #     (prev_z * (1-alpha) + curr_z * alpha).unsqueeze(0)  # 在两点之间的加权位置估计度量
        #     # )
        #         # 提升效率
        #     local_metric = self.estimate_local_riemann_metric_vectorized(
        #         (prev_z * (1-alpha) + curr_z * alpha).unsqueeze(0),  # 在两点之间的加权位置估计度量
        #         model_g
        #         )


        #     # 在黎曼度量下计算线元长度
        #     # ds^2 = dx^T G dx
        #     if local_metric is not None:
        #         # 转换到适当的形状以进行矩阵运算
        #         disp_flat = displacement.view(-1, 1)  # 列向量
        #         local_distance = torch.sqrt(
        #             torch.matmul(torch.matmul(disp_flat.t(), local_metric), disp_flat)
        #         ).item()
        #     else:
        #         # 如果度量估计失败，回退到欧几里得距离
        #         local_distance = torch.norm(displacement, p=2).item()

        #     total_distance += local_distance
        
        # 6.1 并行计算测地距离
        # 计算所有相邻点的欧式位移向量
        displacements = geodesic_path[1:] - geodesic_path[:-1]

        # 计算插值点用于度量估计
        interpolated_points = geodesic_path[:-1] * (1-alpha) + geodesic_path[1:] * alpha

        # 修改——1  并行计算测地距离
        total_distance = self.compute_geodesic_distance_parallel(displacements, interpolated_points, model_g)

        # 修改——2 完全的向量化的计算
        # total_distance = self.compute_geodesic_distance_improved(displacements, interpolated_points, model_g)

        

        # 可以用来画图
        # return total_distance, original_z, proxy_z
        # 输出一个分数
        return total_distance

    def estimate_local_riemann_metric(self, z, epsilon=1e-4):
        """
        估计特征空间中的局部黎曼度量张量
        使用输出空间中的Fisher信息矩阵近似

        参数:
        z: 特征空间中的点
        epsilon: 用于数值计算梯度的扰动大小

        返回:
        局部黎曼度量张量 (Fisher信息矩阵)
        """
        z_dim = z.shape[1]
        batch_size = z.shape[0]

        try:
            # 计算当前点的模型输出
            with torch.no_grad():
                base_output = self.model.model.g(z)

            # 初始化雅可比矩阵
            # 输出空间维度 × 特征空间维度
            output_dim = base_output.shape[1]
            jacobian = torch.zeros((batch_size, output_dim, z_dim), device=z.device)

            # 对每个特征维度计算数值梯度
            for i in range(z_dim):
                # 创建扰动向量
                h = torch.zeros_like(z)
                h[:, i] = epsilon

                # 前向差分计算梯度
                with torch.no_grad():
                    z_plus_h = z + h
                    output_plus_h = self.model.model.g(z_plus_h)

                    # 数值梯度
                    jacobian[:, :, i] = (output_plus_h - base_output) / epsilon

            # 计算Fisher信息矩阵: J^T * J
            batch_metrics = []
            for b in range(batch_size):
                fisher_info = torch.matmul(jacobian[b].t(), jacobian[b])

                # 确保度量张量是正定的
                eigenvalues = torch.linalg.eigvalsh(fisher_info)
                min_eig = torch.min(eigenvalues)

                # 如果有非正特征值，添加小的扰动
                if min_eig <= 0:
                    fisher_info += torch.eye(z_dim, device=z.device) * (abs(min_eig) + 1e-6)

                batch_metrics.append(fisher_info)

            # 返回批次中第一个(或唯一)度量张量
            return batch_metrics[0]

        except Exception as e:
            print(f"估计黎曼度量时出错: {e}")
            return None
    
    def estimate_local_riemann_metric_vectorized(self, z, model, epsilon=1e-4):
        """
        使用PyTorch的functional.jacobian直接计算雅可比矩阵

        参数:
        z: 特征空间中的点
        epsilon: 用于数值计算的容差(在使用直接计算方法时不再需要)

        返回:
        局部黎曼度量张量 (Fisher信息矩阵)
        """
        z_dim = z.shape[1]
        batch_size = z.shape[0]

        try:
            # 使用PyTorch的functional API直接计算雅可比矩阵
            # 这将自动处理所有输出维度
            jacobian = F.jacobian(lambda x: model(x), z)
            # print(f"orginal_output: {jacobian.shape}")
            # jacobian的形状为 [batch_size, output_dim, batch_size, input_dim]
            # 我们需要提取对角线元素以获得每个批次样本的雅可比矩阵
            # 即转换为 [batch_size, output_dim, input_dim]
            jacobian = jacobian.diagonal(dim1=0, dim2=2)  # 提取batch对应的对角线元素
            # print(f"duijiaoxian: {jacobian.shape}")
            jacobian = jacobian.permute(2, 0, 1)  # 调整维度顺序
            # print(f"output: {jacobian.shape}")
            # sys.exit()

            # 计算Fisher信息矩阵: J^T * J
            batch_metrics = []
            for b in range(batch_size):
                fisher_info = torch.matmul(jacobian[b].t(), jacobian[b])

                # 确保度量张量是正定的
                eigenvalues = torch.linalg.eigvalsh(fisher_info)
                min_eig = torch.min(eigenvalues)

                if min_eig <= 0:
                    fisher_info += torch.eye(z_dim, device=z.device) * (abs(min_eig) + 1e-6)

                batch_metrics.append(fisher_info)

            return batch_metrics[0]

        except Exception as e:
            print(f"估计黎曼度量时出错: {e}")
            return None
        
    def compute_geodesic_distance_parallel(self, displacements, interpolated_points, model):
        """并行计算测地距离"""
        total_distance = 0.0
        # 分批次计算以平衡内存使用和计算效率
        batch_size = 40 # 可调整批次大小
        num_segments = displacements.shape[0]

        for i in range(0, num_segments, batch_size):
            # 获取当前批次
            batch_end = min(i + batch_size, num_segments)
            curr_displacements = displacements[i:batch_end]
            curr_interp_points = interpolated_points[i:batch_end].unsqueeze(1)  # 增加批次维度

            # 批量计算局部度量
            batch_metrics = self.estimate_local_riemann_metric_batch(curr_interp_points, model)

            # 计算批量距离
            for j in range(curr_displacements.shape[0]):
                disp = curr_displacements[j].view(-1, 1)  # 列向量
                metric = batch_metrics[j] if batch_metrics[j] is not None else None

                if metric is not None:
                    # 在黎曼度量下计算线元长度
                    local_distance = torch.sqrt(
                        torch.matmul(torch.matmul(disp.t(), metric), disp)
                    ).item()
                else:
                    # 回退到欧几里得距离
                    local_distance = torch.norm(curr_displacements[j], p=2).item()

                total_distance += local_distance

        return total_distance

    def estimate_local_riemann_metric_batch(self, z_batch, model, epsilon=1e-4):
        """批量估计多个点的黎曼度量张量"""
        # 展平批次维度以便处理
        batch_size = z_batch.shape[0]
        z_flattened = z_batch.reshape(-1, z_batch.shape[-1])
        z_dim = z_flattened.shape[1]

        try:
            # 使用PyTorch的functional API直接计算雅可比矩阵
            jacobian = F.jacobian(lambda x: model(x), z_flattened)

            # 重塑雅可比矩阵并提取对角线元素
            jacobian = jacobian.diagonal(dim1=0, dim2=2)
            jacobian = jacobian.permute(2, 0, 1)

            # 批量计算Fisher信息矩阵
            batch_metrics = []
            for b in range(batch_size):
                fisher_info = torch.matmul(jacobian[b].t(), jacobian[b])

                # 确保度量张量是正定的
                eigenvalues = torch.linalg.eigvalsh(fisher_info)
                min_eig = torch.min(eigenvalues)

                if min_eig <= 0:
                    fisher_info += torch.eye(z_dim, device=z_flattened.device) * (abs(min_eig) + 1e-6)

                batch_metrics.append(fisher_info)

            return batch_metrics

        except Exception as e:
            print(f"批量估计黎曼度量时出错: {e}")
            return [None] * batch_size    

    

    def visualize_geodesic_path(self, original_z, proxy_z, num_points=20):
        """
        在特征空间中可视化测地线路径
        使用PCA或t-SNE降维到2D或3D进行可视化

        参数:
        original_z: 原始特征向量
        proxy_z: 代理特征向量
        num_points: 路径上的插值点数
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            # 生成测地线路径上的点
            path_points = []
            for t in torch.linspace(0, 1, num_points):
                # 在特征空间中线性插值
                interp_z = original_z * (1 - t) + proxy_z * t
                path_points.append(interp_z.cpu().detach().numpy())

            # 将路径点堆叠为一个数组
            path_array = np.stack(path_points)

            # 使用PCA降维到2D
            pca = PCA(n_components=2)
            path_2d = pca.fit_transform(path_array)

            # 可视化2D路径
            plt.figure(figsize=(10, 8))
            plt.scatter(path_2d[:, 0], path_2d[:, 1], c=np.arange(num_points), cmap='viridis')
            plt.plot(path_2d[:, 0], path_2d[:, 1], 'r--')
            plt.scatter(path_2d[0, 0], path_2d[0, 1], c='blue', s=100, label='Original Feature')
            plt.scatter(path_2d[-1, 0], path_2d[-1, 1], c='red', s=100, label='Proxy Feature')
            plt.colorbar(label='Path Position')
            plt.legend()
            plt.title('Geodesic Path in Feature Space (PCA Projection)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.grid(True)
            plt.savefig('geodesic_path.png')
            plt.close()

            print(f"测地线路径可视化已保存为 'geodesic_path.png'")

        except Exception as e:
            print(f"可视化测地线路径时出错: {e}")





class BaseIcp(BaseEstimator):
    def __init__(self, nc_function, condition=None):
        self.cal_x, self.cal_y = None, None
        self.nc_function = nc_function

        default_condition = lambda x: 0
        is_default = (callable(condition) and
                      (condition.__code__.co_code ==
                       default_condition.__code__.co_code))

        if is_default:
            self.condition = condition
            self.conditional = False
        elif callable(condition):
            self.condition = condition
            self.conditional = True
        else:
            self.condition = lambda x: 0
            self.conditional = False

    @classmethod
    def get_problem_type(cls):
        return 'regression'

    def fit(self, x, y):
        self.nc_function.fit(x, y)

    def calibrate(self, x, y, increment=False):
        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        if self.conditional:
            category_map = np.array([self.condition((x[i, :], y[i])) for i in range(y.size)])
            self.categories = np.unique(category_map)
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = category_map == cond
                cal_scores = self.nc_function.score(self.cal_x[idx, :], self.cal_y[idx])
                self.cal_scores[cond] = np.sort(cal_scores, 0)[::-1]
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(self.cal_x, self.cal_y)
            self.cal_scores = {0: np.sort(cal_scores, 0)[::-1]}

    def calibrate_batch(self, dataloader):
        if self.conditional:
            raise NotImplementedError

        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score_batch(dataloader)
            self.cal_scores = {0: np.sort(cal_scores, 0)[::-1]}

    def calibrate_per_class(self, data):
        """
        计算每种class的scores
        """
        imgs, labels, one_hot_labels = zip(*data)
        imgs = np.array([img.numpy() for img in imgs])
        one_hot_labels = np.array([label.numpy() for label in one_hot_labels])
        cal_scores = self.nc_function.score(imgs, one_hot_labels)
            
        return cal_scores
    
    def calibrate_per_class_Geodesic_Distance(self, data):
        """
        校准阶段，计算每种class的scores
        """
        imgs, labels, one_hot_labels = zip(*data)
        imgs = np.array([img.numpy() for img in imgs])
        
        # 回归
        # one_hot_labels = np.array([label.numpy() for label in one_hot_labels])
        
        # 分类
        one_hot_labels = np.array([label.numpy() for label in labels])
        
        cal_scores = self.nc_function.score_Geodesic_Distance(imgs, one_hot_labels)
            
        return cal_scores

    def calibrate_all_labels_Geodesic_Distance(self, test_loader, onehot_labels):
        """
        测试阶段，针对一个数据，计算其面对所有可能onehot_label的scores
        返回
        """
        all_scores = []
        i = 0
        for x, y, _ in tqdm(test_loader, desc="calculate test batches", total=len(test_loader), unit="batch", colour='blue'):
            batch_scores = self.nc_function.score_all_labels_Geodesic_Distance(x, onehot_labels)         
            for label, scores in zip(y, batch_scores):
                # print("*" *100)
                # label, scores = torch.tensor(label), torch.tensor(scores)
                # ranks = torch.argsort(scores)
                # print(ranks)
                # print((ranks==label).nonzero(as_tuple=True)[0])
                all_scores.append((label.item(), scores))
            i += 1
            if i >= 2:
                break
        return all_scores
    def calibrate_all_labels_Geodesic_Distance_sony(self, test_loader, onehot_labels):
        """
        测试阶段，针对一个数据，计算其面对所有可能onehot_label的scores
        返回
        减少时间版本
        """
        all_scores = []
        i = 0
        for x, y, _ in tqdm(test_loader, desc="calculate test batches", total=len(test_loader), unit="batch", colour='blue'):
            # 较大数据的时候使用
            x, y = x[:64], y[:64]
            batch_scores = self.nc_function.score_all_labels_Geodesic_Distance(x, onehot_labels)         
            for label, scores in zip(y, batch_scores):
                # print("*" *100)
                # label, scores = torch.tensor(label), torch.tensor(scores)
                # ranks = torch.argsort(scores)
                # print(ranks)
                # print((ranks==label).nonzero(as_tuple=True)[0])
                all_scores.append((label.item(), scores))
            i += 1
            if i >= 2:
                break
        return all_scores


    def _calibrate_hook(self, x, y, increment):
        pass

    def _update_calibration_set(self, x, y, increment):
        if increment and self.cal_x is not None and self.cal_y is not None:
            self.cal_x = np.vstack([self.cal_x, x])
            self.cal_y = np.hstack([self.cal_y, y])
        else:
            self.cal_x, self.cal_y = x, y


class IcpRegressor(BaseIcp):
    def __init__(self, nc_function, condition=None):
        super(IcpRegressor, self).__init__(nc_function, condition)

    def predict(self, x, significance=None):
        self.nc_function.model.model.eval()

        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], self.nc_function.model.model.out_shape, 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], self.nc_function.model.model.out_shape, 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :], self.cal_scores[condition], significance)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction
    
    def predict_sony(self, x, y, label_to_cluster, cal_scores_per_cluster, significance=None):
        self.nc_function.model.model.eval()

        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], self.nc_function.model.model.out_shape, 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], self.nc_function.model.model.out_shape, 2))
        y = [i.item() for i in y]
        y_cluster = [label_to_cluster[i] for i in y]
        for cluster in np.unique(y_cluster):
            idx = cluster == np.array(y_cluster)
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :], cal_scores_per_cluster[cluster], significance)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction

    def if_in_coverage(self, x, y, significance):
        self.nc_function.model.model.eval()
        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])
        result_array = np.zeros(len(x)).astype(bool)
        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                err_dist = self.nc_function.score(x[idx, :], y[idx])
                err_dist_threshold = self.nc_function.err_func.apply_inverse(self.cal_scores[condition], significance)[0][0]
                result_array[idx] = (err_dist < err_dist_threshold)
        return result_array

    def if_in_coverage_batch(self, dataloader, significance):
        self.nc_function.model.model.eval()
        err_dist = self.nc_function.score_batch(dataloader)
        err_dist_threshold = self.nc_function.err_func.apply_inverse(self.cal_scores[0], significance)[0][0]
        result_array = (err_dist < err_dist_threshold)
        print(np.sum(result_array))
        print(len(result_array))
        return result_array
    
    def if_in_coverage_batch_sony(self, cal_scores_cal_data, cal_scores_test_data, significance):
        result = []
        for cluster in cal_scores_cal_data:
            err_dist_threshold = self.nc_function.err_func.apply_inverse(cal_scores_cal_data[cluster], significance)[0][0]    
            result_array = (cal_scores_test_data[cluster] < err_dist_threshold)
            result.extend(result_array)
        print(np.sum(result))
        print(len(result))
        return result
    
    def coverage_length_compute_sony(self, test_loader, label_list_onehot, label_Quantile):
        
        all_scores = []
        y_labels = []
        for x_test, discrete_y_test, _ in tqdm(test_loader):
            y_labels.extend(discrete_y_test)
            batch_scores = torch.zeros(len(x_test), len(label_list_onehot))
            for i, y in enumerate(label_list_onehot):
                result = self.nc_function.score(x_test.numpy(), y.numpy())   #shape: (batch_size,)
                batch_scores[:, i] = torch.tensor(result)   #shape: (args.numclasses, batch_size)
            all_scores.append(batch_scores)    #shape: (data_number, args.numclasses)

        all_scores = torch.cat(all_scores, dim=0)

        result = all_scores < label_Quantile
        length_per_data = result.sum(dim=1, keepdim=True)
        length = length_per_data.float().mean()
        #y_labels和
        # coverage = result[:, y_labels].sum().item() / len(result)
        y_labels = torch.tensor(y_labels)
        coverage = result.gather(1, y_labels.view(-1, 1)).sum().item() / len(result)
        return coverage, length


def calc_p(ncal, ngt, neq, smoothing=False):
    if smoothing:
        return (ngt + (neq + 1) * np.random.uniform(0, 1)) / (ncal + 1)
    else:
        return (ngt + neq + 1) / (ncal + 1)
