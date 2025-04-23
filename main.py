import warnings
import torch
from tqdm import tqdm
import os
import numpy as np
np.warnings.filterwarnings('ignore')

from datasets_cluster import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse

# Cluster FCP
from conformal_cluster import helper as helper_cluster
from conformal_cluster.icp import IcpRegressor as IcpRegressor_cluster, RegressorNc as RegressorNc_cluster
from conformal_cluster.icp import FeatRegressorNc as FeatRegressorNc_cluster
from conformal_cluster.icp import AbsErrorErrFunc, FeatErrorErrFunc
from conformal_cluster.utils import compute_coverage as compute_coverage_cluster, seed_torch, compute_coverage_sony

# score cluster packages
from conformal_cluster.score_cluster_data_utils import get_cifar100_dataloaders, get_places365_dataloaders, split_calibration_test, group_data_by_label, label_to_onehotlabel
from conformal_cluster.score_cluster_data_utils import get_places365_dataloaders_sony, get_dataloaders_iNaturalist, get_dataloaders_iNaturalist_sony
from conformal_cluster.score_cluster_data_utils import get_dataloaders_iNaturalist_sony_1, get_dataloaders_iNaturalist_sony_2, split_cal_and_val_iNaturalist
from conformal_cluster.score_cluster_model_utils import get_model, ModelSplitter
from conformal_cluster.score_cluster_train_utils import WeightedMSE, calculate_class_weights
from conformal_cluster.score_cluster_utils import get_rare_classes, remap_classes, quantile_embedding
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import sys
import pdb
import torchvision
from torchvision import models
warnings.filterwarnings("ignore", category=DeprecationWarning)

def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")

# cluster_fcp
def cluster_fcp(dataloaders_dict, args):
    dir = f"ckpt/{args.data}"
    os.makedirs(dir, exist_ok=True)
    if args.data == "cifar100":
        print("begin load model ...")
        print("*" * 100)
        model_cluster = get_model(dataloaders_dict, args)
        # sys.exit()
        model = ModelSplitter(model_cluster)
    elif args.data == "places365":
        print("begin load model ...")
        print("*" * 100)
        model_cluster = get_model(dataloaders_dict, args)
        # sys.exit()
        model = ModelSplitter(model_cluster)
    elif args.data == "iNaturalist":
        print("begin load model ...")
        print("*" * 100)
        model_cluster = get_model(dataloaders_dict, args)
        print(f"model is ready....")
        # sys.exit()
        model = ModelSplitter(model_cluster)
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda")
    mean_estimator = helper_cluster.MSENet_RegressorAdapter_sony(model=model, device=device, fit_params=None, )

    if float(args.feat_norm) <= 0 or args.feat_norm == "inf":
        args.feat_norm = "inf"
        print("Use inf as feature norm")
    else:
        args.feat_norm = float(args.feat_norm)
        print(f"Use {args.feat_norm} as feature norm")
        
    criterion = torch.nn.CrossEntropyLoss()
    
    
    nc = FeatRegressorNc_cluster(mean_estimator, inv_lr=args.feat_lr, inv_step=args.feat_step, criterion=criterion,
                         feat_norm=args.feat_norm, certification_method=args.cert_method)
    icp = IcpRegressor_cluster(nc)

# 将val数据区分为校准数据集和测试数据集
    
    if args.data == "iNaturalist":
        calibration_loader, test_loader = split_cal_and_val_iNaturalist(dataloaders_dict['val'], args.n_avg, args.num_classes, seed=0, split='balanced')
    else:
        calibration_size = args.n_avg * args.num_classes
        calibration_loader, test_loader = split_calibration_test(dataloaders_dict['val'], calibration_size, args.batch_size, args.num_workers)    
    print(f"calibration_loader length: {len(calibration_loader.dataset)}, test_loader length: {len(test_loader.dataset)}")
   
        
# 1) 按照类别计算scores
    grouped_data = group_data_by_label(calibration_loader.dataset)   #{0：[(image, label, one_hot_label)]}, type: tensor

    cal_scores_per_class = defaultdict(list)

    for label in tqdm(grouped_data, 
                 desc="Calculate all classes", 
                 total=len(grouped_data),
                 unit="class",
                 colour='blue'):
        data = grouped_data[label]
        
# 1.2) 按照类别，使用测地距离计算计算scores
        cal_scores = icp.calibrate_per_class_Geodesic_Distance(data)
        cal_scores_per_class[label] = cal_scores   #针对每个class，其scores是list

# 2) 获得数量较少的类别
    labels = [label.numpy() for _, label, _ in  calibration_loader.dataset]
    # labels = [np.array(label) for _, label, _ in  calibration_loader.dataset]
    rare_classes = get_rare_classes(labels, args.alpha, args.num_classes)
    print(f"rare_classes: {rare_classes}")


# 3) score cluster
# 输出每个簇的class
    # q=[0.5, 0.6, 0.7, 0.8, 0.9]
    q=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]    
    score_embeddings = np.zeros((args.num_classes, len(q)))
    class_number = []
    label_list = []
    if not rare_classes:
        print(f"rare_classes is None")
        for i, label in enumerate(cal_scores_per_class):
            label_list.append(label)
            score_embedding = quantile_embedding(np.array(cal_scores_per_class[label]), q)
            score_embeddings[i, :] = score_embedding
            class_number.append(len(cal_scores_per_class[label]))
        print(score_embeddings.shape)
        kmeans = KMeans(n_clusters=int(args.num_clusters), random_state=0, n_init=60).fit(score_embeddings, sample_weight=np.sqrt(class_number))
        nonrare_class_cluster_assignments = kmeans.labels_  
        print(f'Cluster sizes:', [x[1] for x in Counter(nonrare_class_cluster_assignments).most_common()])
    else:
        print("rare_classes is not None")
        sys.exit()
    
# 3.2) 构建label和聚类簇的映射
    label_to_cluster = {}
    cluster_to_labels = defaultdict(list)
    for label, cluster in zip(label_list, nonrare_class_cluster_assignments):
        label_to_cluster[label] = cluster
        cluster_to_labels[cluster].append(label)

# 3.3) 获得每个cluster簇的scores
    cal_scores_per_cluster = defaultdict(list)
    for cluster in cluster_to_labels:
        cluster_labels = cluster_to_labels[cluster]
        for label in cluster_labels:
            cal_scores_per_cluster[cluster].extend(cal_scores_per_class[label])

# 3.4）计算每种cluster的Quantile及每种class的Quantile
    cluster_Quantile = {}
    print(f"use quantile: {1- args.alpha}")
    for cluster in cal_scores_per_cluster:
        cluster_scores = cal_scores_per_cluster[cluster]

        scores_array = np.array(cluster_scores)
        quantile_values = np.quantile(scores_array, (1-args.alpha))
        cluster_Quantile[cluster] = quantile_values
    # print(f"cluster_Quantile: {cluster_Quantile}")

    class_Quantile = []
    for i in range(args.num_classes):
        cluster = label_to_cluster[i]
        class_Quantile.append(cluster_Quantile[cluster])        
    class_Quantile = np.array(class_Quantile)
    
# 4.1.2)计算所有可能label的discreate_label----分类
    label_list = np.arange(args.num_classes)  #label_list是calculator data
    onehot_labels = torch.tensor(label_list)
   
# 4.2)针对一个测试数据，计算其针对所有可能onehot_label的scores，最后每个数据生成args.num_classes个scores 
    all_scores = icp.calibrate_all_labels_Geodesic_Distance_sony(test_loader, onehot_labels)   #all_scores: (label.item(), scores)


# 5）针对测试数据的计算结果（scores），求预测区间，并最后计算指标

# 5.1) 计算预测区间
    prediction_intervals = []
    for label, scores in all_scores:
        scores_array = np.array(scores)
        prediction_labels = np.array(scores_array < class_Quantile).nonzero()[0]
        prediction_intervals.append((label, prediction_labels))
        
# 5.2) 计算coverage和length
    total_length = 0
    for label, prediction_labels in prediction_intervals:
        total_length += len(prediction_labels)

    length_cluster_fcp = total_length / len(prediction_intervals)
    print(f"length_cluster_fcp: {length_cluster_fcp}")
    
    coverage_list = []
    for label, prediction_labels in prediction_intervals:
        if int(label) in set(prediction_labels):
            coverage_list.append(1)
        else:
            coverage_list.append(0)

    coverage_cluster_fcp = sum(coverage_list) / len(coverage_list)
    print(f"coverage_cluster_fcp: {coverage_cluster_fcp}")

    # 统计每个类别的覆盖数和总样本数
    class_covered = defaultdict(int)  # 每个类别被覆盖的样本数
    class_total = defaultdict(int)    # 每个类别的总样本数
    class_lengths = defaultdict(list) # 存储每个类别的预测区间长度

    for label, prediction_labels in prediction_intervals:
        class_label = int(label)
        class_total[class_label] += 1
        if class_label in set(prediction_labels):
            class_covered[class_label] += 1
        
        # 记录每个类别的预测区间长度
        class_lengths[class_label].append(len(prediction_labels))

    
     # 计算每个类别的coverage
    class_coverage = {}
    for class_label, total in class_total.items():
        class_coverage[class_label] = class_covered[class_label] / total
        
    coverage_values = np.array(list(class_coverage.values()))
    target = 1 - args.alpha  # 提前计算目标值
    CovGap = 100 * np.abs(coverage_values - target).mean()  # 向量化计算
    
    return coverage_cluster_fcp, length_cluster_fcp, CovGap
    

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=0, type=int)
    parser.add_argument('--seed', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument("--data", type=str, default="community", help="meps20 fb1 fb2 blog cifar100 iNaturalist Places365 ImageNet")

    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage error")

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", "--bs", type=int, default=128)
    parser.add_argument("--hidden_size", "--hs", type=int, default=64)
    parser.add_argument("--dropout", "--do", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--no-resume", action="store_true", default=False)

    parser.add_argument("--feat_opt", "--fo", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--feat_lr", "--fl", type=float, default=1e-2)
    parser.add_argument("--feat_step", "--fs", type=int, default=None)
    parser.add_argument("--feat_norm", "--fn", default=-1)
    parser.add_argument("--cert_method", "--cm", type=int, default=0, choices=[0, 1, 2, 3])
    
    # conformal score cluster data_revelant
    parser.add_argument("--num_workers", "--nw", type=int, default=10)
    parser.add_argument("--frac_val", "--fv", type=float, default=0.5, help="val_data proportion")
    parser.add_argument('--target_type', type=str, default='family',
                    help="Only used when dataset==iNaturalist. Options are ['full', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'] ")
    parser.add_argument('--min_train_instances', type=int, default=250,
                    help='Classes with fewer than this many classes in the published train dataset will be filtered out')
    parser.add_argument("--n_avg", type=int, default=30, help="The amount of data for each class")
    
    # conformal score cluster model_revelant
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--model_filename", default="best-cifar100-model", choices=['best-cifar100-model', 'best-iNaturalist-model', 'best-places365-model'])
    parser.add_argument("--feature_extract", action="store_true", default=False, help="False means update all parameters, True means update fc linear parameters")
    
    # conformal score cluster cluster_revelant
    parser.add_argument("--num_clusters", type=int, default=6)
    args = parser.parse_args()

    fcp_coverage_list, fcp_length_list, cp_coverage_list, cp_length_list = [], [], [], []
    if args.data == "cifar100":
        print(f"load cifar100 data ...")
        dataloaders_dict = get_cifar100_dataloaders(args.batch_size, args.frac_val, args.num_classes, args.num_workers)
    elif args.data == "iNaturalist":
        print(f"load iNaturalist data")
        dataloaders_dict = get_dataloaders_iNaturalist_sony_2(args.batch_size, args.num_classes, args.target_type, args.min_train_instances, args.num_workers, args.frac_val)
    elif args.data == "places365":
        print(f"load places365 data")
        dataloaders_dict = get_places365_dataloaders_sony(args.batch_size, args.num_classes, args.num_workers) 

    
    print("load data done ...")
    print("cluster-fcp begin ....")         
    print("*" * 100)
    coverage_cluster_fcp_list, length_cluster_fcp_list, CovGap_list = [], [], []
    for seed in tqdm(args.seed):
        seed_torch(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)     
           
        coverage_cluster_fcp, length_cluster_fcp, CovGap = cluster_fcp(dataloaders_dict, args)   
                
        if isinstance(coverage_cluster_fcp, torch.Tensor):
            coverage_cluster_fcp = coverage_cluster_fcp.detach().cpu().numpy()
        elif not isinstance(coverage_cluster_fcp, np.ndarray):
            coverage_cluster_fcp = np.asarray(coverage_cluster_fcp, dtype=np.float64)
            
        if isinstance(length_cluster_fcp, torch.Tensor):
            length_cluster_fcp = length_cluster_fcp.detach().cpu().numpy()
        elif not isinstance(length_cluster_fcp, np.ndarray):
            length_cluster_fcp = np.asarray(length_cluster_fcp, dtype=np.float64)
        
        coverage_cluster_fcp_list.append(coverage_cluster_fcp)    
        length_cluster_fcp_list.append(length_cluster_fcp)
        CovGap_list.append(CovGap)
        
        print(f"Cluster_FCP coverage: {coverage_cluster_fcp}, Cluster_FCP length: {length_cluster_fcp}")

    print(f'Cluster_FCP coverage: {np.mean(coverage_cluster_fcp_list)} \pm {np.std(coverage_cluster_fcp_list)}',
          f'Cluster_FC length: {np.mean(length_cluster_fcp_list)} \pm {np.std(length_cluster_fcp_list)}',
          f'CovGap: {np.mean(CovGap_list)} \pm {np.std(CovGap_list)}')
    
    