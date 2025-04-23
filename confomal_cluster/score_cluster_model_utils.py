import numpy as np
import pickle
import os, time, copy
import torch
from torchvision.models import resnet50, resnet34, resnet18, resnet101, resnet152
import torch.optim as optim
import torch.nn as nn
import pdb
import re

from conformal_cluster.score_cluster_train_utils import calculate_class_weights, WeightedMSE, GeodesicDistanceLoss
from conformal_cluster.efficient import effnetv2_s, effnetv2_m, effnetv2_l, effnetv2_xl
from conformal_cluster.cifar100 import ResNet9
from tqdm import tqdm
import sys

# lamb训练策略
# from torchvision.transforms import v2
from conformal_cluster.Lamb_optimizer import Lamb
from torch.nn import functional as F
    # 标签平滑
class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth=0.1, class_num=365):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        ''' 
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)	# softmax + log
            target = F.one_hot(target, self.class_num)	# 转换成one-hot
            
            # label smoothing
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*logprobs, 1)
        else:
            # standard cross entropy loss
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))

        return loss.mean()




class ModelSplitter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 常规的选择
        # self.encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        # self.g = model.fc
        
        # 取最后一层
        self.encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten(), *list(model.fc.children())[:-1])
        self.g = list(model.fc.children())[-1]
        
        self.out_shape = 2048

    def forward(self, x):
        output = self.model(x)
        return torch.squeeze(output)

# efficientnet_v2_l split
class ModelSplitter_Efficientnet_v2_l(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten(), model.classifier[0])
        self.g = model.classifier[1]

    def forward(self, x):
        output = self.model(x)
        return torch.squeeze(output)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    # resnet50
    if isinstance(model.fc, nn.Linear):
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
    elif isinstance(model.fc, nn.Sequential):
        for layer in model.fc:
            if isinstance(layer, nn.Linear):
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True
            # 添加的参数
            elif isinstance(layer, nn.BatchNorm1d):
                if layer.affine:
                    layer.weight.requires_grad = True  # gamma参数
                    layer.bias.requires_grad = True    # beta参数
                layer.track_running_stats = True
    # EfficientNetV2
    # model.classifier[1].weight.requires_grad = True
    # model.classifier[1].bias.requires_grad = True

def train_model_cifar100(model, dataloaders, args):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.

    device = torch.device("cpu") if args.device < 0 else torch.device("cuda")
    set_parameter_requires_grad(model, args.feature_extract)

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # The above prints show which layers are being optimized
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params_to_update, lr=args.lr)

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
            # overfitting
            # if phase == 'train' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'train':
            #     val_acc_history.append(epoch_acc.item())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save best model weights
    save_path = os.path.join("./ckpt", args.data, 
                             f"{args.model_filename}_{args.feature_extract}_{args.num_classes}_{args.batch_size}_{args.epochs}_{args.frac_val}_{args.lr}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model_wts, save_path)
    
    # valdata_path = os.path.join("./result", f"{args.model_filename}-valdata_frac={args.frac_val}.npy")
    # vallabels_path = os.path.join("./result", f"{args.model_filename}-vallabels_frac={args.frac_val}.npy")
    # os.makedirs(os.path.dirname(valdata_path), exist_ok=True)
    # np.save(valdata_path, dataloaders['val'].dataset.tensors[0].numpy())
    # np.save(vallabels_path, dataloaders['val'].dataset.tensors[1].numpy())

    return model, val_acc_history

def train_model_cifar100_sony(model, dataloaders, args):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    """
    使用回归数据进行训练
    改变fc层,多加一层?
    """
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda")
    set_parameter_requires_grad(model, args.feature_extract)

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # The above prints show which layers are being optimized
    print("begin calculate class weights")
    # class_weights = calculate_class_weights(dataloaders["val"], args.num_classes)
    print("calculate done ...")
    # class_weights = torch.tensor(class_weights, device=device)
    # criterion = WeightedMSE(class_weights)
    criterion = WeightedMSE()
    # criterion = nn.CrossEntropyLoss() 
    
# Test 测地距离损失
    # criterion_geodesic = GeodesicDistanceLoss(model, criterion, step=2, num_copies=8, noise_std=0.1, inv_lr=0.01)
    # from conformal_cluster.score_cluster_data_utils import label_to_onehotlabel
    # label_list = np.arange(args.num_classes)  #label_list是calculator data
    # label_transform = label_to_onehotlabel(args.num_classes)
    # label_list, label_list_onehot = label_transform(label_list)
    # all_onehot_labels = label_list_onehot.to(dtype=torch.float, device=device)


# 1）优化器1
    optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# 2）优化器2
    # optimizer = optim.SGD(params_to_update,lr=args.lr, momentum=0.9, weight_decay=0.05)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.2)
    
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, discrete_label, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                discrete_label = discrete_label.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

# #Test 加上测地距离损失    
#                     if  phase == "train": 
#                         loss_add = torch.mean(torch.stack([criterion_geodesic(input, label) for input, label in zip(inputs, labels)]))
#                         loss_min = torch.mean(torch.stack(([torch.mean(torch.stack([criterion_geodesic(input, all_onehot_label) for all_onehot_label in all_onehot_labels])) for input in inputs])))
#                         # print(loss)
#                         # print(loss_add)
#                         # print(loss_min)                        
#                         # sys.exit()
#                         loss = loss - loss_add * 10000 + loss_min * 100000
#Test 加上测地距离损失

                    _, preds = torch.min(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == discrete_label.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
        
        scheduler.step()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save best model weights
    save_path = os.path.join("./ckpt", args.data, 
                             f"{args.model_filename}_{args.feature_extract}_{args.num_classes}_{args.batch_size}_{args.epochs}_{args.frac_val}_{args.lr}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model_wts, save_path)
    
    # valdata_path = os.path.join("./result", f"{args.model_filename}-valdata_frac={args.frac_val}.npy")
    # vallabels_path = os.path.join("./result", f"{args.model_filename}-vallabels_frac={args.frac_val}.npy")
    # os.makedirs(os.path.dirname(valdata_path), exist_ok=True)
    # np.save(valdata_path, dataloaders['val'].dataset.tensors[0].numpy())
    # np.save(vallabels_path, dataloaders['val'].dataset.tensors[1].numpy())

    return model, val_acc_history


def train_model_places365_sony(model, dataloaders, args):

    device = torch.device("cpu") if args.device < 0 else torch.device("cuda")
    set_parameter_requires_grad(model, args.feature_extract)

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # The above prints show which layers are being optimized
    print("begin calculate class weights")
    # class_weights = calculate_class_weights(dataloaders["val"], args.num_classes)
    print("calculate done ...")
    # class_weights = torch.tensor(class_weights, device=device)
    # criterion = WeightedMSE(class_weights)
# 损失函数
    # criterion = WeightedMSE()
    # criterion = nn.CrossEntropyLoss() 
# 训练策略，labm,标签平滑的损失CEL
    criterion = CELoss(label_smooth=0.1, class_num=args.num_classes)

# 1）优化器1
    # optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# 2）优化器2
    # optimizer = optim.SGD(params_to_update,lr=args.lr, momentum=0.9, weight_decay=0.05)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.2)
    
# 3）优化器3--lamb
    optimizer = Lamb(params_to_update, lr=args.lr, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.2)
    
    
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 训练策略lamb
            # cutmix = v2.CutMix(num_classes=args.num_classes)
            # mixup = v2.MixUp(num_classes=args.num_classes)
            # cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            
            # Iterate over data.
            for inputs, discrete_label, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                discrete_label = discrete_label.to(device)
                
                # 训练策略lamb
                # inputs, discrete_label = cutmix_or_mixup(inputs, discrete_label)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
# 回归
                    # loss = criterion(outputs, labels)
# 分类
                    loss = criterion(outputs, discrete_label)
# 标签平滑损失

                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == discrete_label.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
        
        scheduler.step()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save best model weights
    save_path = os.path.join("./ckpt", args.data, 
                             f"{args.model_filename}_{args.feature_extract}_{args.num_classes}_{args.batch_size}_{args.epochs}_{args.frac_val}_{args.lr}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model_wts, save_path)
    
    # valdata_path = os.path.join("./result", f"{args.model_filename}-valdata_frac={args.frac_val}.npy")
    # vallabels_path = os.path.join("./result", f"{args.model_filename}-vallabels_frac={args.frac_val}.npy")
    # os.makedirs(os.path.dirname(valdata_path), exist_ok=True)
    # np.save(valdata_path, dataloaders['val'].dataset.tensors[0].numpy())
    # np.save(vallabels_path, dataloaders['val'].dataset.tensors[1].numpy())

    return model, val_acc_history

def train_model_iNaturalist_sony(model, dataloaders, args):

    device = torch.device("cpu") if args.device < 0 else torch.device("cuda")
    set_parameter_requires_grad(model, args.feature_extract)

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

# 损失函数
    # criterion = WeightedMSE()
# 分类损失
    # criterion = nn.CrossEntropyLoss() 
# 训练策略，labm,标签平滑的损失CEL
    criterion = CELoss(label_smooth=0.1, class_num=args.num_classes)

# 1）优化器1
    # optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# 2）优化器2
    # optimizer = optim.SGD(params_to_update,lr=args.lr, momentum=0.9, weight_decay=0.05)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.2)
    
# 3）优化器3--lamb
    optimizer = Lamb(params_to_update, lr=args.lr, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.2)
    
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
        # for phase in ['val']:
            print(f"{phase}")
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 训练策略lamb
            # cutmix = v2.CutMix(num_classes=args.num_classes)
            # mixup = v2.MixUp(num_classes=args.num_classes)
            # cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            
            # Iterate over data.
            idx = 0
            print(f"idx：{idx}")

            for inputs, discrete_label, _ in tqdm(dataloaders[phase]):                   
                
                inputs = inputs.to(device)
                discrete_label = discrete_label.to(device)
                
                # 训练策略lamb
                # inputs, discrete_label = cutmix_or_mixup(inputs, discrete_label)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    # print(outputs.shape)  # 应该是 (batch_size, num_classes)
                    # print(discrete_label.shape)  # 应该是 (batch_size,)
# 回归
                    # loss = criterion(outputs, labels)
# 分类
                    loss = criterion(outputs, discrete_label)
# 标签平滑损失

                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == discrete_label.data)
    
    # 减少测试数据
                if phase == 'val':
                    idx += 1
                    if idx == 66:
                         break
            # raw         
            # epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            elif phase == 'val':
                length = torch.tensor(66 * args.batch_size, dtype=torch.float)
                epoch_loss = running_loss / length
                epoch_acc = running_corrects.double() / length
                
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
        
        if phase == 'train':  # 只在训练阶段调用 scheduler
            scheduler.step()  # 更新学习率
        # torch.cuda.empty_cache()
        print('one epoch is done ...')
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save best model weights
    save_path = os.path.join("./ckpt", args.data, 
                             f"{args.model_filename}_{args.feature_extract}_{args.num_classes}_{args.batch_size}_{args.epochs}_{args.frac_val}_{args.lr}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model_wts, save_path)
    
    # valdata_path = os.path.join("./result", f"{args.model_filename}-valdata_frac={args.frac_val}.npy")
    # vallabels_path = os.path.join("./result", f"{args.model_filename}-vallabels_frac={args.frac_val}.npy")
    # os.makedirs(os.path.dirname(valdata_path), exist_ok=True)
    # np.save(valdata_path, dataloaders['val'].dataset.tensors[0].numpy())
    # np.save(vallabels_path, dataloaders['val'].dataset.tensors[1].numpy())

    return model, val_acc_history

def get_model(dataloaders, args):
    # pattern = re.compile(
    #     rf"{args.model_filename}_{args.feature_extract}_(?P<num_classes>\d+)_(?P<batch_size>\d+)_(?P<epochs>\d+)_(?P<frac_val>[\d.]+)_(?P<lr>[\d.]+).pth"
    # )
    pattern = re.compile(
        rf"{args.model_filename}_{args.feature_extract}_(?P<num_classes>\d+)_.*?_(?P<epochs>\d+)_(?P<frac_val>[\d.]+)_(?P<lr>[\d.]+).pth"
    )
    
    # 权重文件目录
    ckpt_dir = os.path.join("./ckpt", args.data)
    
    # 查找是否有匹配的权重文件
    matched_file = None
    for file_name in os.listdir(ckpt_dir):
        match = pattern.match(file_name)
        if match:
            file_params = match.groupdict()
            print(file_params)
            if (file_name.startswith(f"{args.model_filename}_{args.feature_extract}") and
                int(file_params['num_classes']) == args.num_classes and
                # int(file_params['batch_size']) == args.batch_size and
                int(file_params['epochs']) == args.epochs and
                float(file_params['frac_val']) == args.frac_val and
                float(file_params['lr']) == args.lr):
                matched_file = os.path.join(ckpt_dir, file_name)
                break

    if args.data == "cifar100":
        model = resnet50()
        state_dict = torch.load("./ckpt/cifar100/IMAGENET1K_V2_resnet50.pth")
        model.load_state_dict(state_dict, strict=False)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(512, args.num_classes))
        # 过拟合模拟：
        # model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 512),
        #                     nn.BatchNorm1d(512),
        #                     nn.ReLU(),
        #                     nn.Dropout(p=0),
        #                     nn.Linear(512, args.num_classes))
        # state_dict = torch.load("./ckpt/cifar100/best-cifar100-model_True_100_128_3_0.1_0.01.pth")
        # model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load("./ckpt/cifar100/best-cifar100-model_True_100_128_100_0.1_0.01.pth"))
        # model.load_state_dict(torch.load("./ckpt/cifar100/resnet50cifar100.pth"))  # Acc: 0.272333
        # model.load_state_dict(torch.load("./ckpt/cifar100/best-cifar100-model_False_100_128_40_0.1_0.0001.pth"))  # Acc: 0.95, 0.80
        print("load done ...")
        # sys.exit()
    elif args.data == 'places365':
        model = resnet50(num_classes=365)
    
        # 使用places365预训练的resnet50权重
        checkpoint = torch.load("./ckpt/places365/resnet50_places365.pth.tar", map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)

        # 修改model的fc层
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(512, args.num_classes))
        
        # # 导入fc层微调后的权重
        finetune_state_dict = torch.load("./ckpt/places365/best-places365-model_True_365_512_6_0.5_0.01.pth")
        model.load_state_dict(finetune_state_dict)
        print(f"load model successfully ....")
        
        # print(f"测试精度....")
        # for x, y, onehot_y in dataloaders['val']:
        #     outputs = model(x)
        #     _, preds = torch.min(outputs, 1)
        #     result = torch.sum(preds == y)
        #     print(result / len(y))
        #     sys.exit()
    elif args.data == 'iNaturalist':
        model = resnet50()
        state_dict = torch.load("./ckpt/cifar100/IMAGENET1K_V2_resnet50.pth")
        model.load_state_dict(state_dict, strict=False)
        # 修改model的fc层
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        chekcpoint = torch.load("./ckpt/iNaturalist/best-iNaturalist-model_True_925_512_1_0.5_0.0001.pth")
        model.load_state_dict(chekcpoint)
        # 两层FC层
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(1024, args.num_classes))
        
    device_cluster = torch.device("cpu") if args.device < 0 else torch.device("cuda")
    model = model.to(device_cluster)
    
    if matched_file:
        # 加载匹配的权重文件
        print(f"Loading model from {matched_file}")
        state_dict = torch.load(matched_file, map_location=device_cluster)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model

    # 如果没有找到匹配的权重文件，重新训练模型
    print("No matching model found, retraining...")
    if args.data == "cifar100":
        model, val_acc_history = train_model_cifar100(model, dataloaders, args)     #class
        # model, val_acc_history = train_model_cifar100_sony(model, dataloaders, args)  #regression
    elif args.data == "places365":
        model, val_acc_history = train_model_places365_sony(model, dataloaders, args)  #regression
    elif args.data == "iNaturalist":
        model, val_acc_history = train_model_iNaturalist_sony(model, dataloaders, args)  #regression
    
    return model
if __name__=="__main__":
    model = resnet18()
    print(model)
