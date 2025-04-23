import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, time, copy
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, Subset
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pdb

# 添加
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import gc
import random


# 区分校准数据和测试数据集
def split_calibration_test(dataloader, calibration_size, batch_size, num_workers=1):
    # Get total number of samples
    total_size = len(dataloader.dataset)
    
    # Generate indices and split into calibration and test sets
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    calibration_indices = indices[:calibration_size]
    test_indices = indices[calibration_size:]

    # Create subsets
    calibration_dataset = Subset(dataloader.dataset, calibration_indices)
    test_dataset = Subset(dataloader.dataset, test_indices)

    # Create data loaders
    calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return calibration_loader, test_loader

# 将label转换为Onthot_label
class label_to_onehotlabel():
    epsilon = 1e-6
    def __init__(self, num_classes, noise_std=None):
        self.num_classes = num_classes
        self.noise_std = noise_std
    def convert_onehot(self, label):  # label (num_data,)
        if isinstance(label, int):
            label = torch.tensor(label)
            ones = torch.eye(self.num_classes).float()
            ones = ones * (1 - self.epsilon) + (1 - ones) * self.epsilon
            onehots = ones.index_select(0, label)
        else:
            label = label.view(-1)  # (num_data,)
            ones = torch.eye(self.num_classes).float()
            ones = ones * (1 - self.epsilon) + (1 - ones) * self.epsilon
            onehots = ones.index_select(0, label.to(dtype=torch.int32))
        return onehots
# 加上高斯噪声模糊
    def __call__(self, label):
        if isinstance(label, np.ndarray):
            label = torch.tensor(label)
        onehots = self.convert_onehot(label)  # (self.num_classes, H, W)
        if self.noise_std is not None:
            noise = torch.normal(mean=0, std=self.noise_std, size=onehots.shape)
            onehots = onehots + noise
        double_log_onehots = torch.log(-torch.log(onehots))
        return label, double_log_onehots

class label_to_onehotlabel_sony():
    """
    Returns:0,1 Onehot编码
    """
    epsilon = 1e-6
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def convert_onehot(self, label):  # label (num_data,)
        if isinstance(label, int):
            label = torch.tensor(label)
            ones = torch.eye(self.num_classes).long()
            onehots = ones.index_select(0, label)
        else:
            label = label.view(-1)  # (num_data,)
            ones = torch.eye(self.num_classes).long()
            onehots = ones.index_select(0, label.to(dtype=torch.int64))
        return onehots
# 加上高斯噪声模糊
    def __call__(self, label):
        if isinstance(label, np.ndarray):
            label = torch.tensor(label)
        onehots = self.convert_onehot(label)  # (self.num_classes, H, W)
        return label, onehots


# 获得CIFAR-100的dataloader
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_cifar100_data():
    """
    class condition conformal(cluster CP)使用imagenet预训练的resnet50作为模型，然后用32*32的cifar100数据进行微调
    """
    # Load and unpack data
    orig_train_data = unpickle('./datasets_cluster/cifar-100-python/train')
    orig_test_data = unpickle('./datasets_cluster/cifar-100-python/test')    
    
    tr_imgs = orig_train_data[b'data'].astype(np.float32)
    te_imgs = orig_test_data[b'data'].astype(np.float32)
    tr_labels = torch.tensor(np.array(orig_train_data[b'fine_labels']).astype(int))
    te_labels = torch.tensor(np.array(orig_test_data[b'fine_labels']).astype(int))

    # Fuse train and val sets
    imgs = np.concatenate([tr_imgs, te_imgs], axis=0)
    labels = np.concatenate([tr_labels, te_labels], axis=0)

    imgs = imgs.reshape(imgs.shape[0], 3, 32, 32)
    total_pixels_per_channel = imgs.shape[0] * imgs.shape[2] * imgs.shape[3] 
    means = imgs.sum(axis=2).sum(axis=2).sum(axis=0) / total_pixels_per_channel
    stds = np.sqrt(((imgs - means[None,:,None,None])**2).sum(axis=2).sum(axis=2).sum(axis=0)/total_pixels_per_channel)
    imgs = (imgs - means[None,:,None,None])/stds[None,:,None,None]
    return imgs, labels

def get_cifar100_data_transform():
    """
    将cifar100(3*32*32)转换为(3*224*224),以适应ImageNet的预训练模型
    返回的imgs是一个包含了60000个tensor(3*224*224)的np.ndarray
    """
    # Load and unpack data
    orig_train_data = unpickle('./datasets_cluster/cifar-100-python/train')
    orig_test_data = unpickle('./datasets_cluster/cifar-100-python/test')    
    
    tr_imgs = orig_train_data[b'data']
    te_imgs = orig_test_data[b'data']
    tr_labels = torch.tensor(np.array(orig_train_data[b'fine_labels']).astype(int))
    te_labels = torch.tensor(np.array(orig_test_data[b'fine_labels']).astype(int))

    # Fuse train and val sets
    imgs = np.concatenate([tr_imgs, te_imgs], axis=0)
    labels = np.concatenate([tr_labels, te_labels], axis=0)
    
    # Reshape and normalize images to mean 0 std 1
    imgs = imgs.reshape(imgs.shape[0], 3, 32, 32).astype(np.uint8)
    
    # Convert to PIL images
    from PIL import Image
    imgs_pil = [Image.fromarray(img.transpose(1, 2, 0)) for img in imgs]  # (C, H, W) -> (H, W, C)
    
    # Define transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet standards
    ])
    
    # Apply transforms
    imgs = [data_transforms(img) for img in imgs_pil]
    imgs = np.array([img.numpy() for img in imgs])
    return imgs, labels

def get_cifar100_dataloaders(batch_size=128, frac_val=0.1, num_classes=100, num_workers=1):
    
    assert 0 <= frac_val <= 1

    # imgs, labels = get_cifar100_data_transform()
    # imgs, labels = get_cifar100_data()   #imgs : numpy.ndarray, float64
    # train_imgs, val_imgs, train_labels, val_labels = train_test_split(imgs, labels, test_size=frac_val, random_state=0)
    # label_transform = label_to_onehotlabel(num_classes)
    # train_labels, train_labels_onehot = label_transform(train_labels)   #train_labels[0].dtype: torch.int64, train_labels_onehot[0].dtype: torch.float32
    # val_labels, val_labels_onehot = label_transform(val_labels)

    
# 1）是否使用数据增强       

# 1.1) 不使用数据增强，创建的训练数据和测试数据的datasets

    # image_datasets = {
    #     'train' : TensorDataset(torch.tensor(train_imgs).float(), torch.tensor(train_labels).long(), torch.tensor(train_labels_onehot).long()),
    #     'val' : TensorDataset(torch.tensor(val_imgs).float(), torch.tensor(val_labels).long(), torch.tensor(val_labels_onehot).long())}
    
    
# 1.2) 使用数据增强，创建的训练数据和测试数据的datasets

# 1.2.1） 创建自定义Dataset类来应用transforms
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, labels_onehot, transform=None):
            self.images = images
            self.labels = labels
            self.labels_onehot = labels_onehot
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            label_onehot = self.labels_onehot[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label, label_onehot

# 1.2.2） 数据增强的方式
    data_transforms = {
        'train': transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(6)
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.ToTensor()
            ]),
    }
    
# 1.2.3）使用CustomDataset替代TensorDataset，创建的训练数据和测试数据的datasets
    # image_datasets = {
    #     'train': CustomDataset(torch.tensor(train_imgs).float(), torch.tensor(train_labels).long(), torch.tensor(train_labels_onehot).long(), 
    #                            transform=data_transforms['train']),
    #     'val': CustomDataset(torch.tensor(val_imgs).float(), torch.tensor(val_labels).long(), torch.tensor(val_labels_onehot).long(), 
    #                          transform=data_transforms['val'])}


# 2) 创建训练数据和测试数据的dataloaders
    # dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}


# 3) 224*224的cifar100

    label_transform = label_to_onehotlabel(num_classes)
    
    class CIFAR100WithOneHot(tv.datasets.CIFAR100):
        def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
            super().__init__(root=root, train=train, transform=transform, 
                            target_transform=target_transform, download=download)

        def __getitem__(self, index):
            img, label = super().__getitem__(index)
            # 转换为one-hot编码
            discreat_label, onehot_label = label_transform(label)
            
            return img, torch.tensor(label).long(), torch.squeeze(onehot_label).float()
    
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #默认的标准化参数
                                    ])
    
    trainset = CIFAR100WithOneHot(root='../datasets_cluster/cifar100_resnet', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = CIFAR100WithOneHot(root='../datasets_cluster/cifar100_resnet', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloaders_dict = {'train': trainloader, 'val': testloader}
    
    return dataloaders_dict

class Places365WithOneHot(torch.utils.data.Dataset):
    def __init__(self, original_dataset, label_transform):
        self.dataset = original_dataset
        self.label_transform = label_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        one_hot_label = self.label_transform(label)
        return img, torch.tensor(label).to(dtype=torch.int64), one_hot_label[1][0]   #one_hot_label[1]：[[2.6258, 2.6258, ...-13.8023, ...]],[one_hot_label[1][0]:[2.6258, 2.6258, ...-13.8023, ...]

class iNaturalistWithOneHot(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_classes):
        self.dataset = original_dataset
        one_hot_label = torch.ones((num_classes,))
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, torch.tensor(label).to(dtype=torch.int64), one_hot_label   #one_hot_label[1]：[[2.6258, 2.6258, ...-13.8023, ...]],[one_hot_label[1][0]:[2.6258, 2.6258, ...-13.8023, ...]

# 计算均值和方差
def calc_mean_std(my_dataset):
    image_data_loader = torch.utils.data.DataLoader(
      my_dataset,
      batch_size=512, 
      shuffle=True, 
      num_workers=2
    )

    X, y = iter(image_data_loader).__next__()

    mean = X.mean(dim=(0,2,3))
    std = X.std(dim=(0,2,3))
    return mean, std

def get_places365_data(num_classes=365):
    
    transform_img = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    val_data = datasets.Places365(root="/ssdfs/datahome/u06105/sony/datasets_cluster/places365/val", split="val", download=False, transform=transform_img)
    mean, std = calc_mean_std(val_data)
    print(f"calculate mean std done ...")
    
    transform_img = transforms.Compose([transform_img, transforms.Normalize(mean, std)])
    # 加载训练数据和验证数据
    train_data = datasets.Places365(root="/ssdfs/datahome/u06105/sony/datasets_cluster/places365/train_standard", split="train-standard", download=False, transform=transform_img)
    val_data = datasets.Places365(root="/ssdfs/datahome/u06105/sony/datasets_cluster/places365/val", split="val", download=False, transform=transform_img)

    final_train_idxs = np.arange(len(train_data))
    final_val_idxs = np.arange(len(val_data))
    dataset = torch.utils.data.ConcatDataset([Subset(train_data, final_train_idxs), Subset(val_data, final_val_idxs)])
    
    # 原始onehot-lbale
    # label_transform = label_to_onehotlabel(num_classes)
    
    # 0,1 onehot-label
    label_transform = label_to_onehotlabel_sony(num_classes)
    dataset_with_one_hot = Places365WithOneHot(dataset, label_transform)
    
    return dataset_with_one_hot

# places365数据集
def get_places365_dataloaders(batch_size=128, frac_val=0.1, num_classes=100, num_workers=1):
    assert 0 <= frac_val <= 1
    
    print(f"load data begin ....")

# 数据的类型是tensor,是否要换成numpy
    dataset = get_places365_data(num_classes=num_classes)
    
    generator1 = torch.Generator().manual_seed(0) # For reproducibility
    total_length = len(dataset)
    # print(total_length)
    # sys.exit()
    
# 1.1）使用所有数据
    # unused_data_idx = 0
    # used_idx = total_length
# 1.2）使用部分数据
    unused_data_idx = int(total_length * 0.9)
    used_idx = total_length - unused_data_idx
    # print(used_idx)
    # sys.exit()
    
    train_idx = int(used_idx * (1-frac_val))
    val_idx = used_idx - train_idx
    # print(train_idx, val_idx)
    # sys.exit()
    
    unused_data, train, val = torch.utils.data.random_split(dataset, [unused_data_idx, train_idx, val_idx], generator=generator1)

    # Create training and validation datasets
    image_datasets = {
        'train' : train,
        'val' : val
                     }
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    print(f"load data sucessfull ...")

    return dataloaders_dict

def get_places365_data_sony(num_classes=365):
    """
    使用原始的tran_data和val_data，不重新区分
    """
    
    transform_img_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.1),
        # transforms.RandAugment(num_ops=2, magnitude=6),
        # torchvision.transforms.RandAugment
        transforms.ToTensor(),
    ])
    transform_img_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    val_data = datasets.Places365(root="/ssdfs/datahome/u06105/sony/datasets_cluster/places365/val", split="val", download=False, transform=transform_img_val)
    mean, std = calc_mean_std(val_data)
    print(f"calculate mean std done ...")
    
    transform_img_train = transforms.Compose([transform_img_train, transforms.Normalize(mean, std)])
    transform_img_val = transforms.Compose([transform_img_val, transforms.Normalize(mean, std)])
    # 加载训练数据和验证数据
    train_data = datasets.Places365(root="/ssdfs/datahome/u06105/sony/datasets_cluster/places365/train_standard", split="train-standard", download=False, transform=transform_img_train)
    val_data = datasets.Places365(root="/ssdfs/datahome/u06105/sony/datasets_cluster/places365/val", split="val", download=False, transform=transform_img_val)

    label_transform = label_to_onehotlabel(num_classes)
    train_data_with_onehot = Places365WithOneHot(train_data, label_transform)
    val_data_with_onehot = Places365WithOneHot(val_data, label_transform)
    
    return train_data_with_onehot, val_data_with_onehot

def get_places365_dataloaders_sony(batch_size=128,  num_classes=365, num_workers=1):

    print(f"load data begin ....")

# 数据的类型是tensor,是否要换成numpy
    train_data_with_onehot, val_data_with_onehot = get_places365_data_sony(num_classes=num_classes)

    # Create training and validation datasets
    image_datasets = {
        'train' : train_data_with_onehot,
        'val' : val_data_with_onehot
                     }
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    print(f"load data successfully ...")

    return dataloaders_dict

def group_data_by_label(dataset):
        grouped_data = defaultdict(list)
        # 遍历 dataset 中的所有数据
        for img, label, onehot_label in tqdm(dataset,
                                            desc="Group data", 
                                            total=len(dataset),
                                            colour='blue'):
            if isinstance(label, int):
                grouped_data[label].append((img, label, onehot_label))
            else:
                grouped_data[label.item()].append((img, label, onehot_label))
        
        return grouped_data

# 获得inaturalist与places365数据集
def calc_mean_std(my_dataset):
    image_data_loader = torch.utils.data.DataLoader(
      my_dataset,
      batch_size=512, 
      shuffle=True, 
      num_workers=2
    )

    X, y = iter(image_data_loader).__next__()

    mean = X.mean(dim=(0,2,3))
    std = X.std(dim=(0,2,3))
    return mean, std

def load_and_process_dataset(dset_fn, target_fn, min_train_instances_class):
    # Standard transform
    transform_img = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # here do not use transforms.Normalize(mean, std)
    ])
    train_dataset, val_dataset = dset_fn(transform_img)
    
    # Calculate mean and std dev to use when normalizing
    mean, std = calc_mean_std(val_dataset)
    
    # Filter out rare classes 
    if min_train_instances_class > 0:  
        train_targets = target_fn(train_dataset)
        val_targets = target_fn(val_dataset)
        unique_classes, counts = np.unique(train_targets, return_counts=True)
        counts_large_enough = counts >= min_train_instances_class
        final_classes = unique_classes[ counts_large_enough ]
        final_train_idxs = np.where( np.isin(train_targets, final_classes) )[0]
        final_val_idxs = np.where( np.isin(val_targets, final_classes) )[0]

        # Map class labels to consecutive 0,1,2,...
        label_remapping = {}
        idx = 0
        for k in final_classes:
            label_remapping[k] = idx
            idx += 1
        def transform_label(k):
            return label_remapping[k]
        target_transform = transform_label
        
    else:
        final_train_idxs = np.arange(len(train_dataset))
        final_val_idxs = np.arange(len(val_dataset))
        target_transform = None
    
    transform_img = transforms.Compose([transform_img, transforms.Normalize(mean, std)])
    
    train_dataset, val_dataset = dset_fn(transform_img, target_transform=target_transform)

    dataset = torch.utils.data.ConcatDataset([Subset(train_dataset, final_train_idxs), Subset(val_dataset, final_val_idxs)])

    return dataset

# target_type choices: ['full', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
def get_dataloaders_iNaturalist(dataset='iNaturalist', batch_size=128, frac_val=0.1, target_type="full", min_train_instances_class=0, num_workers=1):
    # Load and unpack data
    if dataset == 'iNaturalist':
        def dset_fn(transform_img, target_transform=None):
            train_dataset = tv.datasets.INaturalist(root = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/train', 
                                                 version = '2021_train', 
                                                 download=False, 
                                                 target_type = target_type,
                                                 transform=transform_img,
                                                 target_transform=target_transform) 
            val_dataset = datasets.INaturalist(root = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/val', 
                                               version = '2021_valid', 
                                               download=False, 
                                               target_type = target_type,
                                               transform=transform_img, 
                                               target_transform=target_transform) 
            return train_dataset, val_dataset
        def target_fn(dset):
            return np.array([x[0] for x in dset.index])

    dataset = load_and_process_dataset(dset_fn, target_fn, min_train_instances_class)

    assert 0 <= frac_val <= 1

    generator1 = torch.Generator().manual_seed(0) # For reproducibility
    train, val = torch.utils.data.random_split(dataset, [1-frac_val, frac_val], generator=generator1)
    
    # Create training and validation datasets
    image_datasets = {
        'train' : train,
        'val' : val
                     }
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}

    return dataloaders_dict

def get_dataloaders_iNaturalist_sony(batch_size=128, num_classes=633, target_type="family", min_train_instances=290, num_workers=10, frac_val=0.5):
    transform_img = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    target_transform = None
    train_dataset = datasets.INaturalist(root = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/train', 
                                         version = '2021_train', 
                                         download=False, 
                                         target_type = target_type,
                                         transform=transform_img,
                                         target_transform=target_transform)
    val_dataset = datasets.INaturalist(root = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/val', 
                                       version = '2021_valid', 
                                       download=False, 
                                       target_type = target_type,
                                       transform=transform_img, 
                                       target_transform=target_transform)
    mean, std = calc_mean_std(val_dataset)
    print(hasattr(train_dataset, 'targets'))
    sys.exit()
    print(f"num_classes:{num_classes}, target_type={target_type}, min_train_instances={min_train_instances}")

# 筛选数据

    train_targets = np.array([x[0] for x in train_dataset.index])
    val_targets = np.array([x[0] for x in val_dataset.index])
    unique_classes, counts = np.unique(train_targets, return_counts=True)
    counts_large_enough = counts >= min_train_instances
    final_classes = unique_classes[ counts_large_enough ]
    final_train_idxs = np.where( np.isin(train_targets, final_classes) )[0]
    final_val_idxs = np.where( np.isin(val_targets, final_classes) )[0]
    # print(final_train_idxs)
    # print(final_val_idxs)
# Map class labels to consecutive 0,1,2,...
    label_remapping = {}
    idx = 0
    for k in final_classes:
        label_remapping[k] = idx
        idx += 1
    def transform_label(k):
        if k in label_remapping:
            return label_remapping[k]
        else:
            return -1
    target_transform = transform_label
    # print(label_remapping)
    # sys.exit()
    
    transform_img_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    transform_img_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    transform_img_train = transforms.Compose([transform_img_train, transforms.Normalize(mean, std)])
    transform_img_val = transforms.Compose([transform_img_val, transforms.Normalize(mean, std)])
    
    train_dataset = datasets.INaturalist(root = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/train', 
                                         version = '2021_train', 
                                         download=False, 
                                         target_type = target_type,
                                         transform=transform_img_train,
                                         target_transform=target_transform) 
    val_dataset = datasets.INaturalist(root = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/val', 
                                       version = '2021_valid', 
                                       download=False, 
                                       target_type = target_type,
                                       transform=transform_img_val, 
                                       target_transform=target_transform) 
    
    
# 原始数据
    # train_dataset = Subset(train_dataset, final_train_idxs)
    # val_dataset = Subset(val_dataset, final_val_idxs)
    # image_datasets = {
    #     'train' : train_dataset,
    #     'val' : val_dataset
    #                  }
    # # Create training and validation dataloaders
    # dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}

# 先分为一个整体，其次再分开
    dataset = torch.utils.data.ConcatDataset([Subset(train_dataset, final_train_idxs), Subset(val_dataset, final_val_idxs)])
    
    
    generator1 = torch.Generator().manual_seed(0) # For reproducibility
    train, val = torch.utils.data.random_split(dataset, [1-frac_val, frac_val], generator=generator1)
    
    # Create training and validation datasets
    image_datasets = {
        'train' : train,
        'val' : val
                     }
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    # dataloaders_dict = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloaders_dict


def get_dataloaders_iNaturalist_sony_1(batch_size=128, num_classes=633, target_type="family", min_train_instances=250, num_workers=10, frac_val=0.5):
    
    # 提取iNaturalist 数据集
    def extract_targets_fast(dataset, batch_size=1024, num_workers=6):
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            targets = []

            for _, target in tqdm(dataloader):  # DataLoader 返回 (data, target)
                targets.extend(target.numpy() if hasattr(target, 'numpy') else target.tolist())

            return targets

    def save_mappings(file_path, valid_classes, old_to_new, train_targets, val_targets, min_train_instances):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'valid_classes': valid_classes,
                'old_to_new': old_to_new,
                'train_targets': train_targets,
                'val_targets': val_targets,
                'min_train_instances': min_train_instances
            }, f)

    def load_mappings(file_path, min_train_instances):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if data['min_train_instances'] != min_train_instances:
            raise ValueError("映射文件中的筛选条件与当前不匹配！")
        return data['valid_classes'], data['old_to_new'], data['train_targets'], data['val_targets']

    class CustomDataset(torch.utils.data.Dataset):
        """自定义数据集，用于存储过滤后的样本和重新映射的标签"""
        def __init__(self, subset, new_targets):
            self.subset = subset
            self.new_targets = new_targets
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            data, _ = self.subset[idx]  # 从 Subset 中获取原始数据
            target = self.new_targets[idx]  # 获取重新映射后的标签
            return data, target, 1

    transform_img = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    target_transform = None
    train_dataset = datasets.INaturalist(root = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/train', 
                                         version = '2021_train', 
                                         download=False, 
                                         target_type = target_type,
                                         transform=transform_img,
                                         target_transform=target_transform)
    val_dataset = datasets.INaturalist(root = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/val', 
                                       version = '2021_valid', 
                                       download=False, 
                                       target_type = target_type,
                                       transform=transform_img, 
                                       target_transform=target_transform)
    mean, std = calc_mean_std(val_dataset)
    
    mappings_file = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/ckpt/iNaturalist/mappings_file.pkl'  # 替换为实际保存路径
    os.makedirs(os.path.dirname(mappings_file), exist_ok=True)
    
    # Step 1_1: 单个数据效率，从数据集中提取目标标签
    # def extract_targets(dataset):
    #     targets = []
    #     for idx in range(len(dataset)):
    #         _, target = dataset[idx]  # 通过 __getitem__ 方法解析目标
    #         targets.append(target)
    #     return targets
     # Step 1_2: batch_size个数据效率，从数据集中提取目标标签
    
    # 检查是否已经存在类别映射文件
    if os.path.exists(mappings_file):
        print(f"加载已保存的类别映射文件: {mappings_file}")
        valid_classes, old_to_new, train_targets, val_targets = load_mappings(mappings_file, min_train_instances)
    else:
        # 提取目标标签
        train_targets = extract_targets_fast(train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_targets = extract_targets_fast(val_dataset, batch_size=batch_size, num_workers=num_workers)

        # 统计类别分布
        label_counts = Counter(train_targets)

        # 筛选出满足条件的类别
        valid_classes = {label: count for label, count in label_counts.items() if count >= min_train_instances}
        print(f"valid_classes number: {len(valid_classes)}")
        # 创建旧标签到新标签的映射
        old_to_new = {old_label: new_label for new_label, old_label in enumerate(sorted(valid_classes.keys()))}

        # 保存映射到本地文件
        save_mappings(mappings_file, valid_classes, old_to_new, train_targets, val_targets, min_train_instances)
        print(f"类别映射文件已保存: {mappings_file}")

    # 提取新一轮的目标标签
    # extracted_train_targets = extract_targets_fast(train_dataset, batch_size=batch_size, num_workers=num_workers)
    # extracted_val_targets = extract_targets_fast(val_dataset, batch_size=batch_size, num_workers=num_workers)

    # # 验证保存的 targets 和重新提取的 targets 是否一致
    # assert train_targets == extracted_train_targets, "train_targets 顺序不一致！"
    # assert val_targets == extracted_val_targets, "val_targets 顺序不一致！"
    # print("目标标签顺序一致！")

    # 获取 train_indices 的 10% 随机样本
    # Step 5: 过滤训练集
    train_indices = [i for i, label in enumerate(train_targets) if label in valid_classes]
    train_indices = random.sample(train_indices, len(train_indices) // 100)
    filtered_train_dataset = Subset(train_dataset, train_indices)
    filtered_train_targets = [old_to_new[train_targets[i]] for i in train_indices]

    # Step 6: 过滤验证集
    val_indices = [i for i, label in enumerate(val_targets) if label in valid_classes]
    filtered_val_dataset = Subset(val_dataset, val_indices)
    filtered_val_targets = [old_to_new[val_targets[i]] for i in val_indices]
    
    # 创建新的自定义数据集
    custom_train_dataset = CustomDataset(filtered_train_dataset, filtered_train_targets)
    custom_val_dataset = CustomDataset(filtered_val_dataset, filtered_val_targets)
    
    # 创建数据加载器
    image_datasets = {
        'train' : custom_train_dataset,
        'val' : custom_val_dataset
                     }
    def worker_init_fn(worker_id):
    # 在每个 worker 进程启动时进行垃圾回收
        gc.collect()
        # seed = worker_id + int(torch.initial_seed()) % (2**32)
    
        # # 设置 Python 随机数种子
        # random.seed(seed)

        # # 设置 NumPy 随机数种子
        # np.random.seed(seed)

        # # 设置 PyTorch 随机数种子
        # torch.manual_seed(seed)

        # # 如果使用的是 CUDA，确保 CUDA 操作的随机性也初始化
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed)
            
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    print(f"label number: {len(valid_classes)}")

    return dataloaders_dict


def get_dataloaders_iNaturalist_sony_2(batch_size=128, num_classes=925, target_type="family", min_train_instances=250, num_workers=10, frac_val=0.5, n_avg=80):

    transform_img_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=10),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
    ])
    transform_img_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # val_data = datasets.Places365(root="/media/zzm/办公应用/data_sony/places365/val", split="val", download=False, transform=transform_img_val)
    val_dataset = datasets.INaturalist(
        root='/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/val',
        version='2021_valid',
        download=False,
        target_type=target_type,
        transform=transform_img_val
    )
    mean, std = calc_mean_std(val_dataset)
    print(f"calculate mean std done ...")
    
    transform_img_train = transforms.Compose([transform_img_train, transforms.Normalize(mean, std)])
    transform_img_val = transforms.Compose([transform_img_val, transforms.Normalize(mean, std)])
    
    def get_targets(dataset, batch_size=1024, num_workers=6, desc="Processing dataset"):
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=False,
            persistent_workers=True
        )
        # 添加总体进度条
        total_items = len(dataset)
        pbar = tqdm(total=total_items, desc=desc)
        
        for _, target in dataloader:
            batch_size = target.size(0)
            pbar.update(batch_size)
            yield from (t.item() for t in target)
        pbar.close()

    class EfficientCustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, label_mapping):
            self.dataset = dataset
            self.indices = indices
            self.label_mapping = label_mapping

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data, old_target = self.dataset[self.indices[idx]]
            return data, torch.tensor(self.label_mapping[old_target]), 1

    def load_or_create_mappings(train_dataset, val_dataset, mappings_file, min_train_instances, batch_size, num_workers):
        if os.path.exists(mappings_file):
            print(f"Loading existing mappings from: {mappings_file}")
            with open(mappings_file, 'rb') as f:
                data = pickle.load(f)
                if data['min_train_instances'] != min_train_instances:
                    raise ValueError("Incompatible min_train_instances!")
                return data
        
        print("\nCreating new mappings file...")
        start_time = time.time()
        
        # 处理训练集
        print("\nStep 1/4: Processing training dataset...")
        train_targets = list(get_targets(train_dataset, batch_size, num_workers, 
                                       desc="Processing training dataset"))
        
        print("\nStep 2/4: Counting labels...")
        label_counts = Counter(train_targets)
        print(f"Found {len(label_counts)} unique labels")
        
        # 处理有效标签
        print("\nStep 3/4: Filtering valid labels...")
        valid_labels = {label for label, count in tqdm(label_counts.items(), 
                                                      desc="Filtering labels") 
                       if count >= min_train_instances}
        print(f"Identified {len(valid_labels)} valid labels with >= {min_train_instances} instances")
        
        label_mapping = {old: new for new, old in enumerate(sorted(valid_labels))}
        
        # 处理验证集
        print("\nStep 4/4: Processing validation dataset...")
        val_targets = list(get_targets(val_dataset, batch_size, num_workers, 
                                     desc="Processing validation dataset"))
        
        # 保存结果
        data = {
            'valid_classes': valid_labels,
            'label_mapping': label_mapping,
            'train_targets': train_targets,
            'val_targets': val_targets,
            'min_train_instances': min_train_instances
        }
        
        # 创建目录并保存
        print("\nSaving mappings file...")
        os.makedirs(os.path.dirname(mappings_file), exist_ok=True)
        with open(mappings_file, 'wb') as f:
            pickle.dump(data, f)
        
        total_time = time.time() - start_time
        print(f"\nMapping creation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return data

    # 加载数据集
    print("\nInitializing datasets...")
    train_dataset = datasets.INaturalist(
        root='/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/train',
        version='2021_train',
        download=False,
        target_type=target_type,
        transform=transform_img_train
    )
    
    val_dataset = datasets.INaturalist(
        root='/share/home/tj06105/backup/lx/FeatureCP_score_cluster/datasets_cluster/inaturalist/val',
        version='2021_valid',
        download=False,
        target_type=target_type,
        transform=transform_img_val
    )

    # 加载或创建映射
    mappings_file = '/share/home/tj06105/backup/lx/FeatureCP_score_cluster/ckpt/iNaturalist/mappings_file.pkl'
    data = load_or_create_mappings(train_dataset, val_dataset, mappings_file, 
                                 min_train_instances, 4096, num_workers)

    # 过滤和采样数据
    print("\nPreparing dataset indices...")
    train_indices = [i for i, label in tqdm(enumerate(data['train_targets']), 
                                          desc="Filtering train indices",
                                          total=len(data['train_targets'])) 
                    if label in data['valid_classes']]
    # 只使用部分train_data
    # train_indices = random.sample(train_indices, len(train_indices) // 100)
    
    val_indices = [i for i, label in tqdm(enumerate(data['val_targets']), 
                                        desc="Filtering validation indices",
                                        total=len(data['val_targets'])) 
                  if label in data['valid_classes']]

    # 创建数据集
    print("\nCreating final datasets...")
    train_dataset = EfficientCustomDataset(train_dataset, train_indices, data['label_mapping'])
    val_dataset = EfficientCustomDataset(val_dataset, val_indices, data['label_mapping'])

# 重新拼接数据集
    dataset_new = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_new, [1-frac_val, frac_val])

    # 配置数据加载器
    dataloader_config = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2
    }

    print("\nInitializing dataloaders...")
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_config),
        'val': torch.utils.data.DataLoader(val_dataset, shuffle=False, **dataloader_config)
    }

    print(f"\nSetup complete! Number of valid classes: {len(data['valid_classes'])}")
    return dataloaders_dict  

    
def split_cal_and_val_iNaturalist(dataloader, n_k, num_classes, seed=0, split='balanced'):
    '''
    专门针对INaturalist数据集的校准集和验证集分割函数
    
    输入:
        dataloader: PyTorch DataLoader对象 (来自INaturalist数据集)
        n_k: 每个类别要选择的样本数(int)或每个类别的样本数数组
        num_classes: 类别总数
        seed: 随机种子
        split: 分割方式 ('balanced' 或 'proportional')
    
    输出:
        cal_loader: 校准集的DataLoader
        val_loader: 验证集的DataLoader
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 获取原始数据集
    original_dataset = dataloader.dataset
    
    # 使用高效的方式收集标签
    print("\nCollecting labels...")
    if hasattr(original_dataset, 'targets'):
        # 如果数据集直接提供targets属性
        y = torch.tensor(original_dataset.targets)
    else:
        # 否则使用高效的批量处理方式获取标签
        all_labels = []
        label_loader = torch.utils.data.DataLoader(
            original_dataset,
            batch_size=1024,  # 使用大批量来加速处理， 16384, 32768或65536
            num_workers=dataloader.num_workers,
            shuffle=False,
            persistent_workers=True
        )
        
        for _, labels, _ in tqdm(label_loader, desc="Collecting labels"):
            all_labels.append(labels)
        y = torch.cat(all_labels, dim=0)
    
    # 处理分割方式
    if split == 'balanced':
        if not hasattr(n_k, '__iter__'):
            n_k = n_k * np.ones((num_classes,), dtype=int)
    elif split == 'proportional':
        assert not hasattr(n_k, '__iter__')
        cts = Counter(y.numpy())
        rarest_class_ct = cts.most_common()[-1][1]
        frac = n_k / rarest_class_ct
        n_k = [int(frac*cts[k]) for k in range(num_classes)]
    else:
        raise Exception('Valid split options are "balanced" or "proportional"')
    
    # 高效地选择样本索引
    print("\nSelecting samples for each class...")
    selected_indices = []
    for k in tqdm(range(num_classes), desc="Processing classes"):
        idx = torch.where(y == k)[0].numpy()
        if len(idx) > 0:  # 确保该类别有样本
            selected_size = min(n_k[k], len(idx))  # 防止n_k大于类别样本数
            selected_idx = np.random.choice(idx, replace=False, size=(selected_size,))
            selected_indices.extend(selected_idx)
    
    selected_indices = torch.tensor(selected_indices)
    # 使用集合操作来高效获取剩余索引
    remaining_indices = torch.tensor(list(set(range(len(y))) - set(selected_indices.tolist())))
    
    # 使用SubsetRandomSampler来避免数据复制
    # cal_sampler = torch.utils.data.SubsetRandomSampler(selected_indices)
    # val_sampler = torch.utils.data.SubsetRandomSampler(remaining_indices)
    # 创建Subset而不是使用Sampler
    cal_dataset = torch.utils.data.Subset(original_dataset, selected_indices)
    val_dataset = torch.utils.data.Subset(original_dataset, remaining_indices)
    
    # 配置新的DataLoader，保持原始配置
    dataloader_config = {
        'batch_size': dataloader.batch_size,
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'persistent_workers': True,
        'prefetch_factor': 2  # 添加预取因子来提高效率
    }
    
    print("\nCreating new dataloaders...")
    # cal_loader = torch.utils.data.DataLoader(
    #     original_dataset,
    #     sampler=cal_sampler,
    #     **dataloader_config
    # )
    
    # val_loader = torch.utils.data.DataLoader(
    #     original_dataset,
    #     sampler=val_sampler,
    #     **dataloader_config
    # )
    cal_loader = torch.utils.data.DataLoader(
        cal_dataset,
        shuffle=True,  # 对校准集进行随机打乱
        **dataloader_config
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,  # 验证集不需要打乱
        **dataloader_config
    )
    
    print(f"\nSplit complete! Calibration set size: {len(selected_indices)}")
    print(f"Validation set size: {len(remaining_indices)}")
    
    return cal_loader, val_loader    
    
    
    
    
    

if __name__=="__main__":
    label_transform = label_to_onehotlabel(365)
    label = torch.tensor([2])
    # label, onthot_label = label_transform(label)
    result = label_transform(label)

    print(result[0])
    print(result[1])
    print(result[1][0])
    print(len(result[1]))
    print(result[1] == result[1][0])