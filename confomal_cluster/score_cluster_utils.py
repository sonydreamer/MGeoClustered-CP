import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def get_quantile_threshold(alpha):
    '''
    对于每一个alpha，计算一个适合cluster_fcp的最小数据量
    Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''
    n = 1
    while np.ceil((n+1)*(1-alpha)/n) > 1:
        n += 1
    return n

def get_rare_classes(labels, alpha, num_classes):
    '''
    获得不适合cluster_fcp的类别
    '''
    thresh = get_quantile_threshold(alpha)
    classes, cts = np.unique(labels, return_counts=True)
    rare_classes = classes[cts < thresh]
        
    # Also included any classes that are so rare that we have 0 labels for it
    zero_ct_classes = np.setdiff1d(np.arange(num_classes), classes)
    rare_classes = np.concatenate((rare_classes, zero_ct_classes))
        
    return rare_classes

def remap_classes(labels, rare_classes):
    '''
    Exclude classes in rare_classes and remap remaining classes to be 0-indexed

    Outputs:
        - remaining_idx: Boolean array the same length as labels. Entry i is True
        iffses  labels[i] is not in rare_clas
        - remapped_labels: Array that only contains the entries of labels that are 
        not in rare_classes (in order) 
        - remapping: Dict mapping old class index to new class index
        '''
    remaining_idx = ~np.isin(labels, rare_classes)

    remaining_labels = labels[remaining_idx]
    remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
    new_idx = 0
    remapping = {}
    for i in range(len(remaining_labels)):
        if remaining_labels[i] in remapping:
            remapped_labels[i] = remapping[remaining_labels[i]]
        else:
            remapped_labels[i] = new_idx
            remapping[remaining_labels[i]] = new_idx
            new_idx += 1
    return remaining_idx, remapped_labels, remapping


def quantile_embedding(samples, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
    '''
    Computes the q-quantiles of samples and returns the vector of quantiles
    给一种class或者一个cluster簇的score计算embedding
    '''
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    return np.quantile(samples, q)