import torch
import numpy as np
from neurotools import util

_distance_metrics_ = ['euclidian', 'pearson', 'spearman', 'dot', 'cosine']


def pdist_general(X: torch.Tensor, metric, **kwargs):
    n = X.shape[0]
    device = X.device
    out_size = (n * (n - 1)) // 2
    dm = torch.zeros(out_size, device=device).float()
    k = 0
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            dm[k] = metric(X[i], X[j], **kwargs)
            k += 1
    return dm


def _dot_pdist(arr: torch.Tensor, normalize=False):
    """
    arr should be 2D < observations(v) x conditions (k) >
    if normalixe is true, equivilant to pairwise cosine similarity
    :param arr:
    :return:
    """
    if len(arr.shape) == 1:
        arr = arr.reshape(1, arr.shape[0])
    k = arr.shape[2]
    if normalize:
        arr = arr / arr.norm(dim=1)[:, None, :]
    outer = torch.matmul(arr.transpose(-1, -2), arr)  # k x k
    indices = torch.triu_indices(k, k, offset=1)
    return outer[:, indices[0], indices[1]]


def _pearson_pdist(arr: torch.Tensor):
    k = arr.shape[1]
    coef = torch.corrcoef(arr)
    indices = torch.triu_indices(k, k, offset=1)
    return coef[indices[0], indices[1]]


def dissimilarity(beta: torch.Tensor, metric='dot'):
    if len(beta.shape) != 3:
        raise IndexError("beta should be 3 dimensional and have batch on dim 0, observations on dim 1 and conditions on dim 2")
    if metric not in _distance_metrics_:
        raise ValueError('metric must be one of ' + str(_distance_metrics_))
    elif metric == 'dot':
        rdm = _dot_pdist(beta, normalize=False)
    elif metric == 'cosine':
        rdm = _dot_pdist(beta, normalize=True)
    elif metric == 'pearson':
        rdm = _pearson_pdist(beta)
    else:
        raise NotImplementedError
    return rdm


def pairwise_rsa(data_region_list, rdm_metric='cosine', pairwise_metric='spearman'):
    """
    Compute rdms for entries in data_region_list, and their pairwise correlations.
    :param data_region_list: a list of 2D tensors, each (data_points x features)
    :param rdm_metric: metric to use when computing rdms
    :param pairwise_metric: metric to use when computing pairwise correlations
    :return: correlations adjacency matrix, list of rdms for each region
    """
    roi_dissimilarity = [dissimilarity(data, metric=rdm_metric) for data in data_region_list]
    dissim = torch.cat(roi_dissimilarity, dim=0)
    if pairwise_metric == 'spearman':
        fxn = util.spearman_correlation
    elif pairwise_metric == 'pearson':
        fxn = util.pearson_correlation
    else:
        raise ValueError
    corr = pdist_general(dissim, fxn)
    return corr, dissim
