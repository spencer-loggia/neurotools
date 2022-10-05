import torch
import numpy as np
from neurotools import util

_distance_metrics_ = ['euclidian', 'pearson', 'spearman', 'dot', 'cosine']


def pdist_general(X, metric, **kwargs):
    n = X.shape[0]
    out_size = (n * (n - 1)) // 2
    dm = torch.zeros(out_size).float()
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


def pairwise_rsa(beta: torch.Tensor, atlas: torch.Tensor, min_roi_dim=5, ignore_atlas_base=True, metric='cosine'):
    if type(beta) is np.ndarray:
        beta = torch.from_numpy(beta)
    if type(atlas) is np.ndarray:
        atlas = torch.from_numpy(atlas)
    atlas = atlas.int()
    unique = torch.unique(atlas)
    if ignore_atlas_base:
        # omit index 0 (unclassified)
        unique = unique[1:]
    unique_filtered = []
    roi_dissimilarity = []
    for roi_id in unique:
        roi_betas = beta[atlas == roi_id].unsqueeze(0)
        if roi_betas.shape[1] > min_roi_dim:
            unique_filtered.append(roi_id)
            rdm = dissimilarity(roi_betas, metric=metric)
            roi_dissimilarity.append(rdm)
    adjacency = torch.zeros([len(unique_filtered), len(unique_filtered)])
    dissim = torch.cat(roi_dissimilarity, dim=0)
    corr = pdist_general(dissim, util.spearman_correlation)
    ind = torch.triu_indices(len(unique_filtered), len(unique_filtered), offset=1)
    adjacency[ind[0], ind[1]] = corr
    adjacency = adjacency.T + adjacency
    return adjacency, unique_filtered, roi_dissimilarity, None