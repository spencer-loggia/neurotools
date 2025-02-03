import itertools

import torch
import numpy as np
from neurotools import util, stats

_distance_metrics_ = ['euclidian', 'pearson', 'spearman', 'dot', 'cosine' ]


def _euclidian_pdist(arr: torch.Tensor, order=2):
    """
    arr should be 3D <batch x observations(v) x conditions (k) >
    if normalize is true, equivalent to pairwise cosine similarity
    :param arr:
    :return:
    """
    arr = arr.unsqueeze(2)
    k = arr.shape[1]
    pd = arr - arr.transpose(1, 2)
    pd = pd.pow(order)
    pd = pd.sum(dim=-1)
    pd = pd.pow(1 / order)
    indices = torch.triu_indices(k, k, offset=1)
    return  pd[:, indices[0], indices[1]]


def _dot_pairwise(arr: torch.Tensor, normalize=False):
    """
    arr should be 3D <batch x observations(v) x conditions (k) >
    if normalize is true, equivalent to pairwise cosine similarity
    :param arr:
    :return:
    """
    if len(arr.shape) == 1:
        arr = arr.reshape(1, arr.shape[0])
    k = arr.shape[1]
    if normalize:
        arr = arr / (arr.norm(dim=2, keepdim=True) + 1e-8)
    outer = torch.matmul(arr, arr.transpose(-1, -2))  # k x k
    indices = torch.triu_indices(k, k, offset=1)
    return outer[:, indices[0], indices[1]]


def _pearson_pairwise(arr: torch.Tensor):
    arr = arr.squeeze()
    k = arr.shape[1]
    coef = torch.corrcoef(arr)
    indices = torch.triu_indices(k, k, offset=1)
    coef = coef[:, indices[0], indices[1]]
    return coef.unsqueeze(0)

def pdist_general(X: torch.Tensor, metric, **kwargs):
    """
    Slow pdist only for small(ish) number of comparisons with arbitrary distance function.
    arr should be 3D <batch x observations(v) x conditions (k) >
    """
    b, n, f = X.shape
    device = X.device
    out_size = (n * (n - 1)) // 2
    dm = torch.zeros(out_size, device=device).float()
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dm[k] = metric(X[:, i, :], X[:, j, :], **kwargs)
            k += 1
    return dm

def dissimilarity(beta: torch.Tensor, metric='dot'):
    if len(beta.shape) == 2:
        beta = beta.unsqueeze(0)
    elif len(beta.shape) != 3:
        raise IndexError("beta should be 3 dimensional and have batch on dim 0, observations on dim 1 and conditions on dim 2")
    if metric not in _distance_metrics_:
        raise ValueError('metric must be one of ' + str(_distance_metrics_))
    elif metric == 'dot':
        rdm = _dot_pairwise(beta, normalize=False)
        rdm = max(rdm) - rdm # this may not be ideal
    elif metric == 'cosine':
        rdm = 1 - _dot_pairwise(beta, normalize=True) # take cosine distance to be 1 - cosine similarity
    elif metric == 'pearson':
        rdm = 1 - _pearson_pairwise(beta)
    elif metric == 'euclidian':
        rdm = _euclidian_pdist(beta)
    else:
        raise NotImplementedError
    if torch.sum(rdm < 0) > 0:
        raise  ValueError("Shouldn't be any negative values in RDM")
    return rdm


def dissimilarity_from_supervised(data, targets, example_weights=None, metric="dot"):
    """
    data is batch, examples, features
    where here batch refers to separate items we're computing the dissimilarity for, and examples refers to
    examples that have a specific target.
    """
    group_labels = list(np.unique(targets))
    group_means = []
    for t in group_labels:
        if example_weights is None:
            example_weights = torch.ones(len(data)) / len(data)
        else:
            assert torch.isclose(example_weights.sum(), torch.tensor([1.]))
        g = data[:, targets == t, :]
        group_mean = torch.sum(g*example_weights[:, None, None], dim=1)  # mean across examples
        group_means.append(group_mean)
    group_means = torch.stack(group_means, dim=1)  # build "condition" (target) dimension
    rdm = dissimilarity(group_means, metric)
    return rdm


def pairwise_rsa(data_region_list, rdm_metric='cosine', pairwise_metric='spearman'):
    """
    Compute rdms for entries in data_region_list, and their pairwise correlations.
    :param data_region_list: a list of 2D tensors, each (data_points x features) (or 3d tensor if all regions are same size)
    :param rdm_metric: metric to use when computing rdms
    :param pairwise_metric: metric to use when computing pairwise correlations
    :return:  adjacency matrix, list of rdms for each region
    """
    roi_dissimilarity = [dissimilarity(data, metric=rdm_metric) for data in data_region_list]
    dissim = torch.cat(roi_dissimilarity, dim=0)
    if pairwise_metric == 'spearman':
        fxn = stats.spearman_correlation
    elif pairwise_metric == 'pearson':
        fxn = stats.pearson_correlation
    else:
        raise ValueError
    corr = pdist_general(dissim, fxn)
    return corr, dissim


def circle_corr(sample, n, metric="rho"):
    """
    determine how well the ordering of samples matches the ordering of a circle with n items.
    Sample is assumed to be the upper triangle of a pairwise distance matrix with size cr(n, 2)
    Args:
        n: number of items in circle
        sample: torch.Tensor <batch, cr(n, 2)>  upper triangle distance matrix.
    Returns: <batch,> spearman rho or kendalls tau on each batch
    """
    # create circle of items and get distance matrix
    inds = 2 * torch.pi * torch.arange(n) / n
    circ_coords = torch.stack([torch.cos(inds), torch.sin(inds)], dim=1).unsqueeze(0)
    circ_dist = _euclidian_pdist(circ_coords).reshape((1, -1))
    gt_ranks = util.get_ranks(circ_dist, ties=True)
    sample_ranks = util.get_ranks(sample, ties=True)
    if metric == "tau":
        gt_idx = torch.argsort(gt_ranks, dim=1)
        gt_ranks = util.take_along_axis(gt_ranks, gt_idx, dim=1)
        sample_ranks = util.take_along_axis(sample_ranks, gt_idx, dim=1)
        m = stats._ktau_from_ranks(gt_ranks, sample_ranks.unsqueeze(0))
    elif metric == "rho":
        m = stats.pearson_correlation(gt_ranks.float(), sample_ranks.float(), dim=1)
    else:
        raise ValueError
    return m



