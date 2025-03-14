import itertools

import torch
import numpy as np
from neurotools import util, stats

_distance_metrics_ = ['euclidean', 'pearson', 'spearman', 'dot', 'cosine', 'mahalanobis']


def _euclidian_pdist(arr: torch.Tensor, order=2):
    """
    arr should be 3D <batch x features (v) x conditions (k) >
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


def mahalanobis_pdist(arr: torch.Tensor, cov=None):
    """
    arr should be 3D <batch x features (v) x conditions (k) >
    if normalize is true, equivalent to pairwise cosine similarity
    :param arr:
    :return:
    """
    if cov is None:
        cov = stats.batch_covariance(arr.transpose(1, 2))
    inv_cov = torch.linalg.pinv(cov)
    arr = arr.unsqueeze(2)
    diffs = (arr - arr.transpose(1, 2))
    # pack diffs
    ds = diffs.shape
    k = ds[1]
    diffs = diffs.reshape((ds[0], -1, ds[-1]))
    pd = (diffs @ inv_cov).reshape((-1, 1, ds[-1]))  # <b * cond^2, feat>
    diffs = diffs.reshape((-1, ds[-1], 1))  # <b * cond^2, feat>
    pd = (pd @ diffs).reshape((ds[0], k, k))  # <b, cond, cond>
    pd[pd < 0] = 1e-20
    pd = torch.nan_to_num(pd, nan=1e-20, posinf=1e-20)
    indices = torch.triu_indices(k, k, offset=1)
    return pd[:, indices[0], indices[1]]


def _dot_pairwise(arr: torch.Tensor, normalize=False):
    """
    arr should be 3D <batch x features(v) x conditions (k) >
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
    """
    arr should be 3D <batch x features(v) x conditions (k) >
    if normalize is true, equivalent to pairwise cosine similarity
    :param arr:
    :return:
    """
    k = arr.shape[1]
    coef = stats.batched_corrcoef(arr)
    indices = torch.triu_indices(k, k, offset=1)
    coef = coef[:, indices[0], indices[1]]
    coef[coef > 1.] = 1.
    coef[coef < -1.] = -1.
    coef = torch.nan_to_num(coef, nan=0., posinf=0., neginf=0.)
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


def dissimilarity(beta: torch.Tensor, metric='dot', cov=None):
    """
    beta: Tensor, <batch, conditions, features>
    return: rdm <batch, cond * (cond - 1) / 2>
    """
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
        rdm[rdm < 0] = 0.
    elif metric == 'pearson':
        rdm = 1 - _pearson_pairwise(beta)
    elif metric == 'euclidean':
        rdm = _euclidian_pdist(beta)
    elif metric == 'mahalanobis':
        rdm = mahalanobis_pdist(beta, cov)
    else:
        raise NotImplementedError
    if torch.sum(rdm < 0) > 0:
        raise  ValueError("Shouldn't be any negative values in RDM")
    return rdm


def dissimilarity_from_supervised(data, targets, example_weights=None, metric="euclidean"):
    """
    data is batch, examples, features
    where here batch refers to separate items we're computing the dissimilarity for, and examples refers to
    examples that have a specific target.
    """
    group_labels = list(np.unique(targets))
    group_means = []
    if metric == "mahalanobis":
        cov = stats.batch_covariance(data.transpose(1, 2))  # batch, features, features
    else:
        cov = None
    for t in group_labels:
        if example_weights is None:
            example_weights = torch.ones(len(data)) / len(data)
        else:
            assert torch.isclose(example_weights.sum(), torch.tensor([1.]))
        g = data[:, targets == t, :]  # <batch, examples, features>
        group_mean = torch.sum(g*example_weights[:, None, None], dim=1)  # mean across examples <batch, variables>
        group_means.append(group_mean)
    group_means = torch.stack(group_means, dim=1)  # <batch, conditions, variables> build "condition" (target) dimension
    rdm = dissimilarity(group_means, metric, cov=cov)
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
    # create circle points
    angles = 2 * torch.pi * torch.arange(n) / n
    x_coords = np.cos(angles)
    y_coords = np.sin(angles)
    points = torch.tensor(np.array([x_coords, y_coords])).float()
    circ_coords = points.transpose(0, 1).unsqueeze(0)
    # compute angles between points
    circ_dist = 1 - _dot_pairwise(circ_coords, normalize=True).reshape((1, -1))
    if metric in ["rho", "tau"]:
        gt_ranks = util.get_ranks(circ_dist, ties=True)
        sample_ranks = util.get_ranks(sample, ties=False)
    else:
        # using pearson
        gt_ranks = circ_dist
        sample_ranks = sample
    if metric == "tau":
        gt_idx = torch.argsort(gt_ranks, dim=1)
        gt_ranks = util.take_along_axis(gt_ranks, gt_idx, dim=1)
        sample_ranks = util.take_along_axis(sample_ranks, gt_idx, dim=1)
        m = stats._ktau_from_ranks(gt_ranks, sample_ranks.unsqueeze(0))
    elif metric == "rho" or metric == "pearson":
        m = stats.pearson_correlation(gt_ranks.float(), sample_ranks.float(), dim=1)
    else:
        raise ValueError
    return m



