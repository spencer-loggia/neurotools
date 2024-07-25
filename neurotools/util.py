import math

import torch
import numpy as np
import networkx as nx


def logistic_func(input, initial, final, log_tau):
    """
    returns the standard logistic function
    Args:
        initial:
        final:
        log_tau:

    Returns:
    """
    input = input.unsqueeze(1)
    num = final.unsqueeze(0)
    exponent = -torch.exp(log_tau) * (input - initial.unsqueeze(0))
    denom = 1 + torch.exp(exponent)
    return num / denom


def gaussian_kernel(kernel_size:tuple, cov:torch.Tensor, integral_resolution=3, renormalize=False):
    cov=cov.squeeze()
    dev = cov.device
    dtype = cov.dtype
    if cov.ndim != 2:
        raise ValueError("covariance matrix should have two dims")
    if len(kernel_size) != cov.shape[0] or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance matrix should be square and have size equal to kernel dimensionality")

    mu = torch.tensor([s / 2 for s in kernel_size], device=dev, dtype=dtype)
    grid = torch.stack(torch.meshgrid([torch.arange(s)
                                       for s in kernel_size]), dim=0).view(len(kernel_size), -1).T.to(dev) # center on each index
    for i in range(5):
        try:
            dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
            break
        except ValueError as e:
            print("WARN:", e, "\n Attempting to fix by add identity...")
            cov = cov + torch.eye(2) * .01
            if i == 4:
                raise ValueError

    # get some samples to construct prob estimate
    locs = torch.stack(torch.meshgrid([torch.arange(integral_resolution)
                                       for _ in kernel_size]), dim=0).view(len(kernel_size), -1).T.float().to(dev)
    if integral_resolution == 1:
        # should sample from middle of intervals if only doing one sampling
        locs += .5
    else:
        # need to scale to (0, 1) if doing multiple samples
        locs = locs / (integral_resolution - 1)
    # approximate cdf on each unit cube in grid
    probs = torch.stack([dist.log_prob(grid + l.unsqueeze(0)).exp() for l in locs], dim=0).mean(dim=0)
    if renormalize:
        probs = probs / torch.sum(probs)
    return probs.view((kernel_size))


def return_from_reward(rewards, gamma):
    """
    Compute the discounted returns for each timestep from a tensor of rewards.

    Parameters:
    - rewards (torch.Tensor): Tensor containing the instantaneous rewards.
    - gamma (float): Discount factor (0 < gamma <= 1).

    Returns:
    - torch.Tensor: Tensor containing the discounted returns.
    """
    # Initialize an empty tensor to store the returns
    returns = torch.zeros_like(rewards)

    # Variable to store the accumulated return, initialized to 0
    G = 0

    # Iterate through the rewards in reverse (from future to past)
    for t in reversed(range(len(rewards))):
        # Update the return: G_t = r_t + gamma * G_{t+1}
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns


def is_converged(loss_history, abs_tol=.00001, consider=100):
    """
    Legacy, use adam / SGDM momentum threshold to estimate convergence.
    A heuristic metric of whether a model has converged
    :param loss_history:
    :param abs_tol: if provided overides rel_tol.
    :param rel_tol: fraction of initial loss to consider as convergence tolerance.
    :param consider: amount of previous loss examples to consider
    :return:
    """
    if len(loss_history) < consider:
        return False
    loss_history = torch.Tensor(loss_history)
    if (loss_history[(-1 * (consider - 1)):]).std() < abs_tol:
        return True
    else:
        return False


def conv_identity_params(in_spatial, desired_kernel, stride=1):
    """
    finds convolution parameters that will maintain the same output shape
    :param in_spatial: spatial dimension to maintain
    :param desired_kernel: desired kernel, actual kernel may be smaller
    :param stride: desired stride. desired output shape scaled by stride / ((stride -1) * kernel)
    :return:

    """
    pad = .1
    in_spatial /= stride
    kernel = min(desired_kernel, in_spatial)
    while round(pad) != pad or pad >= kernel:
        # compute padding that will maintain spatial dims during actual conv
        # pad = (((in_spatial - 1) * stride) - in_spatial + kernel) / 2
        if stride == 1:
            out = in_spatial
        else:
            out = stride * in_spatial / ((stride - 1) * kernel)
        if out.is_integer():
            pad = (stride * (out - 1) - in_spatial + kernel) / 2
        if kernel < 2:
            raise RuntimeError("Could not find kernel pad combination to maintain dimensionality")
        kernel = max(kernel - 1, 0)

    return kernel + 1, int(pad)


def unfold_nd(input_tensor: torch.Tensor, kernel_size: int, padding: int, spatial_dims: int, stride=1):
    pad = [padding] * (2 * spatial_dims)
    batch_size = input_tensor.shape[0]
    channel_size = input_tensor.shape[1]
    padded = torch.nn.functional.pad(input_tensor, pad, "constant", 0)
    for i in range(spatial_dims):
        padded = padded.unfold(dimension=2 + i, size=kernel_size, step=stride)
    kernel_channel_dim = channel_size
    spatial_flat_dim = 1
    for i in range(spatial_dims):
        padded = padded.transpose(2 + i, 2 + spatial_dims + i)
        kernel_channel_dim *= padded.shape[2 + i]
        spatial_flat_dim *= padded.shape[2 + spatial_dims + i]
    # conform with Unfold modules output formatting.
    padded = padded.reshape(batch_size, kernel_channel_dim, spatial_flat_dim)
    return padded


def pearson_correlation(x1: torch.Tensor, x2: torch.Tensor, dim=0):
    """
    compute pearson correlation between two vectors.
    :param x1: vector1 of size n
    :param x2: vector2 of same size n
    :param dim: if ndims > 1, dimension to compute corr along.
    :return:
    """
    x1_mean = torch.mean(x1, dim=dim)
    x2_mean = torch.mean(x2, dim=dim)
    x1c = (x1 - x1_mean.unsqueeze(dim))
    x2c = (x2 - x2_mean.unsqueeze(dim))
    num = torch.sum(x1c * x2c, dim=dim)
    sum_sq = torch.sum(torch.pow(x1c, 2), dim=dim) * torch.sum(torch.pow(x2c, 2), dim=dim)
    denom = torch.sqrt(sum_sq)
    pearson = num / denom
    return pearson


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute spearman correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


def _pad_to_cube(arr: np.ndarray, time_axis=3):
    """
    makes spatial dims have even size.
    :param arr:
    :param time_axis:
    :return:
    """
    if time_axis < np.ndim(arr):
        size = max(arr.shape[:time_axis] + arr.shape[time_axis + 1:])
        print(size)
    else:
        size = max(arr.shape)

    ax_pad = [0] * np.ndim(arr)
    for i in range(len(ax_pad)):
        if i != time_axis:
            ideal = (size - arr.shape[i]) / 2
            ax_pad[i] = (int(np.floor(ideal)), int(np.ceil(ideal)))
        else:
            ax_pad[i] = (0, 0)
    arr = np.pad(arr, ax_pad, mode='constant', constant_values=(0, 0))
    return arr


def atlas_to_list(data_matrix, atlas, ignore_atlas_base=True, min_dim=5):
    """
    converts a matrix and corresponding atlas, to a list with data for each class
    :param data_matrix:
    :param atlas:
    :param ignore_atlas_base: If true, will not create a list entry for items labelled 0 in the atlas.
    :param min_dim: minimum number of datapoint necessary to include an atlas entry
    :return: list of tensors, atlas label that each entry corresponds to.
    """
    if type(data_matrix) is np.ndarray:
        data_matrix = torch.from_numpy(data_matrix)
    if type(atlas) is np.ndarray:
        atlas = torch.from_numpy(atlas)
    atlas = atlas.int()
    unique = torch.unique(atlas)
    unique_filtered = []
    class_data_list = []
    for roi_id in unique:
        if ignore_atlas_base and roi_id == 0:
            continue
        data = data_matrix[atlas == roi_id].unsqueeze(0)
        if data.shape[1] > min_dim:
            unique_filtered.append(roi_id)
            class_data_list.append(data)
    return class_data_list, unique_filtered


def triu_to_square(triu_vector, n, includes_diag=False):
    """
    Converts an upper triangle vector to a full (redundant) symmetrical square matrix.
    :param tri_vector: data point vector
    :param n: size of resulting square
    :param includes_diag: whether the main diagonal is included in triu_vector
    :return: a symmetric square tensor
    """
    if includes_diag:
        offset = 0
    else:
        offset = 1
    adj = torch.zeros((n, n), dtype=triu_vector.dtype, device=triu_vector.device)
    ind = torch.triu_indices(n, n, offset=offset)
    adj[ind[0], ind[1]] = triu_vector
    adj = (adj.T + adj)
    if includes_diag:
        adj = adj - torch.diag(torch.diagonal(adj) / 2)
    return adj
