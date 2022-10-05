import torch
import numpy as np
import networkx as nx


def is_converged(loss_history, abs_tol=.00001, consider=1000):
    """
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
        kernel = max(kernel - 1, 1)

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
    x1_mean = torch.mean(x1, dim=dim)
    x2_mean = torch.mean(x2, dim=dim)
    x1c = (x1 - x1_mean)
    x2c = (x2 - x2_mean)
    num = torch.sum(x1c * x2c)
    sum_sq = torch.sum(torch.pow(x1c, 2)) * torch.sum(torch.pow(x2c, 2))
    denom = torch.sqrt(sum_sq)
    pearson = num / denom
    return pearson


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
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


