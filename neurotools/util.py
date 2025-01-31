import copy
import math
from typing import List, Union

import torch
import numpy as np
from scipy import stats
import itertools


def unfold_nd(input_tensor: torch.Tensor, kernel_size: Union[int, tuple[int]],
              padding: Union[int, tuple[int]], spatial_dims: int, stride=1):
    """ explicitly represents a kernel passed over a space of arbitrary dimensionality. Fixes torch unfold not working
    with more than 2 dims. Useful for arbitrary multidimensional convolutional operations"""
    if type(padding) is int:
        pad = tuple([padding]*2*spatial_dims)
    elif len(padding) == spatial_dims * 2:
        pad = padding
    elif len(padding) == spatial_dims:
        pad = list(itertools.chain(*[[padding[i]] * 2 for i in range(spatial_dims)]))  # abstract pad to both sides of input
    else:
        raise ValueError
    # Assumes torch batch and channel placement (e.g. <b, c, ...>)
    batch_size = input_tensor.shape[0]
    channel_size = input_tensor.shape[1]
    # pad input.
    padded = torch.nn.functional.pad(input_tensor, pad, "constant", 0)
    for i in range(spatial_dims):
        padded = padded.unfold(dimension=2 + i, size=kernel_size[i], step=stride)
    kernel_channel_dim = channel_size
    spatial_flat_dim = 1
    for i in range(spatial_dims):
        padded = padded.transpose(2 + i, 2 + spatial_dims + i)
        kernel_channel_dim *= padded.shape[2 + i]
        spatial_flat_dim *= padded.shape[2 + spatial_dims + i]
    # conform with Unfold modules output formatting.
    padded = padded.reshape(batch_size, kernel_channel_dim, spatial_flat_dim)
    return padded


def take_along_axis(x: torch.Tensor, idx, dim=0):
    """
    A torch implimentation of numpys take along axis. Useful when you need to index an nd array with the results of an
    argsort/max/where along an axis.
    Args:
        x: input tensor
        idx: indexing tensor
        dim: dimension

    Returns: indexing result.

    """
    indexer = np.ogrid[tuple(map(slice, x.shape))]
    indexer[dim] = idx
    return x[indexer]


def get_ranks(x: torch.Tensor, ties=False, rtol=1e-3) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor <batch, features>
        ties: Whether to consider ties
        rtol: percent of spread of data considered equal magnitude, only used with ties

    Returns: ranks tensor <batch, features>

    """
    if ties:
        tol = -1 * torch.log10((torch.max(x) - torch.min(x)) * rtol).item()
        x = torch.round(x, decimals=round(tol))
        idx = torch.argsort(x, dim=1)
        og_ranks = torch.zeros(x.shape, dtype=torch.int)
        # get ranks with ties
        for i, b in enumerate(x):
            ranks = []
            b = b[idx[i]].tolist()
            for j, (_, group) in enumerate(itertools.groupby(b)):
                ranks += [j] * len(list(group))
            ranks = torch.tensor(ranks, dtype=torch.int)
            og_ranks[i, idx[i]] = ranks
    else:
        tmp = x.argsort(dim=1)
        og_ranks = take_along_axis(torch.arange(len(x)), tmp, dim=1)
    return og_ranks


def atlas_from_list(masks: List[np.ndarray], names: List, thresh=.5):
    """
    Create an atlas and lookup file from a list of arrays thresholded at thresh. If two masks overlap, will add
    those indexes to "tie-break" label.
    Args:
        masks: list of masks for different labels
        names: list of names for each label
        thresh: where to threshold background on mask
    Returns: Atlas: np.ndarrray type int, Lookup: pd.DataFrame
    """
    # create atlas
    masks = [np.zeros_like(masks[0])] + masks # add background
    masks = np.stack(masks, axis=0)
    masks = masks > thresh
    tie = (np.sum(masks, axis=0) > 1)
    atlas = np.argmax(masks, axis=0) # 0 needs to indicate background
    atlas[tie] = len(names) + 1
    padded = np.pad(atlas, 1)
    for t in np.argwhere(tie):
        # iterate over ties and replace with nearest
        arr = padded[t[0]-1+1:t[0]+2+1,
                  t[1]-1+1:t[1]+2+1,
                  t[2]-1+1:t[2]+2+1]
        arr = arr[np.nonzero(arr)]
        l = stats.mode(arr.flatten()).mode
        atlas[t[0], t[1], t[2]] = l
    names = names + ["tie"]
    return atlas, names


def gaussian_kernel(kernel_size: tuple, cov: torch.Tensor, integral_resolution=3, renormalize=False):
    """
    A differentiable 2D gaussian kernel! could be easily extended to higher dims. Useful if you want to learn paremeters
    of a distribution that will be convolved over some space. Not currently used in searchlight, but could be if one
    wanted smoothing parameters to be learned.
    Args:
        kernel_size:
        cov:
        integral_resolution:
        renormalize:

    Returns:

    """
    cov = cov.squeeze()
    dev = cov.device
    dtype = cov.dtype
    if cov.ndim != 2:
        raise ValueError("covariance matrix should have two dims")
    if len(kernel_size) != cov.shape[0] or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance matrix should be square and have size equal to kernel dimensionality")

    mu = torch.tensor([s / 2 for s in kernel_size], device=dev, dtype=dtype)
    grid = torch.stack(torch.meshgrid([torch.arange(s)
                                       for s in kernel_size]), dim=0).view(len(kernel_size), -1).T.to(
        dev)  # center on each index
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


def affine_from_params(rotations=(0, 0, 0), scale=(1, 1, 1), translate=(0, 0, 0)):
    """
    generates a 4d PUSH affine transfrom matrix given rotation [deg], scale, and translation arguments.
    Shear not yet implemented.
    Returns:
    ndarry size 4, 4
    """
    scale_mat = np.zeros((3, 3), dtype=float)
    # set diagonal
    scale_mat[np.diag_indices(3)] = np.array(scale, dtype=float)
    # rotation mats
    rotations = np.deg2rad(np.array(rotations))
    x_rot = np.eye(3, dtype=float)
    x_rot[1:, 1:] = np.array([[np.cos(rotations[0]), np.sin(rotations[0])],
                              [-np.sin(rotations[0]), np.cos(rotations[0])]])
    y_rot = np.eye(3, dtype=float)
    y_rot[(0, 0, 2, 2), (0, 2, 0, 2)] = np.array([np.cos(rotations[1]), -np.sin(rotations[1]),
                                                  np.sin(rotations[1]), np.cos(rotations[1])])
    z_rot = np.eye(3, dtype=float)
    z_rot[:2, :2] = np.array([[np.cos(rotations[2]), -np.sin(rotations[2])],
                              [np.sin(rotations[2]), np.cos(rotations[2])]])
    # scale and rot
    affine_3 = scale_mat @ x_rot @ y_rot @ z_rot
    # create 4d affine
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = affine_3
    affine[:3, 3] = np.array(translate)
    return affine


def is_converged(loss_history, optim, batch_size, t, max_lr=.01):
    """
    Legacy, use adam / SGDM momentum threshold to estimate convergence.
    A heuristic metric of whether a model has converged
    :param loss_history:
    :return:
    """
    check_size = min((10000 // batch_size), 250)
    set_lr = 1.
    if ((t + 1) % check_size) == 0:
        # reduce learn rate if not improving and check to stop early...
        d_last_block = np.mean(np.array(loss_history[-3*check_size:-2*check_size]))
        last_block = np.mean(np.array(loss_history[-2*check_size:-1*check_size]))
        block = np.mean(np.array(loss_history[-1*check_size:]))
        lr = 0.
        for g in optim.param_groups:
            lr = g['lr']
            print("LR:", lr)
            if t > 2*check_size and block >= last_block:
                # cool down on plateau
                g['lr'] = g['lr'] * .1
            elif t > 3*check_size and block < last_block < d_last_block:
                # reheat on downward trend
                g['lr'] = min(g['lr'] * 8.0, max_lr)
            set_lr = g['lr']
        print("EPOCH", t, "LOSS", block)
    return optim, set_lr < max_lr * 1e-7


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


def atlas_to_list(data_matrix, atlas, ignore_atlas_base=True, min_dim=0):
    """
    converts a matrix and corresponding atlas, to a list with data for each class
    :param data_matrix: matrix of data. May have a batch dimension.
    :param atlas:
    :param ignore_atlas_base: If true, will not create a list entry for items labelled 0 in the atlas.
    :param min_dim: minimum number of datapoint necessary to include an atlas entry
    :return: list of tensors, atlas label that each entry corresponds to.
    """
    if atlas is None:
        atlas = data_matrix
    if type(data_matrix) is np.ndarray:
        data_matrix = torch.from_numpy(data_matrix)
    if type(atlas) is np.ndarray:
        atlas = torch.from_numpy(atlas)
    if data_matrix.ndim == atlas.ndim:
        data_matrix = data_matrix.unsqueeze(0)
    atlas = atlas.int()
    unique = torch.unique(atlas)
    unique_filtered = []
    class_data_list = []
    for roi_id in unique:
        if ignore_atlas_base and roi_id == 0:
            continue
        data = data_matrix[..., atlas == roi_id]
        if data.shape[1] > min_dim:
            unique_filtered.append(roi_id.detach().cpu().item())
            class_data_list.append(data)
    return class_data_list, unique_filtered


def atlas_to_mask(atlas, lookup: dict, hemi_axis=None):
    """
    converts an atlas to a list of binary masks
    lookup dict roi_name:atlas_val
    """
    if type(atlas) is torch.Tensor:
        atlas = atlas.detach().cpu().numpy()
    atlas = atlas.astype(int)
    out_rois = []
    for roi_id in lookup.keys():
        o = np.zeros_like(atlas).astype(float)
        o[atlas == lookup[roi_id]] = 1.
        if hemi_axis is not None:
            split_dex = atlas.shape[hemi_axis] // 2
            o = np.moveaxis(o, hemi_axis, 0)
            ob = np.zeros_like(o)
            ob[split_dex:] = copy.deepcopy(o[split_dex:])
            o[split_dex:] = 0
            o = np.moveaxis(o, 0, hemi_axis)
            ob = np.moveaxis(ob, 0, hemi_axis)
            out_rois.append(o)
            out_rois.append(ob)
        else:
            out_rois.append(o)
    if hemi_axis is not None:
        names = []
        for n in lookup.keys():
            names.append(n + "_lh")
            names.append(n + "_rh")
    else:
        names = list(lookup.keys())
    return out_rois, names


def triu_to_square(triu_vector, n, includes_diag=False, negate=False):
    """
    Converts an upper triangle vector to a full (redundant) symmetrical square matrix.
    :param tri_vector: data point vector. Either <batch, c, ...> or <c,>
    :param n: size of resulting square
    :param includes_diag: whether the main diagonal is included in triu_vector
    :param negate: whether to negate the lower traingle.
    :return: a symmetric square tensor
    """
    if includes_diag:
        offset = 0
    else:
        offset = 1
    if triu_vector.ndim == 1:
        triu_vector = triu_vector.unsqueeze(0)
    adj = torch.zeros((triu_vector.shape[0], n, n) + triu_vector.shape[2:], dtype=triu_vector.dtype, device=triu_vector.device)
    ind = torch.triu_indices(n, n, offset=offset)
    adj[:, ind[0], ind[1], ...] = triu_vector
    if negate:
        out_adj = (-1 * adj.transpose(1, 2) + adj) + 1e-8
        if includes_diag:
            out_adj = out_adj + torch.diag(torch.diagonal(adj))
    else:
        out_adj = (adj.transpose(1, 2) + adj)
        if includes_diag:
            out_adj = out_adj - torch.diag(torch.diagonal(adj))
    return out_adj


def chance_ce(n_classes):
    with torch.no_grad():
        ce = torch.nn.CrossEntropyLoss()
        pred = torch.ones(1, n_classes) / n_classes
        target = torch.tensor([0], dtype=torch.long)
        cce = ce(pred, target)
        return cce.item()
