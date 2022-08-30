import torch
import networkx as nx


def is_converged(loss_history, abs_tol=.0001, consider=100):
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
    if (loss_history[(-1 * (consider - 1)):]).std() < abs_tol and (loss_history[-1] / loss_history[0]) < .1:
        return True


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

