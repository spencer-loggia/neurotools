import torch
import networkx as nx



def is_converged(loss_history, abs_tol=None, rel_tol=.0001, consider=5):
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
    if abs_tol is None:
        abs_tol = loss_history[0] * rel_tol
    loss_history = torch.Tensor(loss_history)
    comp = loss_history[-1 * consider]
    if abs(loss_history[(-1 * (consider - 1)):] - comp).mean() < abs_tol:
        return True


def conv_identity_params(in_spatial, desired_kernel):
    """
    finds convolution parameters that will maintain the same output shape
    :param in_spatial: spatial dimension to maintain
    :param desired_kernel: desired kernel, actual kernel may be smaller
    :return:
    """
    stride = 1
    pad = .1
    kernel = min(desired_kernel, in_spatial)
    while round(pad) != pad or pad >= kernel:
        # compute padding that will maintain spatial dims during actual conv
        pad = (((in_spatial - 1) * stride) - in_spatial + kernel) / 2
        if kernel < 2:
            raise RuntimeError("Could not find kernel pad combination to maintain dimensionality")
        kernel = max(kernel - 1, 1)

    return kernel + 1, int(pad)
