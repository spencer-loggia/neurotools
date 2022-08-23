import torch
import networkx as nx


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
