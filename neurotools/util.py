import torch


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


def _compute_convolutional_sequence(in_spatial, out_spatial, in_channels, out_channels, mean=0., std=.01):
    """
    Return a spatial rescaling module to spatially scale from one node space to another
    units of another.
    :param in_spatial:
    :param out_spatial:
    :param in_channels:
    :param out_channels:
    :return:
    """
    stride = 1
    pad = .1
    kernel = min(6, in_spatial)
    while round(pad) != pad or pad >= kernel:
        # compute padding that will maintain spatial dims during actual conv
        kernel = max(kernel - 1, 1)
        pad = (((in_spatial - 1) * stride) - in_spatial + kernel) / 2

    # now compute pool size needed to match spatial dims
    if in_spatial >= out_spatial:
        rescale_kernel = int(in_spatial / out_spatial)
        rescale = torch.nn.MaxPool2d(rescale_kernel)
        assert in_spatial / rescale_kernel == out_spatial
    else:
        rescale_kernel = int(out_spatial / in_spatial)
        rescale = torch.nn.Upsample(scale_factor=rescale_kernel, mode='nearest')
        assert in_spatial * rescale_kernel == out_spatial
    conv = torch.nn.Conv2d(kernel_size=int(kernel), padding=int(pad), stride=1, in_channels=int(in_channels),
                           out_channels=int(out_channels), bias=False)
    weights = torch.nn.Parameter(torch.normal(mean, std, size=conv.weight.shape))
    conv.weight = weights
    activation = torch.nn.Tanh()
    return conv, rescale, activation