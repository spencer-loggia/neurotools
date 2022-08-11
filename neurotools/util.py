import torch

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