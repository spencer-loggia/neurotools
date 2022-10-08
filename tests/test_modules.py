from neurotools import modules
import torch


def test_elegant_weighted_convolution():
    states = torch.sigmoid(torch.normal(mean=0, std=.5, size=(4, 2, 16, 16)))
    mod = modules.ElegantWeightedConvolution(in_channels=2, out_channels=2, spatial1=16, spatial2=16,
                                             kernel_size=4, num_nodes=4)
    out = mod(states)
    print('here')


if __name__=='__main__':
    test_elegant_weighted_convolution()