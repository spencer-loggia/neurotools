import torch
import util


class Reverb(torch.nn.Module):

    def __init__(self, spatial1, spatial2, kernel_size, in_channels, out_channels, init_plasticity=.05, init_utility=1.):
        """

        :param spatial1: the spatial size. For now, always 2D, creates a spatial x spatial square.
        :param spatial2: the spatial size. For now, always 2D, creates a spatial x spatial square.
        :param kernel_size: desired kernel size, may be altered to preserve spatial identity mapping
        :param in_channels: number of input channel dimensions
        :param out_channels: number of output channel dimentions
        """
        super().__init__()
        if spatial1 != spatial2:
            raise ValueError("Only square spatial inputs expected currently.")
        weight = torch.empty((1, in_channels, spatial1, spatial2))
        self.weight = torch.nn.init.xavier_normal_(weight)
        self.conv = torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels)
        self.kernel_size, self.pad = util.conv_identity_params(in_spatial=spatial1, desired_kernel=kernel_size)
        self.plasticity = torch.nn.Parameter(torch.ones((in_channels,)).float() * init_plasticity)
        self.utility_factor = torch.nn.Parameter(torch.Tensor([init_utility]))
        self.unfolder = torch.nn.Unfold(kernel_size=self.kernel_size,
                                        padding=self.pad)
        self.folder = torch.nn.Fold(kernel_size=self.kernel_size,
                                    output_size=(spatial1, spatial2),
                                    padding=self.pad)

    def forward(self, x):
        if torch.max(x) > 1 or torch.min(x) < 0:
            print("WARN: Reverb  input activations are expected to have range 0 to 1")
        if x.shape[1:] != self.weight.shape[1:]:
            raise ValueError("input of shape", x.shape,
                             "does not have compatible dimensionality with weights of shape", self.weight.shape)
        weight_unfold = self.unfolder(self.weight)
        x_unfold = self.unfolder(x)
        h = weight_unfold * x_unfold
        h = self.folder(h)
        h = self.conv(h)
        return h

    def update(self):
        raise NotImplementedError
