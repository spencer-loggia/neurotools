import torch
import util


class Reverb(torch.nn.Module):

    def __init__(self, spatial1, spatial2, kernel_size, in_channels, out_channels, init_plasticity=.05):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial1 = spatial1
        self.spatial2 = spatial2
        evo_conv_weight = torch.empty((in_channels, out_channels))
        self.conv = torch.nn.Parameter(torch.nn.init.xavier_normal_(evo_conv_weight))
        self.kernel_size, self.pad = util.conv_identity_params(in_spatial=spatial1, desired_kernel=kernel_size)
        self.plasticity = torch.nn.Parameter(torch.ones((in_channels,)).float() * init_plasticity)
        self.unfolder = torch.nn.Unfold(kernel_size=self.kernel_size,
                                        padding=self.pad)
        self.folder = torch.nn.Fold(kernel_size=self.kernel_size,
                                    output_size=(spatial1, spatial2),
                                    padding=self.pad)
        self.activation_memory = None  # store unfolded most recent activation

    def forward(self, x):
        if torch.max(x) > 1 or torch.min(x) < 0:
            print("WARN: Reverb  input activations are expected to have range 0 to 1")
        if x.shape[1:] != self.weight.shape[1:]:
            raise ValueError("input of shape", x.shape,
                             "does not have compatible dimensionality with weights of shape", self.weight.shape)
        weight_unfold = self.unfolder(self.weight)
        xufld = self.unfolder(x)
        self.activation_memory = xufld.clone()
        local_conv = self.conv.clone()  # (inchan, outchan)
        h1 = weight_unfold * xufld
        h2 = self.folder(h1)  # (1, in_chan, s0, s1)
        h2 = h2.view(self.in_channels, self.spatial1 * self.spatial2)  # (1, s1, s0, in_chan)
        y = h2.transpose(0, 1) @ local_conv
        y = y.transpose(0, 1)
        y = y.view((1, self.out_channels, self.spatial1, self.spatial2))
        return y.clone()

    def update(self, target_activations):
        if torch.max(target_activations) > 1 or torch.min(target_activations) < 0:
            print("WARN: Reverb  input activations are expected to have range 0 to 1")
        if target_activations.shape[2:] != self.weight.shape[2:]:
            raise ValueError("input of shape", target_activations.shape,
                             "does not have compatible dimensionality with weights of shape", self.weight.shape)
        if self.activation_memory is None:
            return
        target_activations = target_activations.view(self.out_channels, self.spatial1 * self.spatial2)
        mem_shape = self.activation_memory.shape
        reverse_conv = self.conv.clone().transpose(0, 1)  # (out_chan, in_chan)
        local_space_target = target_activations.transpose(0, 1) @ reverse_conv
        local_space_target = local_space_target.transpose(0, 1)
        local_space_target = local_space_target.view((1, self.in_channels, self.spatial1, self.spatial2))
        reshaped_local = self.activation_memory.view((self.kernel_size**2,
                                                      self.in_channels,
                                                      self.spatial1,
                                                      self.spatial2))
        delta1 = local_space_target * reshaped_local
        delta2 = self.folder(delta1.reshape(mem_shape))
        local_plast = self.plasticity.clone()
        self.weight = (1 - local_plast.view((1, self.in_channels, 1, 1))) * \
                      self.weight + (local_plast.view((1, self.in_channels, 1, 1)) * delta2)


class ResistiveTensor(torch.nn.Module):
    def __init__(self, shape: tuple, equilibrium=-.1, init_resistivity=.1):
        super().__init__()
        self.data = (torch.ones(shape) * equilibrium).float()
        self.equilibrium = self.data.clone()
        self.resistivity = torch.Tensor([init_resistivity]).float()

    def __add__(self, other):
        self.data = self.data + self.resistivity * (self.equilibrium - self.data)
        self.data = self.data.clone() + other
        return self

    def clone(self):
        new_rt = ResistiveTensor(shape=self.data.shape)
        new_rt.data = self.data.clone()
        new_rt.equilibrium = self.equilibrium.clone()
        new_rt.resistivity = torch.nn.Parameter(self.resistivity.clone())
        return new_rt
