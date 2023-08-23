import numpy as np
import torch
from neurotools import util

class Reverb(torch.nn.Module):

    def __init__(self, spatial1, spatial2, kernel_size, in_channels, out_channels, device='cpu', **kwargs):
        """
        Module that defines connection between two neuronal populations. The weight matrix for this module has an
        intrinsic update function
        :param spatial1: the spatial size. For now, always 2D, creates a spatial x spatial square.
        :param spatial2: the spatial size. For now, always 2D, creates a spatial x spatial square.
        :param kernel_size: desired kernel size, may be altered to preserve spatial identity mapping
        :param in_channels: number of input channel dimensions
        :param out_channels: number of output channel dimentions
        """
        super().__init__()
        if spatial1 != spatial2:
            raise ValueError("Only square spatial inputs expected currently.")
        folded_weight = torch.ones((1, in_channels, spatial1, spatial2)) * .5
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial1 = spatial1
        self.spatial2 = spatial2
        if "init_plasticity" in kwargs:
            init_plasticity = kwargs["init_plasticity"]
        else:
            init_plasticity = .1
        evo_conv_weight = torch.empty((in_channels, out_channels))
        self.conv = torch.nn.Parameter(torch.nn.init.xavier_normal_(evo_conv_weight))
        self.kernel_size, self.pad = util.conv_identity_params(in_spatial=spatial1, desired_kernel=kernel_size)
        self.plasticity = torch.nn.Parameter(torch.ones((in_channels,)).float() * init_plasticity)
        self.unfolder = torch.nn.Unfold(kernel_size=self.kernel_size,
                                        padding=self.pad)
        self.folder = torch.nn.Fold(kernel_size=self.kernel_size,
                                    output_size=(spatial1, spatial2),
                                    padding=self.pad)
        self.weight = self.unfolder(folded_weight)  # 1, channel * kernel * kernel, spatial * spatial
        self.activation_memory = None  # store unfolded most recent activation
        self.device = 'cpu'

        self.to(device)

    def detach(self):
        self.weight = torch.ones_like(self.weight) * .5
        self.activation_memory = None

    def to(self, device):
        self.detach()
        self.weight = self.weight.to(device)
        self.plasticity = torch.nn.Parameter(self.plasticity.to(device))
        self.conv = torch.nn.Parameter(self.conv.to(device))
        self.device = device

    def get_weight(self):
        return self.evo_conv_weight

    def forward(self, x):
        if torch.max(x) > 1 or torch.min(x) < 0:
            print("WARN: Reverb  input activations are expected to have range 0 to 1")
        # unfold x to synaptic space
        xufld = self.unfolder(x)
        # recall the input activations for computing hebbian update in update phase
        self.activation_memory = xufld.clone()
        local_conv = self.conv.clone()  # (inchan, outchan) create a unique node for this param state in comp graph
        h1 = self.weight * xufld  # transmit along each edge
        h2 = self.folder(h1)  # (1, in_chan, s0, s1) sum each receptive field, to output state space
        # map all channels to output state channels  (grad optimized param)
        h2 = h2.view(self.in_channels, self.spatial1 * self.spatial2)  # (1, s1, s0, in_chan)
        y = h2.transpose(0, 1) @ local_conv
        y = y.transpose(0, 1)
        # return output states in torch format
        y = y.view((1, self.out_channels, self.spatial1, self.spatial2))
        return y.clone()

    def update(self, target_activations):
        if torch.max(target_activations) > 1 or torch.min(target_activations) < 0:
            print("WARN: Reverb input activations are expected to have range 0 to 1")
        if self.activation_memory is None:
            return
        # shape of chanel view of synaptic unfolded space
        channel_view = (self.kernel_size ** 2, self.in_channels, self.spatial1, self.spatial2)
        # reverse the channel mapping so source channels receive information about the targets they actually innervate
        target_activations = target_activations.view(self.out_channels, self.spatial1 * self.spatial2)
        reverse_conv = self.conv.clone().transpose(0, 1)  # (out_chan, in_chan)
        local_space_target = target_activations.transpose(0, 1) @ reverse_conv
        local_space_target = local_space_target.transpose(0, 1)
        local_space_target = local_space_target.view((1, self.in_channels, self.spatial1, self.spatial2))

        synaptic_target = self.unfolder(local_space_target).view(channel_view)
        local_activations = self.activation_memory.view(channel_view)
        # get joint activations
        delta = local_activations * synaptic_target
        weight_shape = self.weight.shape
        self.weight = self.weight.view(channel_view)

        local_plast = self.plasticity.clone()  # track plasticity grad param state

        # preform associative update and take a standard unfolded synaptic view of the weights
        self.weight = (1 - local_plast.view((1, self.in_channels, 1, 1))) * \
                      self.weight + (local_plast.view((1, self.in_channels, 1, 1)) * delta)
        self.weight = self.weight.view(weight_shape)


class ElegantReverb(torch.nn.Module):

    def __init__(self, num_nodes, spatial1, spatial2, kernel_size, channels, device='cpu',
                 normalize_conv=True, mask=None, **kwargs):
        """
        serves the same purpose as the standard reverb convolution, but designed to operate on a graph all at once.

        """
        super().__init__()
        self.activation_memory = None
        self.num_nodes = num_nodes
        self.kernel_size, self.pad = util.conv_identity_params(in_spatial=spatial1, desired_kernel=kernel_size)
        if mask is None:
            mask = torch.ones((num_nodes, num_nodes), device=device)
            mask[:, 0] = 0
        self.mask = mask

        # Non-Parametric Weights used for intrinsic update
        weight = torch.empty((num_nodes, num_nodes,
                              spatial1, spatial2,
                              channels, self.kernel_size, self.kernel_size),
                             device=device)  # 8D Tensor.
        self.weight = torch.nn.init.xavier_normal_(weight)

        # Parametric Weight used for default receptive field
        prior = torch.empty((num_nodes, num_nodes, channels, self.kernel_size, self.kernel_size), device=device)
        self.prior = torch.nn.Parameter(torch.nn.init.xavier_normal_(prior))

        # Channel Mapping
        chan_map = torch.empty((num_nodes, num_nodes, channels, channels), device=device)
        self.chan_map = torch.nn.Parameter(torch.nn.init.xavier_normal_(chan_map))

        if "init_plasticity" in kwargs:
            init_plasticity = kwargs["init_plasticity"]
        else:
            init_plasticity = .1
        self.plasticity = torch.nn.Parameter(torch.ones((num_nodes, num_nodes), device=device) * init_plasticity)
        self.device = device
        self.unfolder = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.pad)
        self.folder = torch.nn.Fold(kernel_size=self.kernel_size,
                                    output_size=(spatial1, spatial2),
                                    padding=self.pad)
        self.channels = channels
        self.spatial1 = spatial1
        self.spatial2 = spatial2
        self.normalize = normalize_conv

    # def parameters(self, recurse: bool = True):
    #     params = [self.chan_map, self.plasticity, self.prior]
    #     return params

    def forward(self, x):
        x = x.to(self.device)  # nodes, channels, spatial1, spatial2
        if len(x.shape) != 4:
            raise ValueError("Input Tensor Must Be 4D, not shape", x.shape)
        if x.shape[0] != self.num_nodes:
            raise ValueError("Input Tensor must have number of nodes on batch dimension.")
        if torch.max(x) > 1 or torch.min(x) < 0:
            print("WARN: Reverb  input activations are expected to have range 0 to 1")
        xufld = self.unfolder(x).transpose(1, 2)  # nodes, spatial1 * spatial2, channels * kernel * kernel
        xufld = xufld.view((1, self.num_nodes, self.spatial1 * self.spatial2, self.channels, self.kernel_size ** 2))

        self.activation_memory = xufld.clone()

        conv_weight = self.weight * self.mask.view(self.num_nodes, self.num_nodes, 1, 1, 1, 1, 1)
        prior_weight = self.prior * self.mask.view(self.num_nodes, self.num_nodes, 1, 1, 1)

        combined_weight = conv_weight + prior_weight[:, :, None, None, :, :, :]  # broadcast along spatial 1 and 2
        combined_weight = combined_weight.view((self.num_nodes, self.num_nodes, self.spatial1 * self.spatial2, self.channels,
                                                self.kernel_size ** 2))
        meta_state = combined_weight * xufld

        iter_rule = "uvsck, uvco -> uvsok"
        mapped_meta = torch.einsum(iter_rule, meta_state, self.chan_map)

        mapped_meta = mapped_meta * self.mask.view(self.num_nodes, self.num_nodes, 1, 1, 1)

        ufld_meta = torch.sum(mapped_meta, dim=0)  # sum over input nodes
        ufld_meta = ufld_meta.transpose(2, 3)  # switch the ordering of kernels and channels to original so we can take the correct view on them
        ufld_meta = ufld_meta.reshape(
            (self.num_nodes, self.spatial1 * self.spatial2, self.kernel_size ** 2 * self.channels)
        ).transpose(1, 2)  # finish returning to original unfolded dims
        out = self.folder(ufld_meta)  # nodes, channels, spatial, spatial
        return out

    def get_weight(self):
        return self.out_edge

    def update(self, target_activation):
        """
        intrinsic update
        :param target_activation: the activation of each state after forward pass. (nodes, channel, spatial, spatial)
        :param args:
        :return:
        """
        if len(target_activation.shape) != 4:
            raise ValueError("Input Tensor Must Be 4D, not shape", target_activation.shape)
        if target_activation.shape[0] != self.num_nodes:
            raise ValueError("Input Tensor must have number of nodes on batch dimension.")
        if torch.max(target_activation) > 1 or torch.min(target_activation) < 0:
            print("WARN: Reverb  input activations are expected to have range 0 to 1")
        if self.activation_memory is None:
            return

        # shape of chanel view of synaptic unfolded space
        # channel_view = (self.kernel_size ** 2, self.in_channels, self.spatial1, self.spatial2)

        # reverse the channel mapping so source channels receive information about their targets
        target_activations = target_activation.view(1, self.num_nodes, self.channels,
                                                    self.spatial1 * self.spatial2).transpose(2, 3)
        reverse_conv = self.chan_map.clone().transpose(2, 3)  # (node, node, in_chan, out_chan)

        iter_rule = "uvsc, uvoc -> uvsc"
        target_meta_activations = torch.einsum(iter_rule, target_activations, reverse_conv).transpose(2,
                                                                                                      3)  # source, target, channels, spatial
        target_meta_activations = target_meta_activations.view(self.num_nodes * self.num_nodes, self.channels,
                                                               self.spatial1, self.spatial2)
        ufld_target = self.unfolder(target_meta_activations).transpose(1, 2)  # nodes * nodes, spatial1 * spatial2, channels * kernel * kernel
        ufld_target = ufld_target.view((self.num_nodes, self.num_nodes, self.spatial1 * self.spatial2, self.channels,
                                        self.kernel_size * self.kernel_size))

        coactivation = 2 * self.activation_memory * ufld_target  # so the 2 factor is so that strong coactivation actually increases the weights.

        plasticity = self.plasticity.view(self.num_nodes, self.num_nodes, 1, 1, 1, 1, 1).clone()

        self.weight = (1 - plasticity) * self.weight + \
                      plasticity * coactivation.view((self.num_nodes, self.num_nodes,
                                                      self.spatial1, self.spatial2,
                                                      self.channels, self.kernel_size, self.kernel_size))

    def detach(self, reset_weight=False):
        if reset_weight:
            weight = torch.empty_like(self.weight, device=self.device)
            self.weight = torch.nn.init.xavier_normal_(weight)
        else:
            self.weight = self.weight.detach().clone()
        self.chan_map = torch.nn.Parameter(self.chan_map.detach())
        self.activation_memory = None
        self.prior = torch.nn.Parameter(self.prior.detach())
        self.plasticity = torch.nn.Parameter(self.plasticity.detach())

    def to(self, device):
        self.weight = self.weight.to(device)
        self.chan_map = torch.nn.Parameter(self.chan_map.to(device))
        self.plasticity = torch.nn.Parameter(self.plasticity.to(device))
        self.prior = torch.nn.Parameter(self.prior.to(device))
        self.device = device
        return self


class WeightedConvolution(torch.nn.Module):
    """
    Module that maps one 4D tensor to another using a positive convolution operation, weighted by a
    scalar edge strength. Used as a component of Brain Network Estimation
    """

    def __init__(self, spatial1, spatial2, kernel_size, in_channels, out_channels, device='cpu', **kwargs):
        super().__init__()
        self.kernel_size, self.pad = util.conv_identity_params(in_spatial=spatial1, desired_kernel=kernel_size)
        conv = torch.empty((out_channels, in_channels, self.kernel_size, self.kernel_size), device=device)
        self.conv = torch.nn.Parameter(torch.nn.init.xavier_normal_(conv))
        self.out_edge = torch.nn.Parameter(torch.normal(size=(1,), mean=0., std=.1, device=device))
        self.tanh = torch.nn.Tanh()
        self.unfolder = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.pad)
        self.folder = torch.nn.Fold(kernel_size=self.kernel_size, padding=self.pad,
                                    output_size=(spatial1, spatial2))
        self.out_channel = out_channels
        self.spatial1 = spatial1
        self.spatial2 = spatial2
        self.device = device

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError("Input Tensor Must Be 4D, not shape", x.shape)
        if torch.max(x) > 1 or torch.min(x) < 0:
            print("WARN: Reverb  input activations are expected to have range 0 to 1")

        # unfold x to synaptic space
        xufld = self.unfolder(x).transpose(1, 2)  # 1, spatial1 * spatial2, channels * kernel * kernel
        conv_weight = torch.abs(self.conv.clone()).view(self.conv.size(0),
                                                        -1).t()  # kernel * kernel * inchannels, out_channels
        conv_res = (xufld @ conv_weight).transpose(1, 2)  # 1, out_channels, spatial1 * spatial2
        conv_res = conv_res.view(-1, self.out_channel, self.spatial1, self.spatial2)
        # conv res is standardized so weight comes from edge
        conv_res = conv_res / torch.std(conv_res)
        weighted_out = conv_res * self.out_edge.clone()
        return weighted_out

    def get_weight(self):
        return self.out_edge

    def update(self, *args):
        """
        No intrinsic update for this edge module type.
        """
        pass

    def detach(self):
        return self

    def to(self, device):
        self.conv = torch.nn.Parameter(self.conv.to(device))
        self.out_edge = torch.nn.Parameter(self.out_edge.to(device))
        self.device = device
        return self


class ElegantWeightedConvolution(torch.nn.Module):

    def __init__(self, num_nodes, spatial1, spatial2, kernel_size, channels, device='cpu',
                 normalize_conv=True, inject_noise=True, mask=None, **kwargs):
        """
        serves the same purpose as the standard weighted convolution, but designed to operate on a graph all at once.
        should work better if you have hella v/tRAM and you're not expecting a super sparse graph.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.kernel_size, self.pad = util.conv_identity_params(in_spatial=spatial1, desired_kernel=kernel_size)
        if mask is None:
            mask = torch.ones((num_nodes, num_nodes), device=device)
            mask[:, 0] = 0
        self.mask = mask
        conv = torch.empty((num_nodes, num_nodes, channels, channels, self.kernel_size, self.kernel_size),
                           device=device)  # 6D Tensor.
        conv = torch.nn.init.xavier_normal_(conv)
        self.conv = torch.nn.Parameter(conv)
        out_edge = torch.normal(size=(num_nodes, num_nodes), mean=0., std=.1, device=device)
        out_edge = out_edge * mask
        self.out_edge = torch.nn.Parameter(out_edge)
        self.device = device
        self.unfolder = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.pad)
        self.in_channels = channels
        self.out_channels = channels
        self.softmax = torch.nn.Softmax(dim=3)
        self.spatial1 = spatial1
        self.spatial2 = spatial2
        self.normalize = normalize_conv
        self.inject_noise = inject_noise

    def forward(self, x):
        x = x.to(self.device)
        if len(x.shape) != 4:
            raise ValueError("Input Tensor Must Be 4D, not shape", x.shape)
        if x.shape[0] != self.num_nodes:
            raise ValueError("Input Tensor must have number of nodes on batch dimension.")
        if torch.max(x) > 1 or torch.min(x) < 0:
            print("WARN: Reverb  input activations are expected to have range 0 to 1")
        xufld = self.unfolder(x).transpose(0, 1)  # channels * kernel * kernel, nodes, spatial1 * spatial2
        if self.inject_noise:
            noise = torch.normal(size=self.conv.shape, std=.001, mean=0, device=self.device)
        else:
            noise = 0
        conv_weight = torch.abs((self.conv.clone() + noise).view(self.num_nodes, self.num_nodes, self.out_channels, -1))
        if self.normalize:
            conv_weight = self.softmax(conv_weight)  # nodes, nodes, out_channels, inchannels * kernel * kernel
        conv_weight = conv_weight * self.mask.view(self.num_nodes, self.num_nodes, 1, 1)
        # can't do regular old mat mul cuz don't want to use all n2 weights (just n) for each of n node
        iter_rule = "cus, uvoc -> uvos"
        conv_res = torch.einsum(iter_rule, xufld, conv_weight)  # nodes, nodes, out_channels, spatial1 * spatial,
        masked_outedge = (self.out_edge * self.mask)[:, :, None, None]
        conv_res = conv_res * masked_outedge  # weight by out edge
        conv_res = conv_res.mean(dim=0).view(self.num_nodes, self.out_channels, self.spatial1,
                                             self.spatial2)  # mean over the incoming edges, reshape spatial dims
        return conv_res

    def get_weight(self):
        return self.out_edge

    def update(self, *args):
        """
        No intrinsic update for this edge module type.
        """
        pass

    def detach(self, **kwargs):
        return self

    def to(self, device):
        self.conv = torch.nn.Parameter(self.conv.to(device))
        self.out_edge = torch.nn.Parameter(self.out_edge.to(device))
        self.device = device
        return self


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

