import numpy as np
import torch
from neurotools import util


class Reverb(torch.nn.Module):

    def __init__(self, spatial1, spatial2, kernel_size, in_channels, out_channels, init_plasticity=.05, device='cpu'):
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
        evo_conv_weight = torch.empty((in_channels, out_channels))
        self.conv = torch.nn.Parameter(torch.nn.init.xavier_normal_(evo_conv_weight))
        self.kernel_size, self.pad = util.conv_identity_params(in_spatial=spatial1, desired_kernel=kernel_size)
        self.plasticity = torch.nn.Parameter(torch.ones((in_channels,)).float() * init_plasticity)
        self.unfolder = torch.nn.Unfold(kernel_size=self.kernel_size,
                                        padding=self.pad)
        self.folder = torch.nn.Fold(kernel_size=self.kernel_size,
                                    output_size=(spatial1, spatial2),
                                    padding=self.pad)
        self.weight = self.unfolder(folded_weight)
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


class MDScale:

    def __init__(self, n, embed_dims: int = 2, device='cpu'):
        """
        Computes an embedding of n examples into a `embed_dims` space that attempts to maintain the provided pairwise
        distances between examples
        :param n: number of items
        :param embed_dims: number of dimensions to construct space in
        :param device: device to use
        """
        self.num_items = n
        self.components = embed_dims
        self.mse = torch.nn.MSELoss()
        self.right_latent = None
        self.left_latent = None
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def stress(self, pairwise_target, positions):
        """
        An L2 Norm between distance in embedding space an actual provided pairwise distances
        :param positions:
        :param pairwise_target: upper triangular vector of pairwise distance between examples, size n(n-1) / 2
        :return: torch.Tensor a stress score for the system
        """
        cur_dists = torch.pdist(positions)
        stress = torch.mean(torch.abs(cur_dists - pairwise_target))
        return stress

    def check_dists(self, dist):
        if isinstance(dist, np.ndarray):
            dist = torch.from_numpy(dist).float()
        if np.ndim(dist) == 2 and dist.shape[0] == self.num_items and dist.shape[1] == self.num_items:
            # is a square distance matrix
            sym = (dist.T == dist).all()
            items = dist.shape[0]
            upinds = torch.triu_indices(items, items, offset=1)
            if sym:
                dists = (dist[upinds[0], upinds[1]],)
            else:
                print("Similarity graph is directed. Embedding in and out dissimilarity separately.")
                linds = torch.tril_indices(items, items, offset=-1)
                dists = (dist[upinds[0], upinds[1]],
                         dist[linds[0], linds[1]])

        elif np.ndim(dist) == 1 and (dist.shape[0] * 2) / (self.num_items - 1) == self.num_items:
            # is a triangular distance vector
            dists = (dist,)
        else:
            raise ValueError("Provided distance matrix / vector is malformed.")
        return dists

    def embed(self, dist_vec, max_iter=10000):
        history = []
        cur_iter = 0
        embedding = torch.empty((self.num_items, self.components)).to(self.device)
        embedding = torch.nn.Parameter(torch.nn.init.xavier_normal_(embedding))
        optimizer = torch.optim.Adam(lr=.1, params=[embedding])
        dist_vec = dist_vec.to(self.device)
        cur_iter = 0
        while not util.is_converged(history):
            if cur_iter >= max_iter:
                print("WARNING: Failed to converge in", max_iter, "iterations. "
                      "Could be a solution was found, but the convergence tracker is dumb, just be careful!.")
                break
            optimizer.zero_grad()
            loss = self.stress(dist_vec, embedding)
            history.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            cur_iter += 1
        return embedding

    def fit(self, pairwise_distance: torch.Tensor, max_iter=10000):
        dists = self.check_dists(pairwise_distance)
        self.right_latent = self.embed(dists[0], max_iter=max_iter)
        if len(dists) == 2:
            self.left_latent = self.embed(dists[1], max_iter=max_iter)

    def fit_transform(self, pairwise_distance, max_iter=10000):
        self.fit(pairwise_distance, max_iter=max_iter)
        return self.predict()

    def predict(self):
        if self.left_latent is None:
            return self.right_latent.detach().cpu()
        else:
            return self.right_latent.detach().cpu(), self.left_latent.detach().cpu()


