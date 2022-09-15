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
        self.stress_history = None

    def to(self, device):
        self.device = device
        return self

    def check_dists(self, dist):
        """
        function to check if distances are a vector or matrix, if a matrix returns just the flattened upper triangle if
        symetric, or a tuple of upper and lower (flat) triangles if not symmetric.
        :param num_items: number of items pairwise dists are for. (num rows and cols of distance matrix)
        :param dist:
        :return:
        """
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

    def stress(self, pairwise_target, positions):
        """
        An L2 Norm between distance in embedding space an actual provided pairwise distances
        :param positions:
        :param pairwise_target: upper triangular vector of pairwise distance between examples, size n(n-1) / 2
        :return: torch.Tensor a stress score for the system
        """
        cur_dists = torch.pdist(positions)
        # stress = self.mse(cur_dists, pairwise_target)
        stress = torch.abs(cur_dists - pairwise_target)
        stress = torch.pow(stress, 1.5)
        stress = torch.mean(stress)
        return stress

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
        self.stress_history = history
        return embedding

    def fit(self, pairwise_distance: torch.Tensor, max_iter=10000):
        dists = self.check_dists(pairwise_distance)
        self.right_latent = self.embed(dists[0], max_iter=max_iter)
        print("Right Initial System Tension: ", self.stress_history[0])
        print("Right Final System Tension: ", self.stress_history[-1])
        if len(dists) == 2:
            self.left_latent = self.embed(dists[1], max_iter=max_iter)
            print("Left Initial System Tension: ", self.stress_history[0])
            print("Left Final System Tension: ", self.stress_history[-1])

    def fit_transform(self, pairwise_distance, max_iter=10000):
        self.fit(pairwise_distance, max_iter=max_iter)
        return self.predict()

    def predict(self):
        if self.left_latent is None:
            return self.right_latent.detach().cpu()
        else:
            return self.right_latent.detach().cpu(), self.left_latent.detach().cpu()


class SupervisedEmbed:

    def __init__(self, n_components=2, device='cpu', dist_metric="euclidian",
                 intra_class_weight=1., inter_class_weight=1., sparsity=1):
        """
        Finds three feature weight vectors that maximize the distance between the classes while minimizing the variance
        within each class and maintaining component sparsity.
        :param n_components: number of features to find
        :param device: hardware to use
        """
        self.n_components = n_components
        self.components = None
        self.device = device
        self.metric = dist_metric
        self.sparsity = sparsity
        self.intra_weight = intra_class_weight
        self.inter_weight = inter_class_weight

    def to(self, device):
        if self.components is not None:
            self.components = self.components.to(device)
        self.device = device
        return self

    def feature_l1(self, comp):
        comp_mag = torch.abs(comp)
        return torch.sum(comp_mag.view(-1))

    def fit(self, X, y, max_iter=10000, verbose=False, converge_var=.01):
        """
        :param converge_var:
        :param verbose:
        :param X: items x features
        :param y: target class
        :param max_iter:
        :return:
        """
        n_features = X.shape[1]
        X = X.to(self.device)
        y = y.to(self.device)
        unique_targets = torch.unique(y)
        self.components = torch.empty((n_features, self.n_components))
        self.components = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.components).to(self.device))
        cur_iter = 0
        history = []
        optimizer = torch.optim.Adam(lr=.01, params=[self.components])
        while not util.is_converged(history, abs_tol=converge_var):
            if cur_iter >= max_iter:
                print("WARNING: Failed to converge in", max_iter, "iterations. "
                      "Could be a solution was found, but the convergence tracker is dumb, just be careful!.")
                break
            optimizer.zero_grad()
            std_components = self.get_components()
            embed = X @ std_components
            centers = []
            loss = torch.zeros((1,)).to(self.device)
            for t in unique_targets:
                class_emb = embed[y == t]
                center = torch.mean(class_emb, dim=0)
                centers.append(center)
                var = torch.std(class_emb, dim=0).mean()
                loss += var
            loss = (loss / len(unique_targets)) * self.intra_weight
            centers = torch.stack(centers, dim=0)
            space = torch.pdist(centers)
            space = space.mean()
            loss = loss - (self.inter_weight * space)
            loss = loss + self.sparsity * self.feature_l1(std_components)
            history.append(loss.detach().cpu().item())
            if verbose:
                print("iteration", cur_iter, "loss is", history[-1])
            loss.backward()
            optimizer.step()
            cur_iter += 1
        print("done in", cur_iter, "iterations")

    def get_components(self):
        if self.components is None:
            print("Must fit first.")
            return
        std_components = (self.components - self.components.mean(dim=0).unsqueeze(0)) / self.components.std(dim=0).unsqueeze(0)
        return std_components

    def predict(self, X):
        components = self.get_components()
        embed = X.cpu() @ components
        return embed.detach().cpu()



