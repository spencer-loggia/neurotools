import numpy as np
import torch
from scipy.linalg import eigvals

from neurotools import util


class PCA:
    """
    Just a helper class to provide a quick torch based PCA in sklearn style. Used for initialization by MDS.
    """
    def __init__(self, n_components, device):
        self.n_components = n_components
        self.device = device
        self.components = None
        self.var_exp = None

    def fit(self, X):
        """
        :param X: items x features
        :return:
        """
        cov = torch.cov(X.T.to(self.device))
        eigvals, eigvecs = torch.linalg.eig(cov)
        self.var_exp = torch.abs(eigvals[:self.n_components]) / torch.sum(torch.abs(eigvals))
        self.components = eigvecs[:, :self.n_components]

    def predict(self, X):
        """
        :param X: items x features
        :return:
        """
        X = X.type(torch.complex64)
        return X @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.predict(X)


class MDScale:

    def __init__(self, n, embed_dims: int = 2, initialization="pca", device='cpu'):
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
        if initialization in ["pca", "xavier"]:
            self.initialization = initialization
        else:
            raise ValueError("Initialization must be either 'pca' or 'xavier'.")

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

    def stress(self, pairwise_target, positions, order=2):
        """
        An L2 Norm between distance in embedding space an actual provided pairwise distances
        :param positions:
        :param pairwise_target: upper triangular vector of pairwise distance between examples, size n(n-1) / 2
        :return: torch.Tensor a stress score for the system
        """
        cur_dists = torch.pdist(positions)
        # stress = self.mse(cur_dists, pairwise_target)
        stress = torch.abs(cur_dists - pairwise_target)
        stress = torch.pow(stress, order)
        stress = torch.mean(stress)
        return stress

    def embed(self, dist_vec, max_iter=2000, tol=.001):
        history = []
        cur_iter = 0
        if self.initialization == "xavier":
            embedding = torch.empty((self.num_items, self.components)).to(self.device)
            embedding = torch.nn.Parameter(torch.nn.init.xavier_normal_(embedding))
        elif self.initialization == "pca":
            pca = PCA(n_components=self.components, device=self.device)
            embedding = pca.fit_transform(util.triu_to_square(dist_vec, self.num_items).squeeze()).real.float().detach()
            embedding = torch.nn.Parameter(embedding)
        else:
            raise ValueError

        optimizer = torch.optim.Adam(lr=.1, params=[embedding])
        dist_vec = dist_vec.to(self.device)
        cur_iter = 0
        converged = False
        # loss_history, optim, batch_size, t
        while not converged:
            optimizer, converged = util.is_converged(history, optimizer, 1, cur_iter, max_lr=1000)
            if cur_iter >= max_iter:
                print("WARNING: Failed to converge in", max_iter, "iterations. Could be a solution was found, but the "
                                                                  "convergence tracker is dumb, just be careful!.")
                break
            optimizer.zero_grad()
            loss = self.stress(dist_vec, embedding)
            history.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            cur_iter += 1
        self.stress_history = history
        return embedding

    def fit(self, pairwise_distance: torch.Tensor, max_iter=10000, tol=.001):
        dists = self.check_dists(pairwise_distance)
        self.right_latent = self.embed(dists[0], max_iter=max_iter)
        print("Right Initial System Tension: ", self.stress_history[0])
        print("Right Final System Tension: ", self.stress_history[-1])
        if len(dists) == 2:
            self.left_latent = self.embed(dists[1], max_iter=max_iter, tol=tol)
            print("Left Initial System Tension: ", self.stress_history[0])
            print("Left Final System Tension: ", self.stress_history[-1])

    def fit_transform(self, pairwise_distance, max_iter=10000, tol=.001):
        self.fit(pairwise_distance, max_iter=max_iter, tol=tol)
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
        Used in "Loggia, S. R., Duffield, S. J., Braunlich, K., & Conway, B. R. (2025).
        Color and spatial frequency provide functional signatures of retinotopic visual areas.
        Journal of Neuroscience, 45(2)."

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

    def feature_ln(self, comp, degree):
        return torch.sum(torch.pow(torch.abs(comp), degree), dim=0).mean()


    def fit(self, X, y, max_iter=10000, verbose=False, converge_var=.01, degree=2., dist_mask=None, bootstrap_iter=100):
        """
        :param converge_var:
        :param verbose:
        :param X: items x features
        :param y: target class
        :param max_iter:
        :param dist_mask: can order model to not consider separation between certain targets.
        :return:
        """
        n_features = X.shape[1]
        X = X.to(self.device)
        y = y.to(self.device)
        unique_targets = torch.unique(y)
        if dist_mask is not None:
            triu = torch.triu_indices(len(unique_targets), len(unique_targets), offset=1)
            dist_mask = dist_mask[triu[0], triu[1]]
        self.components = torch.empty((n_features, self.n_components))
        self.components = torch.nn.Parameter(torch.nn.init.kaiming_normal_(self.components).to(self.device))
        cur_iter = 0
        history = []
        optimizer = torch.optim.AdamW(lr=.01, params=[self.components])
        converged = False
        while not converged:
            if cur_iter >= max_iter:
                print("WARNING: Failed to converge in", max_iter, "iterations. Could be a solution was found, but the "
                                                                  "convergence tracker is dumb, just be careful!.")
                break
            optimizer.zero_grad()
            std_components = self.get_norm_components()
            embed = X @ self.components
            centers = []
            loss = torch.zeros((1,)).to(self.device)
            for t in unique_targets:
                class_emb = embed[y == t]
                center = torch.mean(class_emb, dim=0)
                centers.append(center)
                var = torch.abs(torch.cov(class_emb.T)).mean()
                loss += var
            loss = (loss / len(unique_targets)) * self.intra_weight
            centers = torch.stack(centers, dim=0)
            space = torch.pow(torch.pdist(centers), degree)
            if dist_mask is not None:
                space = space * dist_mask
            space = torch.pow(space.mean(), (1 / degree))
            loss = loss - (self.inter_weight * space)
            loss = loss + self.sparsity * self.feature_ln(std_components, 1.00)
            if torch.isnan(loss):
                raise ValueError("SL: numerical instability, loss is non-finite. Try again. "
                                 "If recurring, adjust hyperparams")
            history.append(loss.detach().cpu().item())
            if verbose:
                print("iteration", cur_iter, "loss is", history[-1])
            loss.backward()
            optimizer.step()
            optimizer, converged = util.is_converged(history, optimizer, 1, max_lr=.1, t=cur_iter)
            cur_iter += 1
        print("done in", cur_iter, "iterations")

    def get_norm_components(self):
        if self.components is None:
            print("Must fit first.")
            return
        return self.components / torch.linalg.norm(self.components, dim=0)

    def predict(self, X):
        embed = X.cpu() @ self.components
        return embed.detach().cpu()

