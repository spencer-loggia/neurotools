import numpy as np
import torch
from torch.special import gammainc
from neurotools import util, geometry, embed


def batch_covariance(x: torch.Tensor):
    """
    X : Tensor, batch, examples, features
    """
    x = x.transpose(1, 2)
    means = x.mean(dim=-1, keepdims=True)
    cent = (x - means)
    num = cent @ cent.transpose(1, 2)
    denom = x.shape[2] - 1
    cov = num / denom
    return cov


def symmetric_matrix_sqrt(matrix: torch.Tensor):
    assert matrix.shape[-1] == matrix.shape[-2], "Input must be square matrices"
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0))
    sqrt_matrix = eigvecs @ torch.diag_embed(sqrt_eigvals) @ eigvecs.transpose(-2, -1)
    return sqrt_matrix


def pearson_correlation(x1: torch.Tensor, x2: torch.Tensor, dim=0):
    """
    compute pearson correlation between two vectors in a batched torch friendly wway.
    :param x1: vector1 of size n
    :param x2: vector2 of same size n
    :param dim: if ndims > 1, dimension to compute corr along.
    :return:
    """
    x1_mean = torch.mean(x1, dim=dim)
    x2_mean = torch.mean(x2, dim=dim)
    x1c = (x1 - x1_mean.unsqueeze(dim))
    x2c = (x2 - x2_mean.unsqueeze(dim))
    num = torch.sum(x1c * x2c, dim=dim)
    sum_sq = torch.sum(torch.pow(x1c, 2), dim=dim) * torch.sum(torch.pow(x2c, 2), dim=dim)
    denom = torch.sqrt(sum_sq)
    pearson = num / denom
    return pearson


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute spearman correlation between tow batched vectors in a torch friendly way
    Args:
        x: Shape <n, feat>
        y: Shape <n, feat>
    Returns: rho Tensor <n, 1>
    """
    x_rank = util.get_ranks(x)
    y_rank = util.get_ranks(y)
    rho = pearson_correlation(x_rank, y_rank, dim=1)
    return rho


def _ktau_from_ranks(gt: torch.Tensor, samples: torch.Tensor):
    """
    Computes Kendall's Tau for two tensors without converting to np.

    :param gt: Tensor of shape (batch, features), ground truth rankings.
    :param samples: Tensor of shape (batch, examples, features), sampled rankings.
    :return: Kendall's Tau coefficient tensor of shape (batch, examples).
    """
    batch, examples, features = samples.shape

    # Expand ground truth to match samples shape
    batch_gt_expanded = gt.unsqueeze(1).expand(-1, examples, -1)

    triu = torch.triu_indices(features, features, offset=1)
    # Compute pairwise sign comparisons
    gt_diffs = (batch_gt_expanded.unsqueeze(-2) - batch_gt_expanded.unsqueeze(-1))[..., triu[0, :], triu[1, :]]
    sample_diffs = (samples.unsqueeze(-2) - samples.unsqueeze(-1))[..., triu[0, :], triu[1, :]]

    # get expected number of ties:
    _, counts = gt.unique(return_counts=True, dim=1)
    counts = counts.view((batch, -1))
    gt_ties = torch.sum(counts , dim=1).unsqueeze(1) #  <b, 1, 1>
    # sample ties:
    _, counts = samples.unique(return_counts=True, dim=2)
    counts = counts.view((batch, examples, -1))
    s_ties = torch.sum(counts, dim=2)   # <b, ex, 1>

    gt_signs = torch.sign(gt_diffs)
    sample_signs = torch.sign(sample_diffs)

    # Compute concordant and discordant pairs
    concordant = (gt_signs * sample_signs) > 0
    discordant = (gt_signs * sample_signs) < 0

    # Sum over all pairs
    num_concordant = concordant.sum(dim=(-1, -2))
    num_discordant = discordant.sum(dim=(-1, -2))

    # Compute total number of comparisons
    f_chs_2 = features * (features - 1) // 2
    denom = torch.sqrt((f_chs_2 - gt_ties) * (f_chs_2 - s_ties))   # <b, ex, 1>
    # Compute Kendall's Tau
    tau = (num_concordant - num_discordant).float() / denom

    return tau


def gamma_liklihood(X, a, b):
    """
    X: <batch, examples> Tensor
    a: <batch,> alpha params
    b: <batch,> beta params
    Get the log-liklihood of the gamma distribution given DATA X.
    Based on Minka T., 2002, Estimating a Gamma distribution
    Returns: Tensor <batch,> likilhood for each batch
    """
    n = X.shape[1]
    Xm = torch.mean(X, dim=1)
    t1 = n * (a - 1) * torch.mean(torch.log(X), dim=1)
    t2 = n * torch.lgamma(a)
    t3 = n * a * torch.log(b)
    t4 = n * Xm / b
    l = t1 - t2 - t3 - t4
    return l


def gamma_cdf(a, b, x):
    """
    probability greater than x
    """
    low_incomp_gamma = gammainc(a, x / b)
    return low_incomp_gamma


class GammaDistribution:
    """
    Works quite well for >50 examples.
    """

    def __init__(self, alpha, beta, device="cpu"):
        self.alpha = alpha.to(device)
        self.beta = beta.to(device)
        self.device = device

    def mean(self):
        return self.alpha * self.beta

    def cdf(self, thresh):
        """
        Returns prob that X is less than threshold
        Returns
        """
        return gamma_cdf(self.alpha, self.beta, thresh)

    def score(self, X):
        return gamma_liklihood(X, self.alpha, self.beta)


    def fit(self, X, max_iters=1000):
        # initialize to normal approximation (drastically speeds convergence)
        self.beta = X.std(dim=1).to(self.device)
        self.alpha = torch.nn.Parameter(X.mean(dim=1) / self.beta.detach())
        print("MLE fit", len(X), "Gamma distributions")
        optim = torch.optim.Adam(params=[self.alpha], lr=.0001)
        history = []
        for i in range(max_iters):
            optim.zero_grad()
            self.beta = X.mean(dim=1) / self.alpha # This give MLE of beta given data and alpha
            neglogl = -torch.sum(self.score(X))
            neglogl.backward()
            optim.step()
            history.append(neglogl.cpu().detach().item())
            optim, converged = util.is_converged(history, optim, batch_size=X.shape[0], max_lr=.001, t=i)
            if converged:
                break