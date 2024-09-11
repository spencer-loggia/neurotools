import numpy as np
import torch
from torch.special import gammainc
from neurotools import util, geometry, embed


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
            check_size = 50
            if ((i + 1) % check_size) == 0:
                # reduce learn rate if not improving and check to stop early...
                d_last_block = np.mean(np.array(history[-3 * check_size:-2 * check_size]))
                last_block = np.mean(np.array(history[-2 * check_size:-check_size]))
                block = np.mean(np.array(history[-check_size:]))
                lr = 0.
                for g in optim.param_groups:
                    lr = g['lr']
                    if lr < 1e-9:
                        print("Stop gamma fit after", i, "iters")
                        return
                    print("LR:", lr)
                    if i > 2 * check_size and block >= last_block:
                        # cool down on plateu
                        g['lr'] = g['lr'] * .1
                    elif i > 3 * check_size and block < last_block < d_last_block:
                        # reheat on slope
                        g['lr'] = min(g['lr'] * 2.5, .01)




def batch_mancova(data, targets, return_eig=True):
    """
    Follows algorithm spec in Rencher 2002, Methods of Multivariate Analysis
    Compute the mancova for each group present in targets over data.

    Parameters
    ----------
    data: FloatTensor   shape (batch, examples, features). The input data.
    targets: IntTensor   shape (examples). The group that each example belongs to.
    return_eig: bool    whether to return dissim matrix or eigen-things

    Returns
    -------
    FloatTensor    the wilks trace for each item in batch
    """
    group_labels = list(np.unique(targets))
    group_data = []
    group_means = []
    p = data.shape[-1]
    for t in group_labels:
        g = data[:, targets == t, :]
        group_mean = torch.mean(g, dim=1)  # spatial, kernel
        group_data.append(g - group_mean.unsqueeze(1))  # (centered) spatial, batch, kernel
        group_means.append(group_mean)

    # center the unfolded data  across batch for each feature of each kernel
    unfolded_data = data - data.mean(dim=1).unsqueeze(1)

    # total sum of squares and cross-products matrices
    SSCP_T = unfolded_data.transpose(1, 2) @ unfolded_data  # spatial, kernel, kernel
    SSCP_W = torch.zeros(unfolded_data.shape[0], p, p, dtype=torch.float)

    for i, g in enumerate(group_data):
        # withing group sum of squares and cross-products matrices
        SSCP_W += g.transpose(1, 2) @ g  # spatial kernel kernel

    # between groups sum of squares and cross-products matrices
    SSCP_B = SSCP_T - SSCP_W
    # compute multivarte seperation between groupd
    S = torch.linalg.pinv(SSCP_W) @ SSCP_B  # spatial, kernel, kernel

    if return_eig:
        eig_vals, eig_vecs = torch.linalg.eig(S)  # (S, kernel), (S, kernel, kernel)
        return torch.abs(eig_vals), torch.real(eig_vecs)
    else:
        return S





