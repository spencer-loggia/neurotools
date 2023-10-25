import numpy as np
import torch
from neurotools import util, geometry, embed


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