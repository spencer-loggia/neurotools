import torch
from neurotools import util
from typing import Union
import itertools
import numpy as np
import math

"""
Torch style modules.
"""


class SpatialBN(torch.nn.Module):
    """
    Impliments a nonparametric varient of batch normalization over each channel and spatial dimension
    """

    def __init__(self, channels, train=True, n_classes=4, bs_mod = 1., device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_trian = train
        self.last = None
        self.std_memory = None
        self.mean_memory = None
        self.n_classes = n_classes
        self.train_std = None
        self.train_mean = None
        self._bs_mod = bs_mod
        self.offset = torch.nn.Parameter(torch.tensor([0.]*channels, device=device, dtype=torch.float))
        self.scale = torch.nn.Parameter(torch.tensor([1.]*channels, device=device, dtype=torch.float))

    def train(self, mode=True):
        self.to_trian = mode

    def eval(self):
        self.train(mode=False)

    def forward(self, X):
        spatial = X.shape[2:]
        bs = int(math.ceil(X.shape[0] * self._bs_mod))
        iterk = 100
        i = iterk / bs  # batches to 100 examples of each class
        # want running mean at t-100 to contribute .05
        run_discount = 0.
        # want memory mean at t-100 to contribute .25
        mem_discount = .25 ** (1 / i)
        if self.last is None:
            self.last = X.detach().clone()
        m = X.mean(dim=0).view((1, -1,) + spatial)
        s = X.std(dim=0).view((1, -1,) + spatial) + 1e-8
        if self.std_memory is None:
            self.std_memory = s.detach()
            self.mean_memory = m.detach()
            self.train_mean = m
            self.train_std = s
        else:
            self.std_memory = mem_discount * self.std_memory + (1 - mem_discount) * s.detach()
            self.mean_memory = mem_discount * self.mean_memory + (1 - mem_discount) * m.detach()

            self.train_std = run_discount * self.train_std.detach() + (1 - run_discount) * s
            self.train_mean = run_discount * self.train_mean.detach() + (1 - run_discount) * m
        if self.to_trian:
            m = self.train_mean.clone()
            s = self.train_std.clone()
            # add some noise
            # n = torch.empty_like(X)
            # torch.normal(0., s * .01, out=n)
            X = X
        else:
            if self.std_memory is None:
                raise ValueError("Must train SpatialBN first")
            m = self.mean_memory
            s = self.std_memory
        offset = self.offset.view((1, -1,) + tuple([1]*len(spatial)))
        # scale only rebalances.
        scale = len(self.scale) * self.scale / torch.sum(self.scale)
        scale = scale.view((1, -1,) + tuple([1]*len(spatial)))
        X = offset + scale * (X - m) / s
        return X

    def clone(self):
        new_bn = SpatialBN(n_classes=self.n_classes, channels=len(self.offset))
        new_bn.offset = torch.nn.Parameter(self.offset.clone())
        new_bn.scale = torch.nn.Parameter(self.scale.clone())
        if self.mean_memory is not None:
            new_bn.mean_memory = self.mean_memory.clone()
            new_bn.std_memory = self.std_memory.clone()
        if self.train_mean is not None:
            new_bn.train_std = self.train_std.clone()
            new_bn.train_mean = self.train_mean.clone()
        new_bn.to_trian = self.to_trian
        return new_bn


class VarConvND(torch.nn.Module):
    """
    Applies a differented weighted sum to every patch of the input data.
    """
    def __init__(self, kernel_size: Union[tuple[int], int], padding: Union[tuple[int], int], in_channels: int,
                 out_channels: int, spatial: Union[tuple[int], int], ndims=2, stride=1, bias=True,
                 device="cpu", dtype=torch.float, generator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # weight (spatial, c_in * kernel, c_out)
        self.spatial = spatial
        self.padding = padding
        self.kernel_size = kernel_size
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size,) * ndims
        if type(spatial) is int:
            self.spatial = (spatial,) * ndims
        if type(padding) is int:
            self.padding = (padding,) * 2 * ndims
        elif len(padding) == ndims:
            self.padding = list(itertools.chain(*[[padding[i]] * 2 for i in range(ndims)]))
        if generator is None:
            generator = torch.Generator(device=device)
        self.device = device
        self.stride = stride
        self.ndims = ndims
        self.use_bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        s = np.array(self.spatial)
        k = np.array(self.kernel_size)
        self.k = k
        p = np.array(self.padding).reshape((-1, 2)).sum(axis=1)
        # compute input spatial after pad and stride
        in_spatial: np.ndarray = (((s - k + p) / stride) + 1).astype(int)
        weight_shape = (int(np.prod(in_spatial)),
                        in_channels * np.prod(k),
                        out_channels)
        self.in_spatial: list = in_spatial.tolist()
        self.train()
        # initialize weights following xavier protocol
        weights = torch.empty(weight_shape, dtype=dtype, device=device)
        weights = torch.nn.init.kaiming_normal_(weights, nonlinearity="linear", generator=generator)
        # track weights as torch Parameters
        self.weight = torch.nn.Parameter(weights)
        if bias:
            bias = torch.empty(size=self.in_spatial, dtype=dtype, device=device)
            bias = torch.nn.init.xavier_uniform(bias)
            # track weights as torch Parameters
            self.bias = torch.nn.Parameter(bias)

    def _unfold(self, X):
        return  util.unfold_nd(X, kernel_size=self.kernel_size, spatial_dims=self.ndims,
                                padding=self.padding, stride=self.stride)

    def forward(self, X):
        batch_size = X.shape[0]
        ufld_x = self._unfold(X)
        if self.use_bias:
            ufld_x = ufld_x + self.bias.view((1, 1, -1))

        iterrule = "bks,skc->bcs"
        # mapping input kernel to out channels in next layer for each example for each location in space,
        h = torch.einsum(iterrule, ufld_x, self.weight)  # batch (b), hidden channels, spatial (s)
        # fold to next layer.
        h = h.view([batch_size, self.out_channels] + self.in_spatial)
        return h

    def orthoganality(self):
        # give the orthogonality along the channel dimension of the weights
        w = self.weight  # / torch.linalg.norm(self.weight, dim=1).unsqueeze(1) # <s, ck, o>
        # w = w.reshape((-1, self.out_channels))  # <s, ck, o>
        w = w.unsqueeze(2)  # add dimenstion for outer
        # need to compute pairwise dot product for all output channels
        w = w.transpose(2, 3) @ w
        # zero diagonal (don't wont to penalize itselfS
        w = w * (1 - torch.eye(self.out_channels, device=self.device))[None, None, ...]
        o = w.sum(dim=1)  # dot product along feature dims
        o = torch.abs(o).sum()  # combined absolute value of all dot products
        return o

    def train(self, mode: bool = True):
        self.to_train = mode

    def eval(self):
        self.train(mode=False)


class BalancedCELoss(torch.nn.Module):
    """
    """

    def __init__(self, nclasses, device="cpu", rebalance=True, spatial=None, margin=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nclasses = nclasses
        self.loss_fxn = torch.nn.NLLLoss(reduction="none")
        self.exp_scores = int(nclasses * (nclasses - 1) / 2)
        self.use_rebalancing = rebalance
        self.spatial = spatial
        if spatial is None:
            spatial = (1,)
        self.rebalance = (1 / self.nclasses) * torch.ones((self.nclasses,) + (math.prod(spatial),), device=device)
        self.chance_ce = self.loss_fxn(torch.log_softmax(torch.tensor([0.] * 3), dim=0).unsqueeze(0),
                                       torch.tensor([1], dtype=torch.long)).sum()

    def forward(self, X, y, true_target=None):
        """
        Args:
            X: <n, c, ...>
            y: <n,>

        Returns:
        """
        tailing_dim = X.ndim - 2
        y = torch.tile(y.view((-1,) + tuple([1] * tailing_dim)), ((1,) + X.shape[2:]))
        loss = self.loss_fxn(X, y)

        if self.use_rebalancing:
            if self.spatial is None:
                sum_dim = None
            else:
                sum_dim = 0
            if true_target is None:
                true_target = y
            max_mix = .75
            balanced_loss = torch.zeros_like(self.rebalance)
            class_occ = torch.tensor([0] * self.nclasses, device=X.device)
            # compute loss by class
            for c in range(self.nclasses):
                ind = true_target==c
                class_occ[c] = ind.sum()
                if class_occ[c] == 0:
                    class_occ[c] = 1
                    balanced_loss[c] = self.chance_ce
                else:
                    closs = loss[true_target == c].sum(dim=sum_dim)
                    balanced_loss[c] = closs
            with torch.no_grad():
                # need to change into 0 to 1 vector.
                class_occ = class_occ.unsqueeze(1)
                mix_frac = max_mix * class_occ / class_occ.sum()
                mean_loss = (balanced_loss / class_occ) ** 2
                total_m_loss = mean_loss.sum(dim=sum_dim, keepdim=True)
                norm_loss = mean_loss / total_m_loss # sums to one, percent of total loss.
                # mix with past rebalancing using mix factor
                self.rebalance = (1 - mix_frac) * self.rebalance + mix_frac * norm_loss
            # balance all losses by rebalancing factor
            balanced_loss = self.nclasses * balanced_loss * self.rebalance.detach()
            loss = balanced_loss.sum()

        return loss