import copy
import datetime
import itertools
import math
import os
import random
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn
from scipy import ndimage
from torch.nn.functional import conv3d, conv2d
from matplotlib import pyplot as plt
import pickle as pk
from neurotools import util
import sys

try:
    from captum.attr import DeepLift
except Exception:
    print("Captum module not found, deeplift attribution method not available")


def compute_acc(y, yhat, top=1):
    """
    :param y: 1d of target class labels
    :param yhat: 2d of class scores
    :return:
    """
    correct = (torch.argsort(yhat, dim=1)[:, -top:].int() == y.view(-1, 1)).sum(dim=1)
    acc = 100 * torch.count_nonzero(correct) / len(y)
    return acc.detach().cpu().item()


class VarConvND(torch.nn.Module):
    """
    Module to implement stackable searchlight style convolution (filter differs over space)
    """
    def __init__(self, kernel_size: Union[tuple[int], int], padding: Union[tuple[int], int], in_channels: int,
                 out_channels: int, spatial: Union[tuple[int], int], ndims=2, stride=1, bias=True, standardize=False,
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
        self.standardize = standardize
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
        if standardize:
            self.normalizer = SpatialBN(channels=self.in_channels, bs_mod=(1 / np.prod(k)), device=self.device)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, X):
        batch_size = X.shape[0]
        ufld_x = util.unfold_nd(X, kernel_size=self.kernel_size, spatial_dims=self.ndims,
                                padding=self.padding, stride=self.stride)
        if self.use_bias:
            ufld_x = ufld_x + self.bias.view((1, 1, -1))
        if self.standardize:
            # get channels and kernels on seperate dimension, thn combined the kernel and batch dimensions for input to
            # Spatial BN
            ufld_x = ufld_x.reshape([batch_size, self.in_channels, np.prod(self.k), -1])
            ufld_x = ufld_x.transpose(1, 2).reshape([batch_size * np.prod(self.k), self.in_channels, -1])
            ufld_x = self.normalizer(ufld_x) # apply spatial bn
            # return to desired dimmensions
            ufld_x = (ufld_x.reshape([batch_size, np.prod(self.k), self.in_channels, -1]).transpose(1, 2).
                      reshape([batch_size, self.in_channels * np.prod(self.k), -1]))
            # take mean of batch and kernels

        iterrule = "bks,skc->bcs"
        # mapping input kernel to out channels in next layer for each example for each location in space,
        h = torch.einsum(iterrule, ufld_x, self.weight)  # batch (b), hidden channels, spatial (s)
        # fold to next layer.
        h = h.view([batch_size, self.out_channels] + self.in_spatial)
        return h

    def orthoganality(self):
        # give the orthogonality along the channel dimension of the weights
        w = self.weight # / torch.linalg.norm(self.weight, dim=1).unsqueeze(1) # <s, ck, o>
        #w = w.reshape((-1, self.out_channels))  # <s, ck, o>
        w = w.unsqueeze(2) # add dimenstion for outer
        # need to compute pairwise dot product for all output channels
        w = w.transpose(2, 3) @ w
        # zero diagonal (don't wont to penalize itselfS
        w = w * (1 - torch.eye(self.out_channels, device=self.device))[None, None, ...]
        o = w.sum(dim=1) # dot product along feature dims
        o = torch.abs(o).sum() # combined absolute value of all dot products
        return o

    def train(self, mode: bool = True):
        self.to_train = mode

    def eval(self):
        self.train(mode=False)


class BalancedCELoss(torch.nn.Module):
    """
    Gets upper triangle of pairwise log odds for each class
    Evaluates probability at c out of c(c-1)/2 scores for each class.
    Transforms multiclass into a pairwise binary classification problem,
    But allows use of shared network resource. i.e. classes only need
    separate on last layer.
    """

    def __init__(self, nclasses, device="cpu", rebalance=False, margin=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nclasses = nclasses
        self.loss_fxn = torch.nn.NLLLoss(reduction="sum")
        self.exp_scores = int(nclasses * (nclasses - 1) / 2)
        self.use_rebalancing = rebalance
        self.rebalance = (1 / self.nclasses) * torch.ones((self.nclasses,), device=device)

    def forward(self, X, y):
        """
        Args:
            X: <n, c, ...>
            y: <n,>

        Returns:
        """
        tailing_dim = X.ndim - 2
        y = torch.tile(y.view((-1,) + tuple([1] * tailing_dim)), ((1,) + X.shape[2:]))
        loss = self.loss_fxn(X, y)
        return loss


def subset_accuracy(X, y, pairwise_weights=None):
    correct = []
    tailing_dim = X.ndim - 2
    scores = X
    target = y.reshape((-1,) + tuple([1]*tailing_dim))
    pred = torch.argmax(scores, dim=1)
    class_correct = pred == target
    correct.append(class_correct)
    correct = torch.concatenate(correct, dim=0).float()
    acc = torch.mean(correct, dim=0)
    return np.zeros((1, 1)), acc.detach().cpu().numpy().squeeze()


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


class ROISearchlightDecoder():
    def __init__(self, atlas: np.ndarray, lookup: dict, set_names: Tuple[str], nonlinear=True, spatial=(64, 64, 64),
                 in_channels=2, n_classes=2, device="cuda", pairwise_weights=None, sep_class_weighting=False,
                 sessions=None, n_layers=3,
                 base_kernel_size=2, smooth_kernel_sigma=1.0, model_size=1, dropout_prob=.2, seed=42, share_conv=False,
                 reweight_factors=False):
        """
        atlas
        lookup
        set_names
        nonlinear
        spatial
        in_channels
        n_classes
        device
        pairwise_weights
        sessions
        n_layers
        base_kernel_size
        smooth_kernel_sigma
        model_size
        dropout_prob
        """
        self.atlas = atlas
        self.seed = seed
        self.lookup = lookup
        self.roi_names = list(self.lookup.values())
        self.num_rois = len(self.lookup)
        self.set_names = set_names
        self.dim = len(spatial)
        self.in_spatial = spatial
        self.n_classes = n_classes
        self.pad = [0, 1]
        self.channels = in_channels
        self._train_set = set_names[0]
        self._train_model = True
        self._train_mask = True
        self.device = device
        self.sessions = sessions
        self.smooth_sigma = .7
        self.model_size = model_size
        self.sep_class_weight = sep_class_weighting
        self.n_layers = n_layers
        self.base_kernel_size = base_kernel_size
        self.rewieght_factors = reweight_factors

        if pairwise_weights is None:
            self.pairwise_weights = torch.ones((n_classes, n_classes, n_classes), device=self.device)
        else:
            self.pairwise_weights = pairwise_weights  # mask to use for each input class. Some classes should only be compared to some others.
        if len(spatial) == 2:
            drop = torch.nn.Dropout2d
        elif len(spatial) == 3:
            drop = torch.nn.Dropout3d
        self.out_feat = n_classes
        self.search_dropout = drop(p=dropout_prob)
        self.predictor_dropout = torch.nn.Dropout(p=0.25)
        if nonlinear:
            self.activation = torch.nn.Tanh()
        else:
            self.activation = torch.nn.Identity()
        self._smooth_kernel_size = 5
        self.smooth_kernel = self._create_smoothing_kernel(self._smooth_kernel_size)
        self.total_features = 0
        self.share_conv = share_conv
        self.lin_reg_coef = 1e-5 # 1e-2 # 0. # 1e-1 # 1e-3 # 1e-5 #1e-2 #0.00001  # 0.0001
        self.spatial_reg_coef = 1e-6 #1e-6 # 1e-2 #  1e-3 # 1e-7 # 1e-6 # 1e-5 # 1e-7 #0.0001  # 0.0001
        self.mask = np.zeros(self.in_spatial)
        for roi_ind in self.lookup.keys():
            roi = self.atlas == roi_ind
            roi_size = np.count_nonzero(roi)
            self.total_features += roi_size
            self.mask[roi] = 1
        self.total_features = math.prod(spatial)
        self.mask = torch.from_numpy(self.mask).to(self.device).float()
        self.initialize_params()

    def initialize_params(self):
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        if self.n_layers < 1: raise ValueError
        if self.rewieght_factors and self.n_layers != 2: raise ValueError
        self.conv_layers = []
        # choose intermediate channels size balancing dimensionality and number of targets.
        base_chan = max(1, math.log2(self.n_classes * (self.dim - 1)) - 1)
        # this padding scheme ensures size remains constant, but we must reverse every layer so we don't get offset from
        # input
        p = self.base_kernel_size - 1
        if self.dim == 2:
            forward_padding = (0, p, 0, p)
            reverse_padding = (p, 0, p, 0)
        else:
            forward_padding = (0, p, 0, p, 0, p)
            reverse_padding = (p, 0, p, 0, p, 0)

        if self.share_conv:
            # usig full field conv with higher channel count
            mod = 1.  # round(math.log10(self.total_features))
            forward_padding = p
            reverse_padding = 0
        else:
            mod = 1

        self.conv_layers = []
        self.bn_layers = []
        for l in range(self.n_layers):
            in_chan = int(round(mod * math.ceil(base_chan * self.model_size)))
            out_chan = int(round(mod * math.ceil(base_chan * self.model_size)))
            if l == 0:
                in_chan = self.channels
            self.bn_layers.append({s: SpatialBN(device=self.device, channels=in_chan, n_classes=self.n_classes) for s in self.set_names})
            if l == self.n_layers - 1:
                out_chan = self.n_classes
                # factors weighting is applied to last layer before output.
                self.factor_weight = None # {s: torch.nn.Parameter(torch.ones((in_chan,) + self.in_spatial, device=self.device, dtype=torch.float)) for s in self.set_names}
            if l % 2 == 0:
                pad = forward_padding
            else:
                pad = reverse_padding

            if self.share_conv:
                self.conv_layers.append(
                    torch.nn.Conv3d(in_channels=in_chan, out_channels=out_chan, kernel_size=self.base_kernel_size,
                                    padding="same", device=self.device, bias=False))
            else:
                self.conv_layers.append(
                    VarConvND(in_channels=in_chan, out_channels=out_chan, kernel_size=self.base_kernel_size,
                              padding=pad, spatial=self.in_spatial, ndims=self.dim, bias=False, standardize=False,
                              dtype=torch.float, device=self.device, generator=generator))
        base_chan = int(base_chan)
        self.bn = {}
        self.sess_chan_map = {}
        # initialize weights from searchlights to rois
        self.roi_weights = {}
        self.session_weights = {}
        # if sessions weights are provided, we rewieght channels after layer 2 by session. (Defuct)
        if self.sessions is None:
            self.session_weights["all"] = torch.nn.Parameter(torch.ones((base_chan,), device=self.device))
        else:
            for s in self.sessions:
                self.session_weights[s] = torch.nn.Parameter(torch.ones((base_chan,), device=self.device))
        for s in self.set_names:
            if self.sep_class_weight:
                # we fit separate weights for each class vs the others
                weight = torch.empty((self.n_classes,) + self.in_spatial, device=self.device, dtype=torch.float)
            else:
                # just one weight at each loc in space
                weight = torch.empty(self.in_spatial, device=self.device, dtype=torch.float)
            weight = torch.nn.Parameter(torch.nn.init.ones_(weight).float())
            self.roi_weights[s] = weight

    def reset_weights(self, set):
        weight = torch.empty(self.in_spatial, device=self.device, dtype=torch.float)
        weight = torch.nn.Parameter(torch.abs(torch.nn.init.ones_(weight)))
        self.roi_weights[set] = weight

    def train_searchlight(self, set, on=True):
        if set not in self.set_names:
            raise ValueError
        self._train_set = set
        self._train_model = on

    def train_predictors(self, set: str, on=True):
        if set not in self.set_names:
            raise ValueError
        self._train_set = set
        self._train_mask = on

    def eval(self, set):
        if set not in self.set_names:
            raise ValueError
        self._train_model = False
        self._train_set = set
        self._train_mask = False

    def to(self, dev):
        return self

    def _create_smoothing_kernel(self, kernel_size):
        # Create the Gaussian kernel
        sigma = self.smooth_sigma
        if self.dim == 2:
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, kernel_size // 2] = 1
            kernel = ndimage.gaussian_filter(kernel, sigma)

        elif self.dim == 3:
            kernel = np.zeros((kernel_size, kernel_size, kernel_size))
            kernel[kernel_size // 2, kernel_size // 2, kernel_size // 2] = 1
            kernel = ndimage.gaussian_filter(kernel, sigma)
        else:
            raise ValueError
        kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return kernel_tensor

    def gaussian_smoothing(self, tensor, unit_range=True, *args):
        return tensor
        stride = 1  # self.smooth_kernel_size // 2
        oshape = tensor.shape
        sk = self.smooth_kernel.tile((1, 1) + tuple([1] * self.dim)).float()
        if unit_range:
            tensor = (tensor)
        tensor = tensor.view((tensor.shape[0] * tensor.shape[1], 1) + tensor.shape[2:])  # channels to batch
        if self.dim == 3:
            smoothed_tensor = conv3d(tensor, sk, stride=stride, padding=self._smooth_kernel_size // 2)
        elif self.dim == 2:
            smoothed_tensor = conv2d(tensor, sk, stride=stride, padding=self._smooth_kernel_size // 2)
        else:
            raise RuntimeError("Working dimensionality must be either 2 or 3")
        if smoothed_tensor.shape[-self.dim:] != self.in_spatial:
            upsampler = torch.nn.Upsample(size=self.in_spatial, mode="nearest")
            smoothed_tensor = upsampler(smoothed_tensor)
        smoothed_tensor = smoothed_tensor.view(oshape)  # batch to channels
        return smoothed_tensor

    def roi_step(self, spatial_logits, top30=False):
        """
        Get scores for each ROI
        Args:
            stim: <batch, c_in, s1, s2, s3>

        Returns: dict roi logits
        """
        roi_logits = {}
        first_dim = spatial_logits.shape[:2]
        spatial_logits = spatial_logits.reshape(first_dim + (-1,))
        for i, l in enumerate(self.lookup.keys()):
            ryhat = spatial_logits[..., (self.atlas == l).flatten()]  # <batch, cl, roi>
            n = self.lookup[l]
            if top30:
                sinds = torch.argsort(torch.max(ryhat, dim=1)[0].mean(dim=0))
                ryhat = ryhat[:, :, : sinds[-30:]]
            roi_logits[n] = ryhat.logsumexp(dim=2)
        return roi_logits

    def global_step(self, spatial_logits, top30=False):
        """
        Args:
            stim: <batch, c_in, s1, s2, s3>
        Returns: dist roi logits, global logits
        """
        batch_size = spatial_logits.shape[0]

        # generate smmothed weights
        # if not (self._train_model and not self._train_mask):
        weights = self.roi_weights[self._train_set]
        weights = weights.view((1, 1,) + self.in_spatial)
        if self.sep_class_weight:
            # seperate for each class
            weights = self.gaussian_smoothing(weights.view((1, self.n_classes,) + self.in_spatial))
        else:
            weights = self.gaussian_smoothing(weights.view((1, 1,) + self.in_spatial))

        # compute regularization
        reg = torch.tensor([0.], device=self.device)
        spatial_logits = spatial_logits
        # spatial_logits = torch.tanh(spatial_logits)

        if self._train_model:
            # made redundant by orthogonality term
            w_l2 = torch.sum(torch.stack([torch.sum(torch.pow(w.weight, 2))
                                          for w in self.conv_layers]))
            reg += self.spatial_reg_coef * w_l2

        if self._train_mask:
            rw_l2 = torch.sum(torch.square(weights))
            reg += 1.0 * self.lin_reg_coef * rw_l2

        spatial_logits = torch.log_softmax(spatial_logits, dim=1)
        spatial_logits = torch.clip(spatial_logits, -25, 0)

        if top30:
            sinds = torch.argsort(torch.max(spatial_logits, dim=1)[0].mean(dim=0))
            spatial_logits = spatial_logits[:, :, :, sinds[-30:]]

        if self._train_mask:
            weights = self.predictor_dropout(weights)
        weight_dist = torch.log_softmax(weights.flatten(), dim=0).view(weights.shape)
        weight_dist = torch.clip(weight_dist, -25, 0)
        spatial_logits = spatial_logits + weight_dist

        entropy = -.01 * torch.sum(self.total_features * torch.exp(weight_dist) * weight_dist)
        spatial_entropy = -.001 * torch.sum(torch.exp(spatial_logits) * spatial_logits)
        reg = reg - entropy - spatial_entropy

        spatial_logits = spatial_logits.view((batch_size, spatial_logits.shape[1],  -1))

        return spatial_logits, reg

    def shared_cov_step(self, stim, sess=None):
        h = stim.float()
        # run through layers
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if i < len(self.conv_layers) - 1:
                h = self.activation(h)  # <batch, c, x, y, z>
                h = self.bn_layers[i+1][self._train_set](h)
                if self._train_model:
                    h = self.search_dropout(h)

        return h

    def subset(self, X, y, pairwise_weights):
        """
        Extract the logits we actually want to compare
        """
        all_scores = []
        targets = []
        for c in range(self.n_classes):
            d = X[y == c] # get entries corresponding to specific target
            # ask each relevant regressor, is c? early index class is always consider "TRUE"
            # select scores to compare
            scores = d
            target = torch.tensor([c]).to(X.device).to(torch.long)
            if pairwise_weights is not None:
                class_weights = pairwise_weights[c, c].flatten() #.reshape((1, pairwise_weights.shape[1]) + tuple([1] * tailing_dim))
            else:
                class_weights = torch.ones((self.nclasses,))
            scores = scores[np.arange(len(scores))[:, None], torch.nonzero(class_weights).T, ...] # n, rc, ...
            # scores = scores.transpose(0, 1).reshape((n_select, -1)).T
            # need to reindex target
            num_removed = torch.count_nonzero(class_weights.flatten()[:c]==0)
            target = target - num_removed
            target = torch.tile(target, (len(scores),))
            all_scores.append(scores)
            targets.append(target)
        all_scores = torch.concatenate(all_scores, dim=0)
        targets = torch.concatenate(targets, dim=0)
        return all_scores, targets

    def forward(self, stim, targets, sess=None, top30=False):
        # compute regularization penalty
        if self._train_model:
            spatial_logits = self.shared_cov_step(stim, sess=sess)
        else:
            with torch.no_grad():
                spatial_logits = self.shared_cov_step(stim, sess=sess)
        sub_logits, sub_targets = self.subset(spatial_logits, targets, self.pairwise_weights)
        sub_logits = sub_logits.view((-1, sub_logits.shape[1]) + self.in_spatial)
        logits, reg = self.global_step(sub_logits, top30=top30)  # <n, c, c, x, y, z>
        return logits, sub_targets, reg

    def fit(self, dataloader, lr=.01):
        # loss_fxn = torch.nn.MultiMarginLoss(p=2, margin=scale / 2)
        bn_train = False
        if self._train_model or self._train_mask:
            bn_train = True
        for i in range(self.n_layers):
            # set the batch norm layers equal to those for the MAIN idx 0 set on fitting start
            self.bn_layers[i][self._train_set] = self.bn_layers[i][self.set_names[0]].clone()
            self.bn_layers[i][self._train_set].train(mode=bn_train)
            # set the normalizer to the spatial BN layer for this set type
            self.conv_layers[i].normalizer = self.bn_layers[i][self._train_set]
        loss_fxn = BalancedCELoss(nclasses=self.n_classes, device=self.device, rebalance=False)
        loss_history = []

        bn_params = []
        sess_params = [self.session_weights[s] for s in self.session_weights.keys()]
        sess_params = sess_params
        #if self._train_set != self.set_names[0]:
        factor_param = [] # [self.factor_weight[self._train_set]]
        #else:
        #    factor_param = []
        search_params = []
        for l in self.conv_layers:
            search_params += list(l.parameters())
        for b in self.bn_layers:
            bn_params += list(b[self._train_set].parameters())

        if self._train_model and self._train_mask:
            # set optim to use
            params = search_params + [self.roi_weights[self._train_set]]
        elif self._train_mask:
            params = [self.roi_weights[self._train_set]]
        elif self._train_model:
            # if only training model use CE loss
            params = search_params
        else:
            raise ValueError("No train mode is set. Optimization would be pointless.")
        optim = torch.optim.Adam(params=params, lr=lr)
        for i, res in enumerate(dataloader):
            if self.sessions is None:
                stim, target = res
                sess = ["all"] * len(target)
            else:
                if len(res) != 3:
                    raise ValueError("session correction is ON, dataloader must yield <stim, target, sess_id>")
                stim, target, sess = res
            sess = np.array(sess)
            optim.zero_grad()
            stim = torch.from_numpy(stim).float().to(self.device)
            # stim = self.bn_layers[0][self._train_set](stim)
            # track avg img
            # convert targets to tensors
            target = torch.from_numpy(target)
            batch_size = len(target)
            target = target.long().to(self.device)
            glogits, target, reg = self.forward(stim, target, sess=sess)
            # probs = glogits / torch.linalg.norm(glogits, dim=1).unsqueeze(1)
            probs = glogits  # * self.temperature
            loss = torch.tensor([0.], device=self.device)
            # # if we're training the model and not the mask we compute loss at each spot before summing.
            if self._train_model and not self._train_mask:
            #apply loss before mean over searchlights
                loss = loss + loss_fxn(probs, target)
            probs = probs.logsumexp(dim=2)
            #if self._train_mask:
                # apply loss after mean over searchlights
                # need to scale by number of spatial
            loss = loss + self.total_features * loss_fxn(probs, target)
            loss = loss + reg
            cm, acc = subset_accuracy(probs, target)
            loss_history.append(loss.detach().cpu().item())
            print("Loss Epoch", i, ":", loss.detach().cpu().item(),
                  "Reg:", reg.detach().cpu().item(),
                  "ACC:", acc)
            optim, _ = util.is_converged(loss_history, optim, batch_size, i)
            # apply gradients
            loss.backward()
            optim.step()
            # compute loss independently for each logit set

    def predict(self, dataloader, top30=False):
        # get accuaracies and confusion matrices for each roi and whole brain
        # set train mask and model to false
        _recall = (self._train_model, self._train_mask)
        for i in range(self.n_layers):
            self.bn_layers[i][self._train_set].train(mode=False)
            # set the normalizer to the spatial BN layer for this set type
            self.conv_layers[i].normalizer = self.bn_layers[i][self._train_set]
        self.eval(self._train_set)
        roi_accs = {roi: 0. for roi in self.roi_names}
        roi_cm = {roi: np.zeros((self.n_classes, self.n_classes)) for roi in self.roi_names}
        spatial_accs = []
        spatial_sals = []
        roi_accs["global"] = 0.
        roi_cm["global"] = np.zeros((self.n_classes, self.n_classes))
        count = 1
        session_data = {}  # build sictionary of sessions
        for i, res in enumerate(dataloader):
            if len(res) == 3:
                stim, target, sess = res
                if self.sessions is None:
                    send_sess = ["all"] * len(target)
                else:
                    send_sess = sess
                use_sess = True
            elif len(res) == 2:
                stim, target = res
                sess = send_sess = ["all"] * len(target)
                use_sess = False
            else:
                raise ValueError
            sess = np.array(sess)
            send_sess = np.array(send_sess)
            unique_sess = np.unique(sess)
            # convert targets to tensors
            # convert stiulus to a parameter do we store its gradient by defualt.
            with torch.no_grad():
                stim = torch.from_numpy(stim).float().to(self.device)
                #stim = self.bn_layers[0][self._train_set](stim).detach()
            target = torch.from_numpy(target)
            target = target.long().to(self.device)
            sgy_hat, target, reg = self.forward(stim, target, top30=False, sess=send_sess)
            gy_hat = torch.logsumexp(sgy_hat, dim=2)
            with torch.no_grad():
                roi_logits = self.roi_step(sgy_hat, top30=top30)
                for k in roi_logits.keys():
                    rcm, racc = subset_accuracy(roi_logits[k], target)
                    roi_accs[k] += racc
                    roi_cm[k] += rcm.squeeze()
                cm, acc = subset_accuracy(gy_hat, target)
                # acc = compute_acc(target, gy_hat, pairwise=True, top=1)
                roi_accs["global"] += acc
                roi_cm["global"] += cm
                print("Epoch", i, "ACC:", acc)

                # compute spatial accuracy map
                _, acc_map = subset_accuracy(sgy_hat, target, pairwise_weights=self.pairwise_weights)
                acc_map = acc_map.reshape(self.in_spatial)
                spatial_accs.append(acc_map)
            #differand = -1 * pairloss(sgy_hat, target, pairwise_weights=self.pairwise_weights)
            sal = sgy_hat[np.arange(len(target)), target].reshape((-1,) + self.in_spatial)
            # differand.backward()
            # if linear, this sal is directly equivalent to the sum product of the weight chain.
            # sal = stim.grad.detach().cpu().numpy()
            # sal = sal.reshape((-1, self.channels,) + tuple(self.in_spatial))
            spatial_sals.append(sal.detach().cpu().numpy())
            # propogate gradient to inputs
            if use_sess:
                for s in unique_sess:
                    sind = np.nonzero(sess == s)
                    gl = gy_hat[sind]
                    t = target[sind]
                    cm, acc = subset_accuracy(gl, t, pairwise_weights=self.pairwise_weights)
                    if s in session_data:
                        session_data[s]["acc"] += acc * len(t)
                        session_data[s]["count"] += len(t)
                    else:
                        session_data[s] = {"acc": acc * len(t),
                                           "count": len(t)}
            count += 1
        self.train_model, self._train_mask = _recall
        roi_accs = {k: roi_accs[k] / count for k in roi_accs.keys()}
        roi_cm = {k: roi_cm[k] / np.sum(roi_cm[k], axis=1)[None, :] for k in
                  roi_accs.keys()}  # normalize confusion matrices
        if use_sess:
            sessions = sorted(list(session_data.keys()))
            session_accs = [session_data[k]["acc"] / session_data[k]["count"] for k in sessions]
            fig, ax = plt.subplots(1)
            fig.set_size_inches(18, 18)
            ax.bar(sessions, height=session_accs)
            ax.set_xticklabels(sessions, visible=True, rotation=40)
            plt.show()
        acc_map = np.stack(spatial_accs).mean(axis=0)
        sal_map = np.abs(np.concatenate(spatial_sals)).mean(axis=0)
        return roi_accs, roi_cm, acc_map, sal_map

    def get_saliancy(self, inset, baseline=None):
        if inset not in self.set_names:
            raise ValueError
        if self.sep_class_weight:
            w = self.roi_weights[inset].view((1, self.n_classes,) + self.in_spatial)
        else:
            w = self.roi_weights[inset].view((1, -1,) + self.in_spatial)
            w = w.abs().sum(dim=1, keepdim=True)
        weights = self.gaussian_smoothing(w, unit_range=True).squeeze()
        weights = torch.softmax(weights.flatten(), dim=0).view(weights.shape)
        return weights.detach().cpu().numpy()

    def get_bias(self, inset, class_id):
        if inset not in self.set_names:
            raise ValueError
        b = self.gaussian_smoothing(self.first_bias[inset], unit_range=False)
        b = b[:, class_id].view((1, 1,) + self.in_spatial).squeeze()
        return b.detach().cpu().numpy()

    def get_model_size(self):
        num_params = 0
        for l in self.conv_layers:
            num_params += math.prod(l.weight.shape)
        return num_params