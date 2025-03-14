import math
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn
from scipy import ndimage
from torch.nn.functional import conv3d, conv2d
from neurotools import util
from neurotools.modules import VarConvND, SpatialBN, BalancedCELoss
from neurotools.geometry import dissimilarity_from_supervised
import sys


def compute_acc(y, yhat, top=1):
    """
    :param y: 1d of target class labels
    :param yhat: 2d of class scores
    :return:
    """
    correct = (torch.argsort(yhat, dim=1)[:, -top:].int() == y.view(-1, 1)).sum(dim=1)
    acc = 100 * torch.count_nonzero(correct) / len(y)
    return acc.detach().cpu().item()


def subset_accuracy(X, y):
    """
    Determines accuracy
    :param X:
    :param y:
    :return:
    """
    correct = []
    tailing_dim = X.ndim - 2
    scores = X
    target = y.reshape((-1,) + tuple([1]*tailing_dim))
    pred = torch.argmax(scores, dim=1)
    class_correct = pred == target
    correct.append(class_correct)
    correct = torch.concatenate(correct, dim=0).float()
    acc = torch.mean(correct, dim=0, keepdim=True)
    return acc.detach().cpu().numpy()


class ROISearchlightDecoder():
    def __init__(self, atlas: np.ndarray, lookup: dict, set_names: Tuple[str], nonlinear=False, spatial=(64, 64, 64),
                 in_channels=2, n_classes=2, device="cuda", pairwise_comp=None, n_layers=3, base_kernel_size=2, 
                 smooth_kernel_sigma=1.0, latent_channels=3, dropout_prob=.3, seed=42, share_conv=False, 
                 weight_reg_coef=1e-4, conv_reg_coef=1e-4, combination_mode="stack", use_global_weights=True, mask=None):
        """
        A class that applies a layered searchight to 2D or 3D data, giving a class prediction at each point in space.
        Additionally, can group over regions of space (provided in atlas) and provide performance measure grouped over
        those regions.

        :param atlas: np.ndarray <*spatial>, Integer delineation of all the ROIs, where 0 indicates no assignment.
        :param lookup: dictionary, lookup table for integers in atlas
        :param set_names: Tuple[str], names of different sets of items, e.g. "shape" and "color"
        :param nonlinear: bool, whether to apply nonlinearity between layers.
        :param spatial: Tuple[int] either length 2 or 3, size of the spatial dimension.
        :param in_channels: int, number of input channels.
        :param n_classes: int, number of class options.
        :param device: str, device to use for computation.
        :param pairwise_comp: None or torch.Tensor <nclasses, nclasses,> whether any two classes should be compared.
                              Defualt behavior is all classes are compared to all others.
        :param n_layers: number of layers
        :param base_kernel_size: kernel size for convolutional layers.
        :param smooth_kernel_sigma: standard deviation for smoothing kernel for final weights.
        :param latent_channels: number of latent channels between layers.
        :param dropout_prob: float, dropout probability, e.g. what fraction of weight are randomly zeroed during training.
        :param seed: random seed.
        :param share_conv: bool, whether to share weights between layers.
        :param weight_reg_coef: float, final weight regularization coefficient.
        :param conv_reg_coef: float, spatial conv regularization coefficient.
        :param combination_mode: str, combination mode, either "stack" or "entropy". Entropy compute weights using a
                                heuristic that weights each spot based on prediction confidence. Stack fits a new model
                                create a weighting distribution over all searchlight spot. "entropy" generally produces a
                                higher entropy distribution, while "stack" is more prone to overfitting.
        :param use_global_weights: bool, whether to use global weights. If True, the same weight matrix is trained whenever
                                `fit` is called with train_mask set to True. Otherwise, a separate weight matrix is
                                independently trained for each set. In a cross decoding setting, the former asks -
                                "How well can I decode B where A can be decoded?" whereas the latter asks - "In what
                                subset of the voxels that represent A is B best decoded?".
        """
        self.atlas = atlas
        self.seed = seed
        self.lookup = lookup
        self.roi_names = list(self.lookup.values())
        self.roi_indexes = [atlas.flatten()==int(k) for k in self.lookup.keys()]

        self.num_rois = len(self.lookup)
        self.set_names = set_names
        self.dim = len(spatial)
        self.in_spatial = spatial
        self.n_classes = n_classes
        self.pad = [0, 1]
        self.channels = in_channels
        self.latent_channels = latent_channels
        self._train_set = set_names[0]
        self._train_model = True
        self._train_mask = True
        self.device = device
        self.smooth_sigma = smooth_kernel_sigma
        self.n_layers = n_layers
        self.base_kernel_size = base_kernel_size
        self.latent_state_history = []
        self.use_global_weights = use_global_weights
        self._capture_latent = None # private
        self.mask = mask
        if type(self.mask) == np.ndarray:
            self.mask = torch.from_numpy(self.mask).to(self.device)

        if pairwise_comp is None:
            self.pairwise_comp = torch.ones((n_classes, n_classes, n_classes), device=self.device)
        else:
            self.pairwise_comp = pairwise_comp  # mask to use for each input class. Some classes should only be compared to some others.

        self.out_feat = n_classes
        # dropouts
        self.search_dropout = torch.nn.Dropout(p=dropout_prob)
        self.predictor_dropout = torch.nn.Dropout(p=0.25)
        if nonlinear:
            self.activation = torch.nn.LeakyReLU()
        else:
            self.activation = torch.nn.Identity()
        self.smooth_kernel = self._create_smoothing_kernel()
        self.total_features = 0
        self.share_conv = share_conv
        self.lin_reg_coef = weight_reg_coef
        self.spatial_reg_coef = conv_reg_coef
        for roi_ind in self.lookup.keys():
            roi = self.atlas == roi_ind
            roi_size = np.count_nonzero(roi)
            self.total_features += roi_size
        self.total_features = math.prod(spatial)
        self.combination_mode = combination_mode
        self.initialize_params()

    def initialize_params(self):
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        if self.n_layers < 1: raise ValueError
        # self.conv_layers = []
        # this padding scheme ensures size remains constant, but we must reverse every layer so we don't get offset from
        # input
        p = self.base_kernel_size
        fp = p // 2
        remainder = int(p % 2)
        rp = p // 2 - 1 + remainder
        if self.dim == 2:
            forward_padding = (rp, fp, rp, fp)
            reverse_padding = (fp, rp, fp, rp)
        else:
            forward_padding = (rp, fp, rp, fp, rp, fp)
            reverse_padding = (fp, rp, fp, rp, fp, rp)

        if self.share_conv:
            # usig full field conv with higher channel count
            forward_padding = p
            reverse_padding = rp

        self.conv_layers = []
        self.bn_layers = []
        chan_ass = [self.channels] + [self.latent_channels] * (self.n_layers - 1) + [self.n_classes] # channel sequence
        for l in range(self.n_layers):
            in_chan = chan_ass[l]
            out_chan = chan_ass[l+1]
            self.bn_layers.append({s: SpatialBN(device=self.device, channels=in_chan, n_classes=self.n_classes) for s in self.set_names})
            # ensure we maintain same input / output spatial dims and that we reamin cenetered
            if l % 2 == 0:
                pad = forward_padding
            else:
                pad = reverse_padding

            if self.share_conv:
                # can use same filter (standard convolutional layer) for all spatial locs if we want,
                self.conv_layers.append(
                    torch.nn.Conv3d(in_channels=in_chan, out_channels=out_chan, kernel_size=self.base_kernel_size,
                                    padding="same", device=self.device, bias=False))
            else:
                # apply seperate filter to every spatial loc.
                self.conv_layers.append(
                    VarConvND(in_channels=in_chan, out_channels=out_chan, kernel_size=self.base_kernel_size,
                              padding=pad, spatial=self.in_spatial, ndims=self.dim, bias=False,
                              dtype=torch.float, device=self.device, generator=generator))
        self.bn = {}
        # initialize weights from searchlights to rois
        # just one weight at each loc in space
        weight = torch.empty(self.in_spatial, device=self.device, dtype=torch.float)
        weight = torch.nn.Parameter(torch.nn.init.ones_(weight).float())
        self.weights = {}
        if self.use_global_weights:
            self.weights["global"] = torch.nn.Parameter(weight.data.clone())
        else:
            for s in self.set_names:
                self.weights[s] = torch.nn.Parameter(weight.data.clone())

    def _update_setnames(self, set_names):
        self.set_names = set_names
        chan_ass = [self.channels] + [self.latent_channels] * (self.n_layers - 1) + [self.n_classes]  # channel sequence
        for l in range(self.n_layers):
            in_chan = chan_ass[l]
            for s in self.set_names:
                if s not in self.bn_layers:
                    self.bn_layers[l][s] = SpatialBN(device=self.device, channels=in_chan, n_classes=self.n_classes)

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

    def _create_smoothing_kernel(self):
        # Create the Gaussian kernel
        sigma = self.smooth_sigma
        kernel_size = int((round(sigma * 2) + .5) * 2) # must be odd.
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
        stride = 1
        oshape = tensor.shape
        sk = self.smooth_kernel.tile((1, 1) + tuple([1] * self.dim)).float()
        ksize = sk.shape[-1]
        if unit_range:
            tensor = (tensor)
        tensor = tensor.view((tensor.shape[0] * tensor.shape[1], 1) + tensor.shape[2:])  # channels to batch
        if self.dim == 3:
            smoothed_tensor = conv3d(tensor, sk, stride=stride, padding=ksize // 2)
        elif self.dim == 2:
            smoothed_tensor = conv2d(tensor, sk, stride=stride, padding=ksize // 2)
        else:
            raise RuntimeError("Working dimensionality must be either 2 or 3")
        if smoothed_tensor.shape[-self.dim:] != self.in_spatial:
            # this should only be triggered if something is mismatched, like kernel size not odd.
            upsampler = torch.nn.Upsample(size=self.in_spatial, mode="nearest")
            smoothed_tensor = upsampler(smoothed_tensor)
        smoothed_tensor = smoothed_tensor.view(oshape)  # batch to channels
        return smoothed_tensor

    def global_step(self, spatial_logits, top30=False):
        """
        Args:
            stim: <batch, c_in, s1, s2, s3>
        Returns: dist roi logits, global logits
        """
        batch_size = spatial_logits.shape[0]
        # select correct weight.
        if self.use_global_weights:
            weights = self.weights["global"]
        else:
            weights = self.weights[self._train_set]

        # compute regularization
        reg = torch.tensor([0.], device=self.device)
        if self._train_model:
            # copute regularization penalty
            w_l2 = torch.sum(torch.stack([torch.sum(torch.pow(w.weight, 2))
                                          for w in self.conv_layers]))
            reg += self.spatial_reg_coef * w_l2
        if self._train_mask:
            # compute stack weight regularization
            w_l2 = torch.sum(torch.square(weights))
            reg += self.lin_reg_coef * w_l2

        # smooth weights.
        weights = weights.view((1, 1,) + self.in_spatial)
        weights = self.gaussian_smoothing(weights.view((1, 1,) + self.in_spatial))

        # compute log probs for each search spot and stabalize
        spatial_logits = torch.log_softmax(spatial_logits, dim=1)
        spatial_logits = torch.clip(spatial_logits, -25, 0) # numerical stability and such
        spatial_logits = spatial_logits.view((batch_size, spatial_logits.shape[1],  -1))

        # legacy
        if top30:
            sinds = torch.argsort(torch.max(spatial_logits, dim=1)[0].mean(dim=0))
            spatial_logits = spatial_logits[:, :, :, sinds[-30:]]

        if not self._train_model:
            if self.combination_mode == "stack":
                # using stacking
                lw = torch.log_softmax(weights.flatten(), dim=0).view(weights.shape)
            elif self.combination_mode == "entropy":
                # using confidence
                lw = torch.log(weights)
            else:
                raise ValueError
            lw = torch.clip(lw, -25, 25)

            # weight prob dist at each searchlight spot by confidence matrix.
            lw = lw.view((1, 1, -1))
            spatial_logits = spatial_logits + lw

        # apply an entropy regularization penalty to all spots.
        spatial_entropy = 1e-6 * torch.sum(torch.exp(spatial_logits) * spatial_logits)
        reg = reg - spatial_entropy
        return spatial_logits, reg

    def shared_cov_step(self, stim):
        h = stim.float()
        # run through layers
        for i, layer in enumerate(self.conv_layers):
            if i == 0:
                h = self.bn_layers[0][self._train_set](h)
            if self._capture_latent is not None and i == self._capture_latent:
                # capture input to final layer - need to access unfolded state of h.
                # kinda jank but this is an odd request
                self.latent_state_history.append(h.detach().cpu())
                continue
            h = layer(h)
            if i < self.n_layers - 1:
                h = self.bn_layers[i+1][self._train_set](h)
                h = self.activation(h)  # <batch, c, x, y, z>
                if self._train_model:
                    h = self.search_dropout(h)
        return h

    def subset(self, X, y, pairwise_comp):
        all_scores = []
        targets = []
        og_labels = []
        for c in range(self.n_classes):
            d = X[y == c]  # get entries corresponding to specific target
            # ask each relevant regressor, is c? early index class is always consider "TRUE"
            # select scores to compare
            scores = d
            target = torch.tensor([c]).to(X.device).to(torch.long)
            if pairwise_comp is not None:
                class_weights = pairwise_comp[
                    c, c].flatten()
            else:
                class_weights = torch.ones((self.n_classes,))
            scores = scores[np.arange(len(scores))[:, None], torch.nonzero(class_weights).T, ...]  # <n, rc, ...>
            # need to reindex target
            num_removed = torch.count_nonzero(class_weights.flatten()[:c] == 0)
            target = torch.tile(target, (len(scores),))
            # save original labels
            og_labels.append(target)
            # save scores
            all_scores.append(scores)
            # reindex and save new target label
            target = target - num_removed
            targets.append(target)
        all_scores = torch.concatenate(all_scores, dim=0)
        targets = torch.concatenate(targets, dim=0)
        og_labels = torch.concatenate(og_labels, dim=0)
        return all_scores, targets, og_labels

    def forward(self, stim, targets, top30=False):
        if self._train_model:
            spatial_logits = self.shared_cov_step(stim)
        else:
            # no need to track gradients!
            with torch.no_grad():
                spatial_logits = self.shared_cov_step(stim)
        if self.mask is None:
            mask = (torch.sum(torch.abs(stim), dim=(0, 1)) > 0)
        else:
            mask = self.mask
        mask = mask.view((1, 1,) + mask.shape)
        spatial_logits = spatial_logits * mask
        # reindex class labels and logits to only compare items in the same group.
        sub_logits, sub_targets, og_labels = self.subset(spatial_logits, targets, self.pairwise_comp)
        sub_logits = sub_logits.view((-1, sub_logits.shape[1]) + self.in_spatial)
        logits, reg = self.global_step(sub_logits, top30=top30)  # <n, c, c, x, y, z>
        return logits, sub_targets, reg, og_labels

    def _get_loss(self, probs, target, loss_fxn, true_target=None):
        loss = torch.tensor([0.], device=self.device)
        # # if we're training the model and not the mask we compute loss at each spot before summing.
        if self._train_model and not self._train_mask:
            # apply loss before mean over searchlights
            loss = loss + loss_fxn(probs, target, true_target=true_target)
        probs = probs.logsumexp(dim=2)
        if self._train_mask:
            # apply loss after mean over searchlights
            # need to scale by number of spatial
            loss = loss + self.total_features * loss_fxn(probs, target, true_target=true_target)
        acc = subset_accuracy(probs, target)
        return loss, acc

    def fit(self, dataloader, lr=.01):
        bn_train = True
        loss_fxn = BalancedCELoss(nclasses=self.n_classes, device=self.device, rebalance=True,
                                  spatial=self.in_spatial)
        for i in range(self.n_layers):
            # set the batch norm layers equal to those for the MAIN idx 0 set on fitting start
            self.bn_layers[i][self._train_set] = self.bn_layers[i][self.set_names[0]].clone()
            self.bn_layers[i][self._train_set].train(mode=bn_train)
            # set the normalizer to the spatial BN layer for this set type

        loss_history = []
        search_params = []
        for l in self.conv_layers:
            search_params += list(l.parameters())
        optims = []

        if self.use_global_weights:
            wkey = "global"
        else:
            wkey = self._train_set
        if self._train_model:
            # set optim to use
            optims += [torch.optim.Adam(params=search_params, lr=lr)]
        elif self._train_mask and self.combination_mode == "stack":
            optims += [torch.optim.Adam(params=[self.weights[wkey]], lr=lr)]

        # count batches
        count = 0
        for i, res in enumerate(dataloader):
            stim, target = res
            for o in optims:
                o.zero_grad()
            stim = torch.from_numpy(stim).float().to(self.device)
            # track avg img
            # convert targets to tensors
            og_target = torch.from_numpy(target)
            batch_size = len(target)
            og_target = og_target.long().to(self.device)
            logprobs, target, reg, og_target = self.forward(stim, og_target)
            loss, acc = self._get_loss(logprobs, target, loss_fxn, true_target=og_target)
            # show us current class balance every 100
            if (i % 100) == 0:
                print(loss_fxn.rebalance.mean(dim=1))
            loss = loss + reg
            if self._train_mask and self.combination_mode == "entropy":
                # we directly estimate confidence from class distribution.
                # compute entropy reduction at each spatial location on each trial
                n_op = logprobs.shape[1]
                # reduction in entropy from chance
                hr = (logprobs * torch.exp(logprobs)).sum(dim=1) + math.log(n_op)
                hr = hr.mean(dim=0).view(self.in_spatial)
                self.weights[wkey].data = self.weights[wkey].data + hr
            else:
                loss.backward()
                for o in range(len(optims)):
                    optims[o], _ = util.is_converged(loss_history, optims[o], batch_size, i)
                    optims[o].step()
            loss_history.append(loss.detach().cpu().item())
            print("Loss Epoch", i, ":", loss.detach().cpu().item(),
                  "Reg:", reg.detach().cpu().item(),
                  "ACC:", acc)
            # apply gradients
            count += 1

        # scale the summed confidence weights by number of batches
        if self._train_mask and self.combination_mode == "entropy":
            self.weights[wkey].data = self.weights[wkey].data / (count + 1)
        return loss_history

    def predict(self, dataloader, top30=False):
        loss_fxn = BalancedCELoss(nclasses=self.n_classes, device=self.device, rebalance=False)
        # get accuaracies and confusion matrices for each roi and whole brain
        # set train mask and model to false
        _recall = (self._train_model, self._train_mask)
        for i in range(self.n_layers):
            self.bn_layers[i][self._train_set].train(mode=False)
            # set the normalizer to the spatial BN layer for this set type
            self.conv_layers[i].normalizer = self.bn_layers[i][self._train_set]
        self.eval(self._train_set)
        roi_accs = {roi: 0. for roi in self.roi_names}
        spatial_accs = []
        spatial_sals = []
        roi_accs["global"] = 0.
        count = 1
        for i, res in enumerate(dataloader):
            if len(res) == 3:
                stim, target, sess = res
            elif len(res) == 2:
                stim, target = res
            else:
                raise ValueError
            # convert targets to tensors
            with torch.no_grad():
                stim = torch.from_numpy(stim).float().to(self.device)
            target = torch.from_numpy(target)
            target = target.long().to(self.device)
            sgy_hat, target, reg, _ = self.forward(stim, target, top30=False)
            gy_hat = torch.logsumexp(sgy_hat, dim=2)

            with torch.no_grad():
                for j, k in enumerate(self.roi_names):
                    idxs = self.roi_indexes[j]
                    rce, racc = self._get_loss(sgy_hat[:, :, idxs], target, loss_fxn)
                    roi_accs[k] += racc
                acc = subset_accuracy(gy_hat, target)
                roi_accs["global"] += acc
                print("Epoch", i, "ACC:", acc)

                # compute spatial accuracy map
                acc_map = subset_accuracy(sgy_hat, target)
                acc_map = acc_map.reshape(self.in_spatial)
                spatial_accs.append(acc_map)
            sal = sgy_hat[np.arange(len(target)), target].reshape((-1,) + self.in_spatial)
            spatial_sals.append(sal.detach().cpu().numpy())
            count += 1
        self.train_model, self._train_mask = _recall
        roi_accs = {k: roi_accs[k] / count for k in roi_accs.keys()}
        acc_map = np.stack(spatial_accs).mean(axis=0)
        sal_map = np.abs(np.concatenate(spatial_sals)).mean(axis=0)
        return roi_accs, acc_map, sal_map

    def get_latent(self, dataloader, level="last", metric="pearson", voxelwise=False):
        # turn on latent state tracking
        if level == "last":
            self._capture_latent = self.n_layers - 1
        else:
            assert type(level) is int
            self._capture_latent = level
        self.latent_state_history = []

        if self.use_global_weights:
            wkey = "global"
        else:
            wkey = self._train_set

        _recall = (self._train_model, self._train_mask)
        for i in range(self.n_layers):
            self.bn_layers[i][self._train_set].train(mode=False)
            # set the normalizer to the spatial BN layer for this set type
            self.conv_layers[i].normalizer = self.bn_layers[i][self._train_set]
        self.eval(self._train_set)

        # loop through data
        targets = []
        with torch.no_grad():
            for i, res in enumerate(dataloader):
                if len(res) == 3:
                    stim, target, _ = res
                elif len(res) == 2:
                    stim, target = res
                else:
                    raise ValueError
                stim = torch.from_numpy(stim).float().to(self.device)
                # run forward step
                with torch.no_grad():
                    self.shared_cov_step(stim)
                targets.append(target)
            # get latent state
            latent = torch.concatenate(self.latent_state_history, dim=0).detach().cpu() # <batch, chan, x, y, z>
            kdim = self.base_kernel_size ** self.dim
            targets = np.concatenate(targets, axis=0)
            layer = self.conv_layers[self._capture_latent]
            # if using the first level, features should be over voxels in kernel. Other wise compute rdm at each spatial
            # location then average over kernel
            if voxelwise:
                ch = latent.shape[1]
                latent = layer._unfold(latent).reshape((latent.shape[0], kdim * ch, -1)) # <batch, k * chan, x * y * z>
                latent = latent.permute((2, 0, 1))
                rdms = dissimilarity_from_supervised(latent, targets, metric=metric).detach().numpy().squeeze()  # <spatial, rdm>
            else:
                latent = latent.reshape((latent.shape[0], latent.shape[1], -1))  # <batch, chan, x * y * z>
                latent = latent.permute((2, 0, 1))
                rdms = dissimilarity_from_supervised(latent, targets, metric=metric).detach().T  # <rdm, spatial>
                rdms = rdms.reshape((rdms.shape[0], 1,) + self.in_spatial)
                rdms = layer._unfold(rdms).reshape((rdms.shape[0], kdim, -1)) # <rdm, kdim, spatial>
                rdms = rdms.mean(dim=1).detach().numpy().T # <spatial, rdm>
            rl_dict = {}
            for j, k in enumerate(self.roi_names):
                idxs = self.roi_indexes[j]
                weights = self.weights[wkey].flatten()
                if self.combination_mode == "stack":
                    weights = torch.softmax(weights, dim=0)
                weights = weights[idxs][:, None].detach().cpu().numpy()
                assert np.sum(weights < 0)  == 0
                r_latent = (rdms[idxs] * weights / weights.sum()).sum(axis=0, keepdims=True)
                rl_dict[k] = r_latent

        self._train_model, self._train_mask = _recall
        # disengage latent state tracking
        self._capture_latent = None
        return rdms, rl_dict

    def get_saliancy(self):
        """
        Return the smoothed weights.
        :param inset: set to get weights for 
        :return: np.ndarray <*spatial>
        """
        if self.use_global_weights:
            wkey = "global"
        else:
            wkey = self._train_set

        w = self.weights[wkey].view((1, -1,) + self.in_spatial)
        weights = self.gaussian_smoothing(w, unit_range=True).squeeze()
        if self.combination_mode == "stack":
            weights = torch.softmax(weights.flatten(), dim=0)
        weight_dist = weights.reshape(self.in_spatial)
        return weight_dist.detach().cpu().numpy()

    def get_model_size(self):
        """
        Return the total number of parameters.
        :return:
        """
        num_params = 0
        for l in self.conv_layers:
            num_params += math.prod(l.weight.shape)
        return num_params
