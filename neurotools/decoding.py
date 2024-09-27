import copy
import datetime
import math
import os
from typing import Tuple, Union, List

import numpy as np
import torch
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


class GlobalMultiStepCrossDecoder:
    """
    Trains a torch.nn compliant model (user provided) m to decode from provided train dataset. Also learns a single mask
    convolved with a fixed size gaussian kernel, over input data that maximizes the m's performance over a different
    modality "cross-decoding" set and a held out train set.

    Overall, the model requires a 4 data generators:
        - in-modality train
        - in-modality test
        - cross-modality train
        - cross-modality test

    and gives model performance over the train and test set, as well as, critically, the final cross decoding maps.
    """

    def __init__(self, decoder: Union[torch.nn.Module, tuple], smooth_kernel_size: int, input_spatial: tuple,
                 input_channels: int, force_mask: torch.Tensor, name: str, n_sets=2, device="cuda", lr=.01,
                 set_names=("shape", "color"), unify_fit=False, sigmoid_mask=True):
        if isinstance(decoder, torch.nn.Module):
            decoder = [decoder]
            print("Using single decoder for all modalities.")
        elif (isinstance(decoder, list) or isinstance(decoder, tuple)) and len(decoder) == n_sets:
            print("Using seperate decoding model for each modality")
        else:
            raise ValueError("Decoder must be a torch Module or a list of Modules with "
                             "length equal to the number of modality sets.")

        self.decoder = [d.to(device) for d in decoder]
        self.spatial = input_spatial
        self.spatial_dims = len(input_spatial)
        self.in_channels = input_channels
        self.smooth_kernel_size = smooth_kernel_size
        self.device = device
        self.set_names = set_names
        self.n_sets = n_sets
        self.unify_fit = unify_fit
        self.name = name
        self._reset_mask()
        self.smooth_kernel = self._create_smoothing_kernel(self.smooth_kernel_size)
        self.sigmoid_mask = sigmoid_mask
        if force_mask is None:
            self.force_mask = torch.ones_like(self.mask_base[0][0])
        else:
            self.set_force_mask(force_mask)
        if not unify_fit:
            self.decode_optim = [torch.optim.Adam(lr=lr, params=d.parameters()) for d in self.decoder]
        else:
            # train mask with in decoder, assumes cross regions will be subset of in regions, probably valid
            self.decode_optim = [torch.optim.Adam(lr=lr, params=list(d.parameters()) + [self.mask_base[ind][ind]])
                                 for ind, d in enumerate(self.decoder)]
        self.restart_marks = []
        self.lfxn = torch.nn.MultiMarginLoss()
        self.loss_histories = [[list() for _ in range(n_sets)] for _ in range(n_sets)]
        self.sal_maps = None
        self.accuracies = [[list() for _ in range(n_sets)] for _ in range(n_sets)]
        self.lr = lr

    def _reset_mask(self):
        if not hasattr(self, "lr"):
            self.lr = .01
        # initial mask is set such as to be in the unstable regime of loss function + regularizer
        self.mask_base = [
            [torch.nn.Parameter(torch.normal(size=(1, 1) + self.spatial, mean=0., std=.1, device=self.device)) for
             _ in range(self.n_sets)] for _ in range(self.n_sets)]
        self.mask_optim = [[torch.optim.Adam(lr=.1, params=[m]) for m in set_masks] for set_masks in
                           self.mask_base]

    def set_force_mask(self, mask):
        self.force_mask = mask.to(self.device).reshape((1, 1) + mask.shape)

    def _create_smoothing_kernel(self, kernel_size):
        # Create the Gaussian kernel
        sigma = 1.  # kernel_size / 6
        if self.spatial_dims == 2:
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, kernel_size // 2] = 1
            kernel = ndimage.gaussian_filter(kernel, sigma)

        elif self.spatial_dims == 3:
            kernel = np.zeros((kernel_size, kernel_size, kernel_size))
            kernel[kernel_size // 2, kernel_size // 2, kernel_size // 2] = 1
            kernel = ndimage.gaussian_filter(kernel, sigma)
        kernel = kernel / np.max(kernel)  # kernel takes average over space instead of sum
        kernel = kernel + np.random.normal(0, .05, size=kernel.shape)  # add sampling uncertainty
        kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return kernel_tensor

    def gaussian_smoothing(self, tensor, *args):
        # Apply the kernel using conv3d function
        stride = 1  # self.smooth_kernel_size // 2
        if self.spatial_dims == 3:
            smoothed_tensor = conv3d(tensor, self.smooth_kernel, stride=stride, padding=self.smooth_kernel_size // 2)
        elif self.spatial_dims == 2:
            smoothed_tensor = conv2d(tensor, self.smooth_kernel, stride=stride, padding=self.smooth_kernel_size // 2)
        else:
            raise RuntimeError("Working dimensionality must be either 2 or 3")
        if smoothed_tensor.shape[-1] != self.spatial[-1]:
            upsampler = torch.nn.Upsample(size=self.spatial, mode="nearest")
            smoothed_tensor = upsampler(smoothed_tensor)
        return smoothed_tensor

    def get_mask(self, mask_base, noise=True, reg=False, use_smooth=True):
        # takes -inf, inf input to range 0, 1. Maintains desirable gradient characteristics.
        s_gain = mask_base
        if noise:
            s_gain = s_gain + torch.normal(0., .01, size=s_gain.shape, device=self.device)
        if use_smooth:
            s_gain = self.gaussian_smoothing(s_gain)
        l_mask = s_gain * self.force_mask
        if hasattr(self, "sigmoid_mask") and self.sigmoid_mask:
            final_mask = torch.sigmoid(l_mask)
            if reg:
                return final_mask, 1 * torch.mean(torch.abs(s_gain + 3))
        else:
            final_mask = l_mask
            if reg:
                return final_mask, 1 * torch.mean(torch.abs(s_gain))
        return final_mask

    def decode_step(self, decoder, in_train, mask):
        # preform a decoding step and return cross-entropy loss value.
        stim, target = in_train.__next__()
        stim = torch.from_numpy(stim).float().to(self.device)
        target = torch.from_numpy(target)
        targets = target.long().to(self.device)
        stim = stim * mask
        y_hat = decoder(stim)
        acc = compute_acc(targets, y_hat)
        tloss = self.lfxn(y_hat, targets)
        return tloss, acc

    def plot_loss_curves(self):
        fig, axs = plt.subplots(2, 2)
        for i in range(self.n_sets):
            for j in range(self.n_sets):
                t = np.arange(len(self.loss_histories[i][j]))
                axs[i, j].plot(t, np.array(self.loss_histories[i][j]))
                axs[i, j].set_title(self.set_names[i] + " -> " + self.set_names[j] + " Loss")
        plt.show()

    def _fit(self, X, in_idx, iters=1000, update=True):
        batch_size = X.batch_size
        X = [X[i] for i in range(self.n_sets)]
        self.sal_maps = None
        local_loss_history = [list() for _ in range(self.n_sets)]
        local_acc = [list() for _ in range(self.n_sets)]
        if len(self.decoder) == self.n_sets:
            decoder = self.decoder[in_idx]
        else:
            decoder = self.decoder[0]

        decoder.eval()
        histories = [[] for _ in range(self.n_sets)]
        for epoch in range(iters):
            for x_idx in range(self.n_sets):
                x_train = X[x_idx]
                x_optim = self.mask_optim[in_idx][x_idx]
                set_mask = self.mask_base[in_idx][x_idx]
                if update:
                    x_mask, mask_regularize = self.get_mask(set_mask, reg=True, noise=True)
                else:
                    x_mask, mask_regularize = self.get_mask(set_mask, reg=True, noise=False)
                mask_loss, acc = self.decode_step(decoder, x_train, x_mask)
                local_loss_history[x_idx].append(mask_loss.detach().cpu().item())
                local_acc[x_idx].append(acc)
                print(epoch, "CROSS-MODALITY LOSS:", local_loss_history[x_idx][-1], "ACC", acc,
                      "(REG:", mask_regularize.detach().cpu().item(), ")")  # "(REG:", l2loss.detach().cpu().item()
                loss2 = mask_loss + .01 * mask_regularize
                histories[x_idx].append(loss2.detach().cpu().item())
                if update:
                    check_size = (2000 // batch_size) + 1
                    if ((epoch + 1) % check_size) == 0:
                        # reduce learn rate if not improving and check to stop early...
                        d_last_block = np.mean(np.array(histories[x_idx][-3 * check_size:-2 * check_size]))
                        last_block = np.mean(np.array(histories[x_idx][-2 * check_size:-check_size]))
                        block = np.mean(np.array(histories[x_idx][-check_size:]))
                        lr = 0.
                        for g in x_optim.param_groups:
                            lr = g['lr']
                            print("LR:", lr)
                            if epoch > 2 * check_size and block > last_block:
                                # cool down on plateu
                                g['lr'] = g['lr'] * .1
                            elif epoch > 3 * check_size and block < last_block < d_last_block:
                                # reheat on slope
                                g['lr'] = min(g['lr'] * 2.5, .05)
                    loss2.backward()
                    print(torch.max(set_mask.grad), torch.min(set_mask.grad))
                    x_optim.step()
                    x_optim.zero_grad()

            sys.stdout.flush()
        for dset in X:
            closed = False
            while closed is False:
                try:
                    # make sure dataloaders die
                    dset.__next__()
                except StopIteration:
                    print("dataloader exhausted.")
                    closed = True
                    pass

        return local_loss_history, local_acc

    def fit(self, X, iters=1000, mask_only=False, reset_mask=False):
        """
        :param X: a list of dataloaders for each set
        :return:
        """
        # train the linear model with data from all sets
        batch_size = X.batch_size
        if not mask_only:
            data_iterators = []
            for i in range(self.n_sets):
                data_iterators.append(X[i])
            histories = [[] for _ in self.decoder]
            for epoch in range(iters):
                for i, dset in enumerate(data_iterators):
                    if len(self.decoder) == self.n_sets:
                        decoder = self.decoder[i]
                        decode_optim = self.decode_optim[i]
                    else:
                        decoder = self.decoder[0]
                        decode_optim = self.decode_optim[0]
                    decoder.train()
                    set_mask = self.mask_base[i][i]
                    if self.unify_fit:
                        in_mask, reg = self.get_mask(set_mask, noise=True, reg=True)
                    else:
                        in_mask = torch.ones_like(set_mask)
                        reg = torch.tensor([0.])
                    decode_loss, acc = self.decode_step(decoder, dset, in_mask)
                    l2loss = .01 * torch.mean(
                        torch.stack([torch.mean(torch.pow(param.data.flatten(), 2)) for param in decoder.parameters()]))
                    print(epoch, "MODEL LOSS:", decode_loss.detach().cpu().item(), "ACC", acc, "(REG:",
                          l2loss.detach().cpu().item(), "MASK REG:", reg.detach().cpu().item(), ")")
                    loss = decode_loss + l2loss
                    histories[i].append(loss.cpu().detach().item())
                    check_size = (2000 // batch_size) + 1
                    if ((epoch + 1) % check_size) == 0:
                        # reduce learn rate if not improving and check to stop early...
                        d_last_block = np.mean(np.array(histories[i][-3 * check_size:-2 * check_size]))
                        last_block = np.mean(np.array(histories[i][-2 * check_size:-check_size]))
                        block = np.mean(np.array(histories[i][-check_size:]))
                        lr = 0.
                        for g in decode_optim.param_groups:
                            lr = g['lr']
                            print("LR:", lr)
                            if epoch > 2 * check_size and block > last_block:
                                # cool down on plateu
                                g['lr'] = g['lr'] * .1
                            elif epoch > 3 * check_size and block < last_block < d_last_block:
                                # reheat on slope
                                g['lr'] = min(g['lr'] * 2.5, .01)
                    loss.backward()
                    if self.unify_fit:
                        print(torch.max(set_mask.grad), torch.min(set_mask.grad))
                    decode_optim.step()
                    decode_optim.zero_grad()

            for dset in data_iterators:
                try:
                    # make sure dataloaders die
                    dset.__next__()
                except StopIteration:
                    print("dataloader exhausted.")
                    pass
        if reset_mask:
            self._reset_mask()
        for i in range(self.n_sets):
            X.resample = False
            if self.unify_fit:
                mask_iters = (iters // 2) + 1
                # if train mask was updated during fit, we initialize the cross set mask to it.
                for j in range(len(self.mask_base[i])):
                    if j != i:
                        self.mask_base[i][j] = torch.nn.Parameter(self.mask_base[i][i].clone())
                        self.mask_optim[i][j] = torch.optim.Adam(lr=.01, params=[self.mask_base[i][j]])
            else:
                # need more iters for mask if  starting randomly
                mask_iters = iters
            X.epochs = mask_iters
            X.mode = "dev"
            local_loss, local_acc = self._fit(X, i, iters=mask_iters)
            X.mode = "train"
            for j in range(self.n_sets):
                self.loss_histories[i][j] += local_loss[j]
                self.accuracies[i][j] += local_acc[j]

    def predict(self, X, iters=20):
        accs = []
        for i in range(self.n_sets):
            accs.append([])
            X.epochs = iters
            X.resample = False
            with torch.no_grad():
                local_loss, local_acc = self._fit(X, i, iters=iters, update=False)
            for j in range(self.n_sets):
                accs[-1].append(np.array(local_acc[j]).mean())
        return accs

    def compute_saliancy(self, X):
        """
        Uses standard gradient of logits wrt input to quantify saliance
        -------
        """
        count = 0
        # list to hold each iteration's input gradients
        sal_map = [[] for _ in range(self.n_sets)]
        # loop through train sets
        for i in range(self.n_sets):
            self.decoder[i].eval()
            sal_map.append([])
            # loop through test sets
            for j in range(self.n_sets):
                # initial ize map to zeros
                sal_map[i].append(torch.zeros(self.spatial))
                data = X[j]
                for k, (stim, target) in enumerate(data):
                    print("batch", k)
                    self.decode_optim[i].zero_grad()
                    # make the stimulus a parameter so we store its gradient by default
                    with torch.no_grad():
                        c_stim = torch.nn.Parameter(torch.from_numpy(stim).float().to(self.device).clone())
                    # apply the mask learned for this model on this set
                    s_gain = self.get_mask(self.mask_base[i][j], noise=False)
                    stim = c_stim * s_gain
                    # compute logits
                    y_hat = self.decoder[i](stim)
                    targets = torch.from_numpy(target).long().to(self.device)
                    # compute gradient of correct yhat with respect to pixels
                    correct = torch.argmax(y_hat, dim=1).int() == targets
                    # get the logits of the correct class
                    correct_yhat = y_hat[torch.arange(len(y_hat)), targets]
                    loss = torch.sum(correct_yhat * correct)
                    loss.backward()
                    grad_data = torch.abs(c_stim.grad.data) * torch.abs(stim)  # we care about effect size, not sign
                    # get gradient magnitude over trials and channels
                    sal = torch.sum(grad_data, dim=(0, 1))
                    # normalize by stim size
                    sal_map[i][j] += sal.detach().cpu()
                    count += torch.count_nonzero(correct).cpu()
                self.decode_optim[i].zero_grad()
                sal_map[i][j] = sal_map[i][j].numpy()
        self.sal_maps = sal_map
        return sal_map

    def deeplift(self, X, ref=None, correct_only=True):
        """
        Use the deeplift algorithm to quantify the saliance of each voxel
        Args:
            X:
            ref: list of refs for each set. Often should be the mean of all classes, defaults to 0
        Returns:
        """

        # need to formulate as pytorch module for DeepLift algo
        class _DecodeModule(torch.nn.Module):
            def __init__(self, decoder, mask, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mask = mask
                self.decoder = decoder
                self.decoder.eval()

            def forward(self, input):
                h = input * self.mask
                yhat = self.decoder(h)
                return yhat

        print("Running DeepLift Attribution...")
        # list to hold each iteration's input gradients
        sal_map = [[] for _ in range(self.n_sets)]
        # loop through train sets
        for i in range(self.n_sets):
            self.decoder[i].eval()
            sal_map.append([])
            if ref is not None:
                lref = ref[i].reshape((1, self.in_channels) + self.spatial)
            else:
                lref = torch.zeros((1, self.in_channels) + self.spatial, device=self.device)
            # loop through test sets
            for j in range(self.n_sets):
                model = _DecodeModule(self.decoder[i],
                                      self.get_mask(self.mask_base[i][j]))
                # send to DeepLift framework
                dl = DeepLift(model)
                sal_map[i].append(torch.zeros(self.spatial))
                data = X[j]
                count = 0
                for stim, target in data:
                    print("batch", count)
                    stim = torch.from_numpy(stim).float().to(self.device).clone()
                    target = torch.from_numpy(target).long().to(self.device)
                    # compute attributions
                    res = dl.attribute(stim, target=target, baselines=lref)
                    sal_map[i][j] += torch.mean(torch.abs(res.detach().cpu()), dim=(0, 1))
                    count += 1
                sal_map[i][j] /= count
                sal_map[i][j] = sal_map[i][j].numpy()
        self.sal_maps = sal_map
        return sal_map


class SearchlightDecoder:

    def __init__(self, kernel_size=3, pad=1, stride=1, spatial=(64, 64, 64), n_classes=6, channels=2, lr=.01, reg=.05,
                 device="cuda",
                 num_layer=3, hidden_channels=2, nonlinear=False, standardization_mode=None, reweight=True,
                 step_epochs=500):
        """
        Class to fit a layered convolutional searchlight over input data with 2 or 3 spatial dimensions.
        Args:
            kernel_size: int, Size of the kernel for each layer.
            pad: int, Padding to use for each layer
            stride: int, stride for each layer
            spatial: tuple, input spatial dimmensions
            n_classes: int, number of classes
            channels: int, number of input data channels
            lr: float, starting (maximum) learning rate
            reg: float, L2 regularization coefficient
            device: str, which hardware to use.
            num_layer: int, number of layers
            hidden_channels: int, number of channels for intermediate (hidden) layers
            nonlinear: bool, whether to use a nonlinear activation function between layers
            standardization_mode: str, default 'none'
            reweight: bool, whether to increase importance of difficult classes over training
        """
        self.kernel_size = kernel_size
        self.kernel = [2] * num_layer
        self.kernel[0] = self.kernel_size
        self.in_spatial = spatial
        self.dim = len(spatial)
        self.stride = stride
        self.loss_history = []
        self.reweight = reweight  # whether to give more weight to incorrect predictions
        self.std_mode = standardization_mode
        self.device = device
        self.n_classes = n_classes
        if n_classes == 2:
            self.binary = True
            self.out_channels = 1
        else:
            self.binary = False
            self.out_channels = n_classes
        self.lr = lr
        self.nonlinear = nonlinear
        if type(pad) == int:
            self.pad = [pad] * num_layer
        else:
            self.pad = pad
        self.channels = channels
        hc = [math.ceil(2 * math.log10(n_classes)) * max(c, 1) for c in
              range(hidden_channels, hidden_channels - num_layer + 1, -1)]
        self.all_channels = [channels] + hc + [self.out_channels]
        self.reg_coef = reg
        self.weights = []
        all_spatials = [np.array(self.in_spatial)]
        # compute the weight dimensions for each layer.
        for i in range(num_layer):
            step = (((all_spatials[-1] - self.kernel[i] + 2 * self.pad[i]) / stride) + 1).astype(int)
            # weights for each filter in this layer. (spatial, c_in * kernel, c_out)
            weight_shape = (int(np.prod(step)),
                            self.all_channels[i] * (self.kernel[i] ** self.dim),
                            self.all_channels[i + 1])
            # initialize weights following xavier protocol
            weights = torch.empty(weight_shape, dtype=torch.float, device=device)
            weights = torch.nn.init.xavier_uniform(weights)
            # track weights as torch Parameters
            self.weights.append(torch.nn.Parameter(weights))
            all_spatials.append(step)
        self.bias = torch.empty(self.in_spatial, device=device)
        self.bias = torch.nn.init.xavier_normal_(self.bias)
        # setup optimization scheme
        self.optim = torch.optim.Adam(self.weights + [self.bias], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_epochs,
                                                         .5)  # reduce the maximum learning rate every step epochs

        self.out_spatials = all_spatials[1:]
        self.class_weights = torch.ones((n_classes, int(np.prod(self.out_spatials[-1]))),
                                        device=device)  # modified if reweight is enabled
        ce = torch.nn.MultiMarginLoss()
        self.chance_ce = ce(torch.zeros((1, self.n_classes)),
                            torch.ones((1,), dtype=torch.long))  # compute maximum theoretical cross entropy.
        print("final unfolded space with dimensionality ", channels * self.kernel[-1] ** self.dim, "x",
              self.out_spatials[-1])

    def to(self, target):
        if "cpu" in target or "cuda" in target:
            # we are changing device, not other attribute (e.g. precision)
            self.device = target
        self.bias = torch.nn.Parameter(self.bias.detach().to(target))
        self.class_weights = self.class_weights.detach().to(target)
        self.weights = [torch.nn.Parameter(w.detach().to(target)) for w in self.weights]
        self.optim = torch.optim.Adam(self.weights + [self.bias], lr=.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.scheduler.step_size,
                                                         .5)
        return self

    def step(self, stim):
        """
        A single model iteration
        Args:
            stim: np.ndarray, size(batch, channels, *in_spatial) the input data
        Returns: torch.Tensor, size(batch, classes, out_spatial) class logits for each example at each point in space

        """
        if type(stim) is np.ndarray:
            stim = torch.from_numpy(stim).float().to(self.device)
        batch_size = stim.shape[0]
        h = stim + self.bias.view((1, 1) + self.in_spatial)
        iterrule = "bks,skc->bcs"
        # unfold, apply weights, and refold in sequence.
        for i in range(len(self.weights)):
            ufld_stim = util.unfold_nd(h, self.kernel[i], self.pad[i], self.dim,
                                       self.stride)  # batch size (b), kernel dims (k), spatial dims (s),
            if self.std_mode == "spatial":
                # designed to eliminate differences in signal intensity as a source of error, idk might not work at all.
                # seems to only hurt, not using
                means = ufld_stim.mean(dim=1).unsqueeze(1)
                std = ufld_stim.std(dim=1).unsqueeze(1)
                ufld_stim = (ufld_stim - means) / std
            # mapping input kernel to out channels in next layer for each example for each location in space,
            h = torch.einsum(iterrule, ufld_stim, self.weights[i])  # batch (b), hidden channels, spatial (s)
            # fold to next layer.
            h = h.view([batch_size, self.all_channels[i + 1]] + list(self.out_spatials[i]))
            if self.nonlinear:
                h = torch.nn.functional.leaky_relu(h, negative_slope=.1)  # we don't use a nonlinearity by default
        y_hat = h.reshape([batch_size, self.out_channels, int(np.prod(self.out_spatials[-1]))])
        # in the binary case, we need only predict a single scalar. Class is sign(y_hat)
        if self.binary:
            y_hat = torch.cat([y_hat, -y_hat], dim=1)
        return y_hat

    def evaluate(self, dataloader, return_logits=False):
        """
        Generate output CE and ACC maps (2d or 3d) given input data using the trained model.
        Args:
            dataloader: generator that returns np.ndarrays of stim data at index 0
            return_logits: bool, whether to return set of all logits.
        Returns: torch.Tensor ACC_map, torch.Tensor CE_map

        """
        ce_tracker = None
        acc_tracker = None
        # a non-reducing loss function gives a separate loss value for every input
        loss_fxn = torch.nn.MultiMarginLoss(reduction="none")
        count = 0
        yhat_all = []
        targets_all = []
        # don't track gradients when evaluating
        with torch.no_grad():
            for i, (stim, target) in enumerate(dataloader):
                # convert targets to tensors
                target = torch.from_numpy(target)
                target = target.long().to(self.device).reshape([-1] + [1] * self.dim)
                targets = torch.tile(target, [1] + list(self.out_spatials[-1]))  # <b, 1, s1, s2, s3>
                batch_size = len(target)
                # compute class logits and unflatten spatial dimensions
                y_hat = self.step(stim).reshape(
                    ([batch_size, self.n_classes] + list(self.out_spatials[-1])))  # <b, c, s1, s2, s3>
                yhat_all.append(y_hat.detach().cpu())
                targets_all.append(target.flatten().detach().cpu().numpy())

                # compute cross entropy and average across examples
                loss = loss_fxn(torch.movedim(y_hat, 1, -1).reshape((-1, y_hat.shape[1])),
                                targets.flatten()).reshape(targets.shape)
                loss = loss.mean(dim=0)
                # compute class prediction and batch accuracy
                if self.dim == 2:
                    correct = torch.argmax(y_hat, dim=1) == targets
                elif self.dim == 3:
                    correct = torch.argmax(y_hat, dim=1) == targets
                else:
                    raise ValueError
                spatial_acc = correct.sum(dim=0) / batch_size
                # update combined acc, ce
                if ce_tracker is None:
                    ce_tracker = self.chance_ce - loss.detach().cpu()
                    acc_tracker = spatial_acc.detach().cpu()
                else:
                    acc_tracker += spatial_acc.detach().cpu()
                    ce_tracker += self.chance_ce - loss.detach().cpu()
                count += 1
        # interpolate away small size changes from certain kernel / pad / layer combos.
        if self.dim == 3:
            method = "trilinear"
        elif self.dim == 2:
            method = "bilinear"
        else:
            raise ValueError
        acc_tracker = torch.nn.functional.interpolate(acc_tracker.unsqueeze(0).unsqueeze(0),
                                                      size=self.in_spatial, mode=method).squeeze()
        ce_tracker = torch.nn.functional.interpolate(ce_tracker.unsqueeze(0).unsqueeze(0),
                                                     size=self.in_spatial, mode=method).squeeze()
        acc_tracker = acc_tracker.detach().cpu().numpy()
        ce_tracker = ce_tracker.detach().cpu().numpy()
        if return_logits:
            yhat_all = torch.cat(yhat_all, dim=0)
            yhat_all = torch.nn.functional.interpolate(yhat_all, size=self.in_spatial, mode=method).squeeze()
            return acc_tracker / count, ce_tracker / count, yhat_all.numpy(), np.concatenate(targets_all, axis=0)
            # return averages across batches.
        return acc_tracker / count, ce_tracker / count

    def loss_reweight(self, y_hat, target, loss):
        # prevents easy to classify class from dominating training dynamics.
        # loss is per spacial index
        with torch.no_grad():
            pred = torch.argmax(y_hat, dim=1)
            for j in range(self.n_classes):
                c_dex = torch.nonzero(target.flatten() == j).reshape((-1))
                # class weight moves toward local class weight from this class
                if len(c_dex) != 0:
                    class_acc = 1 - ((torch.count_nonzero(pred[c_dex, :] == target[c_dex, :], dim=0)) / (
                            6 * len(c_dex)))
                    diff_from_chance = (self.class_weights[j, :] - class_acc)
                    self.class_weights[j, :] = self.class_weights[j, :] - (
                            .04 * len(c_dex) * diff_from_chance[None, :])
                # relative class importance changes, but overall magnitude stays the same.
                self.class_weights = self.n_classes * self.class_weights / torch.sum(self.class_weights, dim=0)[
                                                                           None, :]
                loss[c_dex, :] = loss[c_dex, :] * self.class_weights[j, :][None, :]
                return loss

    def fit(self, dataloader):
        """
        fit the layered searchlight model.
        Args:
            dataloader: generator that returns np.ndarrays of stim data (batch, channels, x, y, z), and np.ndarray of targets (batch,) class labels.
        Returns: None

        """
        loss_fxn = torch.nn.MultiMarginLoss(reduction="none")
        for i, (stim, target) in enumerate(dataloader):
            self.optim.zero_grad()
            # convert targets to tensors
            target = torch.from_numpy(target)
            batch_size = len(target)
            target = target.long().to(self.device).reshape([-1, 1])
            targets = torch.tile(target, [1, int(np.prod(self.out_spatials[-1]))])
            # compute regularization penalty
            l2_penalty = torch.sum(torch.stack([torch.sum(torch.pow(w, 2)) for w in self.weights]))
            # get predictions at each spatial location
            y_hat = self.step(stim)  # (batch, n_classes, spatial)
            # compute loss independently for each logit set
            loss = loss_fxn(y_hat.transpose(1, 2).reshape((-1, y_hat.shape[1])),
                            targets.flatten()).reshape(targets.shape)
            if self.reweight:
                loss = self.loss_reweight(y_hat, targets, loss)

            # collapse loss
            loss = torch.sum(loss) + self.reg_coef * l2_penalty  # (1)
            self.loss_history.append(loss.detach().cpu().item())
            util.is_converged(self.loss_history, self.optim, batch_size, i)
            # apply gradients
            loss.backward()
            self.optim.step()
            self.scheduler.step()


class VarConvND(torch.nn.Module):
    def __init__(self, kernel_size: Union[tuple[int], int], padding: Union[tuple[int], int], in_channels: int,
                 out_channels: int, spatial: Union[tuple[int], int], ndims=2, stride=1, bias=True, device="cpu",
                 dtype=torch.float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # weight (spatial, c_in * kernel, c_out)
        self.spatial = spatial
        self.padding = padding
        self.kernel_size = kernel_size
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size,)*ndims
        if type(spatial) is int:
            self.spatial = (spatial,)*ndims
        if type(padding) is int:
            self.padding = (padding,)*ndims
        self.device = device
        self.stride = stride
        self.ndims = ndims
        self.use_bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        s = np.array(self.spatial)
        k = np.array(self.kernel_size)
        p = np.array(self.padding)
        # compute input spatial after pad and stride
        in_spatial: np.ndarray = (((s - k + 2 * p) / stride) + 1).astype(int)
        weight_shape = (int(np.prod(in_spatial)),
                        in_channels * np.prod(k),
                        out_channels)
        self.in_spatial: list = in_spatial.tolist()
        # initialize weights following xavier protocol
        weights = torch.empty(weight_shape, dtype=dtype, device=device)
        weights = torch.nn.init.xavier_uniform(weights)
        # track weights as torch Parameters
        self.weight = torch.nn.Parameter(weights)
        if bias:
            bias = torch.empty(size=self.in_spatial, dtype=dtype, device=device)
            bias = torch.nn.init.xavier_uniform(bias)
            # track weights as torch Parameters
            self.bias = torch.nn.Parameter(bias)


    def forward(self, X):
        batch_size = X.shape[0]
        ufld_x = util.unfold_nd(X, kernel_size=self.kernel_size, spatial_dims=self.ndims,
                                padding=self.padding, stride=self.stride)
        if self.use_bias:
            ufld_x = ufld_x + self.bias.view((1, 1, -1))
        iterrule = "bks,skc->bcs"
        # mapping input kernel to out channels in next layer for each example for each location in space,
        h = torch.einsum(iterrule, ufld_x, self.weight)  # batch (b), hidden channels, spatial (s)
        # fold to next layer.
        h = h.view([batch_size, self.out_channels] + self.in_spatial)
        return h

def hardmax(X, dim=1):
    inds = torch.argmax(X, dim=1)
    new = torch.zeros_like(X)
    new[inds] = 1.
    return new

class ROISearchlightDecoder():
    """
    Seperate masking decoder for each ROI
    """
    def __init__(self, atlas: np.ndarray, lookup: dict, set_names: Tuple[str],
                 nonlinear=True, spatial=(64, 64, 64), in_channels=2, n_classes=2, device="cuda",
                 share_conv=False):
        self.atlas = atlas
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
        self.lin_reg_coef = 0.0
        self.spatial_reg_coef = 0.0
        self.share_conv = share_conv
        self.device = device
        base_chan = math.ceil(.5 * np.log(self.n_classes)) * 2**self.dim
        if self.dim == 2:
            ConvConstr = torch.nn.Conv2d
        else:
            ConvConstr = torch.nn.Conv3d

        self.conv_2 = VarConvND(in_channels=2 ** self.dim, out_channels=2 * base_chan, kernel_size=2,
                                padding=self.pad[0],
                                spatial=tuple(s + 1 for s in self.in_spatial), ndims=self.dim, bias=True,
                                dtype=torch.float, device=self.device)
        self.conv_1 = ConvConstr(in_channels=in_channels, out_channels=2 ** self.dim, kernel_size=2,
                                 padding=self.pad[1],
                                 dtype=torch.float, device=self.device, bias=False)
        self.conv_3 = ConvConstr(in_channels=2 * base_chan, out_channels=n_classes, kernel_size=1, padding=0,
                                 dtype=torch.float, device=self.device, bias=False)
        self.search_dropout = torch.nn.Dropout(p=.4)
        self.predictor_dropout = torch.nn.Dropout(p=0.4)
        self.activation = torch.nn.LeakyReLU()
        self._smooth_kernel_size = 5
        self.smooth_kernel = self._create_smoothing_kernel(self._smooth_kernel_size)
        # initialize weights from searchlights to rois
        self.roi_weights = {}
        self.total_features = 0
        self.mask = np.zeros(self.in_spatial)
        for roi_ind in self.lookup.keys():
            roi = self.atlas==roi_ind
            roi_size = np.count_nonzero(roi)
            self.total_features += roi_size
            self.mask[roi] = 1
        self.mask = torch.from_numpy(self.mask).to(device).float()
        self.global_weights = {}
        self.temperatures = {}
        for s in self.set_names:
            weight = torch.ones(self.in_spatial, device=device)
            weight = torch.nn.Parameter(weight)
            self.roi_weights[s] = weight
            weight = torch.empty((len(self.roi_weights),), device=self.device)
            weight = torch.nn.Parameter(torch.nn.init.ones_(weight))
            self.global_weights[s] = weight
            self.temperatures[s] = torch.nn.Parameter(torch.tensor([1.], device=self.device))

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
        self.conv_1 = self.conv_1.to(dev)
        self.conv_2 = self.conv_2.to(dev)
        self.conv_3 = self.conv_3.to(dev)
        for sk in self.set_names:
            self.roi_weights[sk] = torch.nn.Parameter(self.roi_weights[sk].to(dev))
        self.mask = self.mask.to(dev)
        return self

    def _create_smoothing_kernel(self, kernel_size):
        # Create the Gaussian kernel
        sigma = 0.75
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

    def gaussian_smoothing(self, tensor, *args):
        stride = 1  # self.smooth_kernel_size // 2
        if self.dim == 3:
            smoothed_tensor = conv3d(tensor, self.smooth_kernel, stride=stride, padding=self._smooth_kernel_size // 2)
        elif self.dim == 2:
            smoothed_tensor = conv2d(tensor, self.smooth_kernel, stride=stride, padding=self._smooth_kernel_size // 2)
        else:
            raise RuntimeError("Working dimensionality must be either 2 or 3")
        if smoothed_tensor.shape[-self.dim:] != self.in_spatial:
            upsampler = torch.nn.Upsample(size=self.in_spatial, mode="nearest")
            smoothed_tensor = upsampler(smoothed_tensor)
        return smoothed_tensor

    def roi_step(self, spatial_logits, top30=False):
        """
        Get scores for each ROI
        Args:
            stim: <batch, c_in, s1, s2, s3>

        Returns: dict roi logits
        """
        spatial_logits, labels = util.atlas_to_list(spatial_logits, self.atlas) # [<batch, cl, roi>, ..., <>]
        spatial_weights = self.gaussian_smoothing(self.roi_weights[self._train_set].view((1, 1,) + self.in_spatial))
        spatial_weights, w_labels = util.atlas_to_list(spatial_weights, self.atlas)
        roi_logits = {}
        for i, l in enumerate(self.lookup.keys()):
            rlogits = spatial_logits[labels.index(l)]  # <batch, cl, roi>
            rw = spatial_weights[w_labels.index(l)]
            # get max perclass
            n = self.lookup[l]
            rstim = rlogits / torch.linalg.norm(rlogits, dim=1).unsqueeze(1)  # convert to probs, so we give equal initial weight to each loc
            # rstim = torch.softmax(rlogits, dim=1)
            ryhat = rstim * rw.view((1, 1, -1))  # batch, cl, roi
            if top30:
                sinds = torch.argsort(torch.max(ryhat, dim=1)[0].mean(dim=0))
                ryhat = ryhat[:, :, sinds[-30:]]
            roi_logits[n] = ryhat.sum(dim=2)
        return roi_logits

    def global_step(self, spatial_logits, top30=False):
        """
        Args:
            stim: <batch, c_in, s1, s2, s3>
        Returns: dist roi logits, global logits
        """
        batch_size = spatial_logits.shape[0]
        weights = self.roi_weights[self._train_set]
        weights = self.gaussian_smoothing(weights.view((1, 1,) + self.in_spatial))
        if self._train_mask:
            weights = self.predictor_dropout(weights)
        spatial_logits = spatial_logits / torch.linalg.norm(spatial_logits, dim=1).unsqueeze(1)
        # spatial_logits = torch.softmax(spatial_logits, dim=1)
        spatial_logits = spatial_logits * weights
        spatial_logits = spatial_logits.view((batch_size, self.n_classes, -1))
        if top30:
            sinds = torch.argsort(torch.max(spatial_logits, dim=1)[0].mean(dim=0))
            spatial_logits = spatial_logits[:, :, sinds[-30:]]
        gyhat = spatial_logits.sum(dim=2) # / math.log(math.prod(self.in_spatial)) # batch, cl  * (self.global_weights[self._train_set]).view((1, 1, -1))
        return gyhat

    def shared_cov_step(self, stim):
        if not self.share_conv:
            raise ValueError("Not set in share conv mode. Run `step` instead.")
        h = self.conv_1(stim)
        h = self.activation(h)
        if self._train_model:
            h = self.search_dropout(h)
        h = self.conv_2(h)
        h = self.activation(h)
        return self.conv_3(h)

    def forward(self, stim, top30=False):
        # compute regularization penalty
        reg = torch.tensor([0.], device=self.device)
        if self._train_model:
            w_l2 = torch.sum(torch.stack([torch.mean(torch.pow(w.weight, 2))
                                          for w in [self.conv_1, self.conv_2, self.conv_3]]))
            reg += 1.0 * self.spatial_reg_coef * w_l2
        if self._train_mask:
            rw_l2 = torch.sum(torch.abs(self.roi_weights[self._train_set]))
            rw_l2 = rw_l2 / self.total_features
            reg += 1.0 * self.lin_reg_coef * rw_l2
        spatial_logits = self.shared_cov_step(stim)
        spatial_logits = spatial_logits.view((-1, self.n_classes) + self.in_spatial)
        logits = self.global_step(spatial_logits, top30=top30)
        return logits, reg

    def fit(self, dataloader):
        scale = math.prod(self.in_spatial) / math.log(math.prod(self.in_spatial))
        # loss_fxn = torch.nn.MultiMarginLoss(p=2, margin=scale / 2)
        loss_fxn = torch.nn.MultiMarginLoss(p=1, margin=2)
        lr = .01
        loss_history = []
        if self._train_model and self._train_mask:
            # set optim to use
            params = (list(self.conv_1.parameters()) +
                      list(self.conv_2.parameters()) +
                      list(self.conv_3.parameters()) +
                      [self.roi_weights[self._train_set]] +
                      [self.temperatures[self._train_set]])
            g_params =  [self.global_weights[self._train_set]]
        elif self._train_mask:
            lr = .02
            params = [self.roi_weights[self._train_set]] + [self.temperatures[self._train_set]]
            g_params = [self.global_weights[self._train_set]]
        elif self._train_model:
            # if only training model use CE loss
            params = (list(self.conv_1.parameters()) +
                      list(self.conv_2.parameters()) +
                      list(self.conv_3.parameters()))
        else:
            raise ValueError("No train mode is set. Optimization would be pointless.")
        optim = torch.optim.Adam(params=params, lr=lr)
        for i, (stim, target) in enumerate(dataloader):
            optim.zero_grad()
            stim = torch.from_numpy(stim).float().to(self.device)
            # track avg img
            # convert targets to tensors
            target = torch.from_numpy(target)
            batch_size = len(target)
            target = target.long().to(self.device)
            glogits, reg = self.forward(stim)
            probs = glogits / torch.linalg.norm(glogits, dim=1).unsqueeze(1)
            loss = loss_fxn(probs, target)
            acc = compute_acc(target, glogits, top=1)
            loss = loss + reg  # (1)
            loss_history.append(loss.detach().cpu().item())
            print("Loss Epoch", i, ":", loss_history[-1], "ACC:", acc)
            optim = util.is_converged(loss_history, optim, batch_size, i)
            # apply gradients
            loss.backward()
            optim.step()
            # compute loss independently for each logit set

    def predict(self, dataloader, top30=False):
        # set train mask and model to false
        _recall = (self._train_model, self._train_mask)
        self.eval(self._train_set)
        roi_accs = {roi: 0. for roi in self.roi_names}
        roi_accs["global"] = 0.
        count = 1
        for i, (stim, target) in enumerate(dataloader):
            # convert targets to tensors
            stim = torch.from_numpy(stim).float().to(self.device)
            target = torch.from_numpy(target)
            target = target.long().to(self.device)
            with torch.no_grad():
                gy_hat, reg = self.forward(stim, top30=top30)
                roi_logits = self.roi_step(self.shared_cov_step(stim), top30=top30)
            for k in roi_logits.keys():
                racc = compute_acc(target, roi_logits[k], top=1)
                roi_accs[k] += racc
            acc = compute_acc(target, gy_hat, top=1)
            roi_accs["global"] += acc
            print("Epoch", i, "ACC:", acc)
            count += 1
        self.train_model, self._train_mask = _recall
        roi_accs = {k:roi_accs[k] / count for k in roi_accs.keys()}
        return roi_accs

    def get_saliancy(self, inset, baseline=None):
        if inset not in self.set_names:
            raise ValueError
        weights = self.gaussian_smoothing(self.roi_weights[inset].view((1, 1,) + self.in_spatial)).squeeze()
        #weights = torch.relu(weights)
        return weights.detach().cpu().numpy()


if __name__=="__main__":
    class TestData:
        def __init__(self, ma, mb):
            self.batch_size = 30
            self.A_mean = ma
            self.B_mean = mb

        def iterator(self, epochs):
            for i in range(epochs):
                A_data = np.random.normal(loc=self.A_mean, scale=1, size=(self.batch_size, 2, 10, 10))
                B_data = np.random.normal(loc=self.B_mean, scale=1, size=(self.batch_size, 2, 10, 10))
                targets = np.concatenate([np.zeros((self.batch_size,)), np.ones((self.batch_size))])
                data = np.concatenate([A_data, B_data])
                shuffle = np.random.choice(np.arange(self.batch_size * 2), (self.batch_size * 2))
                yield data[shuffle], targets[shuffle]

    A = TestData(-.2, .2)
    B = TestData(-.1, .4)

    atlas = np.zeros((10, 10))
    atlas[:, :5] = 1
    atlas[:, 5:] = 2
    lookup = {1: "roi_1", 2:"roi_2"}
    decoder = ROISearchlightDecoder(atlas, lookup, set_names=("A", "B"), in_channels=2, n_classes=2, spatial=(10, 10),
                                    nonlinear=True, device="cpu")
    decoder.fit(A.iterator(500))
    print("done")
    decoder.eval("A")
    print(decoder.predict(A.iterator(100)))
    decoder.train_predictors("B")
    decoder.fit(B.iterator(1000))
    print(decoder.predict(B.iterator(100)))

    sal = decoder.get_saliancy("B")
    from matplotlib import pyplot as plt
    plt.imshow(sal)
    plt.show()