import copy
import datetime
import math
import os
from typing import Tuple, Union

import numpy as np
import torch
from scipy import ndimage
from torch.nn.functional import conv3d, conv2d
from matplotlib import pyplot as plt
import pickle as pk
from neurotools import util
import sys


def compute_acc(y, yhat):
    """
    :param y: 1d of target class labels
    :param yhat: 2d of class scores
    :return:
    """
    correct = torch.argmax(yhat, dim=1).int() == y
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

    def __init__(self, decoder: Union[torch.nn.Module, tuple], smooth_kernel_size: int, input_spatial: tuple, input_channels: int,
                 force_mask: torch.Tensor, name: str, n_sets=2, device="cuda", lr=.01, set_names=("shape", "color")):
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
        self.force_mask = force_mask.to(self.device).reshape((1, 1) + force_mask.shape)
        self.set_names = set_names
        self.n_sets = n_sets
        self.name = name
        self._reset_mask()
        self.smooth_kernel = self._create_smoothing_kernel(self.smooth_kernel_size)
        self.decode_optim = [torch.optim.Adam(lr=lr, params=d.parameters()) for d in self.decoder]
        self.restart_marks = []
        self.lfxn = torch.nn.CrossEntropyLoss()
        self.loss_histories = [[list() for _ in range(n_sets)] for _ in range(n_sets)]
        self.sal_maps = None
        self.accuracies = [[list() for _ in range(n_sets)] for _ in range(n_sets)]
        self.lr = lr

    def _reset_mask(self):
        if not hasattr(self, "lr"):
            self.lr = .01
        # initial mask is set such as to be in the unstable regime of loss function + regularizer
        self.mask_base = [[torch.nn.Parameter(torch.normal(.75,
                                                           .2,
                                                           size=(1, 1) + self.spatial, device=self.device)) for
                           _ in range(self.n_sets)] for _ in range(self.n_sets)]
        self.mask_optim = [[torch.optim.Adam(lr=self.lr, params=[m]) for m in set_masks] for set_masks in self.mask_base]

    def _create_smoothing_kernel(self, kernel_size):
        # Create the Gaussian kernel
        sigma = kernel_size / 4
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
        stride = 2 # self.smooth_kernel_size // 2
        if self.spatial_dims == 3:
            smoothed_tensor = conv3d(tensor, self.smooth_kernel, stride=stride, padding=self.smooth_kernel_size // 2)
        elif self.spatial_dims == 2:
            smoothed_tensor = conv2d(tensor, self.smooth_kernel, stride=stride, padding=self.smooth_kernel_size // 2)
        upsampler = torch.nn.Upsample(size=self.spatial, mode="nearest")
        smoothed_tensor = upsampler(smoothed_tensor)
        return smoothed_tensor

    def get_mask(self, mask_base, noise=True, reg=False):
        # takes -inf, inf input to range 0, 1. Maintains desirable gradient characteristics.
        s_gain = torch.abs(mask_base * self.force_mask)
        b_gain = torch.tanh(s_gain)
        s_gain = self.gaussian_smoothing(b_gain)
        final_mask = torch.tanh(s_gain)  # addition of normal makes estimator unstable in linear regime
        if reg:
            return final_mask, 2*torch.mean(.5*s_gain + torch.pow(torch.e, -torch.pow(b_gain - .5, 2)))
        else:
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

    def _fit(self, X, in_idx, iters=1000):
        X = [X[i] for i in range(self.n_sets)]
        self.sal_maps = None
        local_loss_history = [list() for _ in range(self.n_sets)]
        local_acc = [list() for _ in range(self.n_sets)]
        if len(self.decoder) == self.n_sets:
            decoder = self.decoder[in_idx]
        else:
            decoder = self.decoder[0]

        for epoch in range((iters // 5) + 1):
            for x_idx in range(self.n_sets):
                x_train = X[x_idx]
                x_optim = self.mask_optim[in_idx][x_idx]
                x_mask, mask_regularize = self.get_mask(self.mask_base[in_idx][x_idx], reg=True)

                x_mask = torch.tanh(x_mask)
                mask_loss, acc = self.decode_step(decoder, x_train, x_mask)
                local_loss_history[x_idx].append(mask_loss.detach().cpu().item())
                local_acc[x_idx].append(acc)
                print(epoch, "CROSS-MODALITY LOSS:", local_loss_history[x_idx][-1], "ACC", acc,
                      "(REG:", mask_regularize.detach().cpu().item(), ")") #  "(REG:", l2loss.detach().cpu().item()
                loss2 = mask_loss + mask_regularize
                loss2.backward()
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
        if not mask_only:
            data_iterators = []
            for i in range(self.n_sets):
                data_iterators.append(X[i])
            for epoch in range(iters):
                for i, dset in enumerate(data_iterators):
                    if len(self.decoder) == self.n_sets:
                        decoder = self.decoder[i]
                        decode_optim = self.decode_optim[i]
                    else:
                        decoder = self.decoder[0]
                        decode_optim = self.decode_optim[0]
                    in_mask = self.force_mask.clone()
                    decode_loss, acc = self.decode_step(decoder, dset, in_mask)
                    l2loss = .25 * torch.mean(
                        torch.stack([torch.mean(torch.pow(param.data.flatten(), 2)) for param in decoder.parameters()]))
                    print(epoch, "MODEL LOSS:", decode_loss.detach().cpu().item() , "ACC", acc, "(REG:",
                          l2loss.detach().cpu().item(), ")")
                    loss = decode_loss + l2loss
                    loss.backward()
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
            X.epochs = (iters // 4) + 1
            X.resample = False
            local_loss, local_acc = self._fit(X, i, iters=iters)
            for j in range(self.n_sets):
                self.loss_histories[i][j] += local_loss[j]
                self.accuracies[i][j] += local_acc[j]

    def compute_saliancy(self, X):
        count = 0
        sal_map = [[] for _ in range(self.n_sets)]
        for i in range(self.n_sets):
            sal_map.append([])
            for j in range(self.n_sets):
                sal_map[i].append(torch.zeros(self.spatial))
                data = X[j]
                for k, (stim, target) in enumerate(data):
                    print("batch", k)
                    self.decode_optim[i].zero_grad()
                    with torch.no_grad():
                        c_stim = torch.nn.Parameter(torch.from_numpy(stim).float().to(self.device).clone())
                    s_gain = self.get_mask(self.mask_base[i][j], noise=False)
                    stim = c_stim * s_gain
                    # s_gain = torch.abs(gain)
                    # c_stim = c_stim * s_gain
                    y_hat = self.decoder[i](stim)
                    targets = torch.from_numpy(target).long().to(self.device)
                    # compute gradient of correct yhat with respect to pixels
                    correct = torch.argmax(y_hat, dim=1).int() == targets
                    correct_yhat = y_hat[torch.arange(len(y_hat)), targets]
                    loss = torch.sum(correct_yhat * correct)
                    loss.backward()
                    grad_data = c_stim.grad.data
                    # get gradient magnitude
                    over_chan = torch.sum(grad_data, dim=(0, 1))
                    sal = torch.abs(over_chan)
                    sal_map[i][j] += sal.detach().cpu()
                    count += torch.count_nonzero(correct).cpu()
                self.decode_optim[i].zero_grad()
                sal_map[i][j] = sal_map[i][j].numpy()
        self.sal_maps = sal_map
        return sal_map


class SearchlightDecoder:

    def __init__(self, kernel_size=3, pad=1, stride=1, spatial=(64, 64, 64), n_classes=6, channels=2, lr=.01, reg=.05, device="cuda",
                 num_layer=3, hidden_channels=2, nonlinear=False, standardization_mode=None, reweight=True):
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
        self.pad = pad
        self.channels = channels
        hc = [math.ceil(n_classes / 3) * max(c, 1) for c in range(hidden_channels, hidden_channels - num_layer + 1, -1)]
        self.all_channels = [channels] + hc + [self.out_channels]
        self.reg_coef = reg
        self.weights = []
        all_spatials = [np.array(self.in_spatial)]
        # compute the weight dimensions for each layer.
        for i in range(num_layer):
            step = (((all_spatials[-1] - self.kernel[i] + 2 * pad) / stride) + 1).astype(int)
            # weights for each filter in this layer. (spatial, c_in * kernel, c_out)
            weight_shape = (int(np.prod(step)),
                            self.all_channels[i] * (self.kernel[i]**self.dim),
                            self.all_channels[i + 1])
            # initialize weights following xavier protocol
            weights = torch.empty(weight_shape, dtype=torch.double, device=device)
            weights = torch.nn.init.xavier_uniform(weights)
            # track weights as torch Parameters
            self.weights.append(torch.nn.Parameter(weights))
            all_spatials.append(step)
        # setup optimization scheme
        self.optim = torch.optim.Adam(self.weights, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,1500, .2)  # reduce the maximum learning rate every step epochs
        self.out_spatials = all_spatials[1:]
        self.class_weights = torch.ones((n_classes, int(np.prod(self.out_spatials[-1]))), device=device)  # modified if reweight is enabled
        ce = torch.nn.CrossEntropyLoss()
        self.chance_ce = ce(torch.zeros((1, self.n_classes)), torch.ones((1,), dtype=torch.long)) # compute maximum theoretical cross entropy.
        print("final unfolded space with dimensionality ", channels * self.kernel[-1] ** self.dim, "x", self.out_spatials[-1])

    def step(self, stim):
        """
        A single model iteration
        Args:
            stim: np.ndarray, size(batch, channels, *in_spatial) the input data
        Returns: torch.Tensor, size(batch, classes, out_spatial) class logits for each example at each point in space

        """
        stim = torch.from_numpy(stim).double().to(self.device)
        batch_size = stim.shape[0]
        h = stim
        iterrule = "bks,skc->bcs"
        # unfold, apply weights, and refold in sequence.
        for i in range(len(self.weights)):
            ufld_stim = util.unfold_nd(h, self.kernel[i], self.pad, self.dim, self.stride)  # batch size (b), kernel dims (k), spatial dims (s),
            if self.std_mode == "spatial":
                # designed to eliminate differences in signal intensity as a source of error, idk might not work at all.
                # seems to only hurt, not using
                means = ufld_stim.mean(dim=1).unsqueeze(1)
                std = ufld_stim.std(dim=1).unsqueeze(1)
                ufld_stim = (ufld_stim - means) / std
            # mapping input kernel to out channels in next layer for each example for each location in space,
            h = torch.einsum(iterrule, ufld_stim, self.weights[i])  # batch (b), hidden channels, spatial (s)
            # fold to next layer.
            h = h.view([batch_size, self.all_channels[i+1]] + list(self.out_spatials[i]))
            if self.nonlinear:
                h = torch.relu(h)  # we don't use a nonlinearity by default
        y_hat = h.reshape([batch_size, self.out_channels, int(np.prod(self.out_spatials[-1]))])
        # in the binary case, we need only predict a single scalar. Class is sign(y_hat)
        if self.binary:
            y_hat = torch.cat([y_hat, -y_hat], dim=1)
        return y_hat

    def evaluate(self, dataloader):
        """
        Generate output CE and ACC maps (2d or 3d) given input data using the trained model.
        Args:
            dataloader: generator that returns np.ndarrays of stim data at index 0
        Returns: torch.Tensor ACC_map, torch.Tensor CE_map

        """
        ce_tracker = None
        acc_tracker = None
        # a non-reducing loss function gives a separate loss value for every input
        loss_fxn = torch.nn.CrossEntropyLoss(reduction="none")
        count = 0
        # don't track gradients when evaluating
        with torch.no_grad():
            for i, (stim, target) in enumerate(dataloader):
                # convert targets to tensors
                target = torch.from_numpy(target)
                targets = target.long().to(self.device).reshape([-1] + [1]*self.dim)
                targets = torch.tile(targets, [1] + list(self.out_spatials[-1]))
                batch_size = len(target)
                # compute class logits and unflatten spatial dimensions
                y_hat = self.step(stim).reshape(([batch_size, self.n_classes] + list(self.out_spatials[-1])))
                # compute cross entropy and average across examples
                loss = loss_fxn(y_hat, targets)
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
        # return averages across batches.
        return acc_tracker / count, ce_tracker / count

    def fit(self, dataloader):
        """
        fit the layered searchlight model.
        Args:
            dataloader: generator that returns np.ndarrays of stim data, and np.ndarray of targets class labels.
        Returns: None

        """
        loss_fxn = torch.nn.CrossEntropyLoss(reduction="none")
        for i, (stim, target) in enumerate(dataloader):
            self.optim.zero_grad()
            # convert targets to tensors
            target = torch.from_numpy(target)
            target = target.long().to(self.device).reshape([-1, 1])
            targets = torch.tile(target, [1, int(np.prod(self.out_spatials[-1]))])
            # compute regularization penalty
            l2_penalty = torch.sum(torch.stack([torch.sum(torch.pow(w, 2)) for w in self.weights]))
            # get predictions at each spatial location
            y_hat = self.step(stim)
            # compute loss independently for each logit set
            loss = loss_fxn(y_hat, targets)
            if self.reweight:
                # prevents easy to classify class from dominating training dynamics.
                with torch.no_grad():
                    pred = torch.argmax(y_hat, dim=1)
                    for j in range(self.n_classes):
                        c_dex = torch.nonzero(target.flatten() == j).reshape((-1))
                        # class weight moves toward local class weight from this class
                        if len(c_dex) != 0:
                            class_acc = 1 - ((torch.count_nonzero(pred[c_dex, :] == target[c_dex, :], dim=0)) / (4 * len(c_dex)))
                            diff_from_chance = (self.class_weights[j, :] - class_acc)
                            self.class_weights[j, :] = self.class_weights[j, :] - (.025 * len(c_dex) * diff_from_chance[None, :])
                        # relative class importance changes, but overall magnitude stays the same.
                        self.class_weights = self.n_classes * self.class_weights / torch.sum(self.class_weights, dim=0)[None, :]
                        loss[c_dex, :] = loss[c_dex, :] * self.class_weights[j, :][None, :]
            # collapse loss
            loss = torch.sum(loss) + self.reg_coef * l2_penalty  # (1)
            print("EPOCH", i, "LOSS", loss.detach().cpu().item())
            # apply gradients
            loss.backward()
            self.optim.step()
            self.scheduler.step()
        













