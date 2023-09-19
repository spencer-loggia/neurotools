import copy
import datetime
import os
from typing import Tuple, Union

import numpy as np
import torch
from scipy import ndimage
from torch.nn.functional import conv3d
from matplotlib import pyplot as plt
import pickle as pk
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

class GlobalCrossDecoder:
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

    def __init__(self, decoders: Tuple[torch.nn.Module], smooth_kernel_size: int, input_spatial: tuple, input_channels: int,
                 force_mask: torch.Tensor, name: str, n_sets=2, device="cuda", lr=.01, use_train_mask=False,
                 unique_masks=True, set_names=("shape", "color")):
        self.decoders = [decoder.to(device) for decoder in decoders]
        self.spatial = input_spatial
        self.in_channels = input_channels
        self.smooth_kernel_size = smooth_kernel_size
        self.device = device
        self.force_mask = force_mask.to(self.device).reshape((1, 1) + force_mask.shape)
        self.mask_in_set = use_train_mask
        self.set_names = set_names
        self.n_sets = n_sets
        self.name = name
        self.unique_masks = unique_masks
        self.mask_base = [[torch.nn.Parameter(torch.normal(0, .001, size=(1, 1) + self.spatial, device=device)) for
                           _ in range(n_sets)] for _ in range(n_sets)]
        if self.mask_in_set:
            self.decode_optim = [torch.optim.Adam(lr=lr, params=list(d.parameters()) + [self.mask_base[i][i]])
                                 for i, d in enumerate(self.decoders)]
        else:
            self.decode_optim = [torch.optim.Adam(lr=lr, params=d.parameters()) for i, d in enumerate(self.decoders)]
        self.mask_optim = [[torch.optim.Adam(lr=lr, params=[m]) if i != j else None for i, m in enumerate(set_masks)] for j, set_masks in enumerate(self.mask_base)]
        self.restart_marks = []
        self.lfxn = torch.nn.CrossEntropyLoss()
        self.loss_histories = [[list() for _ in range(n_sets)] for _ in range(n_sets)]
        self.sal_maps = None
        self.accuracies = [[list() for _ in range(n_sets)] for _ in range(n_sets)]

    def gaussian_smoothing(self, tensor, kernel_size):
        # Create the Gaussian kernel
        sigma = kernel_size / 3
        kernel = np.zeros((kernel_size, kernel_size, kernel_size))
        kernel[kernel_size // 2, kernel_size // 2, kernel_size // 2] = 1
        kernel = ndimage.gaussian_filter(kernel, sigma)
        kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Apply the kernel using conv3d function
        smoothed_tensor = conv3d(tensor, kernel_tensor, padding=kernel_size // 2)

        return smoothed_tensor

    def get_mask(self, mask_base, noise=True):
        # takes -inf, inf input to range 0, 1. Maintains desirable gradient characteristics.
        s_gain = mask_base * self.force_mask
        s_gain = torch.nn.functional.tanh(s_gain)
        s_gain = torch.pow(s_gain, 2)
        s_gain = self.gaussian_smoothing(s_gain, kernel_size=self.smooth_kernel_size)
        return s_gain

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
        in_optim = self.decode_optim[in_idx]
        in_optim.zero_grad()
        X = [X[i] for i in range(self.n_sets)]
        in_train = X[in_idx]
        decoder = self.decoder[in_idx]
        self.sal_maps = None
        local_loss_history = [list() for _ in range(self.n_sets)]
        local_acc = [list() for _ in range(self.n_sets)]

        for epoch in range(iters):
            if self.mask_in_set:
                in_mask = self.get_mask(self.mask_base[in_idx][in_idx])
            else:
                in_mask = self.force_mask.clone()
            decode_loss, acc = self.decode_step(decoder, in_train, in_mask)
            l2loss = .22 * torch.mean(
                torch.stack([torch.mean(torch.pow(param.data.flatten(), 2)) for param in decoder.parameters()]))
            if self.mask_in_set:
                l1mask_loss = .04 * torch.sum(in_mask)
            else:
                l1mask_loss = 0.
            local_loss_history[in_idx].append(decode_loss.detach().cpu().item())
            local_acc[in_idx].append(acc)
            print(epoch, "IN-MODALITY LOSS:", local_loss_history[in_idx][-1], "ACC", acc, "(REG:",
                  l2loss.detach().cpu().item(), ")")
            loss = decode_loss + l2loss + l1mask_loss
            loss.backward()
            in_optim.step()
            in_optim.zero_grad()
            for x_idx in range(self.n_sets):
                if x_idx == in_idx:
                    # don't cross decode from the train set!
                    continue
                x_train = X[x_idx]
                x_optim = self.mask_optim[in_idx][x_idx]
                x_mask = self.get_mask(self.mask_base[in_idx][x_idx])
                mask_loss, acc = self.decode_step(decoder, x_train, x_mask)
                # l2loss = .04 * torch.sum(x_mask)
                local_loss_history[x_idx].append(mask_loss.detach().cpu().item())
                local_acc[x_idx].append(acc)
                print(epoch, "CROSS-MODALITY LOSS:", local_loss_history[x_idx][-1], "ACC", acc, "(REG:", l2loss.detach().cpu().item() ,")")
                loss2 = mask_loss + l2loss
                loss2.backward()
                x_optim.step()
                x_optim.zero_grad()

            sys.stdout.flush()
        for dset in X:
            try:
                # make sure dataloaders die
                dset.__next__()
            except StopIteration:
                print("dataloader exhausted.")
                pass

        return local_loss_history, local_acc

    def fit(self, X, iters=1000):
        """
        :param X: a list of dataloaders for each set
        :return:
        """
        for i in range(self.n_sets):
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
                    if not self.mask_in_set and i==j:
                        s_gain = self.force_mask
                    else:
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
        self.mask_base = [[torch.nn.Parameter(torch.normal(.75, .2, size=(1, 1) + self.spatial, device=self.device)) for
                           _ in range(self.n_sets)] for _ in range(self.n_sets)]
        self.mask_optim = [[torch.optim.Adam(lr=self.lr, params=[m]) for m in set_masks] for set_masks in self.mask_base]

    def _create_smoothing_kernel(self, kernel_size):
        # Create the Gaussian kernel
        sigma = kernel_size / 5
        kernel = np.zeros((kernel_size, kernel_size, kernel_size))
        kernel[kernel_size // 2, kernel_size // 2, kernel_size // 2] = 1
        kernel = ndimage.gaussian_filter(kernel, sigma)
        kernel = kernel / np.sum(kernel)
        kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return kernel_tensor

    def gaussian_smoothing(self, tensor, *args):
        # Apply the kernel using conv3d function
        smoothed_tensor = conv3d(tensor, self.smooth_kernel, padding=self.smooth_kernel_size // 2)
        return smoothed_tensor

    def get_mask(self, mask_base, noise=True):
        # takes -inf, inf input to range 0, 1. Maintains desirable gradient characteristics.
        s_gain = mask_base * self.force_mask
        s_gain = torch.tanh(s_gain - .001)
        s_gain = torch.pow(s_gain, 2)
        s_gain = self.gaussian_smoothing(s_gain)
        s_gain = torch.tanh(s_gain + .001)
        s_gain = torch.pow(s_gain, 2)
        s_gain = s_gain / torch.max(s_gain)
        return s_gain

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
                x_mask = self.get_mask(self.mask_base[in_idx][x_idx])
                mask_loss, acc = self.decode_step(decoder, x_train, x_mask)
                l2loss = .00025 * torch.sum(torch.pow(x_mask, 2))
                local_loss_history[x_idx].append(mask_loss.detach().cpu().item())
                local_acc[x_idx].append(acc)
                print(epoch, "CROSS-MODALITY LOSS:", local_loss_history[x_idx][-1], "ACC", acc, "(REG:", l2loss.detach().cpu().item() ,")")
                loss2 = mask_loss + l2loss
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






