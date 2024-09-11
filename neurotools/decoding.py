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
try:
    from captum.attr import DeepLift
except Exception:
    print("Captum module not found, deeplift attribution method not available")


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

    def __init__(self, decoder: Union[torch.nn.Module, tuple], smooth_kernel_size: int, input_spatial: tuple,
                 input_channels: int, force_mask: torch.Tensor, name: str, n_sets=2, device="cuda", lr=.01,
                 set_names=("shape", "color"), unify_fit=False):
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
        if force_mask is None:
            self.force_mask = torch.ones_like(self.mask_base[0][0])
        else:
            self.force_mask = force_mask.to(self.device).reshape((1, 1) + force_mask.shape)
        if unify_fit:
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
        self.mask_base = [[torch.nn.Parameter(torch.normal(size=(1, 1) + self.spatial, mean=0., std=.1, device=self.device)) for
                           _ in range(self.n_sets)] for _ in range(self.n_sets)]
        self.mask_optim = [[torch.optim.Adam(lr=.1, params=[m]) for m in set_masks] for set_masks in
                           self.mask_base]

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
        s_gain = mask_base * self.force_mask
        if noise:
            s_gain = s_gain + torch.normal(0., .01, size=s_gain.shape, device=self.device)
        if use_smooth:
            s_gain = self.gaussian_smoothing(s_gain)
        final_mask = torch.sigmoid(s_gain)
        if reg:
            return final_mask, 1 * torch.mean(torch.abs(s_gain + 3))
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
                    in_mask, reg = self.get_mask(set_mask, noise=True, reg=True)
                    decode_loss, acc = self.decode_step(decoder, dset, in_mask)
                    l2loss = .01 * torch.mean(
                        torch.stack([torch.mean(torch.pow(param.data.flatten(), 2)) for param in decoder.parameters()]))
                    print(epoch, "MODEL LOSS:", decode_loss.detach().cpu().item(), "ACC", acc, "(REG:",
                          l2loss.detach().cpu().item(), "MASK REG:", reg.detach().cpu().item(), ")")
                    loss = decode_loss + l2loss + .00001 * reg # add reg penalty and ver small mask penalty
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
                            if epoch > 2*check_size and block > last_block:
                                # cool down on plateu
                                g['lr'] = g['lr'] * .1
                            elif epoch > 3*check_size and block < last_block < d_last_block:
                                # reheat on slope
                                g['lr'] = min(g['lr'] * 2.5, .01)
                    loss.backward()
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
                mask_iters = iters
                # if train mask was updated during fit, we initialize the cross set mask to it.
                for j in range(len(self.mask_base[i])):
                    if j != i:
                        self.mask_base[i][j] = torch.nn.Parameter(self.mask_base[i][i].clone())
                        self.mask_optim[i][j] = torch.optim.Adam(lr=.01, params=[self.mask_base[i][j]])
            else:
                # need more iters for mask if  starting randomly
                mask_iters = iters
            X.epochs = mask_iters
            local_loss, local_acc = self._fit(X, i, iters=mask_iters)
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
        self.pad = pad
        self.channels = channels
        hc = [math.ceil(2 * math.log10(n_classes)) * max(c, 1) for c in range(hidden_channels, hidden_channels - num_layer + 1, -1)]
        self.all_channels = [channels] + hc + [self.out_channels]
        self.reg_coef = reg
        self.weights = []
        all_spatials = [np.array(self.in_spatial)]
        # compute the weight dimensions for each layer.
        for i in range(num_layer):
            step = (((all_spatials[-1] - self.kernel[i] + 2 * pad) / stride) + 1).astype(int)
            # weights for each filter in this layer. (spatial, c_in * kernel, c_out)
            weight_shape = (int(np.prod(step)),
                            self.all_channels[i] * (self.kernel[i] ** self.dim),
                            self.all_channels[i + 1])
            # initialize weights following xavier protocol
            weights = torch.empty(weight_shape, dtype=torch.double, device=device)
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
        stim = torch.from_numpy(stim).double().to(self.device)
        batch_size = stim.shape[0]
        h = stim + self.bias.view((1, 1) + self.in_spatial)
        iterrule = "bks,skc->bcs"
        # unfold, apply weights, and refold in sequence.
        for i in range(len(self.weights)):
            ufld_stim = util.unfold_nd(h, self.kernel[i], self.pad, self.dim,
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
                targets = torch.tile(target, [1] + list(self.out_spatials[-1])) # <b, 1, s1, s2, s3>
                batch_size = len(target)
                # compute class logits and unflatten spatial dimensions
                y_hat = self.step(stim).reshape(([batch_size, self.n_classes] + list(self.out_spatials[-1])))  # <b, c, s1, s2, s3>
                yhat_all.append(y_hat.detach().cpu())
                targets_all.append(target.flatten().detach().cpu().numpy())

                # compute cross entropy and average across examples
                loss = loss_fxn(y_hat.permute((0, 2, 3, 4, 1)).reshape((-1, y_hat.shape[1])),
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
                # prevents easy to classify class from dominating training dynamics.
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
            # collapse loss
            loss = torch.sum(loss) + self.reg_coef * l2_penalty  # (1)
            self.loss_history.append(loss.detach().cpu().item())
            if ((i + 1) % (2000 // batch_size)) == 0:
                # reduce learn rate if not improving and check to stop early...
                d_last_block = np.mean(np.array(self.loss_history[-300:-200]))
                last_block = np.mean(np.array(self.loss_history[-200:-100]))
                block = np.mean(np.array(self.loss_history[-100:]))
                lr = 0.
                for g in self.optim.param_groups:
                    lr = g['lr']
                    print("LR:", lr)
                    if i > 200 and block > last_block:
                        # cool down on plateu
                        g['lr'] = g['lr'] * .1
                    elif i > 300 and block < last_block < d_last_block:
                        # reheat on slope
                        g['lr'] = min(g['lr'] * 2.5, .01)
                print("EPOCH", i, "LOSS", block)
            # apply gradients
            loss.backward()
            self.optim.step()
            self.scheduler.step()
