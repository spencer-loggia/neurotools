import os
from typing import Tuple

import numpy as np
import torch
from scipy import ndimage
from torch.multiprocessing import Pool
from torch.nn.functional import conv3d
from matplotlib import pyplot as plt
import pickle as pk


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
                 force_mask: torch.Tensor, name: str, n_sets=2, save_dir='.', device="cuda", lr=.01, use_train_mask=False,
                 unique_masks=True, set_names=("shape", "color")):
        self.decoders = [decoder.to(device) for decoder in decoders]
        self.spatial = input_spatial
        self.in_channels = input_channels
        self.smooth_kernel_size = smooth_kernel_size
        self.device = device
        self.force_mask = force_mask.to(self.device).reshape((1, 1) + force_mask.shape)
        self.save_iter = 250
        self.n_sets = n_sets
        self.name = name
        self.save_dir = save_dir
        self.unique_masks = unique_masks
        self.mask_base = [[torch.nn.Parameter(torch.normal(0, .001, size=(1, 1) + self.spatial, device=device)) for
                           _ in range(n_sets)] for _ in range(n_sets)]
        self.decode_optim = [torch.optim.Adam(lr=lr, params=list(d.parameters()) + [self.mask_base[i][i]]) for i, d in enumerate(self.decoders)]
        self.mask_optim = [[torch.optim.Adam(lr=lr, params=[m]) if i != j else None for i, m in enumerate(set_masks)] for j, set_masks in enumerate(self.mask_base)]
        self.restart_marks = []
        self.lfxn = torch.nn.CrossEntropyLoss()
        self.mask_in_set = use_train_mask
        self.loss_histories = [[list() for _ in range(n_sets)] for _ in range(n_sets)]
        self.sal_maps = None
        self.accuracies = [[list() for _ in range(n_sets)] for _ in range(n_sets)]

    def gaussian_smoothing(self, tensor, kernel_size):
        # Create the Gaussian kernel
        sigma = kernel_size / 6
        kernel = np.zeros((kernel_size, kernel_size, kernel_size))
        kernel[kernel_size // 2, kernel_size // 2, kernel_size // 2] = 1
        kernel = ndimage.gaussian_filter(kernel, sigma)
        kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Apply the kernel using conv3d function
        smoothed_tensor = conv3d(tensor, kernel_tensor, padding=kernel_size // 2)

        return smoothed_tensor

    def get_mask(self, mask_base, noise=True):
        s_gain = torch.nn.functional.tanh(mask_base)
        s_gain = torch.pow(s_gain, 2)
        s_gain = self.gaussian_smoothing(s_gain, kernel_size=self.smooth_kernel_size)
        s_gain = s_gain * self.force_mask
        return s_gain

    def decode_step(self, decoder, in_train, mask):
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
        t = np.arange(len(self.in_loss_history))
        fig, axs = plt.subplots(2)
        axs[0].plot(t, np.array(self.in_loss_history))
        axs[0].set_title("In-Modality Loss")
        axs[1].plot(t, np.array(self.x_loss_history))
        axs[1].set_title("Cross-Modality Loss")
        plt.show()

    def _fit(self, X, in_idx, iters=1000):
        in_optim = self.decode_optim[in_idx]
        in_optim.zero_grad()
        in_train = X[in_idx]
        decoder = self.decoders[in_idx]
        for opt in self.mask_optim[in_idx]:
            opt.zero_grad()
        self.sal_map = None
        local_loss_history = [list() for _ in range(self.n_sets)]
        local_acc = [list() for _ in range(self.n_sets)]

        for epoch in range(iters):
            in_mask = self.get_mask(self.mask_base[in_idx][in_idx])
            decode_loss, acc = self.decode_step(decoder, in_train, in_mask)
            l2loss = .25 * torch.mean(
                torch.stack([torch.mean(torch.pow(param.data.flatten(), 2)) for param in decoder.parameters()]))
            l1mask_loss = .015 * torch.sum(in_mask)
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
                l2loss = .015 * torch.sum(x_mask)
                local_loss_history[x_idx].append(mask_loss.detach().cpu().item())
                local_acc[x_idx].append(acc)
                print(epoch, "CROSS-MODALITY LOSS:", local_loss_history[-1], "ACC", acc, "(REG:", l2loss.detach().cpu().item() ,")")
                loss2 = mask_loss + l2loss
                loss2.backward()
                x_optim.step()
                x_optim.zero_grad()

            if ((epoch + 1) % self.save_iter) == 0:
                self.save(tag="restart_snapshot")

        for dset in X:
            try:
                # make sure dataloaders die
                dset.__next__()
            except StopIteration:
                print("dataloader exhausted.")
                pass

        self.save("end_epoch_" + str(iters))

    def fit(self, X:list):
        """
        :param X: a list of dataloaders for each set
        :return:
        """
        for i in range(self.n_sets):
            decoder = self.decoders[i]
            in_mask_base = self.mask_base[i][i]
            in_optimizer = self.decode_optim[i]
            in_set = X[i]
            in_mask = self.get_mask(in_mask_base)


    def compute_saliancy(self, data, use_mask=True):
        sal_map = torch.zeros(self.spatial)
        count = 0
        for i, (stim, target) in enumerate(data):
            print("batch", i)
            self.decode_optim.zero_grad()
            with torch.no_grad():
                c_stim = torch.nn.Parameter(torch.from_numpy(stim).float().to(self.device).clone())
            if use_mask:
                s_gain = self.get_mask(noise=False)
            else:
                s_gain = self.force_mask
            stim = c_stim * s_gain
            # s_gain = torch.abs(gain)
            # c_stim = c_stim * s_gain
            y_hat = self.decoder(stim)
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
            sal_map += sal.detach().cpu()
            count += torch.count_nonzero(correct).cpu()
        self.decode_optim.zero_grad()

        sal_map = sal_map.numpy()
        self.sal_map = sal_map
        return sal_map

    def save(self, tag="temp"):
        fname = self.name + "_xdecode_" + tag + "_iac:" + str(int(self.in_acc[-1])) + "_xac:" + str(int(self.x_acc[-1])) + ".pkl"
        with open(os.path.join(self.save_dir, fname), "wb") as f:
            pk.dump(self, f)







