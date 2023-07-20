import os

import numpy as np
import torch
from scipy import ndimage
from torch.nn.functional import conv3d
from matplotlib import pyplot as plt
import pickle as pk


def gaussian_smoothing(tensor, kernel_size):
    # Create the Gaussian kernel
    sigma = kernel_size / 6
    kernel = np.zeros((kernel_size, kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2, kernel_size // 2] = 1
    kernel = ndimage.gaussian_filter(kernel, sigma)
    kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)

    # Apply the kernel using conv3d function
    smoothed_tensor = conv3d(tensor, kernel_tensor, padding=kernel_size // 2)

    return smoothed_tensor


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
    Trains a torch.nn compliant model (user provided) m to decode from provided train dataset. Also learns a single mask,
    convolved with a fixed size gaussian kernel, over input data that maximizes the m's performance over a different
    modality "cross-decoding" set and a held out train set.

    Overall, the model requires a 4 data generators:
        - in-modality train
        - in-modality test
        - cross-modality train
        - cross-modality test

    and gives model performance over the train and test set, as well as, critically, the final cross decoding maps.
    """
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            xdec = pk.load(f)
        xdec.restart_marks.append(len(xdec.in_loss_history))
        return xdec

    def __init__(self, decoder: torch.nn.Module, smooth_kernel_size: int, input_spatial: tuple, input_channels: int,
                 force_mask: torch.Tensor, name: str, save_dir='.', device="cuda"):
        self.decoder = decoder.to(device)
        self.spatial = input_spatial
        self.in_channels = input_channels
        self.smooth_kernel_size = smooth_kernel_size
        self.device = device
        self.force_mask = force_mask
        self.save_iter = 250
        self.name = name
        self.save_dir = save_dir
        self.mask_base = torch.nn.Parameter(torch.normal(0, .001, size=(1, 1) + self.spatial, device=device))
        self.decode_optim = torch.optim.Adam(lr=.001, params=self.decoder.parameters())
        self.mask_optim = torch.optim.Adam(lr=.001, params=[self.mask_base])
        self.restart_marks = []
        self.lfxn = torch.nn.CrossEntropyLoss()
        self.in_loss_history = []
        self.x_loss_history = []
        self.in_acc = []
        self.x_acc = []

    def get_mask(self):
        s_gain = torch.abs(self.mask_base) * self.force_mask
        s_gain = gaussian_smoothing(s_gain, kernel_size=self.smooth_kernel_size)
        return s_gain

    def decode_step(self, in_train, mask):
        stim, target = in_train.__next__()
        stim = torch.from_numpy(stim).float().to(self.device)
        target = torch.from_numpy(target)
        targets = target.long().to(self.device)

        stim = stim * mask
        y_hat = self.decoder(stim)
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

    def fit(self, in_train, x_train, iters=2000):
        for epoch in range(iters):
            mask = self.get_mask()

            self.decode_optim.zero_grad()
            decode_loss, acc = self.decode_step(in_train, mask)
            l2loss = 5 * torch.mean(
                torch.stack([torch.mean(torch.pow(param.data.flatten(), 2)) for param in self.decoder.parameters()]))
            self.in_loss_history.append(decode_loss.detach().cpu().item())
            self.in_acc.append(acc)
            print("IN-MODALITY LOSS:", self.in_loss_history[-1])
            loss = decode_loss + l2loss
            loss.backward()
            self.decode_optim.step()

            self.mask_optim.zero_grad()
            mask_loss, acc = self.decode_step(x_train, mask)
            l2loss = .04 * torch.sum(mask) + .02 * torch.sum(mask ** 2)
            self.x_loss_history.append(mask_loss.detach().cpu().item())
            self.x_acc.append(acc)
            print("CROSS-MODALITY LOSS:", self.x_loss_history[-1])
            loss = mask_loss + l2loss
            loss.backward()
            self.mask_optim.step()

            if ((epoch + 1) % self.save_iter) == 0:
                self.save(tag="restart_snapshot")

        self.save("end_epoch_" + str(iter))

    def save(self, tag="temp"):
        fname = self.name + "_xdecode_" + tag + "_iac:" + str(int(self.in_acc[-1])) + "_xac:" + str(int(self.x_acc[-1])) + ".pkl"
        with open(os.path.join(self.save_dir, fname), "wb") as f:
            pk.dump(self, f)







