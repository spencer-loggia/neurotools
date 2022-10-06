import numpy as np
import torch
from typing import Tuple


class GratingGenerator:

    def __init__(self, stim_params, res=32):
        """

        Parameters
        ----------
        stim_params: A list a grating specifications, as tuple of cycles, color1, color2
        res
        """
        self.stim_params = stim_params
        self.resolution = res
        self.mem = None

    @staticmethod
    def create_grating(spatial, cycles, color1, color2):
        cycler = np.cos((np.arange(spatial) / spatial) * 2 * np.pi * cycles)
        pos_cycler = (cycler + 1) / 2
        neg_cycler = (cycler - 1) / -2
        pos_color_cyc = np.tile(pos_cycler[None, :], (3, 1)).astype(float)
        neg_color_cyc = np.tile(neg_cycler[None, :], (3, 1)).astype(float)
        grating_pos = pos_color_cyc * color1[:, None]
        grating_neg = neg_color_cyc * color2[:, None]
        grating = grating_pos + grating_neg
        grating = np.tile(grating[:, None, :], (1, spatial, 1))
        grating = grating.reshape((1, 3, spatial, spatial))
        return grating

    def get_batch(self, use_cached=True):
        if self.mem is None or not use_cached:
            gratings = []
            for spec in self.stim_params:
                grating = self.create_grating(self.resolution, spec[0], np.array(spec[1]), np.array(spec[2]))
                gratings.append(grating)
        else:
            gratings = self.mem
        if use_cached:
            self.mem = gratings
        return torch.from_numpy(np.concatenate(gratings, axis=0)).float(), torch.arange(len(self.stim_params))