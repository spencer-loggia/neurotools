import math

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
import cv2


class StratifiedSampler:
    """
    Class to create multiple folds where items in the catagorical "strat_cols" and target col are maximally balanced in
    each fold.
    """
    def __init__(self, full_df, n_folds, target_col, stratify=True, strat_cols=()):
        """
        creates a StratifiedSampler object with n_folds folds.
        Parameters
        ----------
        full_df: Full dataframe of stimulus and targets.
        n_folds: number of cross validation folds
        target_col: col with class labels
        strat_cols: other cols that should have values evenly distributed.
        """
        self.og_df = full_df
        self.n_folds = n_folds
        self.target_col = target_col
        self.strat_col = strat_cols + [self.target_col]
        if stratify:
            self.folds = [pd.DataFrame(columns=full_df.columns) for _ in range(n_folds)]
            self._stratified_split_balanced()
        else:
            # full_df = full_df.sample(frac=1.0)
            self.folds = [full_df.iloc[i::n_folds] for i in range(n_folds)]

    def _stratified_split_balanced(self,):
        df = self.og_df
        strat_cols = self.strat_col
        k = self.n_folds

        # Group the DataFrame by the specified stratification columns
        grouped = df.groupby(strat_cols)

        # Loop over each group and distribute rows in a round-robin manner
        counter = 0  # Counter to keep track of DataFrame index in round-robin
        for _, group in grouped:
            for _, row in group.iterrows():
                self.folds[counter % k] = pd.concat([self.folds[counter % k], pd.DataFrame([row])])
                counter += 1

    def get_train(self, idx):
        data = pd.concat(self.folds[:idx] + self.folds[idx + 1:])
        return data

    def get_test(self, idx):
        return self.folds[idx]

    def get_all(self):
        data = pd.concat(self.folds)
        return data


def chunk_tensor(x: torch.Tensor, chunksize=64, return_indexes=False):
    """
    Breaks a tensor into multiple tensors along the batch dimension such that each piece is less than chunk size GB
    x: input tensor
    chunksize: int, max chunksize in GB
    """
    chunks = []
    indexes = []
    chunk_bytes = chunksize * (1024**3)
    elsize = x.element_size()
    xsize = x.nelement() * elsize
    n_chunks = xsize / chunk_bytes
    batch_size = math.floor(x.shape[0] / n_chunks)
    if batch_size == 0:
        raise RuntimeError("Batch of size less then 1 required to achieve chunk size", chunksize, "GB")
    idx = 0
    while idx < x.shape[0]:
        chunks.append(x[idx:idx+batch_size])
        if return_indexes:
            indexes.append(tuple(range(idx, min(x.shape[0] - 1, idx+batch_size))))
        idx += batch_size
    if return_indexes:
        return chunks, indexes
    else:
        return chunks


def save_video_from_arrays(arrays, vmin, vmax, outfile, fps=30):
    """
    Save a movie of 2D NumPy arrays to disk in MPEG-4 format.

    Parameters:
    - arrays: List of 2D numpy arrays.
    - vmin: Minimum value for normalization.
    - vmax: Maximum value for normalization.
    - outfile: Path to the output video file.
    - fps: Frames per second for the output video (default is 30).
    """
    # Check if arrays is not empty
    if len(arrays) == 0:
        raise ValueError("The input list of arrays is empty.")

    size_im = arrays[0].shape
    scale = [240 / s for s in size_im]
    arrays = [zoom(a, scale, mode="nearest") for a in arrays]

    # Get the dimensions of the first array
    height, width = arrays[0].shape

    # Create a VideoWriter object with MPEG-4 codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height), isColor=False)

    for array in arrays:
        if array.shape != (height, width):
            raise ValueError("All arrays must have the same shape.")

        # Normalize array values to the range 0-255 for 8-bit grayscale image
        normalized_array = np.clip((array - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

        # Write the frame to the video
        video_writer.write(normalized_array)

    # Release the VideoWriter object
    video_writer.release()