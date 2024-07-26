import cv2
import numpy as np
import os
from scipy.ndimage import zoom

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
    if not arrays:
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