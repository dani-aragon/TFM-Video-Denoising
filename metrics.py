"""
Script to generate the metrics of each noise level of the test dataset.
Useful to compare to trained checkpoints during validation.
"""

import os
from PIL import Image

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import metadata as md


if __name__ == "__main__":
    # Paths to the directories.
    clean_dir = os.path.join(md.PATH_TEST, "clean")
    noisy_dir = os.path.join(md.PATH_TEST, "noisy")

    # Paths to the images whose metrics are going to be computed.
    clean = [x for x in os.listdir(clean_dir) if "_2_" in x]
    noisy = [x for x in os.listdir(noisy_dir) if "_2_" in x]

    # Split in noise levels.
    clean = {
        "S": sorted([x for x in clean if "_S" in x]),
        "M": sorted([x for x in clean if "_M" in x]),
        "L": sorted([x for x in clean if "_L" in x])
    }
    noisy = {
        "S": sorted([x for x in noisy if "_S" in x]),
        "M": sorted([x for x in noisy if "_M" in x]),
        "L": sorted([x for x in noisy if "_L" in x])
    }

    # Compute metrics by noise level.
    for level in ("S", "M", "L"):
        psnr_vals = []
        ssim_vals = []
        for c_name, n_name in zip(clean[level], noisy[level]):
            c = np.array(Image.open(os.path.join(clean_dir, c_name)))
            n = np.array(Image.open(os.path.join(noisy_dir, n_name)))
            psnr_vals.append(psnr(c, n, data_range=255))
            ssim_vals.append(ssim(c, n, channel_axis=2, data_range=255))

        print(
            f"[{level}] Mean PSNR: {np.mean(psnr_vals):.3f} dB",
            f"| Mean SSIM: {np.mean(ssim_vals):.4f}."
        )
