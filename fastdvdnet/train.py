"""
Script to train the 3D denoising UNet3D models. Tune hyperparameters in the
corresponding config.py.
"""

import os, sys

from torch.utils.data import DataLoader

# To import local modules.
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import metadata as md
import config as conf
from datasets import Train2DDataset
from utils import train


if __name__ == "__main__":
    # Create dataset.
    dataset = Train2DDataset(md.PATH_TRAIN, conf.CLIP_LEN)
    print(f"Total clips: {len(dataset)}.")

    # Create loader.
    train_loader = DataLoader(
        dataset,
        batch_size=conf.BATCH_SIZE,
        shuffle=True,
        num_workers=conf.NUM_WORKERS,
        persistent_workers=conf.PERSISTENT_WORKERS,
        pin_memory=conf.PIN_MEMORY
    )

    # Create model.
    model = conf.MODEL(conf.BASE_CHANNELS).to(md.DEVICE)

    # Train FastDVDNet.
    train(
        model,
        train_loader,
        conf.NAME_TRAIN,
        conf.NUM_EPOCHS,
        conf.LR,
        conf.LAST_EPOCH,
        conf.OPTIMIZER,
        conf.SCHEDULER
    )
