"""
Configuration script for the 2D denoising FastDVDNet models. Change the
hyperparameters here before running training or evaluation.
"""

import os, sys

import torch.optim as optim

# To import local modules.
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from arch import *


# Dataset hyperparameters.
CLIP_LEN = 5
BATCH_SIZE = 1
NUM_WORKERS = 2
PERSISTENT_WORKERS = True
PIN_MEMORY = True

# Model hyperparameters.
MODEL = FastDVDNet
BASE_CHANNELS = 10

# Train hyperparameters.
NAME_TRAIN = "fastdvdnet10"
NUM_EPOCHS = 20
LR = 1e-3
LAST_EPOCH = 0
OPTIMIZER = optim.AdamW
SCHEDULER = True

# Evaluation hyperparameters.
VAL = False
CKPT_RANGE = range(15, 14, -1)
NAME_TEST = "fastdvdnet10"
