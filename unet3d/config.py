"""
Configuration script for the 3D denoising UNet3D models. Change the
hyperparameters here before running training, evaluation or inference.
"""

import os, sys

import torch.optim as optim

# To import local modules.
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from arch import *


# Dataset hyperparameters.
CLIP_LEN = 4
BATCH_SIZE = 1
NUM_WORKERS = 1
PERSISTENT_WORKERS = True
PIN_MEMORY = True

# Model hyperparameters.
MODEL = UNet3D_Res
BASE_CHANNELS = 16

# Train hyperparameters.
NAME_TRAIN = "unetres16"
NUM_EPOCHS = 20
LR = 1e-3
LAST_EPOCH = 0
OPTIMIZER = optim.AdamW
SCHEDULER = True

# Evaluation hyperparameters.
VAL = False
CKPT_RANGE = range(20, 19, -1)
NAME_TEST = "unetres16"

# Inference hyperparameters.
NAME_INF = "unetres16"
EVALUATOR = 4 # how to evaluate and produce the final clean video
FILES = [
    #"4635084-hd_1920_1080_30fps.mp4",
    #"20250816_210226.mp4",
    "20250821_005503.mp4",
]
