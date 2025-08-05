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
MODEL = UNet3D
BASE_CHANNELS = 16

# Train hyperparameters.
NAME_TRAIN = "unet16"
NUM_EPOCHS = 8
LR = 1e-4
LAST_EPOCH = 0
OPTIMIZER = optim.AdamW
SCHEDULER = False
