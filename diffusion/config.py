import os, sys

import torch.optim as optim
# from torch.optim.lr_scheduler import CosineAnnealingLR

# To import local modules.
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from diffusion.arch import *


# Dataset hyperparameters.
DATASET = "PMUB"
TRAIN_DATAROOT = "data/LD_FD_CT_train"
VAL_DATAROOT = "data/PMUB-val"
SAMPLE_DATAROOT = "data/LD_FD_CT_test"
IMAGE_SIZE = 256
CHANNELS = 1
LOGIT_TRANSFORM = False
UNIFORM_DEQUANTIZATION = False
GAUSSIAN_DEQUANTIZATION = False
RANDOM_FLIP = True
RESCALED = True
NUM_WORKERS = 8

# Model hyperparameters.
TYPE = "sg"
IN_CHANNELS = 2
OUT_CH = 1
CH = 128
CH_MULT = [1, 1, 2, 2, 4, 4]
NUM_RES_BLOCKS = 2
ATTN_RESOLUTIONS = [16]
DROPOUT = 0.0
VAR_TYPE = "fixedsmall"
EMA_RATE = 0.999
EMA = True
RESAMP_WITH_CONV = True

# Diffusion hyperparameters.
BETA_SCHEDULE = "linear"
BETA_START = 0.0001
BETA_END = 0.02
NUM_DIFFUSION_TIMESTEPS = 1000

# Training hyperparameters.
BATCH_SIZE_TRAIN = 16
N_EPOCHS = 10000
N_ITERS = 5000000
SNAPSHOT_FREQ = 100000
VALIDATION_FREQ = 5000000000

# Sampling hyperparameters.
BATCH_SIZE_SAMPLING = 8
CKPT_ID = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1500000, 2000000]
LAST_ONLY_SAMPLING = True

# Intermediate evaluation settings.
BATCH_SIZE_INTER = 59
LAST_ONLY_INTER = True

# FID evaluation settings.
BATCH_SIZE_FID = 128
LAST_ONLY_FID = True

# Optimizer hyperparameters.
WEIGHT_DECAY = 0.0
OPTIMIZER = optim.AdamW
LR = 0.00002
BETA1 = 0.9
AMSGRAD = False
EPS = 1e-8
