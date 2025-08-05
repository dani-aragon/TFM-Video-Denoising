import os

import torch


# Root.
ROOT = os.getenv(
    "ROOT_TFM",
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
)

# Device.
DEVICE = torch.device("cuda")

# Data paths.
PATH_DATA = os.path.join(ROOT, "PVDD")
PATH_TRAIN = os.path.join(PATH_DATA, "train")
PATH_TEST = os.path.join(PATH_DATA, "test")
PATH_EVAL = os.path.join(ROOT, "eval")
PATH_TO_INFER = os.path.join(ROOT, "to_infer")
PATH_INFERED = os.path.join(ROOT, "infered")

# Models paths.
PATH_MODELS = os.path.join(ROOT, "models")
PATH_CHECK = os.path.join(PATH_MODELS, "checkpoints")

# Script paths.
PATH_UNET3D = os.path.join(ROOT, "unet3d")
PATH_FASTDVDNET = os.path.join(ROOT, "fastdvdnet")
