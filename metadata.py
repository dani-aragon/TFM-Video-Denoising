import os

import torch


# Root.
ROOT = os.getenv("ROOT_TFM", "D://Documentos//TFM//TFM-Video-Diffusion//data")

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
PATH_AUTOENC = os.path.join(ROOT, "autoencoder")
PATH_DIFF = os.path.join(ROOT, "diffusion")
