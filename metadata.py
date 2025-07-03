import os

# Paths.
ROOT = "D://Documentos//TFM//TFM-Video-Diffusion"
DATA_PATH = os.path.join(ROOT, "PVDD")
AUTOENC_PATH = os.path.join(ROOT, "autoencoder")
DIFF_PATH = os.path.join(ROOT, "diffusion")

# Dataset hyperparameters.
CLIP_LEN = 4
BATCH_SIZE = 1
