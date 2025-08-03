#!/bin/bash

##############################################################################
# This script receives two arguments:
# 1. The model to be trained (autoencoder or diffusion).
# 2. The GPU(s) to use.
##############################################################################

echo "Training ${1}."

# conda init
# conda activate video

GPU=${2:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export ROOT_TFM="/LUSTRE/users/daragon/TFM-Video-Diffusion/data"
export PYTHONPATH=$PYTHONPATH:/LUSTRE/users/daragon/TFM-Video-Diffusion
# export ROOT_TFM="/mnt/d/Documentos/TFM/TFM-Video-Diffusion/data"
# export PYTHONPATH=$PYTHONPATH:/mnt/d/Documentos/TFM/TFM-Video-Diffusion

# Execute python script.
python /LUSTRE/users/daragon/TFM-Video-Diffusion/${1}/train.py
# python /mnt/d/Documentos/TFM/TFM-Video-Diffusion/${1}/train.py
