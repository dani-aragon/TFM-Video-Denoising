#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V

##############################################################################
# This script receives one argument:
# 1. The model to be trained (autoencoder or diffusion).
##############################################################################

echo "Training ${1}."
source activate base
conda activate video
# export ROOT_TFM="/LUSTRE/users/daragon/TFM-Video-Diffusion/data"
# export PYTHONPATH=$PYTHONPATH:/LUSTRE/users/daragon/TFM-Video-Diffusion
export PYTHONPATH=$PYTHONPATH:/mnt/d/Documentos/TFM/TFM-Video-Diffusion

# Execute python script.
# python /LUSTRE/users/daragon/TFM-Video-Diffusion/${1}/train.py
python /mnt/d/Documentos/TFM/TFM-Video-Diffusion/${1}/train.py
