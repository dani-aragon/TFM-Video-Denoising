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

echo "Using prod instance"
conda activate video
export ROOT_TFM="/LUSTRE/users/daragon/TFM-Video-Diffusion/data"
export PYTHONPATH=$PYTHONPATH:/LUSTRE/users/daragon/TFM-Video-Diffusion

# Execute python script.
python /LUSTRE/users/daragon/TFM-Video-Diffusion/${1}/train.py
