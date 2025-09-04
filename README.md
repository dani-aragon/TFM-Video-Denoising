# TFM Video Denoising

This repository contains Python and Shell code to train, evaluate and infer 2D and 3D convolutional encoder-decoder models for video denoising. This implementation belongs to the Master's Thesis "Eliminación de ruido en vídeo mediante redes neuronales profundas" by Daniel Aragón Santín.

---

## Table of Contents
- [Specifications](#specifications)
- [Installation](#installation)
- [How to use the repository](#how-to-use-the-repository)

---

## Specifications

The specifications of the personal computer used in this implementation are the following.
 - 32 GB RAM
 - 1+ TB SSD
 - RTX 4070 with 12 GB VRAM
 - AMD Ryzen 7 7700 with 8 cores
The OS used was Windows 10.

---

## Installation

In order to run the different scripts of this implementation, follow these steps:

### Create conda environment

1. Install either Anaconda or Miniconda if you have not done it yet.
2. Create a conda environment with Python 3.12. You can do it by naming it "video" as follows:
    ```bash
    conda create --name video python=3.12
    ```
3. Activate the environment:
    ```bash
    conda activate video
    ```
4. Install these dependencies with `pip`:
    - `torch==2.7.0+cu128`
    - `torchvision==2.7.0+cu128`
    - `numpy==2.2.5`
    - `opencv-python==4.11.0.86`
    - `scikit-image==0.25.2`
    - `tqdm==4.67.1`

   You should look at [this](https://pytorch.org/get-started/locally/) in order to install CUDA with `torch`.

### Download the PVDD data

1. Download the training data from [here](https://drive.google.com/drive/folders/1rMbZqd84S1Py6buhNH6suPDnyFJjITLe) and the syntethic noise test data from [here](https://drive.google.com/drive/folders/1TRSlPo1CiBPunJVC1NQmV5oLcLLo0laU).
2. Unpack the `.zip` compressed files with a program such as 7-Zip.

### Download the repository and set up the folder structure 

1. Clone this repository with the following command:
    ```bash
    git clone https://github.com/dani-aragon/TFM-Video-Denoising
    ```
2. Create a folder called `data` inside of the repository folder `TFM-Video-Denoising`. This folder must have the following subfolders:

    ```plaintext
    data/
    ├── eval/
    ├── infered/
    ├── models/
    │   ├── checkpoints/
    ├── PVDD/
    │   ├── test/
    │   ├── train/
    │   └── val/
    └── to_infer/
    ```
3. Move `data/unetres16.pth` to `data/models`, outside `checkpoints` folder.
4. Go to the folder with the PVDD data which you have already downloaded. Move the folders `clean` and `noisy` that are inside `training_data_rgb` to `data/PVDD/train`.
5. Move the folders `clean` and `noisy` that are inside `testing_data_rgb/synNoiseData` to `data/PVDD/test`. Copy them to `data/PVDD/val` as well. Once copied to both folders, split this test synthetic-noise dataset into validation and test by deleting the validation data from `data/PVDD/test` and the test data from `data/PVDD/val`.
6. (Optional) Set `ROOT_TFM` environment variable to the `TFM-Video-Denoising` directory path.
7. (Optional) If you want to use the training executable instead of the `train.py` Python script, change `ROOT_TFM` and `PYTHONPATH` definitions in `train.sh` to the correct path to the repository folder.

---

## How to use the repository

### Infer with the best model

If you want to use the best model achieved in the thesis to do video denoising, follow these steps:

1. Move the video(s) you want to denoise to `data/to_infer`.
2. Open `unet3d/config.py`. Go to the inference hyperparameters and fill the list with exactly the video(s) you want to denoise.
3. Select the evaluator. Number 4 is our recommendation, sacrificing some time for better results.
4. Open `unet3d/infer.py` and run the script. The denoised videos will be saved in `data/infered`.

### Train a model

1. Decide if you want a 2D or a 3D architecture.
2. Open the `config.py` script related to this architecture. Select the hyperparameters you wish.
3. Either execute corresponding `train.py` script or `train.sh` shell executable with proper arguments. You can interrupt the training whenever you want and continue it later.
4. To evaluate it, open `config.py` to select validation or test mode and data.
5. Then execute the corresponding `eval.py`. You can see the denoised frames in `data/eval/check_epoch_model` and check the metrics in the console output.



