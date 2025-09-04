"""
Script to evaluate the 3D denoising UNet3D models via validation and test.
Edit config.py file to set the script to validation or test mode and choose
the model and checkpoints to check.
"""

import os, sys

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# To import local modules.
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import metadata as md
import config as conf
from datasets import Test3DDataset


def evaluate_unet3d(model, dataloader, device, output_dir):
    """Inference and metrics of the given data loader for the 3D model
    "model". For each 4-frame noisy clip, the model returns the denoised 4
    frames, which are saved on "output_dir". To compute the metrics, the two
    outputs of the central frame are averaged and the result is compared to
    the clean central frame through PSNR and SSIM metrics. The function
    returns the average PSNR and SSIM across all the central frames of the
    dataloader.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.to(device).eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(
            tqdm(dataloader, desc="Evaluating clips")
        ):
            noisy = noisy.to(device)
            clean = clean.to(device)

            B, C, T, H, W = noisy.shape

            # Build and evaluate both 4-frame windows.
            winA = noisy[:, :, 0:4, :, :] # [B, C, 4, H, W]
            winB = noisy[:, :, 1:5, :, :] # [B, C, 4, H, W]
            outA = model(winA) # [B, C, 4, H, W]
            outB = model(winB) # [B, C, 4, H, W]

            # Extract the 2 outputs corresponding to the central frame.
            predA_center = outA[:, :, 2, :, :] # [B, C, H, W]
            predB_center = outB[:, :, 1, :, :] # [B, C, H, W]

            # Average of both outputs.
            pred_center = 0.5 * (predA_center + predB_center) # [B, C, H, W]

            # Ground-truth center frame.
            gt_center = clean[:, :, 2, :, :] # [B, C, H, W]

            # Move to CPU once for downstream ops
            pred_cpu = pred_center.cpu()
            gt_cpu = gt_center.cpu()

            # Save and compute metrics by batch element.
            for b in range(B):
                out_img = pred_cpu[b].clamp(0.0, 1.0) # [C, H, W]
                gt_img = gt_cpu[b].clamp(0.0, 1.0) # [C, H, W]

                # Save image.
                pil = to_pil_image(out_img)
                filename = os.path.join(
                    output_dir,
                    f"denoised_{batch_idx:04d}_{b:03d}.png"
                )
                pil.save(filename)

                # Transform to numpy.
                out_np = out_img.numpy().transpose(1, 2, 0)
                gt_np = gt_img.numpy().transpose(1, 2, 0)

                # Compute PSNR and SSIM.
                psnr_val = psnr(gt_np, out_np, data_range=1.0)
                ssim_val = ssim(gt_np, out_np, data_range=1.0, channel_axis=2)

                psnr_sum += float(psnr_val)
                ssim_sum += float(ssim_val)
                count += 1

    psnr_mean = psnr_sum / max(1, count)
    ssim_mean = ssim_sum / max(1, count)
    
    return psnr_mean, ssim_mean


if __name__ == "__main__":
    # Distinguish between validation and test.
    if conf.VAL:
        data_path = md.PATH_VAL
        prefix = "val_"
        print("Validation.")
    else:
        data_path = md.PATH_TEST
        prefix = "test_"
        print("Test.")
    
    # Create datasets.
    datasets = {
        "S": Test3DDataset(data_path, "S", clip_len=5),
        "M": Test3DDataset(data_path, "M", clip_len=5),
        "L": Test3DDataset(data_path, "L", clip_len=5)
    }
    print(f"Total clips in S: {len(datasets['S'])}.")
    print(f"Total clips in M: {len(datasets['M'])}.")
    print(f"Total clips in L: {len(datasets['L'])}.")

    for i in conf.CKPT_RANGE:
        model_name = f"check_{i}_{conf.NAME_TEST}"
        print(f"\nEvaluating {model_name}.")
        model_path = os.path.join(md.PATH_EVAL, prefix + model_name)
        os.makedirs(model_path, exist_ok=True)

        # Create and load model.
        model = conf.MODEL(conf.BASE_CHANNELS).to(md.DEVICE)
        ckpt = torch.load(
            os.path.join(md.PATH_CHECK, model_name + ".pth"),
            map_location=md.DEVICE
        )
        model.load_state_dict(ckpt["model_state"])

        # Iterate over noise levels.
        for lvl in ("S", "M", "L"):
            print(f"Evaluating clips with noise level {lvl}.")

            level_path = os.path.join(model_path, lvl)
            os.makedirs(level_path, exist_ok=True)

            test_loader = DataLoader(
                datasets[lvl],
                batch_size=conf.BATCH_SIZE,
                shuffle=False,
                num_workers=conf.NUM_WORKERS,
                persistent_workers=conf.PERSISTENT_WORKERS,
                pin_memory=conf.PIN_MEMORY
            )

            cur_psnr, cur_ssim = evaluate_unet3d(
                model,
                test_loader,
                device=md.DEVICE,
                output_dir=level_path
            )
            print(
                f"PSNR medio: {cur_psnr:.4f} dB, SSIM medio: {cur_ssim:.4f}."
            )
