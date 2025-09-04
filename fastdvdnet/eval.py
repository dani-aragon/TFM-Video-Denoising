"""
Script to evaluate the 2D denoising FastDVDNet models via validation and test.
Edit config.py file to set the script to validation or test mode and choose
the model and checkpoints to check.
"""

import os, sys

import torch
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm

# To import local modules.
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import metadata as md
import config as conf
from datasets import Test2DDataset


def evaluate_fastdvdnet(model, dataloader, device, output_dir):
    """Inference and metrics of the given data loader for the 2D model
    "model". For each 5-frame noisy clip, the model returns the denoised
    central frame, which is saved on "output_dir" and compared through PSNR
    and SSIM metrics to the clean central frame. The function returns the
    average PSNR and SSIM across all the central frames of the dataloader.
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
            # noisy: [B, C*T, H, W], clean: [B, C, H, W]
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Forward.
            denoised = model(noisy) # [B, C, H, W]

            # Move to CPU to save and compute metrics.
            deno_cpu  = denoised.cpu()
            clean_cpu = clean.cpu()

            B = deno_cpu.size(0)
            for b in range(B):
                out_img = deno_cpu[b] # [C, H, W]
                gt_img  = clean_cpu[b] # [C, H, W]

                # Save image.
                pil = to_pil_image(out_img.clamp(0,1))
                filename = os.path.join(
                    output_dir, f"denoised_{batch_idx:04d}_{b:03d}.png"
                )
                pil.save(filename)

                # Convert to numpy H×W×C en [0,1].
                out_np = out_img.numpy().transpose(1,2,0)
                gt_np  = gt_img.numpy().transpose(1,2,0)

                # Compute PSNR and SSIM.
                psnr_val = psnr(gt_np, out_np, data_range=1.0)
                ssim_val = ssim(
                    gt_np, out_np,
                    data_range=1.0,
                    channel_axis=2
                )

                psnr_sum += psnr_val
                ssim_sum += ssim_val
                count += 1

    psnr_avg = psnr_sum / count
    ssim_avg = ssim_sum / count

    return psnr_avg, ssim_avg


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
        "S": Test2DDataset(data_path, "S"),
        "M": Test2DDataset(data_path, "M"),
        "L": Test2DDataset(data_path, "L")
    }
    print(f"Total clips in S: {len(datasets["S"])}.")
    print(f"Total clips in M: {len(datasets["M"])}.")
    print(f"Total clips in L: {len(datasets["L"])}.")

    for i in conf.CKPT_RANGE:
        # Create model eval folder.
        model_name = f"check_{i}_{conf.NAME_TEST}"
        print(f"Evaluating {model_name}.")
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

            # Create noise level folder.
            level_path = os.path.join(model_path, lvl)
            os.makedirs(level_path, exist_ok=True)

            # Create loaders.
            test_loader = DataLoader(
                datasets[lvl],
                batch_size=conf.BATCH_SIZE,
                shuffle=False,
                num_workers=conf.NUM_WORKERS,
                persistent_workers=conf.PERSISTENT_WORKERS,
                pin_memory=conf.PIN_MEMORY
            )

            cur_psnr, cur_ssim = evaluate_fastdvdnet(
                model,
                test_loader,
                device=md.DEVICE,
                output_dir=level_path
            )
            print(
                f"PSNR medio: {cur_psnr:.4f} dB, SSIM medio: {cur_ssim:.4f}."
            )
