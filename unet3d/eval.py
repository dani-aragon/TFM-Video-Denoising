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


def evaluate_unet3d(model, dataloader, device, output_dir, model_input_order="bcthw"):
    """
    Evalúa una UNet3D usando el protocolo explicado arriba.

    Args:
        model: instancia de tu UNet3D (PyTorch).
        dataloader: DataLoader de Test3DDataset con clip_len=5.
        device: 'cuda' o 'cpu'.
        output_dir: carpeta donde se guardan los frames promediados.
        model_input_order: cómo espera el modelo la entrada:
            - "bcthw" (default): batch, channels, time, H, W  -> shape [B, C, T, H, W]
            - "btchw": batch, time, channels, H, W          -> shape [B, T, C, H, W]
            (Ajusta según tu UNet3D. La mayoría de conv3d en PyTorch usan [B, C, D, H, W].)
    Returns:
        (psnr_mean, ssim_mean)
    """
    os.makedirs(output_dir, exist_ok=True)
    model.to(device).eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(tqdm(dataloader, desc="Evaluating UNet3D clips")):
            # Dataset devuelve: noisy [B, C, T, H, W], clean [B, C, T, H, W]
            # Aseguramos tipo float32
            noisy = noisy.to(device)
            clean = clean.to(device)

            B, C, T, H, W = noisy.shape
            assert T == 5, f"Se esperaba T=5 en el test dataset, pero T={T}"

            # Construir las dos ventanas de 4 frames
            # ventana A: frames 0..3
            # ventana B: frames 1..4
            winA = noisy[:, :, 0:4, :, :]  # [B, C, 4, H, W]
            winB = noisy[:, :, 1:5, :, :]  # [B, C, 4, H, W]

            # Algunos modelos UNet3D esperan [B, T, C, H, W], otros [B, C, T, H, W].
            # Aquí asumimos por defecto [B, C, T, H, W] (model_input_order="bcthw").
            if model_input_order == "btchw":
                # permutar a [B, T, C, H, W]
                inpA = winA.permute(0, 2, 1, 3, 4).contiguous()
                inpB = winB.permute(0, 2, 1, 3, 4).contiguous()
            else:
                # usar [B, C, T, H, W]
                inpA = winA
                inpB = winB

            # Forward pass por ventanas de 4
            outA = model(inpA)  # esperamos [B, C, 4, H, W] o [B, 4, C, H, W] dependiendo del modelo
            outB = model(inpB)

            # Normalizar la salida a [B, C, 4, H, W] para extraer índices con consistencia.
            # Intentamos inferir la forma: si out.shape[1] == C => [B,C,T,H,W]
            # si out.shape[1] == 4 => [B,4,C,H,W]
            def normalize_out(out_tensor):
                # out_tensor puede ser [B, C, 4, H, W] o [B, 4, C, H, W]
                if out_tensor.dim() != 5:
                    raise RuntimeError(f"Salida del modelo con dim inesperada: {out_tensor.dim()}")
                b, d1, d2, d3, d4 = out_tensor.shape
                if d1 == C:
                    # forma [B, C, 4, H, W] -> ok
                    return out_tensor  # [B, C, 4, H, W]
                elif d1 == 4 and d2 == C:
                    # forma [B, 4, C, H, W] -> permutar a [B, C, 4, H, W]
                    return out_tensor.permute(0, 2, 1, 3, 4).contiguous()
                else:
                    # forma inesperada
                    raise RuntimeError(f"Salida del modelo con shape inesperada: {out_tensor.shape}")

            outA = normalize_out(outA)
            outB = normalize_out(outB)

            # Extraer las dos predicciones para el frame central del clip original (índice 2)
            # - outA corresponde a frames [0,1,2,3], por tanto índice temporal 2 dentro de outA es el frame central
            # - outB corresponde a frames [1,2,3,4], por tanto índice temporal 1 dentro de outB es el frame central
            predA_center = outA[:, :, 2, :, :]  # [B, C, H, W]
            predB_center = outB[:, :, 1, :, :]  # [B, C, H, W]

            # Media simple de ambas predicciones
            pred_center = 0.5 * (predA_center + predB_center)  # [B, C, H, W]

            # Ground truth center frame
            gt_center = clean[:, :, 2, :, :]  # [B, C, H, W]

            # Move to CPU once for downstream ops
            pred_cpu = pred_center.cpu()
            gt_cpu = gt_center.cpu()

            # Guarda y calcula métricas por elemento del batch
            for b in range(B):
                out_img = pred_cpu[b].clamp(0.0, 1.0)  # [C, H, W], rango [0,1]
                gt_img = gt_cpu[b].clamp(0.0, 1.0)

                # Guardar imagen
                pil = to_pil_image(out_img)
                filename = os.path.join(output_dir, f"unet3d_{batch_idx:04d}_{b:03d}.png")
                pil.save(filename)

                # numpy HWC
                out_np = out_img.numpy().transpose(1, 2, 0)
                gt_np = gt_img.numpy().transpose(1, 2, 0)

                # PSNR / SSIM
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

    for i in range(20, 19, -1):
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

            psnr_mean, ssim_mean = evaluate_unet3d(
                model,
                test_loader,
                device=md.DEVICE,
                output_dir=level_path,
                model_input_order="bcthw"  # cambia a "btchw" si tu modelo espera [B,T,C,H,W]
            )
            print(f"PSNR medio: {psnr_mean:.4f} dB, SSIM medio: {ssim_mean:.4f}.")
