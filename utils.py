import os

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import device
from tqdm.auto import tqdm

import metadata as md


##############################################################################
# Train function.
def train(
    model: nn.Module,
    loader: DataLoader,
    name: str,
    num_epochs: int,
    lr: float,
    last_epoch: int,
    optimizer: Optimizer,
    use_scheduler: bool,
    save_path: str =md.PATH_CHECK,
    device: device =md.DEVICE
) -> None:
    """_summary_

    Args:
        model (nn.Module): _description_
        loader (DataLoader): _description_
        name (str): _description_
        num_epochs (int): _description_
        lr (float): _description_
        last_epoch (int): _description_
        optimizer (Optimizer): _description_
        use_scheduler (bool): _description_
        save_path (str, optional): _description_. Defaults to md.PATH_CHECK.
        device (device, optional): _description_. Defaults to md.DEVICE.
    """
    print(f"Training {name}.")

    # Initialize optimizer.
    optimizer = optimizer(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # To resume training.
    if last_epoch > 0:
        check_path = os.path.join(
            save_path,
            f"check_{last_epoch}_{name}.pth"
        )
        ckpt = torch.load(check_path, map_location=device)

        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))

        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])

        # Resume scheduler if necessary.
        if use_scheduler:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs-1,
                eta_min=lr/100,
                last_epoch=ckpt.get('epoch', -1)
            )

        print(
            f"Resuming from epoch {ckpt['epoch']}: best_loss={best_loss:.6f}."
        )
        print(f"Starting epoch {start_epoch}.")
    
    # To initialize training.
    else:
        # Initialize scheduler if necessary.
        if use_scheduler:
            scheduler = CosineAnnealingLR(
                optimizer, num_epochs-1, eta_min=lr/100
            )

        start_epoch = 1
        best_loss = float("inf")

    # Training loop.
    for epoch in range(start_epoch, num_epochs+1):
        model.train()
        running_loss  = 0.0
        total_samples = 0

        # Batch loop.
        train_loader = tqdm(
            loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"
        )
        for _, (noisy, gt) in enumerate(train_loader, start=1):
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            out = model(noisy)
            loss = criterion(out, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = noisy.size(0)
            running_loss  += loss.item() * bs
            total_samples += bs
            avg_loss = running_loss / total_samples

            train_loader.set_postfix(
                loss=f"{avg_loss:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )

        epoch_loss = running_loss / total_samples
        print(f"Epoch {epoch} final loss: {epoch_loss:.6f}.")

        # Scheduler step.
        if use_scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            print(f"Adjusted LR: {old_lr:.1e} -> {new_lr:.1e}.")

        # Save checkpoints.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        torch.save(
            {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'sched_state': (
                    scheduler.state_dict()
                    if use_scheduler else None
                ),
                'best_loss': best_loss,
            },
            os.path.join(save_path, f"check_{epoch}_{name}.pth")
        )
    
    print("Training has finished!")
