"""Training loop with early stopping, cosine LR schedule, and checkpointing."""

import torch
import torch.nn as nn

from .config import TrainConfig


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    loss_rs = nn.functional.mse_loss(pred[:, 0], target[:, 0])
    loss_rsh = nn.functional.mse_loss(pred[:, 1], target[:, 1])
    return cfg.w_rs * loss_rs + cfg.w_rsh * loss_rsh


def run_epoch(model, loader, device, cfg: TrainConfig, optimizer=None) -> float:
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    for Xb, envb, yb in loader:
        Xb, envb, yb = Xb.to(device), envb.to(device), yb.to(device)

        if is_train and cfg.train_noise_std > 0:
            # light augmentation on the current channel to improve noise robustness
            Xb = Xb.clone()
            Xb[:, 1, :] += torch.randn_like(Xb[:, 1, :]) * cfg.train_noise_std

        with torch.set_grad_enabled(is_train):
            pred = model(Xb, envb)
            loss = weighted_loss(pred, yb, cfg)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def fit(model, train_loader, val_loader, device, cfg: TrainConfig = TrainConfig(),
        checkpoint_path: str = "checkpoints/best_model.pth", verbose: bool = True):
    """Train `model` with early stopping on validation loss. Returns loss history."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(cfg.epochs):
        train_loss = run_epoch(model, train_loader, device, cfg, optimizer=optimizer)
        scheduler.step()
        val_loss = run_epoch(model, val_loader, device, cfg, optimizer=None)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{cfg.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter > cfg.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return {"train_loss": train_losses, "val_loss": val_losses}
