"""Measurement noise robustness: how prediction quality degrades as sensor noise on the
current channel increases, at inference time (independent of training-time augmentation).
"""

import numpy as np
import pandas as pd
import torch

from .evaluate import compute_metrics

DEFAULT_NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.03]


@torch.no_grad()
def evaluate_under_noise(model, loader, device, noise_std: float = 0.0) -> dict:
    """Evaluate the model with Gaussian noise injected into the current channel at inference time."""
    model.eval()
    pred_Rs, true_Rs, pred_Rsh, true_Rsh = [], [], [], []

    for Xb, envb, yb in loader:
        Xb, envb = Xb.to(device), envb.to(device)

        if noise_std > 0:
            Xb = Xb.clone()
            Xb[:, 1, :] += torch.randn_like(Xb[:, 1, :]) * noise_std

        pred = model(Xb, envb).cpu()
        pred_Rs.append(torch.exp(pred[:, 0]).numpy())
        pred_Rsh.append(torch.exp(pred[:, 1]).numpy())
        true_Rs.append(torch.exp(yb[:, 0]).numpy())
        true_Rsh.append(torch.exp(yb[:, 1]).numpy())

    pred_Rs, true_Rs = np.concatenate(pred_Rs), np.concatenate(true_Rs)
    pred_Rsh, true_Rsh = np.concatenate(pred_Rsh), np.concatenate(true_Rsh)

    m_rs = compute_metrics(true_Rs, pred_Rs)
    m_rsh = compute_metrics(true_Rsh, pred_Rsh)

    return {
        "Rs_MAE": m_rs["MAE"], "Rs_R2": m_rs["R2"], "Rs_RE%": m_rs["RE%"],
        "Rsh_MAE": m_rsh["MAE"], "Rsh_R2": m_rsh["R2"], "Rsh_RE%": m_rsh["RE%"],
    }


def run_noise_sweep(model, loader, device, noise_levels: list = None) -> pd.DataFrame:
    """Evaluate the model at each noise level and return a comparison table."""
    noise_levels = noise_levels or DEFAULT_NOISE_LEVELS
    rows = []

    for noise in noise_levels:
        print(f"Testing noise level: {noise}")
        metrics = evaluate_under_noise(model, loader, device, noise_std=noise)
        metrics["Noise"] = noise
        rows.append(metrics)

    return pd.DataFrame(rows)
