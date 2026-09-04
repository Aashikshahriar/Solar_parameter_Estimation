"""Gradient-based sensitivity analysis: which parts of the I-V curve drive each prediction.

Computes d(Rs)/d(current) and d(Rsh)/d(current) via backprop, averaged over samples, both
as a per-voltage-index profile and aggregated into left/middle/right curve regions.
"""

import numpy as np
import pandas as pd
import torch

DEFAULT_REGIONS = {"Left": (0.0, 0.3), "Middle": (0.3, 0.7), "Right": (0.7, 1.0)}


def compute_gradient_sensitivity(model, loader, device, num_samples: int = 100):
    """Average |d(log Rs)/d(I)| and |d(log Rsh)/d(I)| across `num_samples` curves.

    Returns two arrays of shape (n_points,), each normalized to a [0, 1] max.
    """
    model.eval()
    grad_Rs_all, grad_Rsh_all = [], []

    n = min(num_samples, len(loader.dataset))
    for i in range(n):
        Xb, envb, _ = loader.dataset[i]
        Xb = Xb.unsqueeze(0).to(device)
        envb = envb.unsqueeze(0).to(device)
        Xb.requires_grad_(True)

        pred = model(Xb, envb)

        pred[:, 0].backward(retain_graph=True)
        grad_Rs_all.append(Xb.grad[:, 1, :].detach().cpu().numpy())
        Xb.grad.zero_()

        pred[:, 1].backward()
        grad_Rsh_all.append(Xb.grad[:, 1, :].detach().cpu().numpy())

    sens_Rs = np.mean(np.abs(grad_Rs_all), axis=0)[0]
    sens_Rsh = np.mean(np.abs(grad_Rsh_all), axis=0)[0]

    sens_Rs = sens_Rs / (sens_Rs.max() + 1e-8)
    sens_Rsh = sens_Rsh / (sens_Rsh.max() + 1e-8)

    return sens_Rs, sens_Rsh


def region_sensitivity(sens_Rs: np.ndarray, sens_Rsh: np.ndarray, regions: dict = None) -> pd.DataFrame:
    """Aggregate a per-index sensitivity profile into named curve regions, normalized to [0, 1]."""
    regions = regions or DEFAULT_REGIONS
    n = len(sens_Rs)

    rows = []
    for name, (frac_start, frac_end) in regions.items():
        s, e = int(frac_start * n), int(frac_end * n)
        rows.append({"Region": name, "Rs": sens_Rs[s:e].mean(), "Rsh": sens_Rsh[s:e].mean()})

    df = pd.DataFrame(rows)
    df["Rs"] /= df["Rs"].max()
    df["Rsh"] /= df["Rsh"].max()
    return df
