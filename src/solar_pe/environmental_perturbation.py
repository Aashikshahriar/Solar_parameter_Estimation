"""Environmental perturbation analysis: how prediction quality degrades when the reported
irradiance (G) and/or temperature (T) inputs are noisy, isolating each factor's effect.
"""

import numpy as np
import pandas as pd
import torch

from .evaluate import relative_error_pct

DEFAULT_PERTURB_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


@torch.no_grad()
def _predict_with_env_noise(model, loader, device, g_noise: float, t_noise: float):
    pred_Rs, true_Rs, pred_Rsh, true_Rsh = [], [], [], []

    for Xb, envb, yb in loader:
        Xb, envb = Xb.to(device), envb.to(device)

        env_pert = envb.clone()
        if g_noise > 0:
            env_pert[:, 0] += torch.randn_like(env_pert[:, 0]) * g_noise
        if t_noise > 0:
            env_pert[:, 1] += torch.randn_like(env_pert[:, 1]) * t_noise

        pred = model(Xb, env_pert).cpu()
        pred_Rs.append(torch.exp(pred[:, 0]).numpy())
        pred_Rsh.append(torch.exp(pred[:, 1]).numpy())
        true_Rs.append(torch.exp(yb[:, 0]).numpy())
        true_Rsh.append(torch.exp(yb[:, 1]).numpy())

    return (np.concatenate(pred_Rs), np.concatenate(true_Rs),
            np.concatenate(pred_Rsh), np.concatenate(true_Rsh))


def run_env_perturbation_sweep(model, loader, device, perturb_levels: list = None) -> pd.DataFrame:
    """Perturb irradiance and temperature independently across `perturb_levels` (as standardized
    std-dev noise, since env features are z-scored). Returns one row per level with both factors'
    relative errors side by side.
    """
    perturb_levels = perturb_levels or DEFAULT_PERTURB_LEVELS
    rows = []

    for p in perturb_levels:
        pred_Rs, true_Rs, pred_Rsh, true_Rsh = _predict_with_env_noise(model, loader, device, g_noise=p, t_noise=0.0)
        rs_re_G = relative_error_pct(true_Rs, pred_Rs)
        rsh_re_G = relative_error_pct(true_Rsh, pred_Rsh)

        pred_Rs, true_Rs, pred_Rsh, true_Rsh = _predict_with_env_noise(model, loader, device, g_noise=0.0, t_noise=p)
        rs_re_T = relative_error_pct(true_Rs, pred_Rs)
        rsh_re_T = relative_error_pct(true_Rsh, pred_Rsh)

        print(f"{int(p * 100)}% perturbation | G -> Rs={rs_re_G:.2f}% Rsh={rsh_re_G:.2f}% "
              f"| T -> Rs={rs_re_T:.2f}% Rsh={rsh_re_T:.2f}%")

        rows.append({
            "Perturbation (%)": int(p * 100),
            "G_Rs_RE%": rs_re_G, "G_Rsh_RE%": rsh_re_G,
            "T_Rs_RE%": rs_re_T, "T_Rsh_RE%": rsh_re_T,
        })

    return pd.DataFrame(rows)


def run_env_perturbation_grid(model, loader, device, g_levels: list = None, t_levels: list = None) -> pd.DataFrame:
    """Joint G x T perturbation grid (useful for a heatmap of combined environmental noise)."""
    g_levels = g_levels or [0.00, 0.02, 0.05, 0.10]
    t_levels = t_levels or [0.00, 0.02, 0.05, 0.10]
    rows = []

    for g_noise in g_levels:
        for t_noise in t_levels:
            pred_Rs, true_Rs, pred_Rsh, true_Rsh = _predict_with_env_noise(model, loader, device, g_noise, t_noise)
            rows.append({
                "G_noise": g_noise, "T_noise": t_noise,
                "Rs_RE%": relative_error_pct(true_Rs, pred_Rs),
                "Rsh_RE%": relative_error_pct(true_Rsh, pred_Rsh),
            })

    return pd.DataFrame(rows)
