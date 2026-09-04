"""Model-internal interpretability: attention maps and region perturbation / occlusion analysis.

Complements `sensitivity.py` (gradient-based) with two model-agnostic probes:
- occlusion: zero out a curve region, measure how much the prediction moves
- perturbation: inject noise into a curve region, measure how much the prediction moves
"""

import numpy as np
import pandas as pd
import torch

DEFAULT_REGIONS = {"Left": (0.0, 0.3), "Middle": (0.3, 0.7), "Right": (0.7, 1.0)}


@torch.no_grad()
def extract_attention_map(model, X: torch.Tensor, env: torch.Tensor, device) -> np.ndarray:
    """Run one sample through the model and return the (3, 3) region cross-attention matrix,
    averaged over heads. Requires a model whose forward() supports `return_attn=True`.
    """
    model.eval()
    X = X.unsqueeze(0).to(device) if X.dim() == 2 else X.to(device)
    env = env.unsqueeze(0).to(device) if env.dim() == 1 else env.to(device)

    _, attn_weights = model(X, env, return_attn=True)
    attn = attn_weights.detach().cpu().numpy()

    if attn.ndim == 4:      # (B, heads, 3, 3)
        attn = attn[0].mean(axis=0)
    elif attn.ndim == 3:    # (B, 3, 3)
        attn = attn[0]
    else:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    return attn


def region_perturbation_occlusion(model, loader, device, regions: dict = None,
                                   num_samples: int = 100, noise_std: float = 0.05) -> pd.DataFrame:
    """For each curve region, measure the relative change in Rs/Rsh predictions when that
    region is (a) perturbed with noise and (b) fully occluded (zeroed out).

    Returns a per-sample, per-region DataFrame; average with `.groupby("Region").mean()`.
    """
    model.eval()
    regions = regions or DEFAULT_REGIONS
    n = min(num_samples, len(loader.dataset))

    rows = []
    for i in range(n):
        Xb, envb, _ = loader.dataset[i]
        Xb = Xb.unsqueeze(0).to(device)
        envb = envb.unsqueeze(0).to(device)

        with torch.no_grad():
            base = model(Xb, envb)
        base_Rs = torch.exp(base[:, 0]).item()
        base_Rsh = torch.exp(base[:, 1]).item()

        L = Xb.size(-1)
        for name, (frac_start, frac_end) in regions.items():
            start, end = int(frac_start * L), int(frac_end * L)

            X_noise = Xb.clone()
            X_noise[:, :, start:end] += torch.randn_like(X_noise[:, :, start:end]) * noise_std
            with torch.no_grad():
                pred_noise = model(X_noise, envb)

            X_mask = Xb.clone()
            X_mask[:, :, start:end] = 0.0
            with torch.no_grad():
                pred_mask = model(X_mask, envb)

            Rs_noise, Rsh_noise = torch.exp(pred_noise[:, 0]).item(), torch.exp(pred_noise[:, 1]).item()
            Rs_mask, Rsh_mask = torch.exp(pred_mask[:, 0]).item(), torch.exp(pred_mask[:, 1]).item()

            rows.append({
                "Region": name,
                "Rs_pert": abs(Rs_noise - base_Rs) / (base_Rs + 1e-8),
                "Rsh_pert": abs(Rsh_noise - base_Rsh) / (base_Rsh + 1e-8),
                "Rs_occ": abs(Rs_mask - base_Rs) / (base_Rs + 1e-8),
                "Rsh_occ": abs(Rsh_mask - base_Rsh) / (base_Rsh + 1e-8),
            })

    return pd.DataFrame(rows)
