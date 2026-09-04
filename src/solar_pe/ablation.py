"""Ablation study: isolate the contribution of region pooling, cross-attention, and
environmental conditioning by progressively enabling them on a shared CNN backbone.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .evaluate import relative_error_pct

# name -> (use_regions, use_attn, use_env)
ABLATION_CONFIGS = {
    "Baseline CNN": (False, False, False),
    "+ Regions": (True, False, False),
    "+ Attention": (True, True, False),
    "+ Env (Full)": (True, True, True),
}


class AblationModel(nn.Module):
    """Same backbone as AttentionPV, with region pooling / attention / env fusion each optional."""

    def __init__(self, use_regions: bool = True, use_attn: bool = True, use_env: bool = True):
        super().__init__()
        self.use_regions = use_regions
        self.use_attn = use_attn
        self.use_env = use_env

        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
        )

        if use_attn:
            self.attn = nn.MultiheadAttention(128, 4, batch_first=True)

        if use_env:
            self.env_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
            fc_in = 128 + 32
        else:
            fc_in = 128

        self.fc_Rs = nn.Sequential(nn.Linear(fc_in, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fc_Rsh = nn.Sequential(nn.Linear(fc_in, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)  # (B, 128, L)
        L = feat.size(-1)

        if self.use_regions:
            left = feat[:, :, :int(L * 0.3)].mean(dim=2)
            mid = feat[:, :, int(L * 0.3):int(L * 0.7)].mean(dim=2)
            right = feat[:, :, int(L * 0.7):].mean(dim=2)
            tokens = torch.stack([left, mid, right], dim=1)

            if self.use_attn:
                tokens, _ = self.attn(tokens, tokens, tokens)

            feat = tokens.mean(dim=1)
        else:
            feat = feat.mean(dim=2)

        if self.use_env:
            feat = torch.cat([feat, self.env_fc(env)], dim=1)

        return torch.cat([self.fc_Rs(feat), self.fc_Rsh(feat)], dim=1)


def train_variant(model: nn.Module, train_loader, device, epochs: int = 50, lr: float = 5e-4) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for Xb, envb, yb in train_loader:
            Xb, envb, yb = Xb.to(device), envb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(Xb, envb), yb)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def evaluate_variant(model: nn.Module, loader, device) -> tuple:
    """Returns (Rs_relative_error_pct, Rsh_relative_error_pct)."""
    model.eval()
    pred_Rs, true_Rs, pred_Rsh, true_Rsh = [], [], [], []

    for Xb, envb, yb in loader:
        Xb, envb = Xb.to(device), envb.to(device)
        pred = model(Xb, envb).cpu()

        pred_Rs.append(torch.exp(pred[:, 0]).numpy())
        pred_Rsh.append(torch.exp(pred[:, 1]).numpy())
        true_Rs.append(torch.exp(yb[:, 0]).numpy())
        true_Rsh.append(torch.exp(yb[:, 1]).numpy())

    pred_Rs, true_Rs = np.concatenate(pred_Rs), np.concatenate(true_Rs)
    pred_Rsh, true_Rsh = np.concatenate(pred_Rsh), np.concatenate(true_Rsh)

    return relative_error_pct(true_Rs, pred_Rs), relative_error_pct(true_Rsh, pred_Rsh)


def run_ablation_study(train_loader, test_loader, device, configs: dict = None, epochs: int = 50) -> pd.DataFrame:
    """Train and evaluate each ablation variant. Returns a DataFrame of Rs/Rsh relative error per variant."""
    configs = configs or ABLATION_CONFIGS
    rows = []

    for name, (use_regions, use_attn, use_env) in configs.items():
        print(f"Running ablation variant: {name}")
        model = AblationModel(use_regions, use_attn, use_env)
        model = train_variant(model, train_loader, device, epochs=epochs)
        rs_err, rsh_err = evaluate_variant(model, test_loader, device)
        rows.append({"Model": name, "Rs_RE%": rs_err, "Rsh_RE%": rsh_err})

    return pd.DataFrame(rows)
