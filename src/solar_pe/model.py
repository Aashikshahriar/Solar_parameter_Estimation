"""Model architecture for joint Rs / Rsh estimation from I-V curves.

CNN encoder -> region pooling (left / middle / right of the curve) -> self-attention
across regions -> environment-conditioned dual regression heads.
"""

import torch
import torch.nn as nn


class AttentionPV(nn.Module):
    def __init__(self, in_channels: int = 3, env_dim: int = 2, hidden: int = 128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
        )

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, batch_first=True)

        self.env_fc = nn.Sequential(
            nn.Linear(env_dim, 32),
            nn.ReLU(),
        )

        self.fc_Rs = nn.Sequential(
            nn.Linear(hidden + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.fc_Rsh = nn.Sequential(
            nn.Linear(hidden + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, env: torch.Tensor, return_attn: bool = False):
        h = self.conv(x)
        L = h.size(-1)

        # split the curve into three physically meaningful regions
        left = h[:, :, :L // 3].mean(-1)
        mid = h[:, :, L // 3:2 * L // 3].mean(-1)
        right = h[:, :, 2 * L // 3:].mean(-1)

        regions = torch.stack([left, mid, right], dim=1)  # (B, 3, hidden)
        attn_out, attn_weights = self.cross_attn(regions, regions, regions)
        regions = regions + 0.3 * attn_out  # residual attention

        left_a, _, right_a = regions[:, 0], regions[:, 1], regions[:, 2]

        env_feat = self.env_fc(env)

        # Rs dominates the curve's right (near-Voc) slope; Rsh dominates the left (near-Isc) slope
        Rs_out = self.fc_Rs(torch.cat([right_a, env_feat], dim=1))
        Rsh_out = self.fc_Rsh(torch.cat([left_a, env_feat], dim=1))

        output = torch.cat([Rs_out, Rsh_out], dim=1)

        if return_attn:
            return output, attn_weights
        return output
