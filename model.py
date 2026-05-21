"""
============================================================
model.py
Region-Aware Attention Framework for
Photovoltaic Parameter Estimation
============================================================

This module implements the proposed deep learning
framework for estimating:

    - Series resistance (Rs)
    - Shunt resistance (Rsh)

from photovoltaic current-voltage (I-V)
characteristics and environmental conditions.

Architecture:
    - 1D CNN encoder
    - Region decomposition
    - Cross-region attention
    - Residual attention fusion
    - Environment embedding
    - Dual regression heads

Author: <K. A. Shahriar>
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn as nn

# ============================================================
# REGION-AWARE ATTENTION MODEL
# ============================================================

class AttentionPV(nn.Module):

    """
    Region-aware attention framework for
    photovoltaic parameter estimation.
    """

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # CNN FEATURE ENCODER
        # ----------------------------------------------------
        #
        # Input channels:
        #   1. normalized voltage
        #   2. normalized current
        #   3. current gradient
        #
        # Output:
        #   hierarchical PV feature representation
        #
        # ----------------------------------------------------

        self.conv = nn.Sequential(

            nn.Conv1d(
                in_channels=3,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU()
        )

        # ----------------------------------------------------
        # CROSS-REGION ATTENTION
        # ----------------------------------------------------
        #
        # Models nonlinear dependencies between:
        #   - low-voltage region
        #   - transition region
        #   - high-voltage region
        #
        # ----------------------------------------------------

        self.cross_attn = nn.MultiheadAttention(

            embed_dim=128,

            num_heads=4,

            batch_first=True
        )

        # ----------------------------------------------------
        # ENVIRONMENT EMBEDDING
        # ----------------------------------------------------
        #
        # Inputs:
        #   - irradiance (G)
        #   - temperature (T)
        #
        # ----------------------------------------------------

        self.env_fc = nn.Sequential(

            nn.Linear(2, 32),

            nn.ReLU()
        )

        # ----------------------------------------------------
        # SERIES RESISTANCE HEAD
        # ----------------------------------------------------
        #
        # Uses:
        #   - high-voltage region
        #   - environmental embedding
        #
        # ----------------------------------------------------

        self.fc_Rs = nn.Sequential(

            nn.Linear(128 + 32, 64),

            nn.ReLU(),

            nn.Linear(64, 1)
        )

        # ----------------------------------------------------
        # SHUNT RESISTANCE HEAD
        # ----------------------------------------------------
        #
        # Uses:
        #   - low-voltage region
        #   - environmental embedding
        #
        # ----------------------------------------------------

        self.fc_Rsh = nn.Sequential(

            nn.Linear(128 + 32, 64),

            nn.ReLU(),

            nn.Linear(64, 1)
        )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(

        self,
        x,
        env,
        return_attn=False

    ):

        """
        Parameters
        ----------
        x : tensor
            Input photovoltaic tensor
            Shape:
                (B,3,N)

        env : tensor
            Environmental variables
            Shape:
                (B,2)

        return_attn : bool
            Returns attention weights

        Returns
        -------
        output : tensor
            Predicted:
                - log(Rs)
                - log(Rsh)

        attn_weights : tensor (optional)
            Cross-attention weights
        """

        # ----------------------------------------------------
        # CNN FEATURE EXTRACTION
        # ----------------------------------------------------

        h = self.conv(x)

        # feature length
        L = h.size(-1)

        # ----------------------------------------------------
        # REGION DECOMPOSITION
        # ----------------------------------------------------
        #
        # low-voltage region
        # transition region
        # high-voltage region
        #
        # ----------------------------------------------------

        left = h[:, :, :L//3].mean(-1)

        mid = h[:, :, L//3:2*L//3].mean(-1)

        right = h[:, :, 2*L//3:].mean(-1)

        # shape:
        #   (B,3,128)

        regions = torch.stack(

            [left, mid, right],

            dim=1
        )

        # ----------------------------------------------------
        # CROSS-REGION ATTENTION
        # ----------------------------------------------------

        attn_out, attn_weights = self.cross_attn(

            regions,
            regions,
            regions
        )

        # ----------------------------------------------------
        # RESIDUAL ATTENTION FUSION
        # ----------------------------------------------------

        regions = regions + 0.3 * attn_out

        # unpack regions
        left_a = regions[:,0]

        mid_a = regions[:,1]

        right_a = regions[:,2]

        # ----------------------------------------------------
        # ENVIRONMENT EMBEDDING
        # ----------------------------------------------------

        env_feat = self.env_fc(env)

        # ----------------------------------------------------
        # SERIES RESISTANCE PREDICTION
        # ----------------------------------------------------

        Rs_out = self.fc_Rs(

            torch.cat(

                [right_a, env_feat],

                dim=1
            )
        )

        # ----------------------------------------------------
        # SHUNT RESISTANCE PREDICTION
        # ----------------------------------------------------

        Rsh_out = self.fc_Rsh(

            torch.cat(

                [left_a, env_feat],

                dim=1
            )
        )

        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        output = torch.cat(

            [Rs_out, Rsh_out],

            dim=1
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        if return_attn:

            return output, attn_weights

        else:

            return output

# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    # dummy input
    x = torch.randn(8, 3, 200)

    env = torch.randn(8, 2)

    # initialize model
    model = AttentionPV()

    # forward pass
    output, attn = model(

        x,
        env,

        return_attn=True
    )

    print("\n===================================")
    print("MODEL TEST")
    print("===================================")

    print("Input shape :", x.shape)

    print("Output shape:", output.shape)

    print("Attention shape:", attn.shape)

    print("\nModel initialized successfully.")