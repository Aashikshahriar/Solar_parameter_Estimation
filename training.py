"""
============================================================
training.py
Training Pipeline for Region-Aware Attention
Photovoltaic Parameter Estimation Framework
============================================================

Author: <K. A. Shahriar>
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from model import AttentionPV
from preprocessing import (
    build_preprocessing_pipeline
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)

# ============================================================
# CONFIGURATION
# ============================================================

lr = 5e-4

w_rs = 2.0
w_rsh = 2.0

noise_level = 0.01

wd = 1e-5

epochs = 100

# ============================================================
# LOAD DATA
# ============================================================

train_loader, val_loader, test_loader = (

    build_preprocessing_pipeline(

        filename="pv_dataset.npz",

        batch_size=64
    )
)

# ============================================================
# MODEL
# ============================================================

model = AttentionPV().to(device)

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(

    model.parameters(),

    lr=lr,

    weight_decay=wd
)

# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=80
)

# ============================================================
# LOSS
# ============================================================

mse = nn.MSELoss()

# ============================================================
# TRAINING
# ============================================================

best_val = float("inf")

train_losses = []
val_losses = []

print("\n===================================")
print("TRAINING STARTED")
print("===================================")

for epoch in range(epochs):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0

    for Xb, envb, yb in train_loader:

        Xb = Xb.to(device)
        envb = envb.to(device)
        yb = yb.to(device)

        # ----------------------------------------------------
        # Noise augmentation
        # ----------------------------------------------------

        Xb[:,1,:] += (

            torch.randn_like(Xb[:,1,:])

            * noise_level
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad()

        pred = model(Xb, envb)

        # ----------------------------------------------------
        # Multi-task loss
        # ----------------------------------------------------

        loss_rs = mse(

            pred[:,0],
            yb[:,0]
        )

        loss_rsh = mse(

            pred[:,1],
            yb[:,1]
        )

        loss = (

            w_rs * loss_rs

            +

            w_rsh * loss_rsh
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for Xb, envb, yb in val_loader:

            Xb = Xb.to(device)
            envb = envb.to(device)
            yb = yb.to(device)

            pred = model(Xb, envb)

            loss_rs = mse(

                pred[:,0],
                yb[:,0]
            )

            loss_rsh = mse(

                pred[:,1],
                yb[:,1]
            )

            val_loss += (

                loss_rs + loss_rsh
            ).item()

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    train_loss /= len(train_loader)

    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(

        f"Epoch {epoch+1:03d} | "

        f"Train {train_loss:.4f} | "

        f"Val {val_loss:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val:

        best_val = val_loss

        torch.save(

            model.state_dict(),

            "final_model.pth"
        )

print("\n===================================")
print("TRAINING COMPLETE")
print("===================================")

print("Best validation loss:", best_val)
print("Model saved: final_model.pth")