"""
============================================================
evaluation.py
Evaluation Pipeline for Region-Aware
PV Attention Framework
============================================================

Author: <YOUR NAME>
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np

import torch

import matplotlib.pyplot as plt

from sklearn.metrics import (

    r2_score,
    mean_absolute_error
)

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
# LOAD DATA
# ============================================================

train_loader, val_loader, test_loader = (

    build_preprocessing_pipeline(

        filename="pv_dataset.npz",

        batch_size=64
    )
)

# ============================================================
# LOAD MODEL
# ============================================================

model = AttentionPV().to(device)

model.load_state_dict(

    torch.load(

        "final_model.pth",

        map_location=device
    )
)

model.eval()

print("\nModel loaded successfully.")

# ============================================================
# METRICS
# ============================================================

def rel_err(y, yhat):

    return np.mean(

        np.abs(yhat - y)

        / (y + 1e-8)

    ) * 100

# ============================================================
# EVALUATION
# ============================================================

pred_Rs = []
true_Rs = []

pred_Rsh = []
true_Rsh = []

with torch.no_grad():

    for Xb, envb, yb in test_loader:

        Xb = Xb.to(device)
        envb = envb.to(device)
        yb = yb.to(device)

        pred = model(Xb, envb)

        # inverse log transform
        pred_Rs.append(

            torch.exp(pred[:,0])

            .cpu()
            .numpy()
        )

        pred_Rsh.append(

            torch.exp(pred[:,1])

            .cpu()
            .numpy()
        )

        true_Rs.append(

            torch.exp(yb[:,0])

            .cpu()
            .numpy()
        )

        true_Rsh.append(

            torch.exp(yb[:,1])

            .cpu()
            .numpy()
        )

# ============================================================
# CONCATENATE
# ============================================================

pred_Rs = np.concatenate(pred_Rs)
true_Rs = np.concatenate(true_Rs)

pred_Rsh = np.concatenate(pred_Rsh)
true_Rsh = np.concatenate(true_Rsh)

# ============================================================
# RESULTS
# ============================================================

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(

    "\nRs Results:\n"

    "MAE  :", mean_absolute_error(
        true_Rs,
        pred_Rs
    ),

    "\nR2   :", r2_score(
        true_Rs,
        pred_Rs
    ),

    "\nRE%  :", rel_err(
        true_Rs,
        pred_Rs
    )
)

print(

    "\nRsh Results:\n"

    "MAE  :", mean_absolute_error(
        true_Rsh,
        pred_Rsh
    ),

    "\nR2   :", r2_score(
        true_Rsh,
        pred_Rsh
    ),

    "\nRE%  :", rel_err(
        true_Rsh,
        pred_Rsh
    )
)

# ============================================================
# LOSSLESS SCATTER PLOTS
# ============================================================

plt.figure(figsize=(6,6), dpi=300)

plt.scatter(

    true_Rs,
    pred_Rs,

    alpha=0.3
)

plt.plot(

    [true_Rs.min(), true_Rs.max()],

    [true_Rs.min(), true_Rs.max()],

    'r--'
)

plt.xlabel("True Rs", fontsize=14, fontweight='bold')
plt.ylabel("Predicted Rs", fontsize=14, fontweight='bold')

plt.title(
    "Rs Prediction",
    fontsize=15,
    fontweight='bold'
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Rs_prediction.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================

plt.figure(figsize=(6,6), dpi=300)

plt.scatter(

    true_Rsh,
    pred_Rsh,

    alpha=0.3
)

plt.plot(

    [true_Rsh.min(), true_Rsh.max()],

    [true_Rsh.min(), true_Rsh.max()],

    'r--'
)

plt.xlabel("True Rsh", fontsize=14, fontweight='bold')
plt.ylabel("Predicted Rsh", fontsize=14, fontweight='bold')

plt.title(
    "Rsh Prediction",
    fontsize=15,
    fontweight='bold'
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Rsh_prediction.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\n===================================")
print("Evaluation complete.")
print("===================================")