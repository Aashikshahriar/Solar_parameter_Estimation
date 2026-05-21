"""
============================================================
hyperparameter_tuning.py
Randomized Hyperparameter Search for
Region-Aware PV Attention Framework
============================================================

This script performs randomized hyperparameter
optimization for photovoltaic parameter estimation.

Optimized parameters:
    - learning rate
    - weight decay
    - loss weights
    - noise augmentation

Author: <K. A. Shahriar>
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import AttentionPV
from preprocessing import (
    build_preprocessing_pipeline
)

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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
# TRAIN ONE CONFIGURATION
# ============================================================

def train_one_run(config):

    """
    Trains model for one hyperparameter
    configuration.

    Parameters
    ----------
    config : dict
        Hyperparameter configuration

    Returns
    -------
    best_val : float
        Best validation loss
    """

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = AttentionPV().to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.Adam(

        model.parameters(),

        lr=config["lr"],

        weight_decay=config["wd"]
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=30
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    mse = nn.MSELoss()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val = float("inf")

    for epoch in range(30):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_loss = 0

        for Xb, envb, yb in train_loader:

            Xb = Xb.to(device)
            envb = envb.to(device)
            yb = yb.to(device)

            # -----------------------------------------------
            # Noise augmentation
            # -----------------------------------------------

            Xb[:,1,:] += (

                torch.randn_like(Xb[:,1,:])

                * config["noise"]
            )

            # -----------------------------------------------
            # Forward
            # -----------------------------------------------

            optimizer.zero_grad()

            pred = model(Xb, envb)

            # -----------------------------------------------
            # Multi-task loss
            # -----------------------------------------------

            loss_rs = mse(

                pred[:,0],
                yb[:,0]
            )

            loss_rsh = mse(

                pred[:,1],
                yb[:,1]
            )

            loss = (

                config["w_rs"] * loss_rs

                +

                config["w_rsh"] * loss_rsh
            )

            # -----------------------------------------------
            # Backprop
            # -----------------------------------------------

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # ====================================================
        # VALIDATION
        # ====================================================

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

                    loss_rs.item()
                    + loss_rsh.item()
                )

        # normalize
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(

            f"Epoch {epoch+1:02d} | "

            f"Train {train_loss:.4f} | "

            f"Val {val_loss:.4f}"

        )

        # save best
        if val_loss < best_val:

            best_val = val_loss

    return best_val

# ============================================================
# SEARCH SPACE
# ============================================================

search_space = {

    # learning rate
    "lr": [

        1e-3,
        5e-4,
        1e-4
    ],

    # weight decay
    "wd": [

        1e-5,
        1e-6
    ],

    # Rs loss weight
    "w_rs": [

        1.5,
        2.0,
        3.0
    ],

    # Rsh loss weight
    "w_rsh": [

        2.0,
        3.0,
        4.0
    ],

    # noise augmentation
    "noise": [

        0.005,
        0.01,
        0.02
    ]
}

# ============================================================
# RANDOM SEARCH
# ============================================================

num_trials = 10

results = []

print("\n===================================")
print("STARTING RANDOM SEARCH")
print("===================================")

for trial in range(num_trials):

    # --------------------------------------------------------
    # Random configuration
    # --------------------------------------------------------

    config = {

        k: random.choice(v)

        for k, v in search_space.items()
    }

    print("\n===================================")
    print(f"TRIAL {trial+1}/{num_trials}")
    print("===================================")

    print(config)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    val_loss = train_one_run(config)

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append({

        "config": copy.deepcopy(config),

        "val_loss": val_loss
    })

# ============================================================
# SORT RESULTS
# ============================================================

results = sorted(

    results,

    key=lambda x: x["val_loss"]
)

# ============================================================
# BEST CONFIGURATION
# ============================================================

best = results[0]

print("\n===================================")
print("BEST CONFIGURATION")
print("===================================")

print("Validation Loss:")
print(best["val_loss"])

print("\nHyperparameters:")

for k, v in best["config"].items():

    print(f"{k}: {v}")

# ============================================================
# SAVE RESULTS
# ============================================================

with open(

    "hyperparameter_results.txt",

    "w"

) as f:

    f.write(

        "BEST HYPERPARAMETER CONFIGURATION\n\n"
    )

    f.write(

        f"Validation Loss: {best['val_loss']}\n\n"
    )

    for k, v in best["config"].items():

        f.write(f"{k}: {v}\n")

print("\n===================================")
print("Results saved:")
print("hyperparameter_results.txt")
print("===================================")