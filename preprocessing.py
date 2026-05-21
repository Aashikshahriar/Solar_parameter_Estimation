"""
============================================================
preprocessing.py
Photovoltaic Dataset Preprocessing Pipeline
============================================================

This script performs preprocessing for the synthetic
photovoltaic (PV) dataset generated using the SDM.

Preprocessing includes:
    - Voltage normalization
    - Current normalization
    - Gradient feature extraction
    - Environmental normalization
    - Log transformation of targets
    - Dataset splitting
    - PyTorch dataset preparation

Generated outputs:
    - train_loader
    - val_loader
    - test_loader

Author: <K. A. Shahriar>
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    DataLoader
)

# ============================================================
# RANDOM SEED
# ============================================================

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

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(filename="pv_dataset.npz"):

    """
    Loads generated PV dataset.
    """

    data = np.load(filename)

    V = data["V"]
    I = data["I"]

    Rs = data["Rs"]
    Rsh = data["Rsh"]

    G = data["G"]
    T = data["T"]

    print("\n===================================")
    print("Dataset Loaded")
    print("===================================")

    print("Voltage shape :", V.shape)
    print("Current shape :", I.shape)

    return V, I, Rs, Rsh, G, T

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def preprocess_features(

    V,
    I,
    Rs,
    Rsh,
    G,
    T

):

    """
    Performs feature normalization and
    preprocessing.

    Returns
    -------
    X   : input tensor
    env : environmental tensor
    y   : regression targets
    """

    # --------------------------------------------------------
    # Voltage normalization
    # --------------------------------------------------------

    Voc = np.max(
        V,
        axis=1,
        keepdims=True
    )

    Vn = V / Voc

    # --------------------------------------------------------
    # Current normalization
    # --------------------------------------------------------

    Isc = np.max(
        I,
        axis=1,
        keepdims=True
    )

    In = I / Isc

    # --------------------------------------------------------
    # Current gradient feature
    # --------------------------------------------------------

    dI = np.gradient(
        In,
        axis=1
    )

    # --------------------------------------------------------
    # Environmental normalization
    # --------------------------------------------------------

    G_norm = (

        (G - G.mean())
        / G.std()

    )

    T_norm = (

        (T - T.mean())
        / T.std()

    )

    # --------------------------------------------------------
    # Final input tensor
    # Channels:
    #   1. normalized voltage
    #   2. normalized current
    #   3. current gradient
    # --------------------------------------------------------

    X = np.stack(

        [Vn, In, dI],

        axis=1

    )

    # --------------------------------------------------------
    # Environmental input
    # --------------------------------------------------------

    env = np.stack(

        [G_norm, T_norm],

        axis=1

    )

    # --------------------------------------------------------
    # Log-transform targets
    # --------------------------------------------------------

    y = np.stack(

        [

            np.log(Rs),
            np.log(Rsh)

        ],

        axis=1

    )

    print("\n===================================")
    print("Preprocessing Complete")
    print("===================================")

    print("Input shape :", X.shape)
    print("Env shape   :", env.shape)
    print("Target shape:", y.shape)

    return X, env, y

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

def split_dataset(

    X,
    env,
    y,

    train_ratio=0.70,
    val_ratio=0.15

):

    """
    Splits dataset into:
        train
        validation
        test
    """

    N = len(X)

    idx = np.random.permutation(N)

    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)

    train_idx = idx[:train_end]

    val_idx = idx[train_end:val_end]

    test_idx = idx[val_end:]

    print("\n===================================")
    print("Dataset Split")
    print("===================================")

    print("Train :", len(train_idx))
    print("Val   :", len(val_idx))
    print("Test  :", len(test_idx))

    return (

        train_idx,
        val_idx,
        test_idx

    )

# ============================================================
# PYTORCH DATASET
# ============================================================

class IVDataset(Dataset):

    """
    PyTorch Dataset for PV learning.
    """

    def __init__(

        self,
        X,
        env,
        y

    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.env = torch.tensor(
            env,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (

            self.X[idx],
            self.env[idx],
            self.y[idx]

        )

# ============================================================
# DATALOADER CREATION
# ============================================================

def create_dataloaders(

    X,
    env,
    y,

    train_idx,
    val_idx,
    test_idx,

    batch_size=64

):

    """
    Creates PyTorch dataloaders.
    """

    train_loader = DataLoader(

        IVDataset(

            X[train_idx],
            env[train_idx],
            y[train_idx]

        ),

        batch_size=batch_size,

        shuffle=True
    )

    val_loader = DataLoader(

        IVDataset(

            X[val_idx],
            env[val_idx],
            y[val_idx]

        ),

        batch_size=batch_size,

        shuffle=False
    )

    test_loader = DataLoader(

        IVDataset(

            X[test_idx],
            env[test_idx],
            y[test_idx]

        ),

        batch_size=batch_size,

        shuffle=False
    )

    print("\n===================================")
    print("DataLoaders Created")
    print("===================================")

    return (

        train_loader,
        val_loader,
        test_loader

    )

# ============================================================
# COMPLETE PIPELINE
# ============================================================

def build_preprocessing_pipeline(

    filename="pv_dataset.npz",
    batch_size=64

):

    """
    Complete preprocessing pipeline.
    """

    # load dataset
    V, I, Rs, Rsh, G, T = load_dataset(
        filename
    )

    # preprocess
    X, env, y = preprocess_features(

        V,
        I,

        Rs,
        Rsh,

        G,
        T
    )

    # split
    train_idx, val_idx, test_idx = split_dataset(

        X,
        env,
        y
    )

    # dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(

        X,
        env,
        y,

        train_idx,
        val_idx,
        test_idx,

        batch_size=batch_size
    )

    return (

        train_loader,
        val_loader,
        test_loader
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = (

        build_preprocessing_pipeline(

            filename="pv_dataset.npz",

            batch_size=64
        )
    )

    print("\n===================================")
    print("Preprocessing Pipeline Complete")
    print("===================================")