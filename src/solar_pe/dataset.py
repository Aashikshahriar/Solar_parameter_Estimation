"""Feature engineering, train/val/test splitting, and PyTorch Dataset/DataLoader construction."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig


def load_raw(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def build_features(V: np.ndarray, I: np.ndarray, G: np.ndarray, T: np.ndarray):
    """Turn raw I-V curves + environmental readings into model-ready tensors.

    X: (N, 3, n_points) -> normalized voltage, normalized current, current gradient
    env: (N, 2) -> standardized irradiance, standardized temperature
    """
    Voc = np.max(V, axis=1, keepdims=True)
    Isc = np.max(I, axis=1, keepdims=True)

    Vn = V / Voc
    In = I / Isc
    dI = np.gradient(In, axis=1)

    X = np.stack([Vn, In, dI], axis=1)
    env = np.stack([
        (G - G.mean()) / G.std(),
        (T - T.mean()) / T.std(),
    ], axis=1)

    return X, env


def build_targets(Rs: np.ndarray, Rsh: np.ndarray) -> np.ndarray:
    """Log-transform resistances: they span orders of magnitude and are strictly positive."""
    return np.stack([np.log(Rs), np.log(Rsh)], axis=1)


def split_indices(n: int, cfg: DatasetConfig = DatasetConfig()):
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(n)

    n_train = int(cfg.train_frac * n)
    n_val = int((cfg.train_frac + cfg.val_frac) * n)

    return idx[:n_train], idx[n_train:n_val], idx[n_val:]


class IVCurveDataset(Dataset):
    def __init__(self, X: np.ndarray, env: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.env = torch.tensor(env, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.env[i], self.y[i]


def build_dataloaders(raw_path: str, cfg: DatasetConfig = DatasetConfig(), batch_size: int = 64):
    """Load a raw .npz dataset and return train/val/test DataLoaders."""
    raw = load_raw(raw_path)

    X, env = build_features(raw["V"], raw["I"], raw["G"], raw["T"])
    y = build_targets(raw["Rs"], raw["Rsh"])

    train_idx, val_idx, test_idx = split_indices(len(X), cfg)

    train_loader = DataLoader(
        IVCurveDataset(X[train_idx], env[train_idx], y[train_idx]),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        IVCurveDataset(X[val_idx], env[val_idx], y[val_idx]),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        IVCurveDataset(X[test_idx], env[test_idx], y[test_idx]),
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader
