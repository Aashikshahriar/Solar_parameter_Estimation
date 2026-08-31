"""Plotting helpers for I-V curves, training curves, and prediction quality."""

import matplotlib.pyplot as plt
import numpy as np


def plot_iv_curves(V: np.ndarray, I: np.ndarray, indices, labels=None, save_path: str = None):
    plt.figure(figsize=(8, 6))
    for i, idx in enumerate(indices):
        label = labels[i] if labels else f"Sample {idx}"
        plt.plot(V[idx], I[idx], label=label)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("I-V Curves")
    plt.legend()
    plt.grid(True)
    _save_or_show(save_path)


def plot_loss_curve(history: dict, save_path: str = None):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    _save_or_show(save_path)


def plot_prediction_scatter(true: np.ndarray, pred: np.ndarray, name: str, save_path: str = None):
    plt.figure(figsize=(5, 5))
    plt.scatter(true, pred, alpha=0.3)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"{name} Prediction")
    _save_or_show(save_path)


def _save_or_show(save_path: str = None):
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
