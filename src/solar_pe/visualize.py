"""Plotting helpers for I-V curves, training curves, prediction quality, and the
interpretability / ablation / robustness studies in `solar_pe`."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_sensitivity_map(sens_Rs: np.ndarray, sens_Rsh: np.ndarray, save_path: str = None):
    plt.figure(figsize=(6, 4))
    plt.plot(sens_Rs, label="Rs sensitivity")
    plt.plot(sens_Rsh, label="Rsh sensitivity")
    plt.xlabel("Voltage index")
    plt.ylabel("Relative importance")
    plt.title("Gradient Sensitivity Map")
    plt.legend()
    plt.grid(True)
    _save_or_show(save_path)


def plot_region_bars(df: pd.DataFrame, value_cols: tuple, category_col: str, title: str, save_path: str = None):
    """Grouped bar chart, e.g. region_sensitivity() output or ablation_results per model."""
    plt.figure(figsize=(6, 4))
    x = np.arange(len(df))
    width = 0.35

    plt.bar(x - width / 2, df[value_cols[0]], width, label=value_cols[0])
    plt.bar(x + width / 2, df[value_cols[1]], width, label=value_cols[1])
    plt.xticks(x, df[category_col], rotation=20)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    _save_or_show(save_path)


def plot_attention_map(attn: np.ndarray, save_path: str = None):
    labels = ["Left", "Mid", "Right"]
    plt.figure(figsize=(4, 4))
    im = plt.imshow(attn, cmap="viridis")
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    plt.title("Region Cross-Attention")
    plt.colorbar(im)
    _save_or_show(save_path)


def plot_noise_robustness(df_noise: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df_noise["Noise"], df_noise["Rs_RE%"], marker="o")
    axes[0].set_title("Rs Robustness")
    axes[0].set_xlabel("Noise level")
    axes[0].set_ylabel("Relative error (%)")
    axes[0].grid(True)

    axes[1].plot(df_noise["Noise"], df_noise["Rsh_RE%"], marker="o")
    axes[1].set_title("Rsh Robustness")
    axes[1].set_xlabel("Noise level")
    axes[1].set_ylabel("Relative error (%)")
    axes[1].grid(True)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_env_perturbation_sweep(df_env: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = df_env["Perturbation (%)"]

    axes[0].plot(x, df_env["G_Rs_RE%"], marker="o", label="Irradiance (G)")
    axes[0].plot(x, df_env["T_Rs_RE%"], marker="o", label="Temperature (T)")
    axes[0].set_title("Rs vs Environmental Perturbation")
    axes[0].set_xlabel("Perturbation (%)")
    axes[0].set_ylabel("Relative error (%)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x, df_env["G_Rsh_RE%"], marker="o", label="Irradiance (G)")
    axes[1].plot(x, df_env["T_Rsh_RE%"], marker="o", label="Temperature (T)")
    axes[1].set_title("Rsh vs Environmental Perturbation")
    axes[1].set_xlabel("Perturbation (%)")
    axes[1].set_ylabel("Relative error (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_env_perturbation_heatmap(df_grid: pd.DataFrame, save_path: str = None):
    pivot_rs = df_grid.pivot(index="G_noise", columns="T_noise", values="Rs_RE%")
    pivot_rsh = df_grid.pivot(index="G_noise", columns="T_noise", values="Rsh_RE%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pivot, title in [(axes[0], pivot_rs, "Rs Relative Error"), (axes[1], pivot_rsh, "Rsh Relative Error")]:
        im = ax.imshow(pivot.values)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Temperature perturbation")
        ax.set_ylabel("Irradiance perturbation")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_ablation_results(df_ablation: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(6, 4))
    x = np.arange(len(df_ablation))
    plt.plot(x, df_ablation["Rs_RE%"], marker="o", label="Rs")
    plt.plot(x, df_ablation["Rsh_RE%"], marker="o", label="Rsh")
    plt.xticks(x, df_ablation["Model"], rotation=20)
    plt.ylabel("Relative error (%)")
    plt.title("Ablation Study")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _save_or_show(save_path)


def _save_or_show(save_path: str = None):
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
