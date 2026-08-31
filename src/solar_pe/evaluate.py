"""Inference and metric computation on held-out data."""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def relative_error_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true) / (y_true + 1e-8)) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "RE%": relative_error_pct(y_true, y_pred),
    }


@torch.no_grad()
def predict(model, loader, device) -> dict:
    """Run inference over a loader and undo the log-transform on Rs/Rsh."""
    model.eval()
    pred_Rs, true_Rs, pred_Rsh, true_Rsh = [], [], [], []

    for Xb, envb, yb in loader:
        Xb, envb = Xb.to(device), envb.to(device)
        pred = model(Xb, envb).cpu()

        pred_Rs.append(torch.exp(pred[:, 0]).numpy())
        pred_Rsh.append(torch.exp(pred[:, 1]).numpy())
        true_Rs.append(torch.exp(yb[:, 0]).numpy())
        true_Rsh.append(torch.exp(yb[:, 1]).numpy())

    return {
        "pred_Rs": np.concatenate(pred_Rs),
        "true_Rs": np.concatenate(true_Rs),
        "pred_Rsh": np.concatenate(pred_Rsh),
        "true_Rsh": np.concatenate(true_Rsh),
    }


def evaluate(model, loader, device) -> dict:
    """Full evaluation: predictions plus Rs/Rsh metrics."""
    preds = predict(model, loader, device)
    return {
        "Rs": compute_metrics(preds["true_Rs"], preds["pred_Rs"]),
        "Rsh": compute_metrics(preds["true_Rsh"], preds["pred_Rsh"]),
        "predictions": preds,
    }
