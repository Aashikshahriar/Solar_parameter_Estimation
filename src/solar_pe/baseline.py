"""Classical (non-learned) baseline: per-curve Rs/Rsh estimation via nonlinear least-squares.

Used as a reference point for how much the learned model improves over a standard
curve-fitting approach to the same inverse problem.
"""

import time

import numpy as np
from scipy.optimize import least_squares

from .evaluate import relative_error_pct
from sklearn.metrics import r2_score


def pv_model(V: np.ndarray, Rs: float, Rsh: float, G: float, T: float,
             I0: float = 1e-10, n: float = 1.3) -> np.ndarray:
    """Simplified single-diode approximation used as the curve-fitting objective."""
    Iph = 5 * (G / 1000)
    Vt = 0.0259 * (T / 25)
    exp_term = np.exp(np.clip(V / (n * Vt), -50, 50))
    return Iph - I0 * (exp_term - 1) - V / (Rsh + 1e-6) - Rs * 0.05


def _residuals(params, V, I_true, G, T):
    Rs, Rsh = params
    return pv_model(V, Rs, Rsh, G, T) - I_true


def fit_curve(V: np.ndarray, I: np.ndarray, G: float, T: float,
              x0=(0.5, 1000.0), bounds=((0.001, 10.0), (5.0, 10000.0))):
    """Fit (Rs, Rsh) to a single I-V curve via bounded nonlinear least-squares."""
    lb = [bounds[0][0], bounds[1][0]]
    ub = [bounds[0][1], bounds[1][1]]

    try:
        result = least_squares(_residuals, x0=list(x0), bounds=(lb, ub), args=(V, I, G, T), method="trf")
        return float(result.x[0]), float(result.x[1])
    except Exception:
        return x0


def evaluate_baseline(X_samples, env_samples, y_samples) -> dict:
    """Run the least-squares baseline over a list of (X, env, y) samples and score it.

    X_samples[i] is a (channels, n_points) array with V in channel 0 and I in channel 1
    (matches `solar_pe.dataset.build_features` layout). y is log(Rs), log(Rsh).
    env_samples[i] must be the *raw* (G, T) in physical units (W/m^2, K) -- not the
    standardized env features used by the neural model, since `pv_model` expects real values.
    """
    pred_Rs, true_Rs, pred_Rsh, true_Rsh, runtimes = [], [], [], [], []

    for X, env, y in zip(X_samples, env_samples, y_samples):
        V, I = X[0], X[1]
        G, T = env[0], env[1]

        start = time.time()
        Rs_est, Rsh_est = fit_curve(V, I, G, T)
        runtimes.append(time.time() - start)

        pred_Rs.append(Rs_est)
        pred_Rsh.append(Rsh_est)
        true_Rs.append(float(np.exp(y[0])))
        true_Rsh.append(float(np.exp(y[1])))

    return {
        "Model": "Newton-Raphson (least-squares)",
        "Rs_RE%": relative_error_pct(np.array(true_Rs), np.array(pred_Rs)),
        "Rsh_RE%": relative_error_pct(np.array(true_Rsh), np.array(pred_Rsh)),
        "Rs_R2": float(r2_score(true_Rs, pred_Rs)),
        "Rsh_R2": float(r2_score(true_Rsh, pred_Rsh)),
        "runtime_per_sample_s": float(np.mean(runtimes)),
    }
