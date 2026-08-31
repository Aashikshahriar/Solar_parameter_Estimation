"""Single-diode PV model: forward simulation of I-V curves for synthetic dataset generation.

Given (Rs, Rsh, G, T), solves the implicit single-diode equation for current at each
voltage point via Newton-Raphson, then adds measurement noise to emulate real sensors.
"""

import numpy as np

from .config import PHYSICS, DatasetConfig


def solve_current(V_point: float, Iph: float, I0: float, Rs: float, Rsh: float, T: float,
                   iters: int = 50) -> float:
    """Newton-Raphson solve of the implicit single-diode I-V equation at one operating point."""
    q, k, n, Ns = PHYSICS.q, PHYSICS.k, PHYSICS.n, PHYSICS.Ns
    I = Iph  # initial guess

    for _ in range(iters):
        exp_term = np.exp((q * (V_point + I * Rs)) / (n * Ns * k * T))
        f = I - Iph + I0 * (exp_term - 1) + (V_point + I * Rs) / Rsh
        df = 1 + I0 * exp_term * (q * Rs / (n * Ns * k * T)) + Rs / Rsh
        I = I - f / df

    return I


def generate_sample(cfg: DatasetConfig = DatasetConfig(), rng: np.random.Generator = None):
    """Simulate one randomized I-V curve with associated ground-truth Rs, Rsh, G, T."""
    rng = rng or np.random.default_rng()

    G = rng.uniform(*cfg.g_range)
    T = rng.uniform(*cfg.t_range)

    Rs = rng.uniform(*cfg.rs_range)
    Rsh = rng.uniform(*cfg.rsh_range)

    Iph = PHYSICS.Isc_ref * (G / PHYSICS.G_ref) * (1 + PHYSICS.alpha_Isc * (T - PHYSICS.T_ref))
    I0 = 1e-10 * (T / PHYSICS.T_ref) ** 3
    Voc = (PHYSICS.n * PHYSICS.Ns * PHYSICS.k * T / PHYSICS.q) * np.log(Iph / I0 + 1)

    V = np.linspace(0, 1.1 * Voc, cfg.n_points)
    I = np.array([solve_current(v, Iph, I0, Rs, Rsh, T) for v in V])

    noise = rng.normal(0, cfg.noise_level * PHYSICS.Isc_ref, size=I.shape)
    I_noisy = I + noise

    return V, I_noisy, Rs, Rsh, G, T


def generate_dataset(cfg: DatasetConfig = DatasetConfig(), verbose: bool = True):
    """Simulate `cfg.n_samples` I-V curves and stack them into arrays."""
    rng = np.random.default_rng(cfg.seed)

    V_all, I_all, Rs_all, Rsh_all, G_all, T_all = [], [], [], [], [], []

    for i in range(cfg.n_samples):
        V, I, Rs, Rsh, G, T = generate_sample(cfg, rng)
        V_all.append(V)
        I_all.append(I)
        Rs_all.append(Rs)
        Rsh_all.append(Rsh)
        G_all.append(G)
        T_all.append(T)

        if verbose and i % 2000 == 0:
            print(f"Generated {i}/{cfg.n_samples}")

    return (
        np.array(V_all), np.array(I_all),
        np.array(Rs_all), np.array(Rsh_all),
        np.array(G_all), np.array(T_all),
    )


def save_dataset(path: str, cfg: DatasetConfig = DatasetConfig(), verbose: bool = True):
    """Generate and persist a synthetic dataset to a compressed .npz file."""
    V, I, Rs, Rsh, G, T = generate_dataset(cfg, verbose=verbose)
    np.savez(path, V=V, I=I, Rs=Rs, Rsh=Rsh, G=G, T=T)
    if verbose:
        print(f"Dataset saved to: {path}")
    return path
