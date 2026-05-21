"""
============================================================
data_generation.py
Synthetic Photovoltaic Dataset Generation using SDM
============================================================

This script generates synthetic photovoltaic (PV)
current-voltage (I-V) characteristics using the
Single-Diode Model (SDM) with Newton-Raphson solving.

Generated dataset includes:
    - Voltage curves
    - Current curves
    - Series resistance (Rs)
    - Shunt resistance (Rsh)
    - Irradiance (G)
    - Temperature (T)

The dataset is saved as:
    pv_dataset.npz

Author: <K. A. Shahriar>
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np

# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(42)

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

q = 1.602e-19        # Electron charge (C)
k = 1.381e-23        # Boltzmann constant (J/K)

# ============================================================
# REFERENCE CONDITIONS
# ============================================================

G_ref = 1000         # Reference irradiance (W/m^2)
T_ref = 298          # Reference temperature (K)

Isc_ref = 9.0        # Reference short-circuit current (A)
alpha_Isc = 0.0005   # Temperature coefficient

# ============================================================
# PV MODULE PARAMETERS
# ============================================================

n = 1.2              # Diode ideality factor
Ns = 60              # Number of series-connected cells

# ============================================================
# NEWTON-RAPHSON CURRENT SOLVER
# ============================================================

def solve_current(V_point, Iph, I0, Rs, Rsh, T):

    """
    Solves the implicit SDM current equation
    using Newton-Raphson iteration.

    Parameters
    ----------
    V_point : float
        Voltage sample

    Iph : float
        Photocurrent

    I0 : float
        Saturation current

    Rs : float
        Series resistance

    Rsh : float
        Shunt resistance

    T : float
        Operating temperature

    Returns
    -------
    I : float
        Estimated current
    """

    # initial guess
    I = Iph

    for _ in range(50):

        exp_term = np.exp(

            (q * (V_point + I * Rs)) /
            (n * Ns * k * T)

        )

        # SDM equation
        f = (

            I
            - Iph
            + I0 * (exp_term - 1)
            + (V_point + I * Rs) / Rsh

        )

        # derivative
        df = (

            1
            + I0 * exp_term *
            (q * Rs / (n * Ns * k * T))
            + Rs / Rsh

        )

        # Newton update
        I = I - f / df

    return I

# ============================================================
# GENERATE SINGLE SAMPLE
# ============================================================

def generate_sample(noise_level=0.02): #change the default noise level here

    """
    Generates one synthetic PV sample.

    Parameters
    ----------
    noise_level : float
        Gaussian noise level

    Returns
    -------
    V : ndarray
        Voltage curve

    I_noisy : ndarray
        Noisy current curve

    Rs : float
        Series resistance

    Rsh : float
        Shunt resistance

    G : float
        Irradiance

    T : float
        Temperature
    """

    # --------------------------------------------------------
    # Random operating conditions # can be tuned to specific ranges if desired
    # --------------------------------------------------------

    G = np.random.uniform(200, 1200)
    T = np.random.uniform(283, 343)

    # --------------------------------------------------------
    # Random degradation parameters
    # --------------------------------------------------------

    Rs = np.random.uniform(0.1, 1.0)
    Rsh = np.random.uniform(50, 500)

    # --------------------------------------------------------
    # Photocurrent
    # --------------------------------------------------------

    Iph = (

        Isc_ref
        * (G / G_ref)
        * (1 + alpha_Isc * (T - T_ref))

    )

    # --------------------------------------------------------
    # Saturation current
    # --------------------------------------------------------

    I0 = 1e-10 * (T / T_ref)**3

    # --------------------------------------------------------
    # Open-circuit voltage
    # --------------------------------------------------------

    Voc = (

        (n * Ns * k * T / q)
        * np.log(Iph / I0 + 1)

    )

    # --------------------------------------------------------
    # Voltage sweep
    # --------------------------------------------------------

    V = np.linspace(

        0,
        1.1 * Voc,
        200

    )

    # --------------------------------------------------------
    # Solve current
    # --------------------------------------------------------

    I = np.array([

        solve_current(
            v,
            Iph,
            I0,
            Rs,
            Rsh,
            T
        )

        for v in V

    ])

    # --------------------------------------------------------
    # Add Gaussian noise
    # --------------------------------------------------------

    noise = np.random.normal(

        0,
        noise_level * Isc_ref,
        size=I.shape

    )

    I_noisy = I + noise

    return V, I_noisy, Rs, Rsh, G, T

# ============================================================
# GENERATE COMPLETE DATASET
# ============================================================

def generate_dataset(
    N=20000,
    noise_level=0.02
):

    """
    Generates the complete PV dataset.

    Parameters
    ----------
    N : int
        Number of samples

    noise_level : float
        Gaussian noise intensity

    Returns
    -------
    Tuple of arrays
    """

    V_all = []
    I_all = []

    Rs_all = []
    Rsh_all = []

    G_all = []
    T_all = []

    for i in range(N):

        V, I, Rs, Rsh, G, T = generate_sample(
            noise_level
        )

        V_all.append(V)
        I_all.append(I)

        Rs_all.append(Rs)
        Rsh_all.append(Rsh)

        G_all.append(G)
        T_all.append(T)

        if i % 500 == 0:

            print(
                f"Generated {i}/{N}"
            )

    return (

        np.array(V_all),
        np.array(I_all),

        np.array(Rs_all),
        np.array(Rsh_all),

        np.array(G_all),
        np.array(T_all)

    )

# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(

    filename="pv_dataset.npz",
    N=20000,
    noise_level=0.02

):

    """
    Saves generated dataset.
    """

    V, I, Rs, Rsh, G, T = generate_dataset(

        N=N,
        noise_level=noise_level

    )

    np.savez(

        filename,

        V=V,
        I=I,

        Rs=Rs,
        Rsh=Rsh,

        G=G,
        T=T

    )

    print("\n===================================")
    print(f"Dataset saved to:")
    print(os.path.abspath(filename))
    print("===================================")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    save_dataset(

        filename="pv_dataset.npz",

        N=20000,

        noise_level=0.02

    )