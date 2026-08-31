"""Central configuration: physical constants, dataset defaults, and training hyperparameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConstants:
    q: float = 1.602e-19       # electron charge (C)
    k: float = 1.381e-23       # Boltzmann constant (J/K)
    G_ref: float = 1000.0      # reference irradiance (W/m^2)
    T_ref: float = 298.0       # reference temperature (K)
    Isc_ref: float = 9.0       # reference short-circuit current (A)
    alpha_Isc: float = 0.0005  # short-circuit current temperature coefficient
    n: float = 1.2             # diode ideality factor
    Ns: int = 60                # number of series-connected cells


@dataclass(frozen=True)
class DatasetConfig:
    n_samples: int = 20000
    n_points: int = 200        # voltage samples per I-V curve
    noise_level: float = 0.02  # relative current noise (fraction of Isc_ref)
    rs_range: tuple = (0.1, 1.0)
    rsh_range: tuple = (50.0, 500.0)
    g_range: tuple = (200.0, 1200.0)
    t_range: tuple = (283.0, 343.0)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # remaining fraction goes to test
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 100
    patience: int = 15
    w_rs: float = 2.0          # loss weight for series resistance
    w_rsh: float = 2.0         # loss weight for shunt resistance
    train_noise_std: float = 0.01  # augmentation noise added to current channel during training


PHYSICS = PhysicsConstants()
DATASET = DatasetConfig()
TRAIN = TrainConfig()
