"""CLI: generate a synthetic PV I-V curve dataset via the single-diode model.

Usage:
    python scripts/generate_dataset.py --out data/pv_dataset.npz --n 20000 --noise 0.02
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_pe.config import DatasetConfig
from solar_pe.physics import save_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/pv_dataset.npz", help="output .npz path")
    parser.add_argument("--n", type=int, default=DatasetConfig.n_samples, help="number of curves")
    parser.add_argument("--noise", type=float, default=DatasetConfig.noise_level, help="relative current noise")
    parser.add_argument("--seed", type=int, default=DatasetConfig.seed)
    args = parser.parse_args()

    cfg = DatasetConfig(n_samples=args.n, noise_level=args.noise, seed=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_dataset(args.out, cfg)


if __name__ == "__main__":
    main()
