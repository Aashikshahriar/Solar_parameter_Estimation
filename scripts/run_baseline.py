"""CLI: run the classical least-squares baseline on the same test split used for the model.

Usage:
    python scripts/run_baseline.py --data data/pv_dataset.npz --num-samples 50
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from solar_pe.baseline import evaluate_baseline
from solar_pe.config import DatasetConfig
from solar_pe.dataset import build_targets, load_raw, split_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pv_dataset.npz")
    parser.add_argument("--num-samples", type=int, default=50, help="curve-fitting is slow; evaluate on a subset")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.data)
    _, _, test_idx = split_indices(len(raw["V"]), DatasetConfig())
    test_idx = test_idx[: args.num_samples]

    X_samples = [np.stack([raw["V"][i], raw["I"][i]]) for i in test_idx]
    env_samples = [(raw["G"][i], raw["T"][i]) for i in test_idx]
    y_samples = build_targets(raw["Rs"][test_idx], raw["Rsh"][test_idx])

    results = evaluate_baseline(X_samples, env_samples, y_samples)
    print(json.dumps(results, indent=2))

    with open(results_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
