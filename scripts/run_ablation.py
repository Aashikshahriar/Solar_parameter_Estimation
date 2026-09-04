"""CLI: ablation study over region pooling / attention / environmental conditioning.

Usage:
    python scripts/run_ablation.py --data data/pv_dataset.npz --epochs 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from solar_pe.ablation import run_ablation_study
from solar_pe.config import DatasetConfig
from solar_pe.dataset import build_dataloaders
from solar_pe.visualize import plot_ablation_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pv_dataset.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--results-dir", default="results/ablation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _, test_loader = build_dataloaders(args.data, DatasetConfig())

    df_ablation = run_ablation_study(train_loader, test_loader, device, epochs=args.epochs)
    print(df_ablation)

    df_ablation.to_csv(results_dir / "ablation_results.csv", index=False)
    plot_ablation_results(df_ablation, save_path=str(results_dir / "ablation_plot.png"))


if __name__ == "__main__":
    main()
