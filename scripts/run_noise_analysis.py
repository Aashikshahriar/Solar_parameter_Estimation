"""CLI: measurement noise robustness sweep on a trained model.

Usage:
    python scripts/run_noise_analysis.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from solar_pe.config import DatasetConfig
from solar_pe.dataset import build_dataloaders
from solar_pe.model import AttentionPV
from solar_pe.noise_analysis import run_noise_sweep
from solar_pe.visualize import plot_noise_robustness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pv_dataset.npz")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--results-dir", default="results/noise_analysis")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = build_dataloaders(args.data, DatasetConfig())

    model = AttentionPV().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    df_noise = run_noise_sweep(model, test_loader, device)
    print(df_noise)

    df_noise.to_csv(results_dir / "noise_results.csv", index=False)
    plot_noise_robustness(df_noise, save_path=str(results_dir / "noise_robustness.png"))


if __name__ == "__main__":
    main()
