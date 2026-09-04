"""CLI: environmental (irradiance / temperature) perturbation analysis on a trained model.

Usage:
    python scripts/run_environmental_perturbation.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from solar_pe.config import DatasetConfig
from solar_pe.dataset import build_dataloaders
from solar_pe.environmental_perturbation import run_env_perturbation_grid, run_env_perturbation_sweep
from solar_pe.model import AttentionPV
from solar_pe.visualize import plot_env_perturbation_heatmap, plot_env_perturbation_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pv_dataset.npz")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--results-dir", default="results/environmental_perturbation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = build_dataloaders(args.data, DatasetConfig())

    model = AttentionPV().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    df_sweep = run_env_perturbation_sweep(model, test_loader, device)
    print(df_sweep)
    df_sweep.to_csv(results_dir / "env_perturbation_sweep.csv", index=False)
    plot_env_perturbation_sweep(df_sweep, save_path=str(results_dir / "env_perturbation_sweep.png"))

    df_grid = run_env_perturbation_grid(model, test_loader, device)
    df_grid.to_csv(results_dir / "env_perturbation_grid.csv", index=False)
    plot_env_perturbation_heatmap(df_grid, save_path=str(results_dir / "env_perturbation_heatmap.png"))


if __name__ == "__main__":
    main()
