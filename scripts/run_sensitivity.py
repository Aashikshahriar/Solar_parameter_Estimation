"""CLI: gradient-based sensitivity analysis of a trained model.

Usage:
    python scripts/run_sensitivity.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from solar_pe.config import DatasetConfig
from solar_pe.dataset import build_dataloaders
from solar_pe.model import AttentionPV
from solar_pe.sensitivity import compute_gradient_sensitivity, region_sensitivity
from solar_pe.visualize import plot_region_bars, plot_sensitivity_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pv_dataset.npz")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--results-dir", default="results/sensitivity")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = build_dataloaders(args.data, DatasetConfig())

    model = AttentionPV().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    sens_Rs, sens_Rsh = compute_gradient_sensitivity(model, test_loader, device, num_samples=args.num_samples)
    plot_sensitivity_map(sens_Rs, sens_Rsh, save_path=str(results_dir / "sensitivity_map.png"))

    df_region = region_sensitivity(sens_Rs, sens_Rsh)
    print(df_region)
    df_region.to_csv(results_dir / "region_sensitivity.csv", index=False)
    plot_region_bars(df_region, ("Rs", "Rsh"), "Region", "Region-wise Sensitivity",
                      save_path=str(results_dir / "region_sensitivity.png"))


if __name__ == "__main__":
    main()
