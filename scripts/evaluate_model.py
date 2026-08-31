"""CLI: evaluate a trained checkpoint on the held-out test split.

Usage:
    python scripts/evaluate_model.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from solar_pe.config import DatasetConfig
from solar_pe.dataset import build_dataloaders
from solar_pe.evaluate import evaluate
from solar_pe.model import AttentionPV
from solar_pe.visualize import plot_prediction_scatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pv_dataset.npz")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = build_dataloaders(args.data, DatasetConfig())

    model = AttentionPV().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    outcome = evaluate(model, test_loader, device)
    metrics = {"Rs": outcome["Rs"], "Rsh": outcome["Rsh"]}

    print(json.dumps(metrics, indent=2))
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    preds = outcome["predictions"]
    plot_prediction_scatter(preds["true_Rs"], preds["pred_Rs"], "Rs", save_path=str(results_dir / "rs_scatter.png"))
    plot_prediction_scatter(preds["true_Rsh"], preds["pred_Rsh"], "Rsh", save_path=str(results_dir / "rsh_scatter.png"))


if __name__ == "__main__":
    main()
