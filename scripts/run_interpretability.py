"""CLI: attention map extraction plus region perturbation / occlusion analysis.

Usage:
    python scripts/run_interpretability.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from solar_pe.config import DatasetConfig
from solar_pe.dataset import build_dataloaders
from solar_pe.interpretability import extract_attention_map, region_perturbation_occlusion
from solar_pe.model import AttentionPV
from solar_pe.visualize import plot_attention_map, plot_region_bars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pv_dataset.npz")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--results-dir", default="results/interpretability")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = build_dataloaders(args.data, DatasetConfig())

    model = AttentionPV().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    Xb, envb, _ = test_loader.dataset[0]
    attn = extract_attention_map(model, Xb, envb, device)
    plot_attention_map(attn, save_path=str(results_dir / "attention_map.png"))

    df = region_perturbation_occlusion(model, test_loader, device,
                                        num_samples=args.num_samples, noise_std=args.noise_std)
    df_avg = df.groupby("Region").mean().reset_index()
    print(df_avg)

    df.to_csv(results_dir / "perturbation_occlusion_full.csv", index=False)
    df_avg.to_csv(results_dir / "perturbation_occlusion_summary.csv", index=False)

    plot_region_bars(df_avg, ("Rs_pert", "Rsh_pert"), "Region", "Noise Perturbation",
                      save_path=str(results_dir / "perturbation_plot.png"))
    plot_region_bars(df_avg, ("Rs_occ", "Rsh_occ"), "Region", "Occlusion Analysis",
                      save_path=str(results_dir / "occlusion_plot.png"))


if __name__ == "__main__":
    main()
