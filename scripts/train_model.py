"""CLI: train the attention-based Rs/Rsh estimator on a generated dataset.

Usage:
    python scripts/train_model.py --data data/pv_dataset.npz --epochs 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from solar_pe.config import DatasetConfig, TrainConfig
from solar_pe.dataset import build_dataloaders
from solar_pe.model import AttentionPV
from solar_pe.train import fit
from solar_pe.visualize import plot_loss_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pv_dataset.npz")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--loss-plot", default="results/loss_curve.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.loss_plot).parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(
        args.data, DatasetConfig(), batch_size=args.batch_size,
    )

    model = AttentionPV()
    train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    history = fit(model, train_loader, val_loader, device, train_cfg, checkpoint_path=args.checkpoint)
    plot_loss_curve(history, save_path=args.loss_plot)

    print(f"Best checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
