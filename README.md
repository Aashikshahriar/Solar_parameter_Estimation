# Solar Parameter Estimation

Estimating photovoltaic (PV) module degradation parameters — series resistance (**Rs**) and
shunt resistance (**Rsh**) — directly from I-V curve measurements and environmental
conditions (irradiance, temperature), using a physics-informed synthetic dataset and a
deep learning regressor.

> **Status:** This repository accompanies a manuscript currently in preparation. It documents
> the overall pipeline end-to-end so the work is reproducible in outline, but some
> implementation details, the trained model weights, full experimental results, and ablation
> studies are withheld pending publication. These will be added here once the paper is
> published.

## Pipeline overview

1. **Physics-based data generation** (`solar_pe/physics.py`) — a single-diode PV model is
   simulated across randomized operating conditions (irradiance, temperature) and degradation
   states (Rs, Rsh). Each I-V curve is solved point-by-point with a Newton-Raphson iteration
   and perturbed with sensor-like noise.
2. **Feature engineering** (`solar_pe/dataset.py`) — raw curves are normalized and expanded
   into model-ready tensors (voltage, current, and current gradient channels, plus
   standardized environmental features). Targets are log-transformed since resistances span
   orders of magnitude.
3. **Model** (`solar_pe/model.py`) — a 1D CNN encoder extracts local curve features, which are
   pooled into three curve regions and refined with self-attention before being fused with
   environmental context and passed to two regression heads (one per parameter).
4. **Training** (`solar_pe/train.py`) — weighted multi-task MSE loss, cosine learning-rate
   schedule, noise augmentation, and early stopping on validation loss.
5. **Evaluation** (`solar_pe/evaluate.py`) — MAE, RMSE, R², and relative error (%) on a held-out
   test split.
6. **Classical baseline** (`solar_pe/baseline.py`) — a nonlinear least-squares curve fit is
   provided as a reference point for how the learned model compares to a standard
   optimization-based approach to the same inverse problem.

## Repository structure

```
solar-parameter-estimation/
├── src/solar_pe/         # library code
│   ├── config.py         # physical constants & hyperparameters
│   ├── physics.py        # single-diode simulator (synthetic data generation)
│   ├── dataset.py         # feature engineering, splitting, DataLoaders
│   ├── model.py           # AttentionPV architecture
│   ├── train.py           # training loop
│   ├── evaluate.py        # metrics & inference
│   ├── baseline.py        # classical least-squares baseline
│   └── visualize.py       # plotting utilities
├── scripts/               # CLI entry points
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── run_baseline.py
├── data/                  # generated datasets (gitignored)
├── checkpoints/           # trained weights (gitignored)
└── results/               # metrics & plots (gitignored)
```

## Setup

```bash
git clone <this-repo-url>
cd solar-parameter-estimation

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate a synthetic dataset
python scripts/generate_dataset.py --out data/pv_dataset.npz --n 20000

# 2. Train the model
python scripts/train_model.py --data data/pv_dataset.npz --epochs 100

# 3. Evaluate on the held-out test split
python scripts/evaluate_model.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth

# 4. Run the classical baseline for comparison
python scripts/run_baseline.py --data data/pv_dataset.npz --num-samples 50
```

## Notes

- Dataset generation, training, and evaluation configs live in `solar_pe/config.py` — adjust
  ranges, split fractions, and hyperparameters there rather than hardcoding overrides.
- The classical baseline is comparatively slow (per-curve optimization), so it defaults to a
  small subset of the test split via `--num-samples`.

## Citation

A citation entry will be added here once the associated paper is published.
