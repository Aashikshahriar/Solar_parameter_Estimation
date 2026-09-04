# Solar Parameter Estimation

Estimating photovoltaic (PV) module degradation parameters — series resistance (**Rs**) and
shunt resistance (**Rsh**) — directly from I-V curve measurements and environmental
conditions (irradiance, temperature), using a physics-informed synthetic dataset and a
deep learning regressor.

> **Status:** This repository accompanies a manuscript currently in preparation. It documents
> the overall research pipeline end-to-end so the work is reproducible in outline, but some
> implementation details, the trained model weights, and full experimental results are
> withheld pending publication. These will be added here once the paper is published.

## Pipeline overview

1. **Data generation** (`solar_pe/physics.py`) — a single-diode PV model is simulated across
   randomized operating conditions (irradiance, temperature) and degradation states (Rs, Rsh).
   Each I-V curve is solved point-by-point with a Newton-Raphson iteration and perturbed with
   sensor-like noise.
2. **Feature engineering** (`solar_pe/dataset.py`) — raw curves are normalized and expanded
   into model-ready tensors (voltage, current, and current gradient channels, plus
   standardized environmental features). Targets are log-transformed since resistances span
   orders of magnitude.
3. **Model** (`solar_pe/model.py`) — a 1D CNN encoder extracts local curve features, which are
   pooled into three curve regions and refined with self-attention before being fused with
   environmental context and passed to two regression heads (one per parameter).
4. **Main result: training & evaluation** (`solar_pe/train.py`, `solar_pe/evaluate.py`) —
   weighted multi-task MSE loss, cosine learning-rate schedule, noise augmentation, early
   stopping, and MAE / RMSE / R² / relative-error metrics on a held-out test split.
5. **Classical baseline** (`solar_pe/baseline.py`) — a nonlinear least-squares curve fit,
   included as a reference point for how the learned model compares to a standard
   optimization-based approach to the same inverse problem.
6. **Ablation study** (`solar_pe/ablation.py`) — isolates the contribution of region pooling,
   cross-attention, and environmental conditioning by enabling them one at a time on a shared
   CNN backbone.
7. **Sensitivity analysis** (`solar_pe/sensitivity.py`) — gradient-based attribution of each
   prediction to specific points and regions along the I-V curve.
8. **Interpretability** (`solar_pe/interpretability.py`) — region cross-attention map
   extraction, plus region perturbation and occlusion probes.
9. **Noise robustness** (`solar_pe/noise_analysis.py`) — evaluates prediction quality as
   measurement noise on the current channel increases at inference time.
10. **Environmental perturbation** (`solar_pe/environmental_perturbation.py`) — evaluates
    prediction quality as the irradiance and temperature inputs are perturbed, independently
    and jointly.

## Repository structure

```
solar-parameter-estimation/
├── src/solar_pe/                       # library code
│   ├── config.py                       # physical constants & hyperparameters
│   ├── physics.py                      # single-diode simulator (data generation)
│   ├── dataset.py                      # feature engineering, splitting, DataLoaders
│   ├── model.py                        # AttentionPV architecture (main model)
│   ├── train.py                        # training loop
│   ├── evaluate.py                     # metrics & inference (main result)
│   ├── baseline.py                     # classical least-squares baseline
│   ├── ablation.py                     # ablation study (region/attention/env)
│   ├── sensitivity.py                  # gradient-based sensitivity analysis
│   ├── interpretability.py             # attention maps, perturbation & occlusion
│   ├── noise_analysis.py               # measurement noise robustness sweep
│   ├── environmental_perturbation.py   # irradiance / temperature perturbation
│   └── visualize.py                    # plotting utilities for all of the above
├── scripts/                            # CLI entry points, one per pipeline stage
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── run_baseline.py
│   ├── run_ablation.py
│   ├── run_sensitivity.py
│   ├── run_interpretability.py
│   ├── run_noise_analysis.py
│   └── run_environmental_perturbation.py
├── data/                                # generated datasets (gitignored)
├── checkpoints/                         # trained weights (gitignored)
└── results/                             # metrics & plots (gitignored)
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

# 2. Train the model (main result)
python scripts/train_model.py --data data/pv_dataset.npz --epochs 100

# 3. Evaluate on the held-out test split
python scripts/evaluate_model.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth

# 4. Classical baseline for comparison
python scripts/run_baseline.py --data data/pv_dataset.npz --num-samples 50

# 5. Ablation study
python scripts/run_ablation.py --data data/pv_dataset.npz --epochs 50

# 6. Gradient-based sensitivity analysis
python scripts/run_sensitivity.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth

# 7. Interpretability: attention maps + region perturbation/occlusion
python scripts/run_interpretability.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth

# 8. Measurement noise robustness
python scripts/run_noise_analysis.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth

# 9. Environmental (irradiance/temperature) perturbation
python scripts/run_environmental_perturbation.py --data data/pv_dataset.npz --checkpoint checkpoints/best_model.pth
```

Each analysis script writes its tables (CSV) and figures (PNG) into its own subfolder under
`results/`.

## Notes

- Dataset generation, training, and evaluation configs live in `solar_pe/config.py` — adjust
  ranges, split fractions, and hyperparameters there rather than hardcoding overrides.
- The classical baseline is comparatively slow (per-curve optimization), so it defaults to a
  small subset of the test split via `--num-samples`.
- The ablation study trains four model variants from scratch, so it takes roughly 4x as long
  as a single training run at the same epoch count.

## Citation

A citation entry will be added here once the associated paper is published.
