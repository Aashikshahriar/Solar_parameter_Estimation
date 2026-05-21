# Solar_parameter_Estimation

# Region-Aware Attention Framework for Photovoltaic Parameter Estimation

Physics-guided region-aware attention framework for photovoltaic parameter estimation from current–voltage (I–V) characteristics. The proposed deep learning model combines CNN feature extraction, cross-region attention, and environmental conditioning for robust estimation of series resistance ($R_s$) and shunt resistance ($R_{sh}$).

---

# Overview

Photovoltaic (PV) parameter estimation plays a critical role in:

- photovoltaic diagnostics,
- degradation monitoring,
- fault detection,
- digital twin modeling,
- predictive maintenance,
- and energy efficiency optimization.

Traditional optimization-based approaches such as:

- Newton–Raphson,
- Particle Swarm Optimization (PSO),
- Genetic Algorithms (GA),

often suffer from:

- high computational cost,
- sensitivity to initialization,
- convergence instability,
- and degraded robustness under noisy operating conditions.

To address these limitations, this repository introduces a **Region-Aware Attention Framework** that explicitly models physically meaningful photovoltaic operating regions using deep representation learning and attention mechanisms.

---

# Proposed Framework

The proposed framework consists of:

## 1. CNN Feature Encoder
Extracts hierarchical nonlinear representations from photovoltaic I–V characteristics.

## 2. Region Decomposition
The photovoltaic operating curve is partitioned into:

- Low-voltage region
- Transition region
- High-voltage region

to isolate parameter-sensitive operating behaviors.

## 3. Cross-Region Attention
Models nonlinear interactions between different photovoltaic operating regions.

## 4. Environmental Embedding
Incorporates irradiance and temperature information for adaptive learning.

## 5. Dual Regression Heads
Independently estimate:

- Series resistance ($R_s$)
- Shunt resistance ($R_{sh}$)

---

# Repository Structure

```bash
.
├── data_generation.py
├── preprocessing.py
├── model.py
├── training.py
├── evaluation.py
├── hyperparameter_tuning.py
├── baseline_comparison.py
├── requirements.txt
├── README.md
├── pv_dataset.npz
└── final_model.pth
```

---

# Synthetic Dataset Generation

Synthetic photovoltaic datasets are generated using the **Single-Diode Model (SDM)** combined with Newton–Raphson numerical solving.

The generated dataset includes:

- Current–voltage (I–V) characteristics
- Series resistance ($R_s$)
- Shunt resistance ($R_{sh}$)
- Irradiance ($G$)
- Temperature ($T$)
- Gaussian measurement noise

## Generate Dataset

```bash
python data_generation.py
```

Generated dataset:

```bash
pv_dataset.npz
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Train the proposed framework:

```bash
python training.py
```

The best trained model will be saved as:

```bash
final_model.pth
```

---

# Evaluation

Evaluate the trained model:

```bash
python evaluation.py
```

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Relative Error Percentage (RE%)
- Coefficient of Determination ($R^2$)

for both:

- $R_s$
- $R_{sh}$

---

# Hyperparameter Optimization

Randomized hyperparameter tuning:

```bash
python hyperparameter_tuning.py
```

Optimized parameters include:

- Learning rate
- Weight decay
- Noise augmentation
- Multi-task loss weights

---


# Framework Highlights

- Physics-guided region decomposition
- Cross-region nonlinear interaction learning
- Environment-aware parameter estimation
- Noise-aware robust learning
- Multi-task photovoltaic regression
- High computational efficiency
- Interpretable deep learning framework

---

# Example Pipeline

```text
Synthetic PV Dataset
        ↓
Preprocessing
        ↓
CNN Feature Extraction
        ↓
Region Decomposition
        ↓
Cross-Region Attention
        ↓
Environment Fusion
        ↓
Dual Regression Heads
        ↓
Rs and Rsh Estimation
```

---

# Citation

If you use this work, please cite:

```bibtex
@article{Shahriar_solar_,
  title={Region-Aware Attention Framework for Photovoltaic Parameter Estimation},
  author={K.A.Shahriar},
  year={202}
}
```

---

# License

GNU General Public License v3.0

---

# Acknowledgements

This work was developed for research on robust photovoltaic parameter estimation under nonlinear environmental variability and measurement uncertainty.
