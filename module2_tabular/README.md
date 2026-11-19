# Module 2: Tabular Foundation Models

**Goal:** Predict the energy use intensity (EUI) of real buildings from basic facts (type, size, age, location) with the tabular foundation model **TabPFN v2**, compare it with classic scikit-learn models on per-building accuracy and **stock-level energy bias**, and profile its memory and GPU energy cost.

## Notebooks

1. **[`01_tabular_intro.ipynb`](01_tabular_intro.ipynb)** — beginner walkthrough:
   - In-context learning vs. fitting parameters, leakage-free Chicago 2022 building features.
   - 5-fold CV and learning curves (`n = 25` to `2,000`): TabPFN v2 vs. scikit-learn baselines.
   - Floor-area-weighted stock energy bias and NVML/CodeCarbon operational summary.
2. **[`02_tabular_fms_comparison.ipynb`](02_tabular_fms_comparison.ipynb)** — multi-model & split benchmark (install `pip install -r requirements-extras/tabular.txt`):
   - Compares **Linear regression**, **LightGBM**, **CatBoost**, **XGBoost**, **TabPFN v2** (11.1 M), and **TabICLv2** (28.5 M, BSD-3-Clause).
   - Evaluates 5-fold random CV, 80% quantile interval calibration, **temporal generalization (2021 → 2022 repeat vs. unseen buildings)**, **spatial holdout (`GroupKFold` by `community_area`)**, and training-size crossover (`50` to `2,000`).

## Data and model

| Item | Source | Licence |
| :--- | :--- | :--- |
| Chicago Energy Benchmarking, data year 2022 (2,532 buildings after cleaning, ~1 MB) | [City of Chicago Data Portal](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking/xq83-jr8c) | City of Chicago open data; see the portal's terms of use |
| TabPFN v2 regressor (~44.4 MB) | [Prior-Labs/TabPFN-v2-reg](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) | Prior Labs License (Apache 2.0 with attribution) |

Both are downloaded automatically on the first run. The data is stored in `data/cache/`. TabPFN weights go to `TABPFN_MODEL_CACHE_DIR` if set (see the main README), otherwise to TabPFN's default cache.

Newer TabPFN versions (2.5 and later) need a free Prior Labs account and are licensed for non-commercial use; the notebook explains how to switch.

## Results (5-fold cross-validation, 2,532 buildings)

| Model | MAE (kBtu/ft²) | R² | Stock energy bias (%) | 5-fold Time (s) |
| :--- | ---: | ---: | ---: | ---: |
| Average (baseline) | 30.6 | 0.00 | +1.98 | 0.0 |
| Linear regression | 27.6 | 0.23 | -0.49 | 0.0 |
| Gradient boosting | 27.0 | 0.28 | +0.90 | 1.9 |
| Random forest | 26.2 | 0.29 | +4.07 | 4.7 |
| **TabPFN v2** | **25.5** | **0.31** | -1.31 | 7.7 |

TabPFN is the most accurate per-building model at every training size from 25 to 2,000 buildings (with 500 buildings matching a random forest trained on 2,000), while keeping floor-area-weighted stock energy bias much tighter (-1.31%) than random forest (+4.07%). R² stays modest for all models because type, size, age and location explain only part of a building's energy use (operation and occupancy are not in the data).

## Model cards & operational footprint

| Model | Params (M) | Precision | Disk (MB) | Peak VRAM (GB) | Min Hardware | Licence | Limits |
| :--- | ---: | :--- | ---: | ---: | :--- | :--- | :--- |
| Linear regression | – | `cpu/sklearn` | – | 0.00 | `CPU-ok` | BSD-3-Clause | Linear relationships only |
| Random forest (300 trees) | – | `cpu/sklearn` | – | 0.00 | `CPU-ok` | BSD-3-Clause | CPU multi-core |
| Gradient boosting (Hist) | – | `cpu/sklearn` | – | 0.00 | `CPU-ok` | BSD-3-Clause | Needs >= 20 samples/leaf by default |
| **TabPFN v2** | **11.1** | `float32` | **44.4** | **0.24** | `GPU <= 4 GB` | Prior Labs (Apache 2.0 w/ attr.) | <= 10,000 rows, <= 500 cols |

## Measured runtime & energy

Measured on 2026-11-19 on the same workstation as Module 1 (NVIDIA Quadro RTX 6000, PyTorch 2.11.0+cu128, `tabpfn 10.2.0`), repeat run with data and model cached.

| Step | Time (s) | Peak VRAM (GB) | Avg Power (W) | GPU Energy (J) | mJ / item |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1. Load building data | 0.0 | 0.00 | 32.2 | 0.7 | – |
| 2. 5-fold CV: Average + Linear regression | 0.0 | 0.00 | 55.3 | 2.6 | 0.5 / bldg |
| 2. 5-fold CV: Random forest (CPU) | 4.7 | 0.00 | 54.1* | 253.2* | 100.0 / bldg |
| 2. 5-fold CV: Gradient boosting (CPU) | 1.9 | 0.00 | 25.3* | 48.3* | 19.1 / bldg |
| 2. 5-fold CV: TabPFN v2 (GPU) | 7.7 | 0.24 | 117.7 | 910.5 | 359.6 / bldg |
| 3. Learning curve (7 sizes × 5 repeats × 4 models) | 78.8 | 0.29 | 62.9 | 4,950.7 | 35,362 / fit |
| **Total (profiled steps, NVML GPU)** | **93.1** | **0.29** | **66.2** | **6,166.0** | – |

*\*During CPU-only scikit-learn steps, NVML records the workstation GPU's idle power draw (~25–55 W).*

| Telemetry Scope | Duration (s) | GPU (Wh) | CPU + RAM (Wh, modeled) | Total (Wh) | CO₂ CH 30 g/kWh (mg) | CO₂ EU 230 g/kWh (mg) | CO₂ US 380 g/kWh (mg) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Profiled steps (GPU only, NVML 50 ms) | 93.1 | 1.713 | – | 1.713 | 51.4 | 393.9 | 650.9 |
| Whole notebook (CodeCarbon offline) | 95.6 | 1.496 | 2.771 | 4.268 | 128.0 | 981.6 | 1,621.7 |

Structured machine-readable results are saved to [`benchmarks/results/module2_tabular_01_tabular_intro.json`](../benchmarks/results/module2_tabular_01_tabular_intro.json), [`benchmarks/results/module2_tabular_02_tabular_fms_comparison.json`](../benchmarks/results/module2_tabular_02_tabular_fms_comparison.json), and `data/cache/module2_tabular_runtime.csv`.

## Multi-model & split comparison (`02_tabular_fms_comparison.ipynb`, Chicago 2021–2022)

| Model | CV Random MAE | CV Stock Bias (%) | Temporal Repeat MAE | Temporal Unseen MAE | Spatial Holdout MAE | Peak VRAM (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear regression | 27.54 | +0.19 | 26.44 | 29.99 | 28.97 | 0.00 |
| LightGBM | 27.52 | +0.64 | **19.06** | 29.44 | 28.61 | 0.00 |
| XGBoost | 26.85 | +1.58 | 19.67 | 29.21 | 27.29 | 0.00 |
| CatBoost | 26.34 | +2.17 | 21.53 | 28.39 | 27.24 | 0.00 |
| TabPFN v2 (11.1 M) | 25.68 | -0.72 | 23.01 | **27.92** | 26.73 | 0.78 |
| **TabICLv2** (28.5 M) | **25.42** | -5.94 | 22.95 | 28.51 | **26.68** | 2.71 |

Key takeaways: GBDTs memorize **repeat buildings** across years (`LightGBM 19.06`), whereas **TabPFN v2** generalizes best to **unseen buildings** (`27.92`) with tight stock bias (`-0.72%`), and **TabICLv2** leads random CV, spatial holdout, and `n = 200–2,000` scaling (`24.83` at `n = 2,000`) with calibrated 80% intervals (`80.4%` coverage vs. LightGBM's `64.0%`), though its default predictions show `-5.94%` stock bias. Full benchmark run: `6.276 Wh` NVML GPU (`22,594 J`, `430.9 s`) / `39.467 Wh` CodeCarbon offline (`445.6 s`).
