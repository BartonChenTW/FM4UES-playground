# Module 2: Tabular Foundation Models

**Goal:** Predict the energy use intensity (EUI) of real buildings from basic facts (type, size, age, location) with the tabular foundation model **TabPFN v2**, and compare it with classic scikit-learn models, especially when only a few buildings are known.

## Notebook

[`01_tabular_intro.ipynb`](01_tabular_intro.ipynb) covers:

1. What a tabular FM is: in-context learning instead of training, and how this differs from a random forest.
2. Loading the Chicago Energy Benchmarking 2022 data, and choosing features **without data leakage**.
3. Exploring EUI by building type and construction year.
4. 5-fold cross-validation: average baseline, linear regression, random forest, gradient boosting, TabPFN v2.
5. A learning curve from 25 to 2,000 training buildings (data efficiency).
6. A runtime summary of every step.

## Data and model

| Item | Source | Licence |
| :--- | :--- | :--- |
| Chicago Energy Benchmarking, data year 2022 (2,532 buildings after cleaning, ~1 MB) | [City of Chicago Data Portal](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking/xq83-jr8c) | City of Chicago open data; see the portal's terms of use |
| TabPFN v2 regressor (~45 MB) | [Prior-Labs/TabPFN-v2-reg](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) | Prior Labs License (Apache 2.0 with attribution) |

Both are downloaded automatically on the first run. The data is stored in `data/cache/`. TabPFN weights go to the folder
in the `TABPFN_MODEL_CACHE_DIR` environment variable if set (see the main README), otherwise to TabPFN's default cache.

Newer TabPFN versions (2.5 and later) need a free Prior Labs account and are licensed for non-commercial use; the notebook explains how to switch.

## Results (5-fold cross-validation, 2,532 buildings)

| Model | MAE (kBtu/ft²) | R² |
| :--- | ---: | ---: |
| Average (baseline) | 30.6 | 0.00 |
| Linear regression | 27.6 | 0.23 |
| Gradient boosting | 27.0 | 0.28 |
| Random forest | 26.2 | 0.29 |
| **TabPFN v2** | **25.5** | **0.31** |

TabPFN is the most accurate model at every training size from 25 to 2,000 buildings. With 500 training buildings it matches a random forest trained on 2,000. R² stays modest for all models, because type, size, age and location explain only part of a building's energy use (operation and occupancy are not in the data).

## Measured runtime

Measured on 2026-11-18, same workstation as Module 1 (NVIDIA Quadro RTX 6000, PyTorch 2.11.0+cu128, tabpfn 10.2.0), repeat run with data and model cached. All times are in seconds.

| Step | GPU |
| :--- | ---: |
| 1. Load building data (first run: ~1 MB download, about 2.5 s) | 0.0 |
| 2. 5-fold CV: linear regression / random forest / gradient boosting | 0.0 / 4.2 / 1.8 |
| 2. 5-fold CV: TabPFN v2 | 7.0 |
| 3. Learning curve (7 sizes x 5 repeats x 4 models) | 75.7 |
| **Total of timed steps** | **88.7** |
| Whole notebook, start to finish | about 105 |

The notebook saves its timing table to `data/cache/module2_runtime.csv`.
