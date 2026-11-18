# Module 1: Time-Series Foundation Models

**Goal:** Forecast Swiss hourly electricity demand one day ahead with pretrained time-series foundation models (FMs), without training them on our data ("zero-shot"), test whether weather data improves the forecast, and profile the operational and GPU energy cost (including `fp32` vs. `bf16`/`fp16`).

## Notebook

[`01_demand_forecasting.ipynb`](01_demand_forecasting.ipynb) covers:

1. What a time-series FM is: context, horizon, quantiles and covariates.
2. Loading Swiss electricity demand (Open Power System Data, 2016–2020) and weather (Open-Meteo, 5 Swiss cities).
3. Exploring daily, weekly and seasonal patterns, and the link between temperature and demand.
4. Simple baselines: "same as yesterday" and "same as last week".
5. Zero-shot forecasting with **Chronos-Bolt** (demand only).
6. **Chronos-2** with and without weather covariates.
7. Full-year 2019 evaluation: MAE, MAPE, Peak-5% MAE, Ramp MAE, 80% interval coverage and Winkler score.
8. Precision trade-off: `fp32` vs. `bf16` vs. `fp16` (accuracy, peak VRAM, latency and GPU energy).
9. Operational, memory and energy summary (`fm4ues.profiler`: 50 ms NVML GPU power/energy + offline CodeCarbon).

## Data

| Dataset | Source | Licence |
| :--- | :--- | :--- |
| Swiss hourly load (`CH_load_actual_entsoe_transparency`) | [Open Power System Data – Time series, 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06) (from ENTSO-E Transparency Platform) | See the OPSD page; ENTSO-E data terms apply |
| Hourly temperature and shortwave radiation | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | CC BY 4.0 |

Both are downloaded automatically on the first run into `data/cache/`.

## Results (all 365 days of 2019, day-ahead, 24 h horizon)

| Model | MAE (MW) | MAPE (%) | Peak-5% MAE (MW) | Ramp MAE (MW/h) | 80% coverage (%) | Winkler score (MW) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Last week (baseline) | 401.9 | 6.4 | 328.1 | 168.4 | – | – |
| Yesterday (baseline) | 395.0 | 6.2 | 386.2 | 174.9 | – | – |
| Chronos-Bolt (base) | 254.5 | 4.0 | 171.9 | 131.2 | 78.5 | 1,208.6 |
| Chronos-2 | 247.3 | 3.9 | 185.7 | 125.7 | 77.4 | 1,157.8 |
| **Chronos-2 + weather** | **242.0** | **3.8** | **170.5** | 126.7 | 76.6 | **1,147.7** |

Adding weather reduces overall MAPE only slightly (3.9% → 3.8%), but cuts **Peak-5% MAE** from 185.7 MW to 170.5 MW during the highest-demand hours. Note that future weather is *measured* weather (a perfect forecast), so the weather gain is an upper bound.

## Model cards & precision trade-off

| Model | Params (M) | Precision | Disk (MB) | Peak VRAM (GB) | Min Hardware | Licence | Limits |
| :--- | ---: | :--- | ---: | ---: | :--- | :--- | :--- |
| Chronos-Bolt (base) | 205.3 | `float32` | 821.2 | 4.92 | `GPU <= 8 GB` | Apache-2.0 | Context <= 2,048 steps, univariate only |
| Chronos-2 | 119.5 | `float32` | 477.9 | 2.31 | `GPU <= 4 GB` | Apache-2.0 | Supports past & known-future covariates |

Running **Chronos-Bolt (base)** across all 365 days in half precision on an NVIDIA Quadro RTX 6000 (Turing SM 7.5):

| Precision | MAE (MW) | MAPE (%) | Peak VRAM (GB) | Time (s) | Avg Power (W) | GPU Energy (J) | mJ / day |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp32` | 254.5 | 4.00 | 4.92 | 1.07 | 183.9 | 197.6 | 541.4 |
| `bf16` | 254.4 | 4.00 | 2.76 | 1.85 | 178.2 | 329.7 | 903.3 |
| **`fp16`** | **254.5** | **4.00** | **2.59** | **0.87** | **89.9** | **80.3** | **219.9** |

`fp16` cuts peak VRAM nearly in half and uses **2.5× less GPU energy** than `fp32` with identical forecast accuracy. `bf16` is slower and consumes more energy here because Turing (SM 7.5) emulates `bfloat16`, whereas Ampere and newer GPUs (SM 8.0+) accelerate `bf16` natively.

## Measured runtime & energy

Measured on 2026-11-19 on a Windows Server 2022 workstation: NVIDIA Quadro RTX 6000 (24 GB, 260 W TDP), driver 581.15, PyTorch 2.11.0+cu128, `chronos-forecasting 2.3.2`, repeat run with data and weights cached.

| Step | Time (s) | Peak VRAM (GB) | Avg Power (W) | GPU Energy (J) | mJ / day |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1–3. Load data & baselines | 0.1 | 0.00 | 40.6 | 3.5 | 0.9 |
| 4. Load Chronos-Bolt | 2.2 | 0.82 | 52.4 | 113.0 | – |
| 5. Chronos-Bolt: forecast 365 days (`fp32`) | 1.4 | 4.92 | 157.4 | 214.4 | 587.3 |
| 6. Load Chronos-2 | 1.4 | 0.49 | 74.6 | 105.4 | – |
| 7. Chronos-2 (demand only): 365 days | 2.1 | 2.29 | 161.5 | 344.3 | 943.4 |
| 8. Chronos-2 + weather: 365 days | 5.3 | 2.31 | 220.6 | 1,172.1 | 3,211.2 |
| 9. Precision sweep (`fp32` + `bf16` + `fp16`) | 3.8 | 4.92 | – | 607.6 | – |
| **Total (profiled steps, NVML GPU)** | **16.2** | **4.92** | **157.9** | **2,560.3** | – |

| Telemetry Scope | Duration (s) | GPU (Wh) | CPU + RAM (Wh, modeled) | Total (Wh) | CO₂ CH 30 g/kWh (mg) | CO₂ EU 230 g/kWh (mg) | CO₂ US 380 g/kWh (mg) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Profiled steps (GPU only, NVML 50 ms) | 16.2 | 0.711 | – | 0.711 | 21.3 | 163.6 | 270.3 |
| Whole notebook (CodeCarbon offline) | 34.8 | 0.879 | 1.388 | 2.267 | 68.0 | 521.4 | 861.4 |

Structured machine-readable metrics and step logs are saved to [`benchmarks/results/module1_timeseries_01_demand_forecasting.json`](../benchmarks/results/module1_timeseries_01_demand_forecasting.json) and `data/cache/module1_timeseries_runtime.csv`.
