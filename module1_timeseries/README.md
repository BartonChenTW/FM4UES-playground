# Module 1: Time-Series Foundation Models

**Goal:** Forecast Swiss hourly electricity demand one day ahead with pretrained time-series foundation models (FMs), without training them on our data ("zero-shot"), and test whether weather data improves the forecast.

## Notebook

[`01_demand_forecasting.ipynb`](01_demand_forecasting.ipynb) covers:

1. What a time-series FM is: context, horizon, quantiles and covariates.
2. Loading Swiss electricity demand (Open Power System Data, 2016–2020) and weather (Open-Meteo, 5 Swiss cities).
3. Exploring daily, weekly and seasonal patterns, and the link between temperature and demand.
4. Simple baselines: "same as yesterday" and "same as last week".
5. Zero-shot forecasting with **Chronos-Bolt** (demand only).
6. **Chronos-2** with and without weather covariates.
7. Evaluation over all 365 days of 2019: MAE, MAPE and 80% interval coverage.
8. A runtime summary of every step.

## Data

| Dataset | Source | Licence |
| :--- | :--- | :--- |
| Swiss hourly load (`CH_load_actual_entsoe_transparency`) | [Open Power System Data – Time series, 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06) (from ENTSO-E Transparency Platform) | See the OPSD page; ENTSO-E data terms apply |
| Hourly temperature and shortwave radiation | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | CC BY 4.0 |

Both are downloaded automatically on the first run into `data/cache/`.

## Results (all days of 2019, day-ahead, 24 h horizon)

| Model | MAE (MW) | MAPE (%) | 80% interval coverage (%) |
| :--- | ---: | ---: | ---: |
| Last week (baseline) | 401.9 | 6.4 | – |
| Yesterday (baseline) | 395.0 | 6.2 | – |
| Chronos-Bolt (base) | 254.5 | 4.0 | 78.5 |
| Chronos-2 | 247.3 | 3.9 | 77.4 |
| Chronos-2 + weather | 242.0 | 3.8 | 76.6 |

The future weather is the *measured* weather (a perfect weather forecast), so the weather gain is an upper bound.

## Measured runtime

Measured on 2026-11-17 on a Windows Server 2022 workstation: NVIDIA Quadro RTX 6000 (24 GB), driver 581.15, PyTorch 2.11.0+cu128, chronos-forecasting 2.3.2. All times are in seconds.

| Step | GPU, first run (downloads) | GPU, repeat run (cached) | CPU only (cached) |
| :--- | ---: | ---: | ---: |
| 1. Load electricity demand | 7.7 | 0.1 | 0.1 |
| 2. Load weather | 1.4 | 0.0 | 0.0 |
| 3. Baseline forecasts | 0.0 | 0.0 | 0.0 |
| 4. Load Chronos-Bolt | 13.8 | 2.3 | 2.3 |
| 5. Chronos-Bolt: forecast 365 days | 1.5 | 1.5 | 9.7 |
| 6. Load Chronos-2 | 9.9 | 1.7 | 1.6 |
| 7. Chronos-2 (demand only): forecast 365 days | 2.1 | 2.0 | 12.5 |
| 8. Chronos-2 + weather: forecast 365 days | 5.1 | 5.1 | 35.9 |
| **Total of timed steps** | **41.5** | **12.7** | **62.1** |
| Whole notebook, start to finish (incl. imports, plots, kernel start) | 75 | 41–52 | 91 |

The GPU makes the forecasting steps about 7× faster (8.6 s vs. 58.1 s). Download times depend on your network.
The notebook saves its own timing table to `data/cache/module1_runtime.csv`, so you can compare your machine.
