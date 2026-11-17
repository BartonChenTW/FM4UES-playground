# Data Directory

## Policy

Only **small, public sample files** (e.g., tiny CSVs for quick smoke tests) should be committed to this directory. Larger datasets are downloaded directly by the notebooks into a git-ignored cache folder when you run them.

## Data Sources and Licences

All datasets are downloaded automatically by the notebooks on first run. Nothing is committed.

| Dataset | Used in | Cached as | Source | Licence |
| :--- | :--- | :--- | :--- | :--- |
| Swiss hourly electricity load (`CH_load_actual_entsoe_transparency`), 2015–2020 | Modules 1, 3 | `data/cache/opsd_ch_load.csv` | [Open Power System Data – Time series 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06) (from ENTSO-E Transparency Platform) | See the OPSD page; ENTSO-E data terms apply |
| Hourly temperature and shortwave radiation, 5 Swiss cities, 2016–2020 | Module 1 | `data/cache/openmeteo_ch_weather.csv` | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | CC BY 4.0 |
| Chicago Energy Benchmarking, data year 2022 | Module 2 | `data/cache/chicago_benchmarking_2022.csv` | [City of Chicago Data Portal](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking/xq83-jr8c) | City of Chicago open data; see the portal's terms of use |
| EuroSAT RGB, test split (5,400 Sentinel-2 images) | Module 3 | Hugging Face cache | [EuroSAT](https://github.com/phelber/EuroSAT) via [blanchon/EuroSAT_RGB](https://huggingface.co/datasets/blanchon/EuroSAT_RGB) | MIT |

The notebooks also write their runtime tables (`moduleN_runtime.csv`) to `data/cache/`.
