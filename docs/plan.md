# Project Plan & Roadmap — FM4UES-playground

> Working roadmap and phased TODO list for expanding **FM4UES-playground** into a comprehensive, energy-domain-rigorous learning and benchmarking playground for foundation models (FMs) in urban energy systems (UES).

---

## 1. Improvement Pillars

1. **More Examples, Model Families & Modalities**
   - Add energy-native, physics-aware graph neural solvers (**GridFM / GENCO**) for power grid steady-state analysis (AC Power Flow & AC-OPF).
   - Deepen multimodal coverage with domain-pretrained Earth Observation / remote-sensing FMs (**RemoteCLIP**, **SigLIP 2**, **Prithvi-EO-2.0 / Clay / TerraMind**), modern compact VLMs (**Qwen2.5-VL** / **SmolVLM**), and cross-modal image + tabular fusion.
   - Broaden time-series and tabular model diversity beyond Chronos and TabPFN v2 (**TimesFM 2.5**, **TiRex / TiRex-2**, **Moirai 2.0**, **TabICL / TabICLv2**, **Mitra / Mitra-v2**) alongside strong tuned ML baselines (**LightGBM**, **CatBoost**).
2. **Energy-Domain Model Performance Comparisons**
   - Evaluate models with UES-relevant metrics (peak-hour error, ramp error, CRPS/Pinball/Winkler scores, portfolio aggregate EUI bias, power-balance residuals, $N-1$ feasibility) and test harder operational regimes (PV/wind generation, temporal splits, spatial holdouts).
   - Highlight honest "when *not* to use an FM" lessons (pretraining leakage caveats, latency/scale crossovers, constraint violations).
   - Aggregate structured benchmark outputs into a unified leaderboard and Pareto frontier comparison page.
3. **Operational Profile & Energy Footprint ("How much energy does it cost to model energy?")**
   - Instrument notebooks with parameter counts, checkpoint disk size, peak GPU VRAM and host RAM, latency/throughput, approximate FLOPs, hardware tier recommendations, and measured GPU power/energy (Joules, Wh) plus grid CO₂e estimates.
   - Measure precision/quantization trade-offs (`fp32` vs. `bf16`) across accuracy, VRAM, speed, and energy.

---

## 2. Agreed Decisions

| Topic | Decision | Notes |
| :--- | :--- | :--- |
| **Environment strategy** | **Separate optional files** | Keep [environment.yml](../environment.yml) and [requirements.txt](../requirements.txt) lean for the core `01_*.ipynb` notebooks + lightweight profiling deps (`nvidia-ml-py`, `psutil`). Place heavy or CUDA-wheel-sensitive stacks (`gridfm-graphkit`/`torch-scatter`, `terratorch`, `autogluon`, `timesfm`, etc.) in `requirements-extras/*.txt`. |
| **Execution order** | **Start from Phase 1** | Build the shared profiling & metrics backbone first, retrofit Modules 1–3, then expand model benchmarks (Phase 2), Module 4 GridFM (Phase 3), and deeper multimodal + benchmark hub (Phase 4). |
| **Energy telemetry backend** | **Option C (Hybrid)** | High-frequency `nvidia-ml-py` (`pynvml`, 50 ms polling) per step for accurate GPU Joules/Wh on short inference calls + `codecarbon` offline whole-notebook tracker (`country_iso_code="CHE"`) written to `data/cache/`. |

---

## 3. Architecture & Design Overview

### Shared Package (`fm4ues/`) & Benchmark Registry
Currently, [`01_demand_forecasting.ipynb`](../module1_timeseries/01_demand_forecasting.ipynb), [`01_tabular_intro.ipynb`](../module2_tabular/01_tabular_intro.ipynb), and [`01_encoder_decoder.ipynb`](../module3_multimodal/01_encoder_decoder.ipynb) duplicate a local `timed()` helper and write ad-hoc CSVs into `data/cache/`. We will extract a tiny, readable package `fm4ues/`:
- `fm4ues/profiler.py` — `profile(step, model=..., n_items=...)` context manager, `inspect_model()`, peak VRAM/RAM tracking, GPU power/energy sampler, hardware banner, and structured export.
- `fm4ues/metrics.py` — shared UES evaluation metrics across forecasting, tabular, multimodal, and grid tasks.
- `fm4ues/plotting.py` — consistent plot styling and optional export to `docs/assets/img/`.
- `benchmarks/results/` — lightweight tracked JSON/CSV result records (kept separate from gitignored `data/cache/`) to feed `docs/benchmarks.html`.

### Operational Metrics Captured per Model / Step

| Dimension | Metrics | Collection Mechanism |
| :--- | :--- | :--- |
| **Model Card & Size** | Total/active parameters, dtype/precision (`fp32`/`bf16`/`fp16`/`int8`), disk checkpoint size (MB/GB), license, auth/gated status, hard limits (context/rows/graph size) | `inspect_model()` + Hugging Face cache inspection + explicit model metadata dict |
| **Memory & Hardware** | Peak GPU VRAM allocated & reserved (GB), peak host RAM (GB), cold-start load time vs. steady-state inference time, GPU vs. CPU speedup, minimum GPU tier tag (`CPU-ok`, `<=4 GB`, `<=8 GB`, `16-24 GB`) | `torch.cuda.reset_peak_memory_stats()`, `torch.cuda.max_memory_allocated()`, `psutil.Process().memory_info().rss` |
| **Compute Cost** | Latency per item (ms/forecast, ms/building, ms/tile, ms/OPF solve), throughput (items/s), estimated FLOPs on a representative batch, illustrative cloud cost (\$/1M inferences) | `time.perf_counter()` + `torch.cuda.synchronize()`, optional `torch.utils.flop_counter.FlopCounterMode` |
| **Power, Energy & CO₂e** | Avg & peak GPU power (W), GPU energy per step (J and Wh), energy per inference (mJ / µWh), reference CO₂e across **CH (~30 g/kWh)**, **EU avg (~230 g/kWh)**, and **US avg (~380 g/kWh)**, plus optional whole-run system estimate | High-frequency `nvidia-ml-py` (`pynvml`) polling (~50–100 ms trapezoidal integration) + optional `codecarbon` offline tracker |

### Energy-Domain Evaluation Metrics by Task

| Module / Domain | Current Metrics | Added Energy-Domain Metrics |
| :--- | :--- | :--- |
| **Time-Series** (Load, PV, Wind) | MAE, MAPE, 80% coverage | Peak-hour error (top 5% load), ramp-rate error ($\Delta$MW/h), CRPS / mean Pinball loss / Winkler score, holiday & extreme-weather slices, pretraining leakage note, stylized balancing/dispatch cost penalty |
| **Tabular** (Building Stock EUI) | MAE, $R^2$ | Portfolio / district aggregate bias (%), quantile calibration (80% & 90% coverage), data-efficiency crossover vs. tuned GBDTs ($n = 25 \dots 10{,}000$), temporal (2021→2022) & spatial (`community_area`) generalization gap |
| **Multimodal / Geospatial** | Overall 10-class accuracy | Per-class F1 on UES-relevant classes (Residential, Industrial, Highway), few-shot efficiency curve (5 / 10 / 50 / 200 labels), multimodal lift ($\Delta$MAE, $\Delta R^2$ over tabular-only), VLM factual/numeric precision audit |
| **Grid / Power Flow** (`GridFM`) | *(new)* | Max & mean active/reactive power mismatch (MW / Mvar / p.u.), voltage & line thermal limit violation rates (%), optimality gap (%) vs. classical AC-OPF, $N-1$ contingency feasibility rate, wall-clock speedup & energy-per-solve |

---

## 4. Phased Implementation TODO

### Repo Housekeeping
- [x] Reconcile/merge branch `module1-local-gpu` into `main` so GitHub Pages publishes the full 4-module playground (`v0.2.0`).
- [x] Add [`scripts/run_notebooks.py`](../scripts/run_notebooks.py) to execute notebooks top-to-bottom cleanly via `nbconvert` (with `--smoke`, `--audit-only`, `--module`, `--all`, and `--check-urls` external endpoint health checks).
- [ ] Complete post-expansion author review checklist in [`docs/to_review.md`](to_review.md) before cutting `v0.3.0` / `v1.0.0`.

### Phase 1: Operations & Profiling Backbone *(Completed)*
- [x] Finalize energy measurement backend (**Option C: Hybrid** `nvidia-ml-py` 50 ms per-step + offline `codecarbon` whole-notebook tracker) and pin profiling dependencies (`psutil==7.2.2`, `nvidia-ml-py==13.615.71`, `codecarbon==3.3.2`) in [`requirements.txt`](../requirements.txt).
- [x] Create [`fm4ues/__init__.py`](../fm4ues/__init__.py), [`fm4ues/profiler.py`](../fm4ues/profiler.py) (`RunProfiler.step()`, `inspect_model()`, VRAM/RAM/latency/throughput/FLOPs/energy/CO₂e tracking, hardware tier tag, CSV/JSON export), and [`fm4ues/metrics.py`](../fm4ues/metrics.py).
- [x] Retrofit [`module1_timeseries/01_demand_forecasting.ipynb`](../module1_timeseries/01_demand_forecasting.ipynb), [`module2_tabular/01_tabular_intro.ipynb`](../module2_tabular/01_tabular_intro.ipynb), and [`module3_multimodal/01_encoder_decoder.ipynb`](../module3_multimodal/01_encoder_decoder.ipynb) to use `fm4ues.profiler`, display Model Cards + Operational/Energy summary tables, compute UES metrics (Peak-5% MAE, Ramp MAE, Winkler score, floor-area-weighted stock energy bias, UES-class F1), and export structured results to `benchmarks/results/`.
- [x] Add a `fp32` vs. `bf16` vs. `fp16` precision trade-off experiment in Module 1 demonstrating accuracy, peak VRAM (-47% in `fp16`), runtime, and Joules (2.5× lower in `fp16` on Turing SM 7.5, while emulated `bf16` increases energy).
- [x] Re-run all three `01_*.ipynb` notebooks end-to-end with committed outputs; update [`module1_timeseries/README.md`](../module1_timeseries/README.md), [`module2_tabular/README.md`](../module2_tabular/README.md), [`module3_multimodal/README.md`](../module3_multimodal/README.md), [`README.md`](../README.md), [`docs/module1.html`](module1.html), [`docs/module2.html`](module2.html), [`docs/module3.html`](module3.html), and [`docs/playbook.html`](playbook.html) (adding **Model operations & licensing cheat sheet** and **Energy footprint & energy ROI**).

### Phase 2: Energy-Domain Performance Comparisons (Modules 1 & 2) *(Completed)*
- [x] Create [`requirements-extras/tsfm.txt`](../requirements-extras/tsfm.txt) (`timesfm`, `tirex-ts`, `lightgbm`; documenting separate-env isolation for `uni2ts`) and [`requirements-extras/tabular.txt`](../requirements-extras/tabular.txt) (`tabicl`, `lightgbm`, `catboost`, `xgboost`; documenting separate-env isolation for `autogluon.tabular[mitra]`).
- [x] Build [`module1_timeseries/02_tsfm_benchmark_energy.ipynb`](../module1_timeseries/02_tsfm_benchmark_energy.ipynb): compare **Chronos-Bolt**, **Chronos-2**, **TimesFM 2.5**, and **TiRex** (xLSTM) against trained **LightGBM + Ridge** baselines on Swiss load (`CH_load_actual_entsoe_transparency`) **and** German hourly solar PV (`DE_solar_generation_actual`, clean across 2017–2019 without reporting breaks). Report peak-hour/daytime error, ramp error, CRPS/Winkler scores, raw negative-prediction rate, stress slices, pretraining leakage caveats, and stylized hourly imbalance cost.
- [x] Build [`module2_tabular/02_tabular_fms_comparison.ipynb`](../module2_tabular/02_tabular_fms_comparison.ipynb): compare **TabPFN v2** and **TabICLv2** against tuned **LightGBM / CatBoost / XGBoost** on Chicago Energy Benchmarking 2021–2022, adding temporal splits (2021 → 2022 repeat vs. unseen buildings), neighborhood spatial holdout (`community_area`), portfolio aggregate bias, 80% quantile calibration, and `n = 50–2,000` scaling curves.
- [x] Re-run Phase 2 notebooks top-to-bottom with committed outputs; update [`module1_timeseries/README.md`](../module1_timeseries/README.md), [`module2_tabular/README.md`](../module2_tabular/README.md), [`README.md`](../README.md), [`docs/module1.html`](module1.html), [`docs/module2.html`](module2.html), and [`docs/playbook.html`](playbook.html).

### Phase 3: New Module 4 — Graph & Physics-Aware Grid FMs (`GridFM` / `GENCO`) *(Completed)*
- [x] Create [`requirements-extras/gridfm.txt`](../requirements-extras/gridfm.txt) (`gridfm-graphkit==0.9.0.3`, `gridfm-datakit==1.1.0`, `torch_scatter`, `torch_geometric`) pinning CUDA 12.8 PyTorch + PyG wheels and verify compatibility with the local workstation environment.
- [x] Build & execute [`module4_grid/01_gridfm_genco.ipynb`](../module4_grid/01_gridfm_genco.ipynb): load held-out IEEE 30-bus and 118-bus contingency scenarios (`splits["test"]` from `gridfm/{pf,opf}_small_case{30,118}_ieee`), evaluate 4 pretrained **GENCO Base** checkpoints (`gridfm/genco-{pf,opf}-datakit`, Apache-2.0) vs. DC baselines and classical AC/DC solvers (`runtime_data.parquet`), reporting optimality gap (`0.144%` / `1.537%` vs. DC `8.141%` / `4.812%`), power-balance residuals (`|ΔP|`), constraint violation rates (`0.00%` Vm/Pg/angle vs. soft Qg/thermal), `batch=1` vs. `batch=32` speedup (`47.6×` on 118-bus OPF), and NVML + CodeCarbon energy per solve.
- [x] Create [`module4_grid/README.md`](../module4_grid/README.md) and [`docs/module4.html`](module4.html); add Module 4 links to [`README.md`](../README.md) and the site navigation across `docs/*.html`.

### Phase 4: Deeper Multimodal / Geospatial & Unified Benchmark Hub *(Completed)*
- [x] Create [`requirements-extras/multimodal.txt`](../requirements-extras/multimodal.txt) (`open-clip-torch==4.2.2`, `timm==1.0.31`, `qwen-vl-utils==0.0.14`, with `torchvision.models.swin_v2_b` loading SatlasPretrain Sentinel-2 weights cleanly).
- [x] Build & execute [`module3_multimodal/02_geospatial_energy_fms.ipynb`](../module3_multimodal/02_geospatial_energy_fms.ipynb): compare **CLIP ViT-B/32** (87.5%) vs. **SigLIP 2 Base** (89.3%) vs. **RemoteCLIP ViT-B/32** (92.8%) vs. **SatlasPretrain Swin-V2-Base** (97.1% overall; 87.7% at 5 labels/class) on EuroSAT (including *Residential*, *Industrial*, *Highway* F1 and `5/10/50/150` few-shot curves) plus 20 cm rooftop solar PV siting (`Cyrille37/solar-panels-IGN-bdortho`: SigLIP 2 0.962 AUC / Satlas 0.880 F1).
- [x] Build & execute [`module3_multimodal/03_vlm_and_multimodal_fusion.ipynb`](../module3_multimodal/03_vlm_and_multimodal_fusion.ipynb):
  - **Part A (Modern VLM):** Compare **BLIP base**, **SmolVLM-500M-Instruct**, and **Qwen2.5-VL-3B-Instruct** on 20 cm rooftop PV JSON inspection (highlighting SmolVLM's 90% bare-roof false positives vs. Qwen2.5-VL-3B's 0.750 F1 at 193 J/roof vs. ~0.3–1.4 J/img linear probes) and synthetic EPC/utility bill extraction (100% exact across 60 fields on Qwen2.5-VL-3B vs. 76.7% on SmolVLM and 0% on BLIP).
  - **Part B (Cross-Modal Fusion):** Pair frozen zoom-19 Esri Chicago aerial footprints (600 buildings) with Module 2 tabular features via strictly leakage-free 5-fold CV (`PCA-16`), measuring zero net lift over TabPFN v2 (R² 0.182 → 0.173/0.182) due to physical indoor observability limits.
- [x] Build [`scripts/build_benchmark_summary.py`](../scripts/build_benchmark_summary.py) and [`docs/benchmarks.html`](benchmarks.html) generating cross-module leaderboards, CSV summaries (`summary_models.csv`, `summary_steps.csv`, `summary_energy.csv`), and Pareto frontier plots (`docs/assets/img/benchmarks_pareto.png`) from all 8 `benchmarks/results/*.json` files.

### Parking Lot (Future / Out of Current Scope)
- [ ] Module 5 exploration: weather/climate FMs (e.g., **Aurora** / **ClimaX**) for renewable resource downscaling, or energy-domain LLM + tool use generating validated `pandapower` / `PyPSA` networks.
