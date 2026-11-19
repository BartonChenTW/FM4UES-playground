# Changelog

All notable changes to **FM4UES-playground** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Author review of all 8 notebooks, narrative explanations, and visual figures tracked in [`docs/to_review.md`](docs/to_review.md).

## [0.2.0] - 2026-11-19

Major expansion across Phases 1–4 adding operational/energy telemetry, multi-model energy-domain benchmarks, power-grid foundation models, geospatial/VLM evaluation, and a generated Benchmarks Hub.

### Added
- **Version & review tracking:** [`VERSION`](VERSION) (`0.2.0`), `fm4ues.__version__`, [`CHANGELOG.md`](CHANGELOG.md), and the post-expansion review checklist [`docs/to_review.md`](docs/to_review.md).
- **Shared profiling & UES metrics package ([`fm4ues/`](fm4ues/__init__.py)):**
  - `RunProfiler` / `inspect_model()` in [`fm4ues/profiler.py`](fm4ues/profiler.py) combining 50 ms `nvidia-ml-py` per-step GPU telemetry (power, Joules/Wh, peak VRAM/RAM, FLOPs, hardware tier) with whole-notebook offline `codecarbon` estimates and CO₂e across CH (30 g/kWh), EU (230 g/kWh), and US (380 g/kWh).
  - Domain evaluation helpers in [`fm4ues/metrics.py`](fm4ues/metrics.py) (Peak-5% MAE, Ramp MAE, CRPS, Winkler score, floor-area-weighted stock energy bias, UES class F1).
- **Optional dependency isolation ([`requirements-extras/`](requirements-extras/)):** separate `tsfm.txt`, `tabular.txt`, `gridfm.txt`, and `multimodal.txt` files so the core environment stays lean.
- **New benchmark & domain notebooks:**
  - Module 1 [`02_tsfm_benchmark_energy.ipynb`](module1_timeseries/02_tsfm_benchmark_energy.ipynb): Swiss demand + German solar PV across Chronos-Bolt, Chronos-2 (+/- weather), TimesFM 2.5 (200M), TiRex (35M xLSTM), LightGBM, and Ridge.
  - Module 2 [`02_tabular_fms_comparison.ipynb`](module2_tabular/02_tabular_fms_comparison.ipynb): Chicago 2021–2022 EUI across TabPFN v2, TabICLv2, LightGBM, CatBoost, XGBoost, and Linear regression (random CV, 80% calibration, temporal repeat vs. unseen split, spatial `community_area` holdout, `n = 50–2,000` curve).
  - Module 4 [`01_gridfm_genco.ipynb`](module4_grid/01_gridfm_genco.ipynb): Physics-aware GENCO Base (`GNS_heterogeneous`, Apache-2.0) on held-out IEEE 30-bus & 118-bus AC Power Flow (`N-1`) and AC-OPF (`N-k`) contingencies vs. DC approximations and classical solver runtimes (`47.6×` batched speedup vs. AC-OPF on 118-bus, alongside soft thermal/`Qg`/`|ΔP|` feasibility caveats).
  - Module 3 [`02_geospatial_energy_fms.ipynb`](module3_multimodal/02_geospatial_energy_fms.ipynb): CLIP ViT-B/32, SigLIP 2 Base, RemoteCLIP ViT-B/32, and SatlasPretrain Swin-V2-Base on EuroSAT (zero-shot, linear probe, `5/10/50/150`-shot curves, UES class F1) and 20 cm IGN BD ORTHO rooftop PV detection.
  - Module 3 [`03_vlm_and_multimodal_fusion.ipynb`](module3_multimodal/03_vlm_and_multimodal_fusion.ipynb): BLIP base vs. SmolVLM-500M-Instruct vs. Qwen2.5-VL-3B-Instruct on rooftop PV JSON inspection and EPC/utility bill OCR (60 fields), plus 5-fold CV Chicago aerial + tabular fusion.
- **Automation, structured exports & Benchmarks Hub:**
  - [`scripts/run_notebooks.py`](scripts/run_notebooks.py) (`--smoke`, `--audit-only`, `--check-urls`, `--all`) and [`scripts/build_benchmark_summary.py`](scripts/build_benchmark_summary.py).
  - Exported JSON logs for all 8 notebooks in [`benchmarks/results/`](benchmarks/results/), rolled up into [`benchmarks/summary_models.csv`](benchmarks/summary_models.csv), [`benchmarks/summary_steps.csv`](benchmarks/summary_steps.csv), [`benchmarks/summary_energy.csv`](benchmarks/summary_energy.csv), [`docs/assets/img/benchmarks_pareto.png`](docs/assets/img/benchmarks_pareto.png), and [`docs/benchmarks.html`](docs/benchmarks.html).

### Changed
- Retrofitted [`01_demand_forecasting.ipynb`](module1_timeseries/01_demand_forecasting.ipynb) (including `fp32` / `bf16` / `fp16` precision sweep), [`01_tabular_intro.ipynb`](module2_tabular/01_tabular_intro.ipynb), and [`01_encoder_decoder.ipynb`](module3_multimodal/01_encoder_decoder.ipynb) with `fm4ues` instrumentation and JSON exports.
- Updated [`README.md`](README.md), module READMEs, and the GitHub Pages site (`docs/*.html`) with Module 4, Benchmarks Hub navigation, operational/energy tables, and a site-wide `v0.2.0` release badge.
- Merged `module1-local-gpu` into `main` so GitHub Pages reflects the complete 4-module, 8-notebook playground.

## [0.1.0] - 2026-11-18

Initial baseline playground and GitHub Pages walkthrough (`411384b`).

### Added
- Core three-module local-GPU notebooks: [`01_demand_forecasting.ipynb`](module1_timeseries/01_demand_forecasting.ipynb), [`01_tabular_intro.ipynb`](module2_tabular/01_tabular_intro.ipynb), and [`01_encoder_decoder.ipynb`](module3_multimodal/01_encoder_decoder.ipynb).
- Initial GitHub Pages documentation site under `docs/` (`index.html`, `module1.html`, `module2.html`, `module3.html`, `playbook.html`).

---

## Versioning & Release Workflow

1. Update [`VERSION`](VERSION) (using `MAJOR.MINOR.PATCH`).
2. Add the new release entry in [`CHANGELOG.md`](CHANGELOG.md) and update the header version badge across `docs/*.html` (`index.html`, `module1.html`–`module4.html`, `playbook.html`).
3. Re-generate `docs/benchmarks.html` (`python scripts/build_benchmark_summary.py`) so it embeds the new `VERSION` in its header/footer, re-run verification (`python scripts/run_notebooks.py --smoke` and `python scripts/run_notebooks.py --audit-only`), commit, push to `main`, create an annotated tag (`git tag -a vX.Y.Z -m "..."`), and push the tag (`git push origin vX.Y.Z`).
