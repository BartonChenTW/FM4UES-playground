# Review Checklist (`v0.2.0` -> Next Release)

> **Status:** All four expansion phases are published live on [`main`](https://github.com/BartonChenTW/FM4UES-playground/tree/main) and the [GitHub Pages website](https://bartonchentw.github.io/FM4UES-playground/) as [`v0.2.0`](https://github.com/BartonChenTW/FM4UES-playground/releases/tag/v0.2.0).
> **Why this file exists:** The automated benchmark runs and HTML/JSON pipelines passed, while the **notebooks, pedagogical pacing, and narrative framing still need an author review pass** before cutting `v0.3.0` (or `v1.0.0`). Check items off as you review or refine each notebook.

---

## 1. Notebooks to Review (Modules 1–4)

### Module 1 — Time-Series Foundation Models
- [ ] [`module1_timeseries/01_demand_forecasting.ipynb`](../module1_timeseries/01_demand_forecasting.ipynb) · *Status: Updated in Phase 1* · ([Module 1 README](../module1_timeseries/README.md) · [Website page](module1.html))
  - Verify beginner flow and explanatory markdown around the added `fm4ues.profiler` calls and structured JSON export ([`module1_timeseries_01_demand_forecasting.json`](../benchmarks/results/module1_timeseries_01_demand_forecasting.json)).
  - Review UES metric framing: Peak-5% MAE (`170.5 MW`), Ramp MAE (`126.7 MW/h`), 80% Winkler score (`1,147.7 MW`), and the perfect-weather-forecast upper-bound caveat.
  - Check the `fp32` vs. `bf16` vs. `fp16` precision sweep commentary — confirm Turing SM 7.5 `bf16` emulation (`329.7 J`) vs. `fp16` (`80.3 J`, `2.5×` lower GPU energy) is clear for readers on Ampere/Ada GPUs.
- [ ] [`module1_timeseries/02_tsfm_benchmark_energy.ipynb`](../module1_timeseries/02_tsfm_benchmark_energy.ipynb) · *Status: New in Phase 2* · ([Module 1 README](../module1_timeseries/README.md) · [Website page](module1.html#benchmark))
  - Check transition from Notebook 01 and setup instructions for [`requirements-extras/tsfm.txt`](../requirements-extras/tsfm.txt).
  - Review contrast between smooth Swiss load (where univariate zero-shot FMs `Chronos-2`/`Chronos-Bolt`/`TiRex`/`TimesFM 2.5` beat `LightGBM`) and German solar PV (`DE_solar_generation_actual`, where weather+lag `LightGBM` `681.1 MW` and `Ridge` `781.6 MW` beat all univariate FMs `1,061.6–1,318.6 MW`).
  - Check raw negative-prediction diagnostics on solar PV, pretraining-cutoff leakage caveats, and stylized hourly imbalance-cost framing.

### Module 2 — Tabular Foundation Models
- [ ] [`module2_tabular/01_tabular_intro.ipynb`](../module2_tabular/01_tabular_intro.ipynb) · *Status: Updated in Phase 1* · ([Module 2 README](../module2_tabular/README.md) · [Website page](module2.html))
  - Verify beginner pacing after adding `fm4ues.profiler`, Model Cards, and floor-area-weighted stock energy bias (`TabPFN v2` `-1.31%` vs. Random Forest `+4.07%`).
  - Check that calling `inspect_model` after `.fit()` and pinning `ModelVersion.V2` (`Prior-Labs/TabPFN-v2-reg`, Apache-2.0 w/ attribution) are well explained.
- [ ] [`module2_tabular/02_tabular_fms_comparison.ipynb`](../module2_tabular/02_tabular_fms_comparison.ipynb) · *Status: New in Phase 2* · ([Module 2 README](../module2_tabular/README.md) · [Website page](module2.html#benchmark))
  - Review setup instructions for [`requirements-extras/tabular.txt`](../requirements-extras/tabular.txt) and split design (2022 random 5-fold CV, 2021 → 2022 temporal repeat vs. unseen buildings, and spatial `GroupKFold` by `community_area`).
  - Verify intuition for the GBDT vs. TabPFN split divergence (`LightGBM` `19.06` on repeat buildings vs. `TabPFN v2` `27.92` on unseen buildings) and `TabICLv2`'s `-5.94%` stock energy bias alongside its `80.4%` interval calibration.

### Module 3 — Multimodal, Geospatial EO & Local VLMs
- [ ] [`module3_multimodal/01_encoder_decoder.ipynb`](../module3_multimodal/01_encoder_decoder.ipynb) · *Status: Updated in Phase 1* · ([Module 3 README](../module3_multimodal/README.md) · [Website page](module3.html))
  - Check intro narrative after adding `fm4ues.profiler`, EuroSAT UES-class F1 (*Residential*, *Industrial*, *Highway*), and the encoding-vs-decoding GPU energy comparison (`CLIP ~0.50 J/img` vs. `BLIP ~5.95 J/caption`).
- [ ] [`module3_multimodal/02_geospatial_energy_fms.ipynb`](../module3_multimodal/02_geospatial_energy_fms.ipynb) · *Status: New in Phase 4* · ([Module 3 README](../module3_multimodal/README.md) · [Website page](module3.html#partD))
  - Review setup note for [`requirements-extras/multimodal.txt`](../requirements-extras/multimodal.txt) and Windows single-worker download note (`max_workers=1`) for `Cyrille37/solar-panels-IGN-bdortho`.
  - Check pedagogical clarity on `[0, 1]` Sentinel-2 RGB scaling for `SatlasPretrain Swin-V2-Base` (`97.1%` EuroSAT linear probe; `87.7%` 5-shot; `0.880` IGN PV F1), `SigLIP 2 Base` (`0.962` IGN PV AUC), and `RemoteCLIP` (`92.8%` probe vs. `32.9%` zero-shot).
  - Visually review sample selection and titles in `m3_rooftop_pv_examples.png` and `m3_geospatial_benchmark.png`.
- [ ] [`module3_multimodal/03_vlm_and_multimodal_fusion.ipynb`](../module3_multimodal/03_vlm_and_multimodal_fusion.ipynb) · *Status: New in Phase 4* · ([Module 3 README](../module3_multimodal/README.md) · [Website page](module3.html#partE))
  - Review Part A VLM audit framing (`BLIP base` vs. `SmolVLM-500M-Instruct` vs. `Qwen2.5-VL-3B-Instruct`): highlight `SmolVLM`'s `90.0%` bare-roof PV false positives vs. `Qwen2.5-VL-3B`'s `100.0%` EPC OCR exact match (`192.8 J/roof` vs. `0.3–1.4 J/img` probes).
  - Verify clarity that the 12 EPC / utility bill cards are synthetic templates testing strict JSON extraction rather than real utility PII.
  - Review Part B Chicago aerial–tabular 5-fold CV fusion (`PCA-16`, zoom-19 Esri imagery): check explanation of why `Ridge` gains modest lift (`R² 0.051 -> 0.094`) while `TabPFN v2` gains zero lift (`0.182` tabular vs. `0.173`/`0.182` fused).

### Module 4 — Physics-Aware Power Grid Foundation Models
- [ ] [`module4_grid/01_gridfm_genco.ipynb`](../module4_grid/01_gridfm_genco.ipynb) · *Status: New in Phase 3* · ([Module 4 README](../module4_grid/README.md) · [Website page](module4.html))
  - Check installation flow for [`requirements-extras/gridfm.txt`](../requirements-extras/gridfm.txt) and single-worker snapshot download handling on Windows.
  - Review contingency-ensemble explanation ($N-1$ PF vs. $N-k$ OPF up to $N-5$ / $N-10$) and AC ground-truth unit verification (`|ΔP|` sanity check).
  - Verify the core message ("fast ≠ certified feasible"): `batch=32` hits `47.6×` speedup (`3.40 ms/solve`, `366.9 mJ/solve`) with `0.00%` `Vm`/`Pg`/angle violations via `bound_with_sigmoid`, while soft bounds leave `0.84%` thermal violations and up to `204.9 MW` max nodal `|ΔP|` on 118-bus OPF.

---

## 2. Shared Package, Scripts & Dependencies

- [ ] [`fm4ues/profiler.py`](../fm4ues/profiler.py) & [`fm4ues/metrics.py`](../fm4ues/metrics.py): review ModelCard schema, Option C telemetry assumptions (50 ms NVML polling + CodeCarbon country override `"CHE"`), and CO₂e factors (CH `30`, EU `230`, US `380 gCO₂/kWh`).
- [ ] [`requirements.txt`](../requirements.txt) & `requirements-extras/`: review version pins in [`tsfm.txt`](../requirements-extras/tsfm.txt), [`tabular.txt`](../requirements-extras/tabular.txt), [`gridfm.txt`](../requirements-extras/gridfm.txt), and [`multimodal.txt`](../requirements-extras/multimodal.txt).
- [ ] [`scripts/run_notebooks.py`](../scripts/run_notebooks.py) & [`scripts/build_benchmark_summary.py`](../scripts/build_benchmark_summary.py): verify CLI flags, JSON schema assumptions, and HTML/CSV regeneration flow.
- [ ] [`CHANGELOG.md`](../CHANGELOG.md) & [`VERSION`](../VERSION): verify `v0.1.0` / `v0.2.0` release notes and future bump workflow match your repository conventions.

---

## 3. Website Pages & Figures (`docs/`)

- [ ] Check all rendered pages on `https://bartonchentw.github.io/FM4UES-playground/` (light & dark mode, header `.version-badge`, mobile wrapping):
  - [`index.html`](index.html)
  - [`module1.html`](module1.html)
  - [`module2.html`](module2.html)
  - [`module3.html`](module3.html)
  - [`module4.html`](module4.html)
  - [`playbook.html`](playbook.html)
  - [`benchmarks.html`](benchmarks.html) (auto-generated by `scripts/build_benchmark_summary.py`)
- [ ] Review key Phase 2–4 figures visually:
  - [`m1_benchmark_pareto.png`](assets/img/m1_benchmark_pareto.png)
  - [`m2_benchmark_splits.png`](assets/img/m2_benchmark_splits.png)
  - [`m3_geospatial_benchmark.png`](assets/img/m3_geospatial_benchmark.png) & [`m3_rooftop_pv_examples.png`](assets/img/m3_rooftop_pv_examples.png)
  - [`m3_vlm_audit.png`](assets/img/m3_vlm_audit.png) & [`m3_multimodal_fusion.png`](assets/img/m3_multimodal_fusion.png)
  - [`m4_grid_topology.png`](assets/img/m4_grid_topology.png) & [`m4_genco_results.png`](assets/img/m4_genco_results.png)
  - [`benchmarks_pareto.png`](assets/img/benchmarks_pareto.png)
- [ ] Decide whether module download buttons (`module3.html`, `module4.html`) should keep pointing to `blob/module1-local-gpu/...` or switch to `blob/main/...` (both point to identical commits at `v0.2.0`).

---

## 4. Workflow After You Edit Any Notebook

```bash
# 1. Re-execute an edited notebook top-to-bottom in the fm4ues kernel
python scripts/run_notebooks.py --notebook module2_tabular/01_tabular_intro.ipynb

# 2. Rebuild Benchmarks Hub CSVs, Pareto plot, and docs/benchmarks.html
python scripts/build_benchmark_summary.py

# 3. Verify imports, JSON schemas, and clean committed outputs across all 8 notebooks
python scripts/run_notebooks.py --smoke
python scripts/run_notebooks.py --audit-only
```

Once every review item above is checked off, bump [`VERSION`](../VERSION) (e.g., `0.3.0`), update [`CHANGELOG.md`](../CHANGELOG.md) and HTML header badges, push to `main`, and tag the new release.
