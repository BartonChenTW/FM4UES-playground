# Module 4: Grid Foundation Models (`GridFM / GENCO`)

**Goal:** Move from tabular and time-series data to **physical network graphs**, evaluating **GENCO Base** (`GNS_heterogeneous`, Apache-2.0) on held-out AC Power Flow (PF) and AC Optimal Power Flow (AC-OPF) contingency test sets for **IEEE 30-bus** and **IEEE 118-bus** grids, and testing whether neural speed translates into certified physical feasibility.

## Notebook

1. **[`01_gridfm_genco.ipynb`](01_gridfm_genco.ipynb)** — requires `pip install -r requirements-extras/gridfm.txt`:
   - Evaluates 4 official **GENCO Base** checkpoints (`20.07 M` parameters, 12 heterogeneous GNN layers, 8 attention heads, hidden dim 48) on held-out `splits["test"]` scenarios (partitions `0..14`, `0` train/val overlap).
   - Contrasts **AC Power Flow** vs. **AC-OPF**, **DC linear approximations** (`Va_dc`, `p_mw_dc`), and **classical solver wall times** (`runtime_data.parquet`).
   - Measures electrical accuracy, economic optimality gap, nodal AC power-balance residuals (`|ΔP|`, `|ΔQ|`), operational limit violation rates, `batch=1` vs. `batch=32` speedup, and NVML + CodeCarbon energy consumption.

## Data and models

| Item | Source | Licence |
| :--- | :--- | :--- |
| GENCO Base checkpoints (`PF` & `OPF`, ~80.4–80.5 MB each) | [`gridfm/genco-pf-datakit`](https://huggingface.co/gridfm/genco-pf-datakit), [`gridfm/genco-opf-datakit`](https://huggingface.co/gridfm/genco-opf-datakit) | Apache-2.0 |
| Held-out test scenarios (`pf_small_case30_ieee`, `opf_small_case30_ieee`, `pf_small_case118_ieee`, `opf_small_case118_ieee`) | [`gridfm` HF Datasets](https://huggingface.co/gridfm/datasets) | CC-BY-4.0 / datakit terms |

All test splits in `gridfm-datakit` are **contingency ensembles** rather than intact-only benchmarks: the PF splits (`404` scenarios each) consist 100% of single contingencies ($N-1$: generator or branch outage), while OPF tests multi-outage $N-k$ regimes up to $N-5$ on 30-bus (`355` scenarios) and $N-10$ on 118-bus (`404` scenarios).

## Results (held-out `splits["test"]` scenarios from partitions `0..14`)

### 1. Accuracy, economic dispatch & physics residuals

| Task · Grid | N test | Vm MAE (pu) | Va MAE (deg) | DC Va MAE (deg) | Mean \|ΔP\| (MW) | P95 \|ΔP\| (MW) | Max \|ΔP\| (MW) | OPF \|ΔC\| (%) | DC \|ΔC\| (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **PF · IEEE 30-bus** | 404 | 0.0001 | 0.022° | 1.621° | 0.0040 | 0.0131 | 0.050 | – | – |
| **OPF · IEEE 30-bus** | 355 | 0.0018 | 0.028° | 1.771° | 0.0061 | 0.0206 | 0.368 | **0.144%** | 8.141% |
| **PF · IEEE 118-bus** | 404 | 0.0001 | 0.285° | 1.831° | 0.0883 | 0.2817 | 2.016 | – | – |
| **OPF · IEEE 118-bus** | 404 | 0.0110 | 1.497° | 3.095° | 1.4738 | 4.6520 | **204.912** | **1.537%** | 4.812% |

- **Constraint feasibility on OPF (where bounds are active):** GENCO enforces `Vm` and `Pg` limits via `bound_with_sigmoid`, yielding **0.00%** `Vm`, **0.00%** active-generation `Pg`, and **0.00%** branch-angle violation rates on both 30-bus and 118-bus OPF. Soft constraints remain nonzero: `Qg` violation rate is **0.20%** (30-bus) / **0.07%** (118-bus), and branch thermal limit (`rate_a`) exceedance is **0.55%** (`0.001 MVA` mean) on 30-bus and **0.84%** (`0.039 MVA` mean) on 118-bus (vs. **0.00%** on AC-OPF ground truth).
- **Contingency depth impact (118-bus OPF):** Mean `|ΔP|` rises from `1.084 MW` (line-intact scenarios) and `1.205 MW` ($N-1$) to `1.487 MW` (line outages) and `1.519 MW` ($N-2..10$), with economic gap moving from `1.208%` ($N-1$) to `1.593%` ($N-k$).

### 2. Solve latency, speedup & energy footprint

| Benchmark | Classical AC (ms) | Classical DC (ms) | GENCO b=1 (ms) | GENCO b=32 (ms) | Speedup vs AC (b=32) | Peak VRAM (GB) | GPU W (b=32) | mJ / solve (b=32) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **PF · IEEE 30-bus** | 1.32 | 0.45 | 74.17 | 3.043 | 0.4x | 0.128 | 54.5 | 166.37 |
| **OPF · IEEE 30-bus** | 33.57 | 6.75 | 71.48 | 3.069 | **10.9x** | 0.128 | 55.8 | 171.85 |
| **PF · IEEE 118-bus** | 4.26 | 1.40 | 72.98 | 3.186 | **1.3x** | 0.270 | 112.1 | 358.99 |
| **OPF · IEEE 118-bus** | 161.76 | 42.44 | 70.98 | 3.400 | **47.6x** | 0.270 | 107.6 | 366.86 |

- **Batching matters:** Single-shot GPU inference (`batch=1`, `71–74 ms/solve`) is kernel-launch-bound and *slower* than CPU Newton-Raphson for small PF (`1.3–4.3 ms`), whereas batched screening (`batch=32`) runs in `3.0–3.4 ms/solve` — beating even CPU DC-OPF (`12.5x` faster on 118-bus) and achieving **`47.6x` speedup over classical AC-OPF** (`161.76 ms` vs `3.40 ms`).
- **Fast ≠ certified feasible:** Because max nodal `|ΔP|` reaches `204.9 MW` and thermal/reactive bounds are soft, use GENCO for **rapid contingency screening or warm-starting AC-OPF solvers**, paired with a post-hoc feasibility verifier.

## Measured runtime & whole-notebook energy

Measured on NVIDIA Quadro RTX 6000 (`24 GB`, PyTorch `2.11.0+cu128`, `gridfm-graphkit==0.9.0.3`, cached parquets & checkpoints).

| Telemetry Scope | Duration (s) | GPU (Wh) | CPU + RAM (Wh, modeled) | Total (Wh) | CO₂ CH 30 g/kWh (mg) | CO₂ EU 230 g/kWh (mg) | CO₂ US 380 g/kWh (mg) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Profiled steps (GPU only, NVML 50 ms) | 80.1 | 0.7832 | – | 0.7832 | 23.50 | 180.14 | 297.62 |
| Whole notebook (CodeCarbon offline) | 117.6 | 0.9261 | 2.9416 | 3.8677 | 116.03 | 889.57 | 1,469.72 |

Structured results are exported to [`benchmarks/results/module4_grid_01_gridfm_genco.json`](../benchmarks/results/module4_grid_01_gridfm_genco.json) and figures to `docs/assets/img/m4_grid_topology.png` and `docs/assets/img/m4_genco_results.png`. See also the [Module 4 site page](../docs/module4.html).
