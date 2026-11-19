#!/usr/bin/env python3
"""Aggregate benchmarks/results/*.json into CSVs, Pareto plots, and docs/benchmarks.html."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
BENCH_DIR = ROOT / "benchmarks"
IMG_PATH = ROOT / "docs" / "assets" / "img" / "benchmarks_pareto.png"
HTML_PATH = ROOT / "docs" / "benchmarks.html"
VERSION_PATH = ROOT / "VERSION"


def _read_version() -> str:
    return VERSION_PATH.read_text(encoding="utf-8").strip() if VERSION_PATH.exists() else "0.2.0"


def _esc(val: Any) -> str:
    return html.escape("-" if val is None else str(val))


def _fmt(val: Any, spec: str = ".2f", suffix: str = "") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "–"
    return f"{float(val):{spec}}{suffix}"


def load_all_results() -> dict[str, dict[str, Any]]:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON result files found in {RESULTS_DIR}")
    return {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in files}


def build_csv_exports(payloads: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows, step_rows, energy_rows = [], [], []
    for nb_key, p in payloads.items():
        for m in p.get("models", []):
            model_rows.append({"notebook": nb_key, "module": p.get("module"), **m})
        for s in p.get("steps", []):
            step_rows.append({"notebook": nb_key, "module": p.get("module"), **s})
        nv = p.get("nvml_totals") or {}
        cc = p.get("codecarbon") or {}
        tot_wh = cc.get("total_energy_wh")
        energy_rows.append(
            {
                "notebook": nb_key,
                "module": p.get("module"),
                "nvml_duration_s": nv.get("duration_s"),
                "nvml_gpu_j": nv.get("gpu_energy_j"),
                "nvml_gpu_wh": nv.get("gpu_energy_wh"),
                "nvml_co2_ch_mg": nv.get("co2_ch_mg"),
                "nvml_co2_eu_mg": nv.get("co2_eu_mg"),
                "nvml_co2_us_mg": nv.get("co2_us_mg"),
                "cc_duration_s": cc.get("duration_s"),
                "cc_gpu_wh": cc.get("gpu_energy_wh"),
                "cc_cpu_ram_wh": (
                    round((cc.get("cpu_energy_wh", 0.0) or 0.0) + (cc.get("ram_energy_wh", 0.0) or 0.0), 5)
                    if cc
                    else None
                ),
                "cc_total_wh": tot_wh,
                "cc_co2_ch_mg": round(tot_wh * 30.0, 2) if tot_wh is not None else None,
                "cc_co2_eu_mg": round(tot_wh * 230.0, 2) if tot_wh is not None else None,
                "cc_co2_us_mg": round(tot_wh * 380.0, 2) if tot_wh is not None else None,
            }
        )

    df_models = pd.DataFrame(model_rows)
    df_steps = pd.DataFrame(step_rows)
    df_energy = pd.DataFrame(energy_rows)

    df_models.to_csv(BENCH_DIR / "summary_models.csv", index=False)
    df_steps.to_csv(BENCH_DIR / "summary_steps.csv", index=False)
    df_energy.to_csv(BENCH_DIR / "summary_energy.csv", index=False)
    return df_models, df_steps, df_energy


def plot_pareto_figure(payloads: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.8))
    ax_ts, ax_tab, ax_geo, ax_grid = axes.ravel()

    # Panel A: Module 1 Swiss Load MAE vs GPU Energy (J)
    m1 = payloads["module1_timeseries_02_tsfm_benchmark_energy"]["metrics"]
    ts_points = [
        ("Chronos-Bolt", m1["Load :: Chronos-Bolt (base)"], (-12, 16), "#2563eb"),
        ("Chronos-2", m1["Load :: Chronos-2"], (-26, -28), "#0ea5e9"),
        ("Chronos-2 +weather", m1["Load :: Chronos-2 + weather"], (-25, -24), "#0d9488"),
        ("TiRex", m1["Load :: TiRex"], (16, 16), "#d97706"),
        ("TimesFM 2.5", m1["Load :: TimesFM 2.5"], (0, 14), "#dc2626"),
    ]
    for lbl, d, off, col in ts_points:
        ax_ts.scatter(d["GPU Energy (J)"], d["MAE (MW)"], s=115, color=col, edgecolor="white", lw=1.2, zorder=4)
        ax_ts.annotate(
            f"{lbl}\n{d['MAE (MW)']:.1f} MW",
            (d["GPU Energy (J)"], d["MAE (MW)"]),
            textcoords="offset points",
            xytext=off,
            ha="center",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.7},
        )
    ax_ts.axhline(m1["Load :: LightGBM (+ weather & lags)"]["MAE (MW)"], ls="--", color="#475569", lw=1.2, label=f"LightGBM ({m1['Load :: LightGBM (+ weather & lags)']['MAE (MW)']:.1f} MW, CPU)")
    ax_ts.axhline(m1["Load :: Ridge (+ weather & lags)"]["MAE (MW)"], ls=":", color="#94a3b8", lw=1.2, label=f"Ridge ({m1['Load :: Ridge (+ weather & lags)']['MAE (MW)']:.1f} MW, CPU)")
    ax_ts.set_xlabel("GPU energy across 365 daily forecasts (J)")
    ax_ts.set_ylabel("MAE on 2019 Swiss load (MW, lower is better)")
    ax_ts.set_title("A · Time-Series FMs: Accuracy vs. GPU Energy", fontweight="bold")
    ax_ts.set_xlim(50, 1380)
    ax_ts.set_ylim(230, 315)
    ax_ts.grid(alpha=0.25)
    ax_ts.legend(fontsize=8, loc="upper right")

    # Panel B: Module 2 Tabular Random CV vs Temporal Unseen MAE + VRAM bubble
    m2 = payloads["module2_tabular_02_tabular_fms_comparison"]["metrics"]
    tab_models = [
        ("Linear reg", "#94a3b8", (18, 12)),
        ("LightGBM", "#64748b", (-22, -24)),
        ("XGBoost", "#f59e0b", (18, 10)),
        ("CatBoost", "#2563eb", (18, -22)),
        ("TabICLv2", "#7c3aed", (22, 10)),
        ("TabPFN v2", "#0d9488", (-18, 14)),
    ]
    key_map = {
        "Linear reg": "Linear regression",
        "LightGBM": "LightGBM",
        "XGBoost": "XGBoost",
        "CatBoost": "CatBoost",
        "TabICLv2": "TabICLv2",
        "TabPFN v2": "TabPFN v2",
    }
    for short_n, col, off in tab_models:
        k = key_map[short_n]
        sp = m2[f"Splits :: {k}"]
        cv = m2[f"CV_2022 :: {k}"]
        vram = float(cv.get("Peak VRAM (GB)", 0.0) or 0.0)
        sz = 90 + vram * 140
        ax_tab.scatter(sp["CV Random MAE"], sp["Temporal Unseen MAE"], s=sz, color=col, alpha=0.9, edgecolor="white", lw=1.2, zorder=4)
        vtxt = "CPU" if vram < 0.05 else f"{vram:.2f}GB"
        ax_tab.annotate(
            f"{short_n} ({vtxt})",
            (sp["CV Random MAE"], sp["Temporal Unseen MAE"]),
            textcoords="offset points",
            xytext=off,
            ha="center",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.7},
        )
    ax_tab.set_xlabel("Random 5-fold CV MAE (kBtu/ft², lower is better)")
    ax_tab.set_ylabel("Unseen buildings 2022 MAE (kBtu/ft², lower is better)")
    ax_tab.set_title("B · Tabular FMs vs. GBDTs (size ~ Peak VRAM)", fontweight="bold")
    ax_tab.set_xlim(24.8, 28.4)
    ax_tab.set_ylim(27.3, 30.6)
    ax_tab.grid(alpha=0.25)

    # Panel C: Module 3 Vision Probes vs Generative VLMs on Rooftop PV F1 vs Energy (log mJ/img)
    g3 = payloads["module3_multimodal_02_geospatial_energy_fms"]
    v3 = payloads["module3_multimodal_03_vlm_and_multimodal_fusion"]["metrics"]
    c_items = [
        ("CLIP probe", g3["metrics"]["CLIP ViT-B/32"]["mj_per_img_eurosat"], g3["metrics"]["CLIP ViT-B/32"]["pv_val_f1"], (-24, 16), "#64748b"),
        ("RemoteCLIP probe", g3["metrics"]["RemoteCLIP ViT-B/32"]["mj_per_img_eurosat"], g3["metrics"]["RemoteCLIP ViT-B/32"]["pv_val_f1"], (-12, -32), "#d97706"),
        ("SigLIP 2 probe", g3["metrics"]["SigLIP 2 Base (16/224)"]["mj_per_img_eurosat"], g3["metrics"]["SigLIP 2 Base (16/224)"]["pv_val_f1"], (30, -32), "#2563eb"),
        ("Satlas probe", g3["metrics"]["SatlasPretrain Swin-V2-B"]["mj_per_img_eurosat"], g3["metrics"]["SatlasPretrain Swin-V2-B"]["pv_val_f1"], (22, 16), "#0d9488"),
        ("BLIP captioner", v3["VLM :: BLIP captioning base"]["a1_j_per_img"] * 1000.0, v3["VLM :: BLIP captioning base"]["pv_f1"], (0, 14), "#94a3b8"),
        ("SmolVLM-500M", v3["VLM :: SmolVLM-500M-Instruct"]["a1_j_per_img"] * 1000.0, v3["VLM :: SmolVLM-500M-Instruct"]["pv_f1"], (-36, -26), "#dc2626"),
        ("Qwen2.5-VL-3B", v3["VLM :: Qwen2.5-VL-3B-Instruct"]["a1_j_per_img"] * 1000.0, v3["VLM :: Qwen2.5-VL-3B-Instruct"]["pv_f1"], (-38, 14), "#7c3aed"),
    ]
    for lbl, mj, f1v, off, col in c_items:
        ax_geo.scatter(mj, f1v, s=110, color=col, edgecolor="white", lw=1.2, zorder=4)
        ax_geo.annotate(
            f"{lbl}\nF1={f1v:.2f}",
            (mj, f1v),
            textcoords="offset points",
            xytext=off,
            ha="center",
            fontsize=7.8,
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.7},
        )
    ax_geo.set_xscale("log")
    ax_geo.set_xlabel("GPU energy per image (mJ, log scale)")
    ax_geo.set_ylabel("20 cm Rooftop PV F1 score (higher is better)")
    ax_geo.set_title("C · Vision Probes vs. Generative VLMs (300–500x Energy Gap)", fontweight="bold")
    ax_geo.set_ylim(0.50, 0.96)
    ax_geo.set_xlim(90, 500000)
    ax_geo.grid(True, which="both", alpha=0.22)

    # Panel D: Module 4 GridFM Latency vs Electrical/OPF Speedup
    m4 = payloads["module4_grid_01_gridfm_genco"]["metrics"]
    grid_keys = ["GENCO Base (PF · 30-bus)", "GENCO Base (OPF · 30-bus)", "GENCO Base (PF · 118-bus)", "GENCO Base (OPF · 118-bus)"]
    g_short = ["PF-30", "OPF-30", "PF-118", "OPF-118"]
    xg = np.arange(len(grid_keys))
    w = 0.24
    ax_grid.bar(xg - w, [m4[k]["ac_solver_ms"] for k in grid_keys], width=w, color="#475569", label="Classical AC solver")
    ax_grid.bar(xg, [m4[k]["genco_batch1_ms"] for k in grid_keys], width=w, color="#f59e0b", label="GENCO (batch=1)")
    b32 = ax_grid.bar(xg + w, [m4[k]["genco_batch32_ms"] for k in grid_keys], width=w, color="#0d9488", label="GENCO (batch=32)")
    for i, bar in enumerate(b32):
        sp = m4[grid_keys[i]]["speedup_vs_ac_b32"]
        ax_grid.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.18, f"{sp:.1f}x" if sp < 10 else f"{sp:.0f}x", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0f766e")
    ax_grid.set_yscale("log")
    ax_grid.set_xticks(xg)
    ax_grid.set_xticklabels(g_short)
    ax_grid.set_ylabel("Solve time (ms / scenario, log scale)")
    ax_grid.set_title("D · GridFM / GENCO Solve Time & AC-OPF Speedup", fontweight="bold")
    ax_grid.set_ylim(0.8, 360)
    ax_grid.grid(axis="y", which="both", alpha=0.22)
    ax_grid.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_benchmarks_html(
    payloads: dict[str, dict[str, Any]],
    df_models: pd.DataFrame,
    df_energy: pd.DataFrame,
) -> str:
    ver = _esc(_read_version())
    hw = next(iter(payloads.values()))["hardware"]

    # Deduplicated representative model card rows ordered by module
    seen_cards = set()
    card_trs = []
    mod_labels = {
        "module1_timeseries": "1 · Time series",
        "module2_tabular": "2 · Tabular",
        "module3_multimodal": "3 · Multimodal",
        "module4_grid": "4 · GridFM",
    }
    for _, r in df_models.iterrows():
        key = (r["module"], r["name"])
        if key in seen_cards:
            continue
        seen_cards.add(key)
        params_txt = "–" if pd.isna(r["params_m"]) else (f"{r['params_m']:.2f} M" if r["params_m"] >= 0.01 else f"{int(r['params_total']):,}")
        disk_txt = _fmt(r["disk_mb"], ".1f")
        vram_txt = _fmt(r["peak_vram_gb"], ".2f")
        card_trs.append(
            "<tr>"
            f"<td>{_esc(mod_labels.get(r['module'], r['module']))}</td>"
            f"<td><b>{_esc(r['name'])}</b></td>"
            f"<td>{_esc(r['family'])}</td>"
            f"<td class=\"num\">{params_txt}</td>"
            f"<td><code>{_esc(r['dtype'])}</code></td>"
            f"<td class=\"num\">{disk_txt}</td>"
            f"<td class=\"num\">{vram_txt}</td>"
            f"<td>{_esc(r['hardware_tier'])}</td>"
            f"<td>{_esc(r['license'])}</td>"
            f"<td>{_esc(r['limits'] or '–')}</td>"
            "</tr>"
        )

    # Module 1 table (Swiss Load + German Solar)
    m1 = payloads["module1_timeseries_02_tsfm_benchmark_energy"]["metrics"]
    m1_rows = []
    for k, d in m1.items():
        task, _, mname = k.partition(" :: ")
        sec_col = "MAPE (%)" if task == "Load" else "nMAE (% mean)"
        sec_val = d.get("MAPE (%)") if task == "Load" else d.get("nMAE (% mean)")
        pk_val = d.get("Peak-5% MAE (MW)") if task == "Load" else d.get("Daytime MAE (MW)")
        m1_rows.append(
            "<tr>"
            f"<td>{_esc(task)}</td><td><b>{_esc(mname)}</b></td>"
            f"<td class=\"num\">{_fmt(d.get('MAE (MW)'), '.1f')}</td>"
            f"<td class=\"num\">{_fmt(sec_val, '.2f', '%')}</td>"
            f"<td class=\"num\">{_fmt(pk_val, '.1f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('Ramp MAE (MW/h)'), '.1f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('CRPS (MW)'), '.1f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('Imbalance (kEUR)'), ',.0f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('GPU Energy (J)'), '.1f')}</td>"
            "</tr>"
        )

    # Module 2 table (Splits across 2021-2022)
    m2 = payloads["module2_tabular_02_tabular_fms_comparison"]["metrics"]
    m2_rows = []
    for k, d in m2.items():
        if not k.startswith("Splits :: "):
            continue
        mname = k.split(" :: ", 1)[1]
        cv = m2.get(f"CV_2022 :: {mname}", {})
        m2_rows.append(
            "<tr>"
            f"<td><b>{_esc(mname)}</b></td>"
            f"<td class=\"num\">{_fmt(d.get('CV Random MAE'), '.2f')}</td>"
            f"<td class=\"num\">{_fmt(cv.get('R²'), '.3f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('Temporal Repeat MAE'), '.2f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('Temporal Unseen MAE'), '.2f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('Spatial Holdout MAE'), '.2f')}</td>"
            f"<td class=\"num\">{_fmt(cv.get('Stock bias (%)'), '+.2f', '%')}</td>"
            f"<td class=\"num\">{_fmt(cv.get('Peak VRAM (GB)'), '.2f')}</td>"
            "</tr>"
        )

    # Module 3 Geospatial + VLM + Fusion tables
    g3 = payloads["module3_multimodal_02_geospatial_energy_fms"]["metrics"]
    g3_rows = []
    for r in g3.values():
        fs5 = r["few_shot"]["5"]["mean"]
        fs50 = r["few_shot"]["50"]["mean"]
        g3_rows.append(
            "<tr>"
            f"<td><b>{_esc(r['name'])}</b></td>"
            f"<td class=\"num\">{_fmt(r.get('eurosat_zs_acc_pct'), '.1f', '%')}</td>"
            f"<td class=\"num\">{_fmt(r.get('eurosat_probe_acc_pct'), '.1f', '%')}</td>"
            f"<td class=\"num\">{fs5:.1f}% / {fs50:.1f}%</td>"
            f"<td class=\"num\">{_fmt(r.get('eurosat_f1_residential'), '.3f')}</td>"
            f"<td class=\"num\">{_fmt(r.get('eurosat_f1_industrial'), '.3f')}</td>"
            f"<td class=\"num\">{_fmt(r.get('pv_val_auc'), '.3f')}</td>"
            f"<td class=\"num\">{_fmt(r.get('pv_val_f1'), '.3f')}</td>"
            f"<td class=\"num\">{_fmt(r.get('mj_per_img_eurosat'), '.1f')}</td>"
            "</tr>"
        )

    v3 = payloads["module3_multimodal_03_vlm_and_multimodal_fusion"]["metrics"]
    v3_rows = []
    for k, d in v3.items():
        if not k.startswith("VLM :: "):
            continue
        v3_rows.append(
            "<tr>"
            f"<td><b>{_esc(d['name'])}</b></td>"
            f"<td class=\"num\">{_fmt(d.get('pv_acc_pct'), '.1f', '%')}</td>"
            f"<td class=\"num\">{_fmt(d.get('pv_f1'), '.3f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('bare_roof_fp_pct'), '.1f', '%')}</td>"
            f"<td class=\"num\">{_fmt(d.get('a2_json_valid_pct'), '.1f', '%')}</td>"
            f"<td class=\"num\">{_fmt(d.get('a2_field_acc_pct'), '.1f', '%')}</td>"
            f"<td class=\"num\">{_fmt(d.get('a2_hallucination_or_miss_pct'), '.1f', '%')}</td>"
            f"<td class=\"num\">{_fmt(d.get('a1_sec_per_img'), '.2f', ' s')}</td>"
            f"<td class=\"num\">{_fmt(d.get('a1_j_per_img'), '.1f')} / {_fmt(d.get('a2_j_per_card'), '.1f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('peak_vram_gb'), '.2f')}</td>"
            "</tr>"
        )

    f3_rows = []
    for k, d in v3.items():
        if not k.startswith("Fusion :: "):
            continue
        f3_rows.append(
            "<tr>"
            f"<td>{_esc(d['modality'])}</td><td><b>{_esc(d['regressor'])}</b></td>"
            f"<td class=\"num\">{_fmt(d.get('mae_kbtu_sqft'), '.2f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('r2'), '.3f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('stock_bias_pct'), '+.2f', '%')}</td>"
            "</tr>"
        )

    # Module 4 table
    m4 = payloads["module4_grid_01_gridfm_genco"]["metrics"]
    m4_rows = []
    for d in m4.values():
        m4_rows.append(
            "<tr>"
            f"<td><b>{_esc(d['label'])}</b></td>"
            f"<td class=\"num\">{int(d['n_test_scenarios'])}</td>"
            f"<td class=\"num\">{_fmt(d.get('vm_mae_pu'), '.4f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('va_mae_deg'), '.3f', '°')}</td>"
            f"<td class=\"num\">{_fmt(d.get('dp_mean_mw'), '.4f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('opt_abs_gap_pct'), '.3f', '%')}</td>"
            f"<td class=\"num\">{_fmt(d.get('dc_opt_abs_gap_pct'), '.3f', '%')}</td>"
            f"<td class=\"num\">{_fmt(d.get('ac_solver_ms'), '.2f')}</td>"
            f"<td class=\"num\">{_fmt(d.get('genco_batch32_ms'), '.2f')} ({_fmt(d.get('speedup_vs_ac_b32'), '.1f', '×')})</td>"
            f"<td class=\"num\">{_fmt(d.get('mj_per_solve_b32'), '.1f')}</td>"
            "</tr>"
        )

    # Whole-playground energy rollup
    energy_trs = []
    tot_nvml_wh = float(df_energy["nvml_gpu_wh"].fillna(0).sum())
    tot_cc_wh = float(df_energy["cc_total_wh"].fillna(0).sum())
    tot_nvml_s = float(df_energy["nvml_duration_s"].fillna(0).sum())
    tot_cc_s = float(df_energy["cc_duration_s"].fillna(0).sum())
    for _, er in df_energy.iterrows():
        energy_trs.append(
            "<tr>"
            f"<td><code>{_esc(er['notebook'])}</code></td>"
            f"<td class=\"num\">{_fmt(er['nvml_duration_s'], '.1f')}</td>"
            f"<td class=\"num\">{_fmt(er['nvml_gpu_wh'], '.4f')}</td>"
            f"<td class=\"num\">{_fmt(er['cc_duration_s'], '.1f')}</td>"
            f"<td class=\"num\">{_fmt(er['cc_total_wh'], '.4f')}</td>"
            f"<td class=\"num\">{_fmt(er['cc_co2_ch_mg'], '.1f')}</td>"
            f"<td class=\"num\">{_fmt(er['cc_co2_eu_mg'], ',.1f')}</td>"
            f"<td class=\"num\">{_fmt(er['cc_co2_us_mg'], ',.1f')}</td>"
            "</tr>"
        )
    energy_trs.append(
        "<tr class=\"best\">"
        "<td><b>All 8 notebooks combined</b></td>"
        f"<td class=\"num\"><b>{tot_nvml_s:.1f}</b></td>"
        f"<td class=\"num\"><b>{tot_nvml_wh:.4f}</b></td>"
        f"<td class=\"num\"><b>{tot_cc_s:.1f}</b></td>"
        f"<td class=\"num\"><b>{tot_cc_wh:.4f}</b></td>"
        f"<td class=\"num\"><b>{tot_cc_wh * 30.0:,.1f}</b></td>"
        f"<td class=\"num\"><b>{tot_cc_wh * 230.0:,.1f}</b></td>"
        f"<td class=\"num\"><b>{tot_cc_wh * 380.0:,.1f}</b></td>"
        "</tr>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmarks Hub · FM4UES</title>
<meta name="description" content="Unified cross-module benchmark leaderboard, operational footprint, Pareto frontiers, and energy cost generated from benchmarks/results/*.json.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css">
<script src="assets/site.js"></script>
</head>
<body>

<header class="site-header">
  <div class="inner">
    <div class="brand-group">
      <a class="brand" href="index.html">FM4UES <span>playground</span></a>
      <a class="version-badge" href="https://github.com/BartonChenTW/FM4UES-playground/releases/tag/v{ver}" title="Release v{ver} — changelog &amp; tags">v{ver}</a>
    </div>
    <nav class="site-nav">
      <a href="index.html">Home</a>
      <a href="module1.html">1 · Time series</a>
      <a href="module2.html">2 · Tabular</a>
      <a href="module3.html">3 · Multimodal</a>
      <a href="module4.html">4 · GridFM</a>
      <a href="playbook.html">Playbook</a>
      <a href="benchmarks.html" aria-current="page">Benchmarks</a>
      <a class="gh" href="https://github.com/BartonChenTW/FM4UES-playground">GitHub</a>
      <button class="theme-toggle" type="button" aria-label="Toggle dark mode">◐</button>
    </nav>
  </div>
</header>

<main>
<div class="prose">

<div class="hero">
  <p class="eyebrow">Auto-generated from benchmarks/results/*.json</p>
  <h1>Cross-Module Energy &amp; Operational Benchmark Hub</h1>
  <p class="lede">
    Every table and Pareto plot on this page is generated directly by <code>scripts/build_benchmark_summary.py</code> from executed
    notebook telemetry (<b>{len(payloads)}</b> notebooks, <b>{len(card_trs)}</b> unique model entries, <b>{len(df_models)}</b> logged cards,
    and <b>{tot_nvml_wh:.2f} Wh</b> total measured GPU electricity).
  </p>
  <div class="actions">
    <a class="btn primary" href="../benchmarks/summary_models.csv">Download summary_models.csv</a>
    <a class="btn" href="../benchmarks/summary_energy.csv">Download summary_energy.csv</a>
  </div>
</div>

<dl class="facts">
  <div><dt>Workstation</dt><dd>{_esc(hw.get('gpu_name'))} ({_fmt(hw.get('vram_total_gb'), '.1f')} GB VRAM, SM {_esc(hw.get('compute_capability'))})</dd></div>
  <div><dt>Stack</dt><dd>Python {_esc(hw.get('python'))} · PyTorch {_esc(hw.get('pytorch'))}</dd></div>
  <div><dt>Telemetry</dt><dd>NVML 50 ms GPU energy + CodeCarbon offline system model</dd></div>
  <div><dt>Grand total</dt><dd>{tot_nvml_wh:.2f} Wh GPU · {tot_cc_wh:.2f} Wh whole-system across {len(payloads)} notebooks</dd></div>
</dl>

<nav class="toc">
  <b>Sections</b>
  <a href="#pareto">Pareto frontiers</a>
  <a href="#cards">Unified model cards</a>
  <a href="#tsfm">1 · Time series</a>
  <a href="#tabular">2 · Tabular</a>
  <a href="#multimodal">3 · Geospatial, VLMs &amp; fusion</a>
  <a href="#grid">4 · GridFM</a>
  <a href="#energy">Energy &amp; CO₂e rollup</a>
</nav>

<section id="pareto">
  <h2>Cross-module accuracy vs. energy &amp; latency Pareto frontiers</h2>
  <figure>
    <div class="frame"><img src="assets/img/benchmarks_pareto.png" alt="Four-panel cross-module Pareto chart covering Time-Series FMs, Tabular FMs vs GBDTs, Vision Probes vs VLMs, and GridFM GENCO solve speedups." loading="lazy"></div>
    <figcaption>Synthesized directly from all 8 executed notebooks: covariates matter more than zero-shot TSFM scaling (A), TabPFN v2 leads unseen buildings at 0.36 GB VRAM (B), frozen EO probes beat generative VLMs by ~300–500× lower energy on rooftop PV (C), and batched GENCO accelerates 118-bus AC-OPF 47.6× (D).</figcaption>
  </figure>
</section>

<section id="cards">
  <h2>Unified model cards &amp; hardware requirements</h2>
  <div class="table-wrap"><table>
    <tr><th>Module</th><th>Model</th><th>Family</th><th class="num">Params</th><th>Precision</th><th class="num">Disk (MB)</th><th class="num">Peak VRAM (GB)</th><th>Min Hardware</th><th>Licence</th><th>Hard Limits</th></tr>
    {"".join(card_trs)}
  </table></div>
</section>

<section id="tsfm">
  <h2>1 · Time-series foundation models (Swiss Load &amp; German Solar PV, 2019)</h2>
  <div class="table-wrap"><table>
    <tr><th>Task</th><th>Model</th><th class="num">MAE (MW)</th><th class="num">MAPE / nMAE</th><th class="num">Peak / Day MAE</th><th class="num">Ramp MAE</th><th class="num">CRPS (MW)</th><th class="num">Imbalance (kEUR)</th><th class="num">GPU J</th></tr>
    {"".join(m1_rows)}
  </table></div>
</section>

<section id="tabular">
  <h2>2 · Tabular FMs vs. GBDTs across validation splits (Chicago 2021–2022)</h2>
  <div class="table-wrap"><table>
    <tr><th>Model</th><th class="num">CV MAE</th><th class="num">CV R²</th><th class="num">Temporal Repeat MAE</th><th class="num">Temporal Unseen MAE</th><th class="num">Spatial Holdout MAE</th><th class="num">Stock Bias</th><th class="num">VRAM (GB)</th></tr>
    {"".join(m2_rows)}
  </table></div>
</section>

<section id="multimodal">
  <h2>3 · Geospatial encoders, local VLMs &amp; aerial–tabular fusion</h2>
  <h3>3a · Geospatial &amp; remote-sensing encoders (EuroSAT 5,400 + IGN 20 cm PV)</h3>
  <div class="table-wrap"><table>
    <tr><th>Backbone</th><th class="num">EuroSAT ZS</th><th class="num">EuroSAT Probe</th><th class="num">Few-Shot (5 / 50 cls)</th><th class="num">Res F1</th><th class="num">Ind F1</th><th class="num">PV Val AUC</th><th class="num">PV Val F1</th><th class="num">mJ / img</th></tr>
    {"".join(g3_rows)}
  </table></div>

  <h3>3b · Local VLM structured extraction &amp; hallucination audit</h3>
  <div class="table-wrap"><table>
    <tr><th>VLM</th><th class="num">A1 PV Acc</th><th class="num">A1 PV F1</th><th class="num">Bare-Roof FP</th><th class="num">A2 JSON Valid</th><th class="num">A2 EPC Exact</th><th class="num">A2 Miss/Halluc</th><th class="num">s / roof</th><th class="num">J / roof | card</th><th class="num">VRAM (GB)</th></tr>
    {"".join(v3_rows)}
  </table></div>

  <h3>3c · Aerial + tabular multimodal fusion (600 Chicago buildings, 5-fold CV)</h3>
  <div class="table-wrap"><table>
    <tr><th>Modality</th><th>Regressor</th><th class="num">MAE (kBtu/ft²)</th><th class="num">R²</th><th class="num">Stock Bias</th></tr>
    {"".join(f3_rows)}
  </table></div>
</section>

<section id="grid">
  <h2>4 · GridFM / GENCO power-grid surrogates (IEEE 30-bus &amp; 118-bus contingencies)</h2>
  <div class="table-wrap"><table>
    <tr><th>Benchmark</th><th class="num">N test</th><th class="num">Vm MAE (pu)</th><th class="num">Va MAE</th><th class="num">Mean |ΔP| (MW)</th><th class="num">GENCO |ΔC|</th><th class="num">DC |ΔC|</th><th class="num">AC Solver (ms)</th><th class="num">GENCO b=32 (Speedup)</th><th class="num">mJ / solve</th></tr>
    {"".join(m4_rows)}
  </table></div>
</section>

<section id="energy">
  <h2>Per-notebook energy &amp; grid carbon rollup (NVML + CodeCarbon)</h2>
  <div class="table-wrap"><table>
    <tr><th>Notebook</th><th class="num">NVML Time (s)</th><th class="num">NVML GPU (Wh)</th><th class="num">CC Time (s)</th><th class="num">CC System (Wh)</th><th class="num">CO₂ CH (mg)</th><th class="num">CO₂ EU (mg)</th><th class="num">CO₂ US (mg)</th></tr>
    {"".join(energy_trs)}
  </table></div>
</section>

<nav class="pager">
  <a href="playbook.html"><small>Previous</small>← Playbook</a>
  <a href="index.html" style="text-align:right"><small>Back to</small>Home</a>
</nav>

</div>
</main>

<footer>
  <div class="inner">
    Generated by <code>scripts/build_benchmark_summary.py</code> · FM4UES playground ·
    Release <a href="https://github.com/BartonChenTW/FM4UES-playground/releases/tag/v{ver}">v{ver}</a> ·
    <a href="https://github.com/BartonChenTW/FM4UES-playground/blob/main/CHANGELOG.md">Changelog</a> ·
    <a href="https://github.com/BartonChenTW/FM4UES-playground/blob/main/docs/to_review.md">Review checklist</a>.
  </div>
</footer>

</body>
</html>
"""


def main() -> int:
    payloads = load_all_results()
    df_models, df_steps, df_energy = build_csv_exports(payloads)
    plot_pareto_figure(payloads)
    HTML_PATH.write_text(render_benchmarks_html(payloads, df_models, df_energy), encoding="utf-8")
    print(f"Loaded {len(payloads)} JSON files -> wrote CSVs, {IMG_PATH.relative_to(ROOT).as_posix()}, and {HTML_PATH.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
