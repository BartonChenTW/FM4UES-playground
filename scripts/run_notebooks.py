#!/usr/bin/env python3
"""Execute, audit, and health-check FM4UES-playground notebooks cleanly.

Usage examples:
  python scripts/run_notebooks.py --list
  python scripts/run_notebooks.py --audit-only
  python scripts/run_notebooks.py --check-urls
  python scripts/run_notebooks.py --smoke
  python scripts/run_notebooks.py --module module4_grid
  python scripts/run_notebooks.py --all --kernel fm4ues
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTEBOOK_ORDER = [
    Path("module1_timeseries/01_demand_forecasting.ipynb"),
    Path("module1_timeseries/02_tsfm_benchmark_energy.ipynb"),
    Path("module2_tabular/01_tabular_intro.ipynb"),
    Path("module2_tabular/02_tabular_fms_comparison.ipynb"),
    Path("module3_multimodal/01_encoder_decoder.ipynb"),
    Path("module3_multimodal/02_geospatial_energy_fms.ipynb"),
    Path("module3_multimodal/03_vlm_and_multimodal_fusion.ipynb"),
    Path("module4_grid/01_gridfm_genco.ipynb"),
]

DATA_ENDPOINTS = [
    ("OPSD Time Series", "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"),
    ("Open-Meteo Archive", "https://archive-api.open-meteo.com/v1/archive?latitude=46.95&longitude=7.45&start_date=2019-01-01&end_date=2019-01-01&hourly=temperature_2m"),
    ("Chicago Benchmarking", "https://data.cityofchicago.org/resource/xq83-jr8c.csv?$limit=1"),
    ("HF Blanchon EuroSAT", "https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/README.md"),
    ("HF GridFM OPF", "https://huggingface.co/gridfm/genco-opf-datakit/resolve/main/README.md"),
    ("Esri World Imagery", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/18/97468/67282"),
]

FORBIDDEN_STRINGS = [
    "F:\\Barton",
    "F:/Barton",
    "C:\\Users",
    "C:/Users",
]
CLUTTER_PATTERNS = ["it/s]", "%|"]


def discover_notebooks(module: str | None = None, notebook: str | None = None) -> list[Path]:
    if notebook:
        p = Path(notebook)
        rel = p.relative_to(ROOT) if p.is_absolute() else p
        if not (ROOT / rel).exists():
            raise FileNotFoundError(f"Notebook not found: {rel}")
        return [rel]

    found = [p for p in NOTEBOOK_ORDER if (ROOT / p).exists()]
    # Include any extra notebooks outside canonical order cleanly
    for p in sorted(ROOT.glob("module*/*.ipynb")):
        rel = p.relative_to(ROOT)
        if ".ipynb_checkpoints" not in rel.parts and rel not in found:
            found.append(rel)

    if module:
        found = [p for p in found if p.parts[0] == module or p.parts[0].startswith(module)]
    return found


def audit_notebook(rel_path: Path) -> list[str]:
    issues: list[str] = []
    full_path = ROOT / rel_path
    nb = json.loads(full_path.read_text(encoding="utf-8"))
    has_outputs = False

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for pat in FORBIDDEN_STRINGS:
            if pat in src:
                issues.append(f"cell {idx} src contains local path fragment {pat!r}")

        outputs = cell.get("outputs", [])
        if outputs:
            has_outputs = True
        for out in outputs:
            if out.get("output_type") == "error":
                issues.append(f"cell {idx} has error {out.get('ename')}: {out.get('evalue')}")
            text = "".join(out.get("text", [])) + "".join(out.get("data", {}).get("text/plain", []))
            for pat in FORBIDDEN_STRINGS:
                if pat in text:
                    issues.append(f"cell {idx} output contains local path fragment {pat!r}")
            for pat in CLUTTER_PATTERNS:
                if pat in text:
                    issues.append(f"cell {idx} output contains progress-bar noise {pat!r}")

    if not has_outputs:
        issues.append("notebook has no committed outputs")

    expected_json = ROOT / "benchmarks" / "results" / f"{rel_path.parts[0]}_{rel_path.stem}.json"
    if not expected_json.exists():
        issues.append(f"missing structured results JSON: {expected_json.relative_to(ROOT).as_posix()}")
    else:
        try:
            json.loads(expected_json.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"invalid JSON {expected_json.name}: {exc}")

    return issues


def check_urls() -> int:
    failures = 0
    for name, url in DATA_ENDPOINTS:
        t0 = time.perf_counter()
        try:
            method = "HEAD" if "time_series_60min" in url else "GET"
            req = urllib.request.Request(url, method=method, headers={"User-Agent": "FM4UES-playground/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                ms = (time.perf_counter() - t0) * 1000.0
                print(f"OK   {resp.status:3d} ({ms:6.0f} ms) | {name:22s} | {url}")
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"FAIL ERR ({ms:6.0f} ms) | {name:22s} | {exc}")
            failures += 1
    return failures


def run_smoke() -> int:
    sys.path.insert(0, str(ROOT))
    import torch
    import fm4ues

    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    print(f"fm4ues exports: {len(fm4ues.__all__)} symbols")
    jsons = sorted((ROOT / "benchmarks" / "results").glob("*.json"))
    for jf in jsons:
        json.loads(jf.read_text(encoding="utf-8"))
    print(f"Validated {len(jsons)} JSON result file(s) under benchmarks/results/")
    return 0


def execute_notebook(rel_path: Path, kernel: str, timeout: int) -> int:
    print(f"\n=== Executing {rel_path.as_posix()} (kernel={kernel}) ===")
    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(ROOT / rel_path),
    ]
    res = subprocess.run(cmd, cwd=ROOT)
    dt = time.perf_counter() - t0
    print(f"Finished {rel_path.as_posix()} in {dt:.1f} s (exit={res.returncode})")
    return res.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit, health-check, and run FM4UES notebooks.")
    parser.add_argument("--list", action="store_true", help="List notebooks in execution order.")
    parser.add_argument("--audit-only", action="store_true", help="Audit committed notebooks & JSON exports without executing.")
    parser.add_argument("--check-urls", action="store_true", help="Check external dataset/model endpoint reachability.")
    parser.add_argument("--smoke", action="store_true", help="Quick package + GPU + JSON health check.")
    parser.add_argument("--module", type=str, default=None, help="Filter notebooks by module folder prefix (e.g. module3_multimodal).")
    parser.add_argument("--notebook", type=str, default=None, help="Execute a single notebook path.")
    parser.add_argument("--all", action="store_true", help="Execute all discovered notebooks.")
    parser.add_argument("--kernel", type=str, default="fm4ues", help="Jupyter kernel name (default: fm4ues).")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-cell execution timeout in seconds.")
    args = parser.parse_args()

    nbs = discover_notebooks(module=args.module, notebook=args.notebook)

    if args.list:
        for nb in nbs:
            print(nb.as_posix())
        return 0

    if args.smoke:
        return run_smoke()

    if args.check_urls:
        return 1 if check_urls() else 0

    if args.audit_only:
        failed = 0
        for nb in nbs:
            issues = audit_notebook(nb)
            if issues:
                print(f"FAIL {nb.as_posix()}:")
                for iss in issues:
                    print(f"  - {iss}")
                failed += 1
            else:
                print(f"OK   {nb.as_posix()}")
        return 1 if failed else 0

    if not (args.all or args.module or args.notebook):
        parser.print_help()
        return 0

    for nb in nbs:
        rc = execute_notebook(nb, args.kernel, args.timeout)
        if rc != 0:
            return rc
        issues = audit_notebook(nb)
        if issues:
            print(f"Post-execution audit warnings for {nb.as_posix()}: {issues}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
