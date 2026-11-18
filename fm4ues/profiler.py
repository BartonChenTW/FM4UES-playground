"""Lightweight model, hardware, compute and energy profiler for FM4UES notebooks."""

from __future__ import annotations

import inspect
import json
import logging
import os
import platform
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import psutil
import torch

try:
    import pynvml

    _NVML_IMPORT_OK = True
except ImportError:
    pynvml = None
    _NVML_IMPORT_OK = False

try:
    from codecarbon import OfflineEmissionsTracker

    _CODECARBON_IMPORT_OK = True
except ImportError:
    OfflineEmissionsTracker = None
    _CODECARBON_IMPORT_OK = False


# Reference grid carbon intensities (gCO2eq / kWh) for sensitivity comparison
CARBON_INTENSITY_G_PER_KWH: dict[str, float] = {
    "CH": 30.0,
    "EU": 230.0,
    "US": 380.0,
}

# Illustrative single-GPU cloud price (USD / hour) for order-of-magnitude cost context
REFERENCE_GPU_USD_PER_HOUR: float = 1.00


def _nvml_is_usable() -> bool:
    if not (_NVML_IMPORT_OK and torch.cuda.is_available()):
        return False
    try:
        pynvml.nvmlInit()
        pynvml.nvmlDeviceGetHandleByIndex(0)
        return True
    except Exception:
        return False
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_hardware_info(device: str) -> dict[str, Any]:
    """Collect non-sensitive hardware and software details for reproducibility."""
    cuda_active = device == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 0
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "device": "cuda" if cuda_active else "cpu",
        "gpu_name": None,
        "vram_total_gb": None,
        "compute_capability": None,
        "gpu_power_limit_w": None,
        "fp16_native": None,
        "bf16_native": None,
        "nvml_available": _nvml_is_usable(),
        "codecarbon_available": _CODECARBON_IMPORT_OK,
    }
    if cuda_active:
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(props.total_memory / 1e9, 2)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        info["fp16_native"] = cap >= (7, 0)
        info["bf16_native"] = cap >= (8, 0)
        if info["nvml_available"]:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info["gpu_power_limit_w"] = round(
                    pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0, 1
                )
            except Exception:
                pass
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
    return info


def print_hardware_banner(device: str, cache_dir: Path, root_dir: Path) -> dict[str, Any]:
    """Print a compact hardware & telemetry summary and return the hardware dict."""
    hw = get_hardware_info(device)
    print(f"Python   : {hw['python']} | OS: {hw['os']} | RAM: {hw['ram_total_gb']:.0f} GB")
    print(f"PyTorch  : {hw['pytorch']}")
    if hw["device"] == "cuda":
        bf16_note = "native bf16" if hw["bf16_native"] else "native fp16, emulated bf16"
        tdp_note = f", TDP {hw['gpu_power_limit_w']:.0f} W" if hw["gpu_power_limit_w"] else ""
        print(
            f"Device   : cuda ({hw['gpu_name']}, {hw['vram_total_gb']:.1f} GB VRAM, "
            f"SM {hw['compute_capability']}{tdp_note}, {bf16_note})"
        )
    else:
        print("Device   : cpu")
    nvml_txt = "NVML GPU power (50 ms)" if hw["nvml_available"] else "NVML unavailable"
    cc_txt = "CodeCarbon offline (system)" if hw["codecarbon_available"] else "CodeCarbon unavailable"
    print(f"Telemetry: {nvml_txt} + {cc_txt}")
    print(f"Data dir : {cache_dir.relative_to(root_dir)}")
    return hw


class _StepResourceSampler:
    """Polls host RAM and (when CUDA is active) NVIDIA GPU power & utilisation."""

    def __init__(self, device: str, interval_s: float = 0.05) -> None:
        self.use_nvml = device == "cuda" and _NVML_IMPORT_OK and torch.cuda.is_available()
        self.interval_s = interval_s
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._nvml_active = False
        self._samples: list[tuple[float, float | None, float | None]] = []
        self.peak_rss_bytes: int = 0

    def _read_once(self) -> None:
        now = time.perf_counter()
        try:
            rss = self._proc.memory_info().rss
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
        except Exception:
            pass
        power_w: float | None = None
        util_pct: float | None = None
        if self._nvml_active and self._handle is not None:
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                util_pct = float(pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            except Exception:
                pass
        self._samples.append((now, power_w, util_pct))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._read_once()

    def start(self) -> None:
        if self.use_nvml:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_active = True
            except Exception:
                self._nvml_active = False
        self._read_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._read_once()
        if self._nvml_active:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        power_pts = [(t, p) for t, p, _ in self._samples if p is not None]
        util_pts = [u for _, _, u in self._samples if u is not None]

        energy_j: float | None = None
        avg_power_w: float | None = None
        peak_power_w: float | None = None
        if len(power_pts) >= 2:
            e_j = 0.0
            for (t0, p0), (t1, p1) in zip(power_pts, power_pts[1:]):
                dt = max(0.0, t1 - t0)
                e_j += 0.5 * (p0 + p1) * dt
            total_dt = max(1e-9, power_pts[-1][0] - power_pts[0][0])
            energy_j = e_j
            avg_power_w = e_j / total_dt
            peak_power_w = max(p for _, p in power_pts)

        return {
            "peak_ram_gb": self.peak_rss_bytes / 1e9 if self.peak_rss_bytes else None,
            "avg_power_w": avg_power_w,
            "peak_power_w": peak_power_w,
            "gpu_util_pct": (sum(util_pts) / len(util_pts)) if util_pts else None,
            "energy_j": energy_j,
            "energy_wh": (energy_j / 3600.0) if energy_j is not None else None,
        }


@dataclass
class ModelCard:
    """Operational profile and licence metadata for a model."""

    name: str
    family: str = ""
    modality: str = ""
    source: str = ""
    license: str = ""
    gated: bool = False
    limits: str = ""
    params_total: int | None = None
    params_m: float | None = None
    trainable_params: int | None = None
    dtype: str = ""
    weight_memory_mb: float | None = None
    disk_mb: float | None = None
    peak_vram_gb: float | None = None
    hardware_tier: str = "CPU-ok"
    notes: str = ""


def _unwrap_torch_modules(model: Any) -> list[torch.nn.Module]:
    """Return underlying PyTorch nn.Module(s) for wrappers like Chronos or TabPFN."""
    if model is None:
        return []
    if isinstance(model, torch.nn.Module):
        return [model]
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        return [model.model]
    if hasattr(model, "model_") and isinstance(model.model_, torch.nn.Module):
        return [model.model_]
    if hasattr(model, "models_") and isinstance(model.models_, (list, tuple)):
        mods = [m for m in model.models_ if isinstance(m, torch.nn.Module)]
        if mods:
            return mods
    return []


def _hf_cache_dir() -> Path:
    hub_env = os.environ.get("HF_HUB_CACHE")
    if hub_env:
        return Path(hub_env)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_disk_mb(hf_repo_id: str) -> float | None:
    """Compute cached disk footprint (MB) for one HF repo without double-counting symlinks."""
    repo_dir = _hf_cache_dir() / f"models--{hf_repo_id.replace('/', '--')}"
    if not repo_dir.exists():
        return None
    seen_inodes: set[tuple[int, int]] = set()
    seen_paths: set[Path] = set()
    total_bytes = 0
    for sub in ("blobs", "snapshots"):
        base = repo_dir / sub
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            try:
                resolved = f.resolve()
                st = resolved.stat()
                ino_key = (st.st_dev, st.st_ino)
                if st.st_ino and ino_key in seen_inodes:
                    continue
                if resolved in seen_paths:
                    continue
                if st.st_ino:
                    seen_inodes.add(ino_key)
                seen_paths.add(resolved)
                total_bytes += st.st_size
            except OSError:
                continue
    return round(total_bytes / 1e6, 1) if total_bytes > 0 else None


def _path_disk_mb(path: Path | str | None) -> float | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return round(p.stat().st_size / 1e6, 1)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1e6, 1) if total > 0 else None


def _recommend_hardware_tier(
    peak_vram_gb: float | None,
    weight_memory_mb: float | None,
    cpu_viable: bool,
    dtype: str,
) -> str:
    if dtype == "cpu/sklearn":
        return "CPU-ok"
    if cpu_viable and (weight_memory_mb or 0.0) < 10.0 and (peak_vram_gb or 0.0) < 0.2:
        return "CPU-ok"
    est_gb = max(peak_vram_gb or 0.0, ((weight_memory_mb or 0.0) / 1000.0) * 1.5)
    if est_gb <= 4.0:
        return "GPU <= 4 GB"
    if est_gb <= 8.0:
        return "GPU <= 8 GB"
    if est_gb <= 24.0:
        return "GPU 16-24 GB"
    return "GPU > 24 GB"


def inspect_model(
    model: Any,
    *,
    name: str,
    family: str = "",
    modality: str = "",
    source: str = "",
    license: str = "",
    gated: bool = False,
    limits: str = "",
    hf_repo_id: str | None = None,
    disk_path: Path | str | None = None,
    disk_mb: float | None = None,
    cpu_viable: bool = True,
    notes: str = "",
) -> ModelCard:
    """Inspect a PyTorch, Hugging Face or scikit-learn model into a ModelCard."""
    modules = _unwrap_torch_modules(model)
    params_total: int | None = None
    params_m: float | None = None
    trainable_params: int | None = None
    dtype_str = "cpu/sklearn"
    weight_memory_mb: float | None = None

    if modules:
        all_params = [p for m in modules for p in m.parameters()]
        params_total = sum(p.numel() for p in all_params)
        trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
        params_m = round(params_total / 1e6, 2)
        dtypes = sorted({str(p.dtype).replace("torch.", "") for p in all_params})
        dtype_str = "/".join(dtypes) if dtypes else "unknown"
        weight_bytes = sum(p.numel() * p.element_size() for p in all_params)
        buf_bytes = sum(b.numel() * b.element_size() for m in modules for b in m.buffers())
        weight_memory_mb = round((weight_bytes + buf_bytes) / 1e6, 1)

    resolved_disk = _path_disk_mb(disk_path)
    if resolved_disk is None and hf_repo_id:
        resolved_disk = _hf_repo_disk_mb(hf_repo_id)
    if resolved_disk is None:
        resolved_disk = disk_mb

    tier = _recommend_hardware_tier(None, weight_memory_mb, cpu_viable, dtype_str)

    return ModelCard(
        name=name,
        family=family,
        modality=modality,
        source=source,
        license=license,
        gated=gated,
        limits=limits,
        params_total=params_total,
        params_m=params_m,
        trainable_params=trainable_params,
        dtype=dtype_str,
        weight_memory_mb=weight_memory_mb,
        disk_mb=resolved_disk,
        hardware_tier=tier,
        notes=notes,
    )


def estimate_flops(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float | None:
    """Best-effort FLOP count for one representative PyTorch forward pass."""
    try:
        logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
        from torch.utils.flop_counter import FlopCounterMode

        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                fn(*args, **kwargs)
        return float(fc.get_total_flops())
    except Exception:
        return None


class RunProfiler:
    """Records step timing, VRAM, RAM, NVML GPU energy and optional CodeCarbon run totals."""

    def __init__(
        self,
        module: str,
        notebook: str,
        device: str,
        cache_dir: Path,
        results_dir: Path,
        *,
        enable_codecarbon: bool = True,
        country_iso_code: str = "CHE",
        codecarbon_interval_s: float = 5.0,
    ) -> None:
        self.module = module
        self.notebook = notebook
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path(results_dir)
        self.enable_codecarbon = enable_codecarbon and _CODECARBON_IMPORT_OK
        self.country_iso_code = country_iso_code
        self.codecarbon_interval_s = codecarbon_interval_s

        self.hardware = get_hardware_info(device)
        self.timings: dict[str, float] = {}
        self.steps: list[dict[str, Any]] = []
        self.models: dict[str, ModelCard] = {}
        self.model_cpu_viable: dict[str, bool] = {}
        self.metrics: dict[str, dict[str, Any]] = {}
        self.codecarbon_summary: dict[str, Any] | None = None
        self._tracker: Any = None
        self._run_started = False

    def start_run(self) -> None:
        """Start the optional whole-notebook CodeCarbon offline tracker."""
        if not self.enable_codecarbon or self._run_started:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cc_logger = logging.getLogger("codecarbon")
            cc_logger.setLevel(logging.CRITICAL)
            cc_logger.propagate = False

            base_sig = inspect.signature(OfflineEmissionsTracker.__mro__[1].__init__)
            valid = set(base_sig.parameters.keys())
            kwargs: dict[str, Any] = {
                "country_iso_code": self.country_iso_code,
                "project_name": f"{self.module}_{self.notebook}",
                "measure_power_secs": self.codecarbon_interval_s,
                "output_dir": str(self.cache_dir),
                "output_file": f"{self.module}_codecarbon.csv",
                "save_to_file": True,
                "log_level": "error",
                "allow_multiple_runs": True,
            }
            filtered = {k: v for k, v in kwargs.items() if k in valid or k == "country_iso_code"}
            self._tracker = OfflineEmissionsTracker(**filtered)
            self._tracker.start()
            self._run_started = True
        except Exception:
            self._tracker = None
            self._run_started = False

    def stop_run(self) -> dict[str, Any] | None:
        """Stop the CodeCarbon tracker and store whole-notebook system estimates."""
        if not self._run_started or self._tracker is None:
            return self.codecarbon_summary
        try:
            self._tracker.stop()
            data = getattr(self._tracker, "final_emissions_data", None)
            if data is not None:
                self.codecarbon_summary = {
                    "duration_s": round(float(data.duration), 2),
                    "country_iso_code": data.country_iso_code,
                    "gpu_power_w": round(float(data.gpu_power), 2),
                    "cpu_power_w": round(float(data.cpu_power), 2),
                    "ram_power_w": round(float(data.ram_power), 2),
                    "gpu_energy_wh": round(float(data.gpu_energy) * 1000.0, 5),
                    "cpu_energy_wh": round(float(data.cpu_energy) * 1000.0, 5),
                    "ram_energy_wh": round(float(data.ram_energy) * 1000.0, 5),
                    "total_energy_wh": round(float(data.energy_consumed) * 1000.0, 5),
                    "emissions_g_co2eq": round(float(data.emissions) * 1000.0, 5),
                    "note": (
                        "GPU measured via NVML; on Windows, CPU uses TDP x load "
                        "and RAM uses a capacity heuristic (~3 W per 8 GB)."
                    ),
                }
        except Exception:
            self.codecarbon_summary = None
        finally:
            self._run_started = False
            self._tracker = None
        return self.codecarbon_summary

    def register_model(self, card: ModelCard, *, cpu_viable: bool = True) -> ModelCard:
        """Register or update a ModelCard in the profiler."""
        self.models[card.name] = card
        self.model_cpu_viable[card.name] = cpu_viable
        return card

    def record_metrics(self, model_name: str, metrics: dict[str, Any]) -> None:
        """Store evaluation metrics (MAE, MAPE, F1, etc.) for a model or variant."""
        clean = {
            k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for k, v in metrics.items()
        }
        self.metrics[model_name] = clean

    @contextmanager
    def step(
        self,
        name: str,
        *,
        n_items: int | None = None,
        item_unit: str = "item",
        model_name: str | None = None,
    ):
        """Measure wall-clock latency, peak VRAM/RAM, GPU power and energy for one block."""
        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        sampler = _StepResourceSampler(self.device)
        sampler.start()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            res = sampler.stop()

        peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if use_cuda else None
        peak_vram_res_gb = (torch.cuda.max_memory_reserved() / 1e9) if use_cuda else None

        ms_per_item = (elapsed * 1000.0 / n_items) if n_items else None
        items_per_sec = (n_items / elapsed) if (n_items and elapsed > 0) else None
        energy_j = res["energy_j"]
        energy_wh = res["energy_wh"]
        mj_per_item = (energy_j * 1000.0 / n_items) if (energy_j is not None and n_items) else None
        usd_per_1m = (
            (elapsed / 3600.0) * REFERENCE_GPU_USD_PER_HOUR * (1_000_000.0 / n_items)
            if (use_cuda and n_items)
            else None
        )

        record = {
            "step": name,
            "model_name": model_name,
            "seconds": round(elapsed, 3),
            "n_items": n_items,
            "item_unit": item_unit if n_items else None,
            "ms_per_item": round(ms_per_item, 3) if ms_per_item is not None else None,
            "items_per_sec": round(items_per_sec, 1) if items_per_sec is not None else None,
            "peak_vram_gb": round(peak_vram_gb, 3) if peak_vram_gb is not None else None,
            "peak_vram_reserved_gb": (
                round(peak_vram_res_gb, 3) if peak_vram_res_gb is not None else None
            ),
            "peak_ram_gb": (
                round(res["peak_ram_gb"], 2) if res["peak_ram_gb"] is not None else None
            ),
            "avg_power_w": (
                round(res["avg_power_w"], 1) if res["avg_power_w"] is not None else None
            ),
            "peak_power_w": (
                round(res["peak_power_w"], 1) if res["peak_power_w"] is not None else None
            ),
            "gpu_util_pct": (
                round(res["gpu_util_pct"], 1) if res["gpu_util_pct"] is not None else None
            ),
            "energy_j": round(energy_j, 2) if energy_j is not None else None,
            "energy_wh": round(energy_wh, 5) if energy_wh is not None else None,
            "mj_per_item": round(mj_per_item, 2) if mj_per_item is not None else None,
            "co2_ch_mg": (
                round(energy_wh * CARBON_INTENSITY_G_PER_KWH["CH"], 3)
                if energy_wh is not None
                else None
            ),
            "co2_eu_mg": (
                round(energy_wh * CARBON_INTENSITY_G_PER_KWH["EU"], 3)
                if energy_wh is not None
                else None
            ),
            "co2_us_mg": (
                round(energy_wh * CARBON_INTENSITY_G_PER_KWH["US"], 3)
                if energy_wh is not None
                else None
            ),
            "usd_per_1m_items": round(usd_per_1m, 3) if usd_per_1m is not None else None,
            "flops_per_run": None,
        }
        self.timings[name] = elapsed
        self.steps.append(record)

        parts = [f"[profile] {name}: {elapsed:.1f} s"]
        if peak_vram_gb is not None and peak_vram_gb > 0:
            parts.append(f"VRAM {peak_vram_gb:.2f} GB")
        if energy_j is not None and res["avg_power_w"] is not None:
            parts.append(f"GPU {res['avg_power_w']:.0f} W ({energy_j:.1f} J)")
        if ms_per_item is not None:
            parts.append(f"{ms_per_item:.2f} ms/{item_unit}")
        print(" | ".join(parts))

    def record_step_flops(self, step_name: str, total_flops: float | None) -> None:
        """Attach an estimated FLOP count to an already-recorded step."""
        if total_flops is None:
            return
        for s in reversed(self.steps):
            if s["step"] == step_name:
                s["flops_per_run"] = float(total_flops)
                break

    def _sync_model_vram(self) -> None:
        for name, card in self.models.items():
            vrams = [
                s["peak_vram_gb"]
                for s in self.steps
                if s["model_name"] == name and s["peak_vram_gb"] is not None
            ]
            if vrams:
                card.peak_vram_gb = round(max(vrams), 3)
            card.hardware_tier = _recommend_hardware_tier(
                card.peak_vram_gb,
                card.weight_memory_mb,
                self.model_cpu_viable.get(name, True),
                card.dtype,
            )

    def model_card_table(self) -> pd.DataFrame:
        """Return a display-friendly Model Card table for all registered models."""
        self._sync_model_vram()
        rows = []
        for card in self.models.values():
            rows.append(
                {
                    "Model": card.name,
                    "Params (M)": card.params_m,
                    "Precision": card.dtype,
                    "Disk (MB)": card.disk_mb,
                    "Peak VRAM (GB)": card.peak_vram_gb,
                    "Min Hardware": card.hardware_tier,
                    "Licence": card.license,
                    "Limits": card.limits,
                }
            )
        return pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()

    def summary_table(self) -> pd.DataFrame:
        """Return the per-step operational summary table (time, memory, power, energy, CO2)."""
        if not self.steps:
            return pd.DataFrame()
        rows = []
        for s in self.steps:
            rows.append(
                {
                    "Step": s["step"],
                    "Time (s)": round(s["seconds"], 1),
                    "Peak VRAM (GB)": (
                        round(s["peak_vram_gb"], 2) if s["peak_vram_gb"] is not None else None
                    ),
                    "Avg Power (W)": s["avg_power_w"],
                    "GPU Energy (J)": (
                        round(s["energy_j"], 1) if s["energy_j"] is not None else None
                    ),
                    "mJ / item": s["mj_per_item"],
                    "ms / item": (
                        round(s["ms_per_item"], 2) if s["ms_per_item"] is not None else None
                    ),
                    "CO2 CH (mg)": (
                        round(s["co2_ch_mg"], 2) if s["co2_ch_mg"] is not None else None
                    ),
                    "CO2 EU (mg)": (
                        round(s["co2_eu_mg"], 2) if s["co2_eu_mg"] is not None else None
                    ),
                }
            )
        df = pd.DataFrame(rows).set_index("Step")
        total_time = round(sum(s["seconds"] for s in self.steps), 1)
        gpu_steps = [s for s in self.steps if s["energy_j"] is not None]
        total_j = round(sum(s["energy_j"] for s in gpu_steps), 1) if gpu_steps else None
        total_wh = (sum(s["energy_j"] for s in gpu_steps) / 3600.0) if gpu_steps else None
        gpu_dt = sum(s["seconds"] for s in gpu_steps)
        max_vram = max(
            (s["peak_vram_gb"] for s in self.steps if s["peak_vram_gb"] is not None),
            default=None,
        )
        df.loc["Total (profiled steps)"] = {
            "Time (s)": total_time,
            "Peak VRAM (GB)": round(max_vram, 2) if max_vram is not None else None,
            "Avg Power (W)": round(total_j / max(1e-9, gpu_dt), 1) if total_j is not None else None,
            "GPU Energy (J)": total_j,
            "mJ / item": None,
            "ms / item": None,
            "CO2 CH (mg)": (
                round(total_wh * CARBON_INTENSITY_G_PER_KWH["CH"], 2)
                if total_wh is not None
                else None
            ),
            "CO2 EU (mg)": (
                round(total_wh * CARBON_INTENSITY_G_PER_KWH["EU"], 2)
                if total_wh is not None
                else None
            ),
        }
        return df

    def energy_summary_table(self) -> pd.DataFrame:
        """Compare NVML per-step GPU measurements against CodeCarbon whole-notebook totals."""
        energies_j = [s["energy_j"] for s in self.steps if s["energy_j"] is not None]
        nvml_j = sum(energies_j) if energies_j else 0.0
        nvml_wh = nvml_j / 3600.0
        rows = [
            {
                "Scope": "Profiled steps (GPU only, NVML 50 ms)",
                "Duration (s)": round(sum(s["seconds"] for s in self.steps), 1),
                "GPU (Wh)": round(nvml_wh, 4),
                "CPU + RAM (Wh, modeled)": None,
                "Total (Wh)": round(nvml_wh, 4),
                "CO2 CH 30g/kWh (mg)": round(nvml_wh * CARBON_INTENSITY_G_PER_KWH["CH"], 2),
                "CO2 EU 230g/kWh (mg)": round(nvml_wh * CARBON_INTENSITY_G_PER_KWH["EU"], 2),
                "CO2 US 380g/kWh (mg)": round(nvml_wh * CARBON_INTENSITY_G_PER_KWH["US"], 2),
            }
        ]
        if self.codecarbon_summary is not None:
            cc = self.codecarbon_summary
            tot_wh = cc["total_energy_wh"]
            rows.append(
                {
                    "Scope": "Whole notebook (CodeCarbon offline: GPU + CPU/RAM model)",
                    "Duration (s)": round(cc["duration_s"], 1),
                    "GPU (Wh)": round(cc["gpu_energy_wh"], 4),
                    "CPU + RAM (Wh, modeled)": round(
                        cc["cpu_energy_wh"] + cc["ram_energy_wh"], 4
                    ),
                    "Total (Wh)": round(tot_wh, 4),
                    "CO2 CH 30g/kWh (mg)": round(tot_wh * CARBON_INTENSITY_G_PER_KWH["CH"], 2),
                    "CO2 EU 230g/kWh (mg)": round(tot_wh * CARBON_INTENSITY_G_PER_KWH["EU"], 2),
                    "CO2 US 380g/kWh (mg)": round(tot_wh * CARBON_INTENSITY_G_PER_KWH["US"], 2),
                }
            )
        return pd.DataFrame(rows).set_index("Scope")

    def save(self) -> tuple[Path, Path]:
        """Export step CSV to `data/cache/` and structured JSON to `benchmarks/results/`."""
        self._sync_model_vram()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        gpu_label = self.hardware["gpu_name"] or "CPU only"
        csv_path = self.cache_dir / f"{self.module}_runtime.csv"
        runtime_df = self.summary_table().assign(device=gpu_label)
        runtime_df.to_csv(csv_path)

        energies_j = [s["energy_j"] for s in self.steps if s["energy_j"] is not None]
        nvml_j = sum(energies_j) if energies_j else None
        nvml_wh = (nvml_j / 3600.0) if nvml_j is not None else None
        nvml_totals = (
            {
                "duration_s": round(sum(s["seconds"] for s in self.steps), 3),
                "gpu_energy_j": round(nvml_j, 2),
                "gpu_energy_wh": round(nvml_wh, 5),
                "co2_ch_mg": round(nvml_wh * CARBON_INTENSITY_G_PER_KWH["CH"], 3),
                "co2_eu_mg": round(nvml_wh * CARBON_INTENSITY_G_PER_KWH["EU"], 3),
                "co2_us_mg": round(nvml_wh * CARBON_INTENSITY_G_PER_KWH["US"], 3),
            }
            if nvml_j is not None and nvml_wh is not None
            else None
        )

        payload = {
            "module": self.module,
            "notebook": self.notebook,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hardware": self.hardware,
            "models": [asdict(c) for c in self.models.values()],
            "steps": self.steps,
            "metrics": self.metrics,
            "nvml_totals": nvml_totals,
            "codecarbon": self.codecarbon_summary,
        }
        json_path = self.results_dir / f"{self.module}_{self.notebook}.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return csv_path, json_path
