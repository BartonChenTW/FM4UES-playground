"""Shared evaluation metrics and operational/energy profiling for FM4UES-playground."""

from fm4ues.metrics import (
    class_f1_scores,
    crps_from_quantiles,
    interval_coverage,
    mae,
    mape,
    peak_hour_mae,
    pinball_loss,
    portfolio_bias_pct,
    ramp_rate_mae,
    rmse,
    stylized_imbalance_cost,
    winkler_score,
)
from fm4ues.profiler import (
    CARBON_INTENSITY_G_PER_KWH,
    ModelCard,
    RunProfiler,
    estimate_flops,
    get_hardware_info,
    inspect_model,
    print_hardware_banner,
)

__all__ = [
    "CARBON_INTENSITY_G_PER_KWH",
    "ModelCard",
    "RunProfiler",
    "class_f1_scores",
    "crps_from_quantiles",
    "estimate_flops",
    "get_hardware_info",
    "inspect_model",
    "interval_coverage",
    "mae",
    "mape",
    "peak_hour_mae",
    "pinball_loss",
    "portfolio_bias_pct",
    "print_hardware_banner",
    "ramp_rate_mae",
    "rmse",
    "stylized_imbalance_cost",
    "winkler_score",
]
