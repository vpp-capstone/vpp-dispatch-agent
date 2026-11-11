"""Evaluate 2023-calibrated wind P10/P50/P90 intervals on 2024."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vpp.generation_forecast.paths import RESULTS_DIR
from vpp.generation_forecast.wind_uncertainty import (
    fit_wind_residual_quantiles,
    uncertainty_portfolio_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def pinball(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = actual - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def main() -> None:
    args = parse_args()
    folds = pd.read_parquet(args.reports / "wind_final_fold_predictions.parquet")
    folds["dynamic"] = folds["final_prediction"]
    calibration = folds[folds["fold"].isin(["2023_q3", "2023_q4"])]
    calibrator = fit_wind_residual_quantiles(calibration)
    holdout_rows = pd.read_parquet(args.reports / "wind_final_2024_predictions.parquet")
    holdout_rows["dynamic"] = holdout_rows["final_prediction"]
    holdout = uncertainty_portfolio_frame(holdout_rows)
    intervals = calibrator.predict(
        holdout["dynamic"].to_numpy(),
        holdout["forecast_day_offset"].to_numpy(),
        holdout["wind_speed_ms"].to_numpy(),
    )
    actual = holdout["actual"].to_numpy()
    capacity = holdout["capacity_kw"].to_numpy()
    inside = (actual >= intervals[:, 0]) & (actual <= intervals[:, 2])
    summary = {
        "calibration_years": [2023],
        "evaluation_years": [2024],
        "locked_2025_loaded": False,
        "nominal_coverage_pct": 80.0,
        "empirical_coverage_pct": float(100.0 * np.average(inside, weights=capacity)),
        "mean_interval_width_capacity_factor": float(
            np.average(intervals[:, 2] - intervals[:, 0], weights=capacity)
        ),
        "pinball_loss": {
            "p10": pinball(actual, intervals[:, 0], 0.1),
            "p50": pinball(actual, intervals[:, 1], 0.5),
            "p90": pinball(actual, intervals[:, 2], 0.9),
        },
        "groups": len(calibrator.grouped),
        "fallback_offsets": len(calibrator.by_offset),
    }
    output = holdout[["issue_time_utc", "valid_time_utc", "actual", "dynamic"]].copy()
    output[["p10", "p50", "p90"]] = intervals
    output.to_parquet(args.reports / "wind_uncertainty_2024_predictions.parquet")
    (args.reports / "wind_uncertainty_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
