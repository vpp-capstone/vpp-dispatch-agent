"""Deterministic wind forecast trajectory post-processing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_wind_prediction(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    method: str,
    strength: float,
) -> np.ndarray:
    """Smooth only within each farm and forecast issue trajectory."""
    working = frame[["asset_id", "issue_time_utc", "lead_hour"]].copy()
    working["prediction"] = np.asarray(prediction, dtype=float)
    ordered = working.sort_values(["asset_id", "issue_time_utc", "lead_hour"])
    groups = ordered.groupby(["asset_id", "issue_time_utc"], observed=True, sort=False)
    if method == "median3":
        smooth = groups["prediction"].transform(
            lambda values: values.rolling(3, center=True, min_periods=1).median()
        )
    elif method == "mean3":
        smooth = groups["prediction"].transform(
            lambda values: values.rolling(3, center=True, min_periods=1).mean()
        )
    elif method == "savgol5":
        from scipy.signal import savgol_filter

        smooth = groups["prediction"].transform(
            lambda values: (
                savgol_filter(values.to_numpy(), 5, 2, mode="interp")
                if len(values) >= 5
                else values.to_numpy()
            )
        )
    else:
        raise ValueError(f"unknown wind smoothing method: {method}")
    blended = (1.0 - strength) * ordered["prediction"] + strength * smooth
    result = pd.Series(blended.to_numpy(), index=ordered.index).reindex(frame.index)
    return np.clip(result.to_numpy(), 0.0, 1.0)


def limit_wind_ramp(
    frame: pd.DataFrame, prediction: np.ndarray, maximum_change: float
) -> np.ndarray:
    """Apply a symmetric capacity-factor ramp limit inside each issued forecast."""
    working = frame[["asset_id", "issue_time_utc", "lead_hour"]].copy()
    working["prediction"] = np.asarray(prediction, dtype=float)
    ordered = working.sort_values(["asset_id", "issue_time_utc", "lead_hour"])
    result = ordered["prediction"].to_numpy(copy=True)
    grouped_positions = ordered.groupby(
        ["asset_id", "issue_time_utc"], observed=True, sort=False
    ).indices.values()
    for positions in grouped_positions:
        for current, previous in zip(positions[1:], positions[:-1], strict=True):
            result[current] = np.clip(
                result[current],
                result[previous] - maximum_change,
                result[previous] + maximum_change,
            )
    restored = pd.Series(result, index=ordered.index).reindex(frame.index)
    return np.clip(restored.to_numpy(), 0.0, 1.0)
