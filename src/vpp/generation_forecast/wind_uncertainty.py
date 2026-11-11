"""Residual-quantile uncertainty intervals for wind portfolio forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vpp.generation_forecast.wind_regime import wind_speed_regime

BOUNDARIES = (3.0, 7.0)


def weighted_quantile(
    values: np.ndarray, quantiles: tuple[float, ...], weights: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    ordered_values = np.asarray(values, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative = cumulative / ordered_weights.sum()
    return np.interp(quantiles, cumulative, ordered_values)


@dataclass
class WindResidualQuantiles:
    """Forecast-horizon and wind-regime conditional residual quantiles."""

    grouped: dict[tuple[int, int], np.ndarray]
    by_offset: dict[int, np.ndarray]
    global_quantiles: np.ndarray
    boundaries: tuple[float, ...] = BOUNDARIES

    def corrections(
        self, forecast_day_offset: np.ndarray, wind_speed_ms: np.ndarray
    ) -> np.ndarray:
        offsets = np.asarray(forecast_day_offset, dtype=int)
        regimes = wind_speed_regime(np.asarray(wind_speed_ms), self.boundaries)
        result = np.empty((len(offsets), 3), dtype=float)
        for index, (offset, regime) in enumerate(zip(offsets, regimes, strict=True)):
            result[index] = self.grouped.get(
                (int(offset), int(regime)),
                self.by_offset.get(int(offset), self.global_quantiles),
            )
        # Keep the operational point forecast inside its uncertainty interval.
        result[:, 0] = np.minimum(result[:, 0], 0.0)
        result[:, 2] = np.maximum(result[:, 2], 0.0)
        return result

    def predict(
        self,
        point_capacity_factor: np.ndarray,
        forecast_day_offset: np.ndarray,
        wind_speed_ms: np.ndarray,
    ) -> np.ndarray:
        point = np.asarray(point_capacity_factor, dtype=float)
        corrections = self.corrections(forecast_day_offset, wind_speed_ms)
        result = np.clip(point[:, None] + corrections, 0.0, 1.0)
        result[:, 1] = point
        result[:, 0] = np.minimum(result[:, 0], result[:, 1])
        result[:, 2] = np.maximum(result[:, 2], result[:, 1])
        return result


def uncertainty_portfolio_frame(rows: pd.DataFrame) -> pd.DataFrame:
    working = rows.copy()
    working["dynamic_kwh"] = working["dynamic"] * working["capacity_kw"]
    working["speed_x_capacity"] = (
        working["aws_proxy_wind_speed_ms"] * working["capacity_kw"]
    )
    result = (
        working.groupby(["issue_time_utc", "valid_time_utc"], observed=True)
        .agg(
            forecast_day_offset=("forecast_day_offset", "first"),
            actual_kwh=("generation_kwh", "sum"),
            dynamic_kwh=("dynamic_kwh", "sum"),
            capacity_kw=("capacity_kw", "sum"),
            speed_x_capacity=("speed_x_capacity", "sum"),
        )
        .reset_index()
    )
    result["actual"] = result["actual_kwh"] / result["capacity_kw"]
    result["dynamic"] = result["dynamic_kwh"] / result["capacity_kw"]
    result["wind_speed_ms"] = result["speed_x_capacity"] / result["capacity_kw"]
    result["residual"] = result["actual"] - result["dynamic"]
    result["regime"] = wind_speed_regime(
        result["wind_speed_ms"].to_numpy(), BOUNDARIES
    )
    return result


def fit_wind_residual_quantiles(
    rows: pd.DataFrame, minimum_group_rows: int = 200
) -> WindResidualQuantiles:
    frame = uncertainty_portfolio_frame(rows)
    quantiles = (0.1, 0.5, 0.9)

    def calculate(group: pd.DataFrame) -> np.ndarray:
        return weighted_quantile(
            group["residual"].to_numpy(), quantiles, group["capacity_kw"].to_numpy()
        )

    global_quantiles = calculate(frame)
    by_offset = {
        int(offset): calculate(group)
        for offset, group in frame.groupby("forecast_day_offset", observed=True)
        if len(group) >= minimum_group_rows
    }
    grouped = {
        (int(offset), int(regime)): calculate(group)
        for (offset, regime), group in frame.groupby(
            ["forecast_day_offset", "regime"], observed=True
        )
        if len(group) >= minimum_group_rows
    }
    return WindResidualQuantiles(grouped, by_offset, global_quantiles)
