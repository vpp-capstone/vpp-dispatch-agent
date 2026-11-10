"""Feature and portfolio-model experiments for generation forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

SOLAR_IMPROVEMENT_FEATURES = [
    "clear_sky_irradiance_proxy_w_m2",
    "clear_sky_index",
    "radiation_x_solar_elevation",
    "cloud_x_solar_elevation",
    "shortwave_change_1h_w_m2",
    "shortwave_change_3h_w_m2",
    "cloud_change_1h_pct",
    "cloud_change_3h_pct",
    "regional_shortwave_mean_w_m2",
    "regional_shortwave_std_w_m2",
    "regional_shortwave_range_w_m2",
    "regional_cloud_mean_pct",
    "regional_cloud_std_pct",
    "regional_cloud_range_pct",
    "shortwave_minus_regional_mean_w_m2",
    "cloud_minus_regional_mean_pct",
    "regional_shortwave_change_1h_w_m2",
    "regional_shortwave_change_3h_w_m2",
    "regional_cloud_change_1h_pct",
    "regional_cloud_change_3h_pct",
]
WIND_IMPROVEMENT_FEATURES = [
    "wind_speed_squared",
    "wind_speed_cubed",
    "wind_speed_change_6h_ms",
    "wind_speed_forward_change_1h_ms",
    "wind_speed_forward_change_3h_ms",
    "wind_speed_neighbor_mean_3h_ms",
    "wind_speed_neighbor_range_3h_ms",
    "wind_u_change_1h_ms",
    "wind_u_change_3h_ms",
    "wind_u_change_6h_ms",
    "wind_v_change_1h_ms",
    "wind_v_change_3h_ms",
    "wind_v_change_6h_ms",
    "regional_wind_speed_mean_ms",
    "regional_wind_speed_std_ms",
    "regional_wind_speed_range_ms",
    "regional_wind_u_mean_ms",
    "regional_wind_v_mean_ms",
    "wind_speed_minus_regional_mean_ms",
    "wind_u_minus_regional_mean_ms",
    "wind_v_minus_regional_mean_ms",
    "regional_wind_speed_change_1h_ms",
    "regional_wind_speed_change_3h_ms",
    "regional_wind_speed_change_6h_ms",
]


def enforce_portfolio_physics(
    energy_type: str, frame: pd.DataFrame, prediction: np.ndarray
) -> np.ndarray:
    """Clip portfolio capacity factors and force solar nighttime output to zero."""
    result = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    if energy_type == "solar":
        night = frame["is_daylight_portfolio_mean"].le(0) | frame[
            "shortwave_radiation_w_m2_portfolio_mean"
        ].le(0)
        result[night.to_numpy()] = 0.0
    return result


def _exact_group_change(
    frame: pd.DataFrame,
    value: str,
    hours: int,
    group_columns: list[str],
) -> pd.Series:
    groups = frame.groupby(group_columns, observed=True, sort=False)
    previous_value = groups[value].shift(hours)
    previous_lead = groups["lead_hour"].shift(hours)
    exact = frame["lead_hour"].sub(previous_lead).eq(hours)
    return frame[value].sub(previous_value).where(exact, 0.0)


def _exact_group_forward_change(
    frame: pd.DataFrame,
    value: str,
    hours: int,
    group_columns: list[str],
) -> pd.Series:
    groups = frame.groupby(group_columns, observed=True, sort=False)
    next_value = groups[value].shift(-hours)
    next_lead = groups["lead_hour"].shift(-hours)
    exact = next_lead.sub(frame["lead_hour"]).eq(hours)
    return next_value.sub(frame[value]).where(exact, 0.0)


def add_wind_improvement_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add wind trajectory and two-grid spatial-gradient forecast features."""
    original_index = frame.index
    result = frame.sort_values(["asset_id", "issue_time_utc", "lead_hour"]).copy()
    trajectory = ["asset_id", "issue_time_utc"]
    speed = result["wind_speed_80m_ms"]
    result["wind_speed_squared"] = speed**2
    result["wind_speed_cubed"] = speed**3
    result["wind_speed_change_6h_ms"] = _exact_group_change(
        result, "wind_speed_80m_ms", 6, trajectory
    )
    for hours in (1, 3):
        result[f"wind_speed_forward_change_{hours}h_ms"] = _exact_group_forward_change(
            result, "wind_speed_80m_ms", hours, trajectory
        )
    groups = result.groupby(trajectory, observed=True, sort=False)
    previous_speed = groups["wind_speed_80m_ms"].shift(1)
    next_speed = groups["wind_speed_80m_ms"].shift(-1)
    previous_lead = groups["lead_hour"].shift(1)
    next_lead = groups["lead_hour"].shift(-1)
    previous_speed = previous_speed.where(result["lead_hour"].sub(previous_lead).eq(1), speed)
    next_speed = next_speed.where(next_lead.sub(result["lead_hour"]).eq(1), speed)
    neighbors = pd.concat([previous_speed, speed, next_speed], axis=1)
    result["wind_speed_neighbor_mean_3h_ms"] = neighbors.mean(axis=1)
    result["wind_speed_neighbor_range_3h_ms"] = neighbors.max(axis=1) - neighbors.min(axis=1)
    for component in ("wind_u_80m_ms", "wind_v_80m_ms"):
        short = "wind_u" if component.startswith("wind_u") else "wind_v"
        for hours in (1, 3, 6):
            result[f"{short}_change_{hours}h_ms"] = _exact_group_change(
                result, component, hours, trajectory
            )

    grid = result.drop_duplicates(["gfs_grid_id", "issue_time_utc", "valid_time_utc"])
    regional = (
        grid.groupby(["issue_time_utc", "valid_time_utc"], observed=True)
        .agg(
            lead_hour=("lead_hour", "first"),
            regional_wind_speed_mean_ms=("wind_speed_80m_ms", "mean"),
            regional_wind_speed_std_ms=("wind_speed_80m_ms", "std"),
            regional_wind_speed_min_ms=("wind_speed_80m_ms", "min"),
            regional_wind_speed_max_ms=("wind_speed_80m_ms", "max"),
            regional_wind_u_mean_ms=("wind_u_80m_ms", "mean"),
            regional_wind_v_mean_ms=("wind_v_80m_ms", "mean"),
        )
        .reset_index()
        .sort_values(["issue_time_utc", "lead_hour"])
    )
    regional["regional_wind_speed_std_ms"] = regional["regional_wind_speed_std_ms"].fillna(0.0)
    regional["regional_wind_speed_range_ms"] = regional.pop(
        "regional_wind_speed_max_ms"
    ) - regional.pop("regional_wind_speed_min_ms")
    for hours in (1, 3, 6):
        regional[f"regional_wind_speed_change_{hours}h_ms"] = _exact_group_change(
            regional, "regional_wind_speed_mean_ms", hours, ["issue_time_utc"]
        )
    result = result.merge(
        regional,
        on=["issue_time_utc", "valid_time_utc", "lead_hour"],
        how="left",
        validate="many_to_one",
    )
    result["wind_speed_minus_regional_mean_ms"] = (
        result["wind_speed_80m_ms"] - result["regional_wind_speed_mean_ms"]
    )
    result["wind_u_minus_regional_mean_ms"] = (
        result["wind_u_80m_ms"] - result["regional_wind_u_mean_ms"]
    )
    result["wind_v_minus_regional_mean_ms"] = (
        result["wind_v_80m_ms"] - result["regional_wind_v_mean_ms"]
    )
    result.index = original_index
    return result


def add_solar_improvement_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add clear-sky, trajectory and regional cloud-movement forecast features."""
    original_index = frame.index
    result = frame.sort_values(["asset_id", "issue_time_utc", "lead_hour"]).copy()
    elevation_sin = np.maximum(
        np.sin(np.deg2rad(result["solar_elevation_deg"].to_numpy(dtype=float))), 0.0
    )
    clear_sky = 1361.0 * elevation_sin
    result["clear_sky_irradiance_proxy_w_m2"] = clear_sky
    result["clear_sky_index"] = np.divide(
        result["shortwave_radiation_w_m2"].to_numpy(dtype=float),
        np.maximum(clear_sky, 1.0),
    ).clip(0.0, 1.5)
    result["radiation_x_solar_elevation"] = (
        result["shortwave_radiation_w_m2"].to_numpy(dtype=float) * elevation_sin
    )
    result["cloud_x_solar_elevation"] = (
        result["total_cloud_cover_pct"].to_numpy(dtype=float) * elevation_sin
    )
    trajectory = ["asset_id", "issue_time_utc"]
    for hours in (1, 3):
        result[f"shortwave_change_{hours}h_w_m2"] = _exact_group_change(
            result, "shortwave_radiation_w_m2", hours, trajectory
        )
        result[f"cloud_change_{hours}h_pct"] = _exact_group_change(
            result, "total_cloud_cover_pct", hours, trajectory
        )

    grid = result.drop_duplicates(["gfs_grid_id", "issue_time_utc", "valid_time_utc"])[
        [
            "gfs_grid_id",
            "issue_time_utc",
            "valid_time_utc",
            "lead_hour",
            "shortwave_radiation_w_m2",
            "total_cloud_cover_pct",
        ]
    ]
    regional = (
        grid.groupby(["issue_time_utc", "valid_time_utc", "lead_hour"], observed=True)
        .agg(
            regional_shortwave_mean_w_m2=("shortwave_radiation_w_m2", "mean"),
            regional_shortwave_std_w_m2=("shortwave_radiation_w_m2", "std"),
            regional_shortwave_min_w_m2=("shortwave_radiation_w_m2", "min"),
            regional_shortwave_max_w_m2=("shortwave_radiation_w_m2", "max"),
            regional_cloud_mean_pct=("total_cloud_cover_pct", "mean"),
            regional_cloud_std_pct=("total_cloud_cover_pct", "std"),
            regional_cloud_min_pct=("total_cloud_cover_pct", "min"),
            regional_cloud_max_pct=("total_cloud_cover_pct", "max"),
        )
        .reset_index()
        .sort_values(["issue_time_utc", "lead_hour"])
    )
    regional["regional_shortwave_std_w_m2"] = regional["regional_shortwave_std_w_m2"].fillna(0.0)
    regional["regional_cloud_std_pct"] = regional["regional_cloud_std_pct"].fillna(0.0)
    regional["regional_shortwave_range_w_m2"] = regional.pop(
        "regional_shortwave_max_w_m2"
    ) - regional.pop("regional_shortwave_min_w_m2")
    regional["regional_cloud_range_pct"] = regional.pop("regional_cloud_max_pct") - regional.pop(
        "regional_cloud_min_pct"
    )
    for hours in (1, 3):
        regional[f"regional_shortwave_change_{hours}h_w_m2"] = _exact_group_change(
            regional, "regional_shortwave_mean_w_m2", hours, ["issue_time_utc"]
        )
        regional[f"regional_cloud_change_{hours}h_pct"] = _exact_group_change(
            regional, "regional_cloud_mean_pct", hours, ["issue_time_utc"]
        )
    result = result.merge(
        regional,
        on=["issue_time_utc", "valid_time_utc", "lead_hour"],
        how="left",
        validate="many_to_one",
    )
    result["shortwave_minus_regional_mean_w_m2"] = (
        result["shortwave_radiation_w_m2"] - result["regional_shortwave_mean_w_m2"]
    )
    result["cloud_minus_regional_mean_pct"] = (
        result["total_cloud_cover_pct"] - result["regional_cloud_mean_pct"]
    )
    result.index = original_index
    return result


def portfolio_frame(
    frame: pd.DataFrame, weather_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Collapse asset rows into one row per forecasted portfolio hour."""
    keys = ["issue_time_utc", "valid_time_utc"]
    stable = [
        "lead_hour",
        "forecast_day_offset",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "capacity_kw": ("capacity_kw", "sum"),
        "asset_count": ("asset_id", "nunique"),
        "valid_time_kst": ("valid_time_kst", "first"),
        "dataset_split": ("dataset_split", "first"),
    }
    for column in stable:
        aggregations[column] = (column, "first")
    if "generation_kwh" in frame:
        aggregations["generation_kwh"] = ("generation_kwh", "sum")
    output_features = list(stable)
    for column in weather_features:
        mean_name = f"{column}_portfolio_mean"
        std_name = f"{column}_portfolio_std"
        aggregations[mean_name] = (column, "mean")
        aggregations[std_name] = (column, "std")
        output_features.extend([mean_name, std_name])
    result = frame.groupby(keys, observed=True).agg(**aggregations).reset_index()
    result[output_features] = result[output_features].fillna(0.0)
    if "generation_kwh" in result:
        result["capacity_factor"] = result["generation_kwh"] / result["capacity_kw"]
    return result, output_features
