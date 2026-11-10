"""Build leakage-safe Jeju GFS features for the day-ahead SMP model."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from vpp.generation_forecast.paths import PLANT_MASTER_PATH, RAW_GFS_DIR
from vpp.generation_forecast.weather_quality import discover_daily_files, validate_daily_frame

KST = "Asia/Seoul"
DEFAULT_INPUT = RAW_GFS_DIR / "daily"
DEFAULT_OUTPUT = Path("data/processed/weather/jeju_renewable_weather_features.csv")
EXPECTED_GRIDS = 6
GROUP_KEYS = ["issue_time_utc", "valid_time_utc"]
REQUIRED_COLUMNS = {
    "gfs_grid_id",
    "issue_time_utc",
    "issue_time_kst",
    "source_file_last_modified_utc",
    "bid_deadline_kst",
    "available_before_bid_deadline",
    "valid_time_utc",
    "valid_time_kst",
    "lead_hour",
    "shortwave_radiation_w_m2",
    "temperature_2m_c",
    "temperature_is_interpolated",
    "total_cloud_cover_pct",
    "cloud_cover_is_interpolated",
    "wind_u_80m_ms",
    "wind_v_80m_ms",
    "wind_speed_80m_ms",
}
REGIONAL_AGGREGATIONS = {
    "jeju_shortwave_mean_w_m2": ("shortwave_radiation_w_m2", "mean"),
    "jeju_shortwave_max_w_m2": ("shortwave_radiation_w_m2", "max"),
    "jeju_shortwave_std_w_m2": ("shortwave_radiation_w_m2", "std"),
    "jeju_temperature_mean_c": ("temperature_2m_c", "mean"),
    "jeju_temperature_min_c": ("temperature_2m_c", "min"),
    "jeju_temperature_max_c": ("temperature_2m_c", "max"),
    "jeju_temperature_std_c": ("temperature_2m_c", "std"),
    "jeju_cloud_cover_mean_pct": ("total_cloud_cover_pct", "mean"),
    "jeju_cloud_cover_min_pct": ("total_cloud_cover_pct", "min"),
    "jeju_cloud_cover_max_pct": ("total_cloud_cover_pct", "max"),
    "jeju_cloud_cover_std_pct": ("total_cloud_cover_pct", "std"),
    "jeju_wind_u_mean_ms": ("wind_u_80m_ms", "mean"),
    "jeju_wind_v_mean_ms": ("wind_v_80m_ms", "mean"),
    "jeju_wind_speed_mean_ms": ("wind_speed_80m_ms", "mean"),
    "jeju_wind_speed_min_ms": ("wind_speed_80m_ms", "min"),
    "jeju_wind_speed_max_ms": ("wind_speed_80m_ms", "max"),
    "jeju_wind_speed_std_ms": ("wind_speed_80m_ms", "std"),
}
WEIGHTED_COLUMNS = {
    "shortwave_radiation_w_m2": "shortwave_w_m2",
    "temperature_2m_c": "temperature_c",
    "total_cloud_cover_pct": "cloud_cover_pct",
    "wind_u_80m_ms": "wind_u_ms",
    "wind_v_80m_ms": "wind_v_ms",
    "wind_speed_80m_ms": "80m_wind_speed_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plants", type=Path, default=PLANT_MASTER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when any requested planning date is missing or invalid",
    )
    return parser.parse_args()


def capacity_weights(plants: pd.DataFrame, energy_type: str) -> pd.Series:
    """Return installed-capacity shares by GFS grid for one generation type."""
    required = {"energy_type", "gfs_grid_id", "capacity_mw"}
    missing = required.difference(plants.columns)
    if missing:
        raise ValueError(f"plant master missing columns: {sorted(missing)}")
    selected = plants.loc[plants["energy_type"].eq(energy_type)].dropna(
        subset=["gfs_grid_id", "capacity_mw"]
    )
    grouped = selected.groupby("gfs_grid_id", observed=True)["capacity_mw"].sum()
    if grouped.empty or (grouped <= 0).any():
        raise ValueError(f"no positive {energy_type} capacity weights")
    return grouped / grouped.sum()


def weighted_features(
    frame: pd.DataFrame, weights: pd.Series, prefix: str
) -> pd.DataFrame:
    """Aggregate weather over grids using installed capacity shares."""
    weight_frame = weights.rename("capacity_weight").rename_axis("gfs_grid_id").reset_index()
    selected = frame.merge(weight_frame, on="gfs_grid_id", how="inner", validate="many_to_one")
    if selected.empty:
        raise ValueError(f"no GFS grids match {prefix} capacity weights")
    grouped = selected.groupby(GROUP_KEYS, observed=True, sort=False)
    denominator = grouped["capacity_weight"].sum()
    result = pd.DataFrame(index=denominator.index)
    for source, suffix in WEIGHTED_COLUMNS.items():
        numerator = (selected[source] * selected["capacity_weight"]).groupby(
            [selected[key] for key in GROUP_KEYS], observed=True, sort=False
        ).sum()
        result[f"{prefix}_{suffix}_fleet_weighted"] = numerator / denominator
    speed_cubed = (selected["wind_speed_80m_ms"] ** 3) * selected["capacity_weight"]
    result[f"{prefix}_80m_wind_speed_cubed_fleet_weighted"] = speed_cubed.groupby(
        [selected[key] for key in GROUP_KEYS], observed=True, sort=False
    ).sum() / denominator
    return result.reset_index()


def aggregate_daily_frame(
    frame: pd.DataFrame,
    solar_weights: pd.Series,
    wind_weights: pd.Series,
) -> pd.DataFrame:
    """Convert one 6-grid, 48-hour GFS checkpoint into SMP weather features."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"GFS frame missing columns: {sorted(missing)}")
    if frame.duplicated(["gfs_grid_id", "valid_time_utc"]).any():
        raise ValueError("GFS frame has duplicate grid/target rows")

    grouped = frame.groupby(GROUP_KEYS, observed=True, sort=False)
    regional = grouped.agg(**REGIONAL_AGGREGATIONS).reset_index()
    metadata = grouped.agg(
        issued_at=("issue_time_kst", "first"),
        available_at=("source_file_last_modified_utc", "max"),
        target_ts=("valid_time_kst", "first"),
        lead_hours=("lead_hour", "first"),
        weather_grid_coverage=("gfs_grid_id", "nunique"),
        quality_all_available_before_bid_deadline=(
            "available_before_bid_deadline",
            "all",
        ),
        quality_temperature_interpolation_ratio=("temperature_is_interpolated", "mean"),
        quality_cloud_interpolation_ratio=("cloud_cover_is_interpolated", "mean"),
    ).reset_index()
    solar = weighted_features(frame, solar_weights, "solar")
    wind = weighted_features(frame, wind_weights, "wind")
    output = metadata.merge(regional, on=GROUP_KEYS, validate="one_to_one")
    output = output.merge(solar, on=GROUP_KEYS, validate="one_to_one")
    output = output.merge(wind, on=GROUP_KEYS, validate="one_to_one")
    output["jeju_wind_vector_speed_ms"] = np.hypot(
        output["jeju_wind_u_mean_ms"], output["jeju_wind_v_mean_ms"]
    )
    output["weather_grid_coverage_ratio"] = output["weather_grid_coverage"] / EXPECTED_GRIDS
    for column in ("issued_at", "available_at", "target_ts"):
        output[column] = pd.to_datetime(output[column], utc=True).dt.tz_convert(KST)
    return output.drop(columns=GROUP_KEYS).sort_values("target_ts").reset_index(drop=True)


def dates_between(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end must be on or after --start")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def build_features(
    input_dir: Path,
    plants_path: Path,
    start: date,
    end: date,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read valid daily checkpoints and return a combined feature table and summary."""
    expected_dates = dates_between(start, end)
    files = discover_daily_files(input_dir, start, end)
    plants = pd.read_csv(plants_path)
    solar_weights = capacity_weights(plants, "solar")
    wind_weights = capacity_weights(plants, "wind")
    frames: list[pd.DataFrame] = []
    invalid_dates: list[str] = []
    for planning_day in expected_dates:
        path = files.get(planning_day)
        if path is None:
            continue
        daily = pd.read_parquet(path)
        quality = validate_daily_frame(daily, planning_day)
        if quality["status"] != "PASS":
            invalid_dates.append(planning_day.isoformat())
            continue
        features = aggregate_daily_frame(daily, solar_weights, wind_weights)
        features.insert(0, "planning_day_kst", planning_day.isoformat())
        frames.append(features)

    missing_dates = [day.isoformat() for day in expected_dates if day not in files]
    if strict and (missing_dates or invalid_dates):
        raise ValueError(
            f"missing GFS dates={missing_dates}; invalid GFS dates={invalid_dates}"
        )
    if not frames:
        raise ValueError("no valid GFS checkpoints found")
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["available_at", "target_ts"]
    )
    duplicate_keys = int(combined.duplicated(["available_at", "target_ts"]).sum())
    if duplicate_keys:
        raise ValueError(f"duplicate available_at/target_ts keys: {duplicate_keys}")
    summary = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "expected_planning_days": len(expected_dates),
        "processed_planning_days": len(frames),
        "missing_planning_dates": missing_dates,
        "invalid_planning_dates": invalid_dates,
        "rows": int(len(combined)),
        "feature_columns": [
            column
            for column in combined.select_dtypes(include="number").columns
            if "quality" not in column and "coverage" not in column and column != "lead_hours"
        ],
        "target_start_kst": combined["target_ts"].min().isoformat(),
        "target_end_kst": combined["target_ts"].max().isoformat(),
    }
    return combined.reset_index(drop=True), summary


def main() -> None:
    args = parse_args()
    features, summary = build_features(
        args.input, args.plants, args.start, args.end, strict=args.strict
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {args.output} ({len(features)} rows)")


if __name__ == "__main__":
    main()
