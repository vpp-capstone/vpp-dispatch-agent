"""Build solar and wind training marts from GFS forecasts and generation labels."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from vpp.generation_forecast.paths import (
    PLANT_MASTER_PATH,
    PROCESSED_DIR,
    RAW_GENERATION_DIR,
    RAW_GFS_DIR,
)
from vpp.generation_forecast.weather_quality import discover_daily_files, validate_daily_frame


DEFAULT_GENERATION_DIR = RAW_GENERATION_DIR
DEFAULT_WEATHER_DIR = RAW_GFS_DIR / "daily"
DEFAULT_PLANTS = PLANT_MASTER_PATH
DEFAULT_OUTPUT = PROCESSED_DIR
KST = "Asia/Seoul"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION_DIR)
    parser.add_argument("--weather-dir", type=Path, default=DEFAULT_WEATHER_DIR)
    parser.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def find_one(root: Path, text: str) -> Path:
    matches = [path for path in root.glob("*.csv") if text in path.name]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one CSV containing {text!r}, found {matches}")
    return matches[0]


def quality_labels(*conditions: tuple[pd.Series, str]) -> pd.Series:
    if not conditions:
        raise ValueError("at least one condition is required")
    result = pd.Series("ok", index=conditions[0][0].index, dtype="string")
    for condition, label in conditions:
        result = result.mask(condition & result.eq("ok"), label)
    return result


def load_plants(path: Path) -> pd.DataFrame:
    plants = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "asset_id",
        "plant_name",
        "energy_type",
        "capacity_mw",
        "latitude",
        "longitude",
        "gfs_grid_id",
    }
    missing = required.difference(plants.columns)
    if missing:
        raise ValueError(f"plant master missing columns: {sorted(missing)}")
    if plants["asset_id"].duplicated().any():
        raise ValueError("plant master has duplicate asset_id")
    plants["capacity_kw"] = plants["capacity_mw"] * 1000.0
    return plants


def localize_kst(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.tz_localize(KST, ambiguous="raise", nonexistent="raise")


def load_solar_generation(
    path: Path, plants: pd.DataFrame, target_start: date, target_end: date
) -> pd.DataFrame:
    columns = ["거래일", "발전기코드", "발전기명", "설비용량", "시간", "계량전력량"]
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=columns)
    frame["거래일"] = pd.to_datetime(frame["거래일"], errors="coerce")
    for column in ["설비용량", "시간", "계량전력량"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["거래일"].dt.date.between(target_start, target_end, inclusive="both")
    ].copy()
    frame["plant_name"] = frame["발전기명"].astype("string").str.strip()
    solar_plants = plants[plants["energy_type"].eq("solar")]
    name_map = solar_plants.set_index("plant_name")["asset_id"]
    frame["asset_id"] = frame["plant_name"].map(name_map)
    unknown = sorted(frame.loc[frame["asset_id"].isna(), "plant_name"].dropna().unique())
    if unknown:
        raise ValueError(f"unmapped solar plant names: {unknown}")
    if frame.duplicated(["asset_id", "거래일", "시간"]).any():
        raise ValueError("solar generation has duplicate asset/date/hour")
    frame["valid_time_kst"] = localize_kst(
        frame["거래일"] + pd.to_timedelta(frame["시간"] - 1, unit="h")
    )
    frame = frame.merge(
        solar_plants[["asset_id", "capacity_kw"]], on="asset_id", how="left", validate="many_to_one"
    )
    frame["generation_kwh"] = frame["계량전력량"]
    frame["source_capacity_kw"] = frame["설비용량"]
    frame["generation_quality"] = quality_labels(
        (frame["generation_kwh"].isna(), "missing_generation"),
        (frame["generation_kwh"].lt(0), "negative_generation"),
        (frame["generation_kwh"].gt(frame["capacity_kw"]), "capacity_exceeded"),
    )
    frame["label_is_valid"] = frame["generation_quality"].eq("ok")
    frame["capacity_factor"] = frame["generation_kwh"] / frame["capacity_kw"]
    return frame[
        [
            "asset_id",
            "plant_name",
            "valid_time_kst",
            "generation_kwh",
            "capacity_factor",
            "generation_quality",
            "label_is_valid",
            "source_capacity_kw",
        ]
    ]


def conflicting_keys(frame: pd.DataFrame, keys: list[str], values: list[str]) -> set[tuple]:
    duplicate_rows = frame[frame.duplicated(keys, keep=False)]
    conflicts: set[tuple] = set()
    for key, group in duplicate_rows.groupby(keys, dropna=False):
        if len(group[values].drop_duplicates()) > 1:
            conflicts.add(key if isinstance(key, tuple) else (key,))
    return conflicts


def load_staged_wind(
    path: Path,
    asset_id: str,
    plant_name: str,
    capacity_kw: float,
    target_start: date,
    target_end: date,
) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="cp949")
    frame["년월일"] = pd.to_datetime(frame["년월일"], errors="coerce")
    frame["단계"] = pd.to_numeric(frame["단계"], errors="coerce")
    hour_columns = [str(hour) for hour in range(1, 25)]
    for column in hour_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["년월일"].dt.date.between(target_start, target_end, inclusive="both")
    ].copy()
    keys = ["년월일", "단계"]
    conflicts = conflicting_keys(frame, keys, hour_columns)
    conflict_mask = pd.Series(False, index=frame.index)
    for current in conflicts:
        conflict_mask |= frame["년월일"].eq(current[0]) & frame["단계"].eq(current[1])
    clean = frame.loc[~conflict_mask].drop_duplicates(keys, keep="first")
    melted = clean.melt(
        id_vars=keys,
        value_vars=hour_columns,
        var_name="hour",
        value_name="stage_generation_kwh",
    )
    melted["hour"] = melted["hour"].astype(int)
    melted["valid_time_naive"] = melted["년월일"] + pd.to_timedelta(melted["hour"] - 1, unit="h")
    grouped = melted.groupby("valid_time_naive", as_index=False).agg(
        generation_kwh=("stage_generation_kwh", lambda value: value.sum(min_count=2)),
        stages_present=("단계", "nunique"),
    )
    grouped["valid_time_kst"] = localize_kst(grouped["valid_time_naive"])
    grouped["asset_id"] = asset_id
    grouped["plant_name"] = plant_name
    grouped["generation_quality"] = quality_labels(
        (grouped["stages_present"].ne(2), "stage_conflict_or_missing"),
        (grouped["generation_kwh"].isna(), "missing_generation"),
        (grouped["generation_kwh"].lt(0), "negative_generation"),
        (grouped["generation_kwh"].gt(capacity_kw), "capacity_exceeded"),
    )
    grouped["label_is_valid"] = grouped["generation_quality"].eq("ok")
    grouped["capacity_factor"] = grouped["generation_kwh"] / capacity_kw
    grouped["source_capacity_kw"] = capacity_kw
    return grouped[
        [
            "asset_id",
            "plant_name",
            "valid_time_kst",
            "generation_kwh",
            "capacity_factor",
            "generation_quality",
            "label_is_valid",
            "source_capacity_kw",
        ]
    ]


def load_sangmyeong(
    path: Path,
    asset_id: str,
    plant_name: str,
    capacity_kw: float,
    target_start: date,
    target_end: date,
) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="cp949")
    frame["기준일"] = pd.to_datetime(frame["기준일"], errors="coerce")
    frame = frame[
        frame["기준일"].notna()
        & frame["기준일"].dt.date.between(target_start, target_end, inclusive="both")
    ].copy()
    hour_columns = [f"{hour}시" for hour in range(1, 25)]
    for column in hour_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.duplicated(["기준일", "호기"]).any():
        raise ValueError("Sangmyeong has duplicate date/unit")
    melted = frame.melt(
        id_vars=["기준일", "호기"],
        value_vars=hour_columns,
        var_name="hour_label",
        value_name="generation_raw",
    )
    melted["hour"] = melted["hour_label"].str.replace("시", "", regex=False).astype(int)
    melted["valid_time_naive"] = melted["기준일"] + pd.to_timedelta(melted["hour"] - 1, unit="h")
    unit_change = pd.Timestamp("2025-02-01")
    melted["generation_kwh"] = np.where(
        melted["valid_time_naive"].lt(unit_change),
        melted["generation_raw"] / 1000.0,
        melted["generation_raw"],
    )
    melted["valid_time_kst"] = localize_kst(melted["valid_time_naive"])
    melted["asset_id"] = asset_id
    melted["plant_name"] = plant_name
    melted["generation_quality"] = quality_labels(
        (melted["generation_kwh"].isna(), "missing_generation"),
        (melted["generation_kwh"].lt(0), "negative_generation"),
        (melted["generation_kwh"].gt(capacity_kw), "capacity_exceeded"),
    )
    melted["label_is_valid"] = melted["generation_quality"].eq("ok")
    melted["capacity_factor"] = melted["generation_kwh"] / capacity_kw
    melted["source_capacity_kw"] = capacity_kw
    return melted[
        [
            "asset_id",
            "plant_name",
            "valid_time_kst",
            "generation_kwh",
            "capacity_factor",
            "generation_quality",
            "label_is_valid",
            "source_capacity_kw",
        ]
    ]


def load_wind_generation(
    generation_dir: Path, plants: pd.DataFrame, target_start: date, target_end: date
) -> pd.DataFrame:
    wind = plants[plants["energy_type"].eq("wind")].set_index("asset_id")
    seongsan = load_staged_wind(
        find_one(generation_dir, "성산풍력"),
        "wind_seongsan",
        wind.loc["wind_seongsan", "plant_name"],
        float(wind.loc["wind_seongsan", "capacity_kw"]),
        target_start,
        target_end,
    )
    hangyeong = load_staged_wind(
        find_one(generation_dir, "한경풍력"),
        "wind_hangyeong",
        wind.loc["wind_hangyeong", "plant_name"],
        float(wind.loc["wind_hangyeong", "capacity_kw"]),
        target_start,
        target_end,
    )
    sangmyeong = load_sangmyeong(
        find_one(generation_dir, "상명풍력"),
        "wind_sangmyeong",
        wind.loc["wind_sangmyeong", "plant_name"],
        float(wind.loc["wind_sangmyeong", "capacity_kw"]),
        target_start,
        target_end,
    )
    result = pd.concat([seongsan, hangyeong, sangmyeong], ignore_index=True)
    if result.duplicated(["asset_id", "valid_time_kst"]).any():
        raise ValueError("wind generation has duplicate asset/time after cleaning")
    return result


def load_weather(root: Path, start: date, end: date) -> tuple[pd.DataFrame, list[str]]:
    files = discover_daily_files(root, start, end)
    frames: list[pd.DataFrame] = []
    invalid: list[str] = []
    for current in sorted(files):
        frame = pd.read_parquet(files[current])
        validation = validate_daily_frame(frame, current)
        if validation["status"] != "PASS":
            invalid.append(current.isoformat())
            continue
        frame = frame.copy()
        frame["planning_day_kst"] = current.isoformat()
        frames.append(frame)
    if not frames:
        raise ValueError("no valid GFS daily files found")
    return pd.concat(frames, ignore_index=True), invalid


def solar_elevation_deg(times_utc: pd.Series, latitude: pd.Series, longitude: pd.Series) -> np.ndarray:
    """Approximate solar elevation using NOAA's fractional-year equations."""
    utc = pd.to_datetime(times_utc, utc=True)
    day = utc.dt.day_of_year.to_numpy(dtype=float)
    hour = (
        utc.dt.hour.to_numpy(dtype=float)
        + utc.dt.minute.to_numpy(dtype=float) / 60.0
        + utc.dt.second.to_numpy(dtype=float) / 3600.0
    )
    gamma = 2.0 * math.pi / 365.0 * (day - 1.0 + (hour - 12.0) / 24.0)
    equation = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    solar_minutes = (hour * 60.0 + equation + 4.0 * longitude.to_numpy(dtype=float)) % 1440.0
    hour_angle = np.deg2rad(solar_minutes / 4.0 - 180.0)
    lat = np.deg2rad(latitude.to_numpy(dtype=float))
    cosine_zenith = (
        np.sin(lat) * np.sin(declination)
        + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    )
    zenith = np.arccos(np.clip(cosine_zenith, -1.0, 1.0))
    return 90.0 - np.rad2deg(zenith)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    local = result["valid_time_kst"]
    hour = local.dt.hour
    day = local.dt.dayofyear
    result["hour_kst"] = hour.astype("int8")
    result["month_kst"] = local.dt.month.astype("int8")
    result["day_of_year"] = day.astype("int16")
    result["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    result["day_of_year_sin"] = np.sin(2 * math.pi * day / 365.25)
    result["day_of_year_cos"] = np.cos(2 * math.pi * day / 365.25)
    planning = pd.to_datetime(result["planning_day_kst"]).dt.date
    result["forecast_day_offset"] = np.array(
        [(valid - planned).days for valid, planned in zip(local.dt.date, planning)], dtype="int8"
    )
    # Split by label/valid year, not planning-day year.  A 48-hour forecast
    # issued on 31 December crosses into the next year; using planning year
    # would place the same realized generation timestamp in two splits.
    year = local.dt.year
    result["dataset_split"] = np.select(
        [year.eq(2023), year.eq(2024), year.eq(2025)],
        ["train", "validation", "test"],
        default="other",
    )
    return result


def build_mart(
    weather: pd.DataFrame,
    generation: pd.DataFrame,
    plants: pd.DataFrame,
    energy_type: str,
) -> pd.DataFrame:
    selected_plants = plants[plants["energy_type"].eq(energy_type)].copy()
    selected_plants = selected_plants.rename(
        columns={"latitude": "plant_latitude", "longitude": "plant_longitude"}
    )
    plant_columns = [
        "asset_id",
        "plant_name",
        "energy_type",
        "capacity_kw",
        "embedded_weather_group",
        "plant_latitude",
        "plant_longitude",
        "coordinate_quality",
        "gfs_grid_id",
    ]
    expanded = weather.merge(
        selected_plants[plant_columns], on="gfs_grid_id", how="inner", validate="many_to_many"
    )
    expanded = expanded.merge(
        generation,
        on=["asset_id", "plant_name", "valid_time_kst"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_label"),
    )
    expanded["generation_quality"] = expanded["generation_quality"].fillna("label_missing")
    expanded["label_is_valid"] = expanded["label_is_valid"].fillna(False).astype(bool)
    expanded = add_time_features(expanded)
    if energy_type == "solar":
        expanded["solar_elevation_deg"] = solar_elevation_deg(
            expanded["valid_time_utc"], expanded["plant_latitude"], expanded["plant_longitude"]
        )
        expanded["is_daylight"] = expanded["solar_elevation_deg"].gt(0)
        features = [
            "shortwave_radiation_w_m2",
            "temperature_2m_c",
            "total_cloud_cover_pct",
            "solar_elevation_deg",
            "is_daylight",
        ]
    else:
        features = [
            "wind_u_80m_ms",
            "wind_v_80m_ms",
            "wind_speed_80m_ms",
            "wind_direction_80m_deg",
            "temperature_2m_c",
            "total_cloud_cover_pct",
        ]
    base = [
        "asset_id",
        "plant_name",
        "energy_type",
        "capacity_kw",
        "embedded_weather_group",
        "plant_latitude",
        "plant_longitude",
        "coordinate_quality",
        "gfs_grid_id",
        "model",
        "planning_day_kst",
        "issue_time_utc",
        "issue_time_kst",
        "bid_deadline_kst",
        "available_before_bid_deadline",
        "valid_time_utc",
        "valid_time_kst",
        "lead_hour",
        "forecast_day_offset",
        "dataset_split",
        "hour_kst",
        "month_kst",
        "day_of_year",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    labels = [
        "generation_kwh",
        "capacity_factor",
        "generation_quality",
        "label_is_valid",
        "source_capacity_kw",
    ]
    result = expanded[base + features + labels].sort_values(
        ["planning_day_kst", "asset_id", "valid_time_kst"]
    )
    if result.duplicated(["asset_id", "issue_time_utc", "valid_time_utc"]).any():
        raise ValueError(f"{energy_type} mart has duplicate asset/issue/valid key")
    return result.reset_index(drop=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def quality_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame["generation_quality"].value_counts().items()}


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plants = load_plants(args.plants)
    target_end = args.end + timedelta(days=1)
    solar_path = find_one(args.generation_dir, "태양광")
    solar_generation = load_solar_generation(solar_path, plants, args.start, target_end)
    wind_generation = load_wind_generation(args.generation_dir, plants, args.start, target_end)
    weather, invalid_weather_dates = load_weather(args.weather_dir, args.start, args.end)

    solar_mart = build_mart(weather, solar_generation, plants, "solar")
    wind_mart = build_mart(weather, wind_generation, plants, "wind")

    atomic_parquet(solar_generation, args.output / "solar_generation_clean.parquet")
    atomic_parquet(wind_generation, args.output / "wind_generation_clean.parquet")
    atomic_parquet(solar_mart, args.output / "solar_training_mart.parquet")
    atomic_parquet(wind_mart, args.output / "wind_training_mart.parquet")

    present_dates = set(weather["planning_day_kst"].unique())
    expected_dates = {
        (args.start + timedelta(days=i)).isoformat()
        for i in range((args.end - args.start).days + 1)
    }
    summary = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "weather_planning_days": len(present_dates),
        "missing_weather_dates": sorted(expected_dates - present_dates),
        "invalid_weather_dates": invalid_weather_dates,
        "solar_generation_rows": int(len(solar_generation)),
        "wind_generation_rows": int(len(wind_generation)),
        "solar_mart_rows": int(len(solar_mart)),
        "wind_mart_rows": int(len(wind_mart)),
        "solar_assets": int(solar_mart["asset_id"].nunique()),
        "wind_assets": int(wind_mart["asset_id"].nunique()),
        "solar_label_quality": quality_counts(solar_mart),
        "wind_label_quality": quality_counts(wind_mart),
        "solar_valid_label_pct": round(100 * solar_mart["label_is_valid"].mean(), 6),
        "wind_valid_label_pct": round(100 * wind_mart["label_is_valid"].mean(), 6),
        "split_rows": {
            "solar": {str(k): int(v) for k, v in solar_mart["dataset_split"].value_counts().items()},
            "wind": {str(k): int(v) for k, v in wind_mart["dataset_split"].value_counts().items()},
        },
    }
    (args.output / "training_mart_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
