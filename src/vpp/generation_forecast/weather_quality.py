"""Validate daily NOAA GFS checkpoints used by the generation models."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from vpp.generation_forecast.paths import PROCESSED_DIR, RAW_GFS_DIR

DEFAULT_INPUT = RAW_GFS_DIR / "daily"
DEFAULT_OUTPUT = PROCESSED_DIR
EXPECTED_GRIDS = 6
EXPECTED_HOURS = 48
EXPECTED_ROWS = EXPECTED_GRIDS * EXPECTED_HOURS
DATE_PATTERN = re.compile(r"gfs_(\d{8})\.parquet$")
REQUIRED_COLUMNS = {
    "gfs_grid_id",
    "issue_time_utc",
    "issue_time_kst",
    "bid_deadline_kst",
    "available_before_bid_deadline",
    "valid_time_utc",
    "valid_time_kst",
    "lead_hour",
    "shortwave_radiation_w_m2",
    "temperature_2m_c",
    "total_cloud_cover_pct",
    "wind_u_80m_ms",
    "wind_v_80m_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 when missing or invalid dates exist",
    )
    return parser.parse_args()


def dates_between(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end must be on or after --start")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def discover_daily_files(root: Path, start: date, end: date) -> dict[date, Path]:
    files: dict[date, Path] = {}
    for path in root.rglob("gfs_*.parquet"):
        match = DATE_PATTERN.search(path.name)
        if not match:
            continue
        current = date.fromisoformat(
            f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}"
        )
        if start <= current <= end:
            if current in files:
                raise ValueError(f"duplicate checkpoint date: {current}")
            files[current] = path
    return files


def validate_daily_frame(frame: pd.DataFrame, planning_day: date) -> dict[str, object]:
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    duplicate_keys = None
    null_cells = None
    grid_count = None
    per_grid_hours_ok = False
    lead_min = None
    lead_max = None
    valid_start = None
    valid_end = None
    all_available = False
    issue_count = None
    errors: list[str] = []

    if missing_columns:
        errors.append("missing_columns")
    else:
        duplicate_keys = int(frame.duplicated(["gfs_grid_id", "valid_time_utc"]).sum())
        null_cells = int(frame[list(REQUIRED_COLUMNS)].isna().sum().sum())
        grid_count = int(frame["gfs_grid_id"].nunique())
        counts = frame.groupby("gfs_grid_id")["valid_time_utc"].nunique()
        per_grid_hours_ok = bool(len(counts) == EXPECTED_GRIDS and (counts == EXPECTED_HOURS).all())
        lead_min = int(frame["lead_hour"].min())
        lead_max = int(frame["lead_hour"].max())
        valid_start = frame["valid_time_kst"].min()
        valid_end = frame["valid_time_kst"].max()
        all_available = bool(frame["available_before_bid_deadline"].all())
        issue_count = int(frame["issue_time_utc"].nunique())
        expected_start = pd.Timestamp(planning_day, tz="Asia/Seoul")
        expected_end = expected_start + pd.Timedelta(hours=EXPECTED_HOURS - 1)

        if duplicate_keys:
            errors.append("duplicate_grid_time")
        if null_cells:
            errors.append("missing_values")
        if grid_count != EXPECTED_GRIDS:
            errors.append("grid_count")
        if not per_grid_hours_ok:
            errors.append("per_grid_hour_count")
        if lead_min != 21 or lead_max != 68:
            errors.append("lead_range")
        if valid_start != expected_start or valid_end != expected_end:
            errors.append("valid_time_range")
        if not all_available:
            errors.append("available_after_bid_deadline")
        if issue_count != 1:
            errors.append("multiple_issue_times")
        if not frame["total_cloud_cover_pct"].between(0, 100).all():
            errors.append("cloud_cover_range")
        if (frame["shortwave_radiation_w_m2"] < 0).any():
            errors.append("negative_radiation")

    if len(frame) != EXPECTED_ROWS:
        errors.append("row_count")

    return {
        "planning_day_kst": planning_day.isoformat(),
        "rows": int(len(frame)),
        "grid_count": grid_count,
        "per_grid_48_hours": per_grid_hours_ok,
        "duplicate_grid_time": duplicate_keys,
        "required_null_cells": null_cells,
        "lead_hour_min": lead_min,
        "lead_hour_max": lead_max,
        "valid_time_start_kst": None if valid_start is None else valid_start.isoformat(),
        "valid_time_end_kst": None if valid_end is None else valid_end.isoformat(),
        "issue_time_count": issue_count,
        "available_before_bid_deadline": all_available,
        "missing_columns": ",".join(missing_columns),
        "status": "PASS" if not errors else "FAIL",
        "errors": ",".join(errors),
    }


def run_validation(root: Path, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    expected = dates_between(start, end)
    discovered = discover_daily_files(root, start, end)
    missing = [value for value in expected if value not in discovered]
    rows: list[dict[str, object]] = []
    for current in expected:
        path = discovered.get(current)
        if path is None:
            rows.append(
                {
                    "planning_day_kst": current.isoformat(),
                    "rows": 0,
                    "status": "MISSING",
                    "errors": "checkpoint_missing",
                    "path": "",
                }
            )
            continue
        result = validate_daily_frame(pd.read_parquet(path), current)
        result["path"] = str(path)
        rows.append(result)
    by_day = pd.DataFrame(rows)
    missing_frame = pd.DataFrame(
        {"planning_day_kst": [value.isoformat() for value in missing], "reason": "checkpoint_missing"}
    )
    failed = by_day[by_day["status"].eq("FAIL")]
    summary = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "expected_days": len(expected),
        "present_days": len(discovered),
        "missing_days": len(missing),
        "invalid_present_days": int(len(failed)),
        "pass_days": int(by_day["status"].eq("PASS").sum()),
        "expected_rows": len(expected) * EXPECTED_ROWS,
        "present_rows": int(by_day["rows"].sum()),
        "coverage_pct": round(100 * len(discovered) / len(expected), 6),
        "missing_dates": [value.isoformat() for value in missing],
        "invalid_dates": failed["planning_day_kst"].tolist(),
        "status": "PASS" if not missing and failed.empty else "INCOMPLETE",
    }
    return by_day, missing_frame, summary


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    by_day, missing, summary = run_validation(args.input, args.start, args.end)
    by_day.to_csv(args.output / "gfs_quality_by_day.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(args.output / "gfs_missing_dates.csv", index=False, encoding="utf-8-sig")
    (args.output / "gfs_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
