"""Build audited Jeju renewable targets without mixing different populations.

Standalone data-team entry point; the shared hourly contract is not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SOURCES = {
    "kpx_file": "https://www.data.go.kr/data/15065269/fileData.do",
    "kpx_api": "https://www.data.go.kr/data/15103243/openapi.do",
    "jec_solar": "https://www.data.go.kr/data/15038930/fileData.do",
    "jec_wind": "https://www.data.go.kr/data/15028181/fileData.do",
}
KEYS = ["trade_date", "hour_label", "fuel"]
FUELS = {"태양광": "solar", "풍력": "wind", "solar": "solar", "wind": "wind"}


def read_source(path: Path) -> pd.DataFrame:
    """Read the public CSV encodings and strip whitespace from column names."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype=str)
            frame.columns = frame.columns.str.strip()
            if frame.columns.duplicated().any():
                raise ValueError(f"Duplicate column names: {path.name}")
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unsupported CSV encoding: {path.name}")


def numeric(series: pd.Series) -> pd.Series:
    """Parse numeric source values; blanks and nonfinite values remain missing."""
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).replace([float("inf"), float("-inf")], float("nan"))


def provenance(path: Path, source: str) -> dict:
    """Record the original file checksum without local credentials or paths."""
    return {
        "source_id": source,
        "source_url": SOURCES[source],
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def normalize_kpx(path: Path, *, api_export: bool = False) -> tuple[pd.DataFrame, dict]:
    """Normalize KPX file or explicitly mapped API CSV; preserve trade-hour labels.

    API CSV must first be mapped to the documented canonical columns. This
    adapter intentionally does not guess an unverified remote API schema.
    """
    original = read_source(path)
    frame = original.copy()
    frame.columns = frame.columns.str.replace(" ", "", regex=False)
    frame = frame.rename(columns={
        "거래일": "trade_date", "거래시간": "hour_label", "지역": "region",
        "연료원": "fuel", "전력거래량(MWh)": "traded_energy_mwh",
        "거래일자": "trade_date", "지역명": "region", "발전량(MWh)": "traded_energy_mwh",
    })
    wide_fuels = {
        column: fuel for column in frame.columns
        for label, fuel in [("태양광발전량(mwh)", "solar"), ("풍력발전량(mwh)", "wind")]
        if column.lower() == label
    }
    if wide_fuels:
        frame = frame.melt(
            id_vars=["trade_date", "hour_label", "region"], value_vars=list(wide_fuels),
            var_name="fuel", value_name="traded_energy_mwh",
        )
        frame["fuel"] = frame.fuel.map(wide_fuels)
    if api_export and "fuel" not in frame:
        frame["fuel"] = "solar"
    required = {*KEYS, "region", "traded_energy_mwh"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    frame = frame[frame.region.str.strip().isin(["제주", "제주도", "제주특별자치도"])]
    frame = frame.copy()
    frame["fuel"] = frame.fuel.str.strip().map(FUELS)
    frame = frame[frame.fuel.notna()].copy()
    if api_export and (frame.fuel != "solar").any():
        raise ValueError("The solar API export cannot contain wind observations")
    frame["trade_date"] = pd.to_datetime(frame.trade_date, errors="coerce")
    frame["hour_label"] = numeric(frame.hour_label)
    invalid = (
        frame.trade_date.isna() | frame.hour_label.isna()
        | ~frame.hour_label.between(1, 24)
        | frame.hour_label.mod(1).ne(0)
        | frame.trade_date.ne(frame.trade_date.dt.normalize())
    )
    if invalid.any():
        raise ValueError(f"{path.name}: {int(invalid.sum())} invalid date/hour keys")
    if frame.empty:
        raise ValueError(f"{path.name}: no Jeju renewable rows")
    frame["hour_label"] = frame.hour_label.astype(int)
    frame["raw_value"] = frame.traded_energy_mwh
    frame["traded_energy_mwh"] = numeric(frame.traded_energy_mwh)
    frame["source_file"] = path.name
    frame["source_id"] = "kpx_api" if api_export else "kpx_file"
    duplicate_rows = int(frame.duplicated(KEYS).sum())
    grouped = frame.groupby(KEYS).traded_energy_mwh
    conflicts = grouped.nunique(dropna=False).gt(1)
    conflict_keys = conflicts[conflicts].index
    frame["conflicting_duplicate"] = pd.MultiIndex.from_frame(frame[KEYS]).isin(conflict_keys)
    # Conflicts become unavailable labels; neither first/last nor averaging is evidence.
    conflict_rows = frame.loc[frame.conflicting_duplicate].copy()
    frame = frame.drop_duplicates(KEYS).copy()
    frame.loc[frame.conflicting_duplicate, "traded_energy_mwh"] = float("nan")
    report = {
        **provenance(path, "kpx_api" if api_export else "kpx_file"),
        "original_rows": len(original), "jeju_unique_keys": len(frame),
        "duplicate_key_rows": duplicate_rows,
        "conflicting_keys": int(conflicts.sum()),
        "start_date": frame.trade_date.min().date().isoformat(),
        "end_date": frame.trade_date.max().date().isoformat(),
        "unit": "MWh", "resolution": "hourly",
        "population": "Jeju electricity-market participants; not all Jeju generation",
    }
    report["_conflict_rows"] = conflict_rows
    return frame, report


def audit_daily(path: Path, fuel: str) -> tuple[pd.DataFrame, dict]:
    """Retain JEC daily values in source units, without manufacturing hourly labels."""
    frame = read_source(path)
    if frame.empty:
        raise ValueError(f"{path.name}: no daily observations")
    date_column = frame.columns[0]
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any() or dates.ne(dates.dt.normalize()).any():
        raise ValueError(f"{path.name}: invalid daily date")
    value_columns = list(frame.columns[1:])
    values = frame[value_columns].apply(numeric)
    total_candidates = [c for c in value_columns if "전체" in c]
    if len(total_candidates) != 1:
        raise ValueError(f"{path.name}: expected one source total column")
    total = total_candidates[0]
    components = values.drop(columns=total)
    difference = values[total] - components.sum(axis=1, min_count=len(components.columns))
    mismatch = difference.abs().gt(0.01)
    long = frame.assign(trade_date=dates.dt.strftime("%Y-%m-%d")).melt(
        id_vars="trade_date", value_vars=value_columns,
        var_name="plant_or_total", value_name="raw_value",
    )
    long["value_source_unit"] = numeric(long.raw_value)
    long["fuel"] = fuel
    long["is_reported_total"] = long.plant_or_total.eq(total)
    long["unit_status"] = "unverified_do_not_convert"
    long["source_file"] = path.name
    long["source_id"] = f"jec_{fuel}"
    report = {
        **provenance(path, f"jec_{fuel}"),
        "rows": len(frame), "resolution": "daily", "unit": "unverified",
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
        "missing_dates": len(pd.date_range(dates.min(), dates.max()).difference(dates)),
        "duplicate_date_rows": int(dates.duplicated().sum()),
        "missing_numeric_cells": int(values.isna().sum().sum()),
        "negative_numeric_cells": int(values.lt(0).sum().sum()),
        "total_component_mismatch_days": int(mismatch.sum()),
        "total_component_comparable_days": int(difference.notna().sum()),
        "population": "Jeju Energy Corporation operated plants only",
        "hourly_target_eligible": False,
    }
    return long, report


def build_targets(
    kpx_file: Path, jec_solar: Path, jec_wind: Path, output_dir: Path,
    *, start: str | None = None, end: str | None = None,
    api_csv: Path | None = None,
) -> pd.DataFrame:
    """Save labels, separate daily references, provenance and a quality report.

    No interpolation, no cross-population filling, no API overwrites. The API
    is compared on overlapping keys until unit/population equivalence is approved.
    """
    main, file_report = normalize_kpx(kpx_file)
    conflicts = file_report.pop("_conflict_rows")
    lower = pd.Timestamp(start) if start else main.trade_date.min()
    upper = pd.Timestamp(end) if end else main.trade_date.max()
    if lower > upper or lower != lower.normalize() or upper != upper.normalize():
        raise ValueError("start/end must be inclusive dates with start <= end")
    calendar = pd.MultiIndex.from_product(
        [pd.date_range(lower, upper), range(1, 25), ["solar", "wind"]], names=KEYS,
    ).to_frame(index=False)
    targets = calendar.merge(main, on=KEYS, how="left", validate="one_to_one")
    targets["quality_flag"] = "observed"
    targets.loc[targets.source_file.isna(), "quality_flag"] = "missing_hour"
    targets.loc[
        targets.source_file.notna() & targets.traded_energy_mwh.isna(), "quality_flag"
    ] = "missing_or_invalid_value"
    targets.loc[targets.conflicting_duplicate.eq(True), "quality_flag"] = "conflicting_duplicate"
    targets.loc[targets.traded_energy_mwh.lt(0), "quality_flag"] = "negative_review"
    targets["target_available"] = targets.quality_flag.eq("observed")
    targets["is_imputed"] = False
    targets["includes_existing_ess"] = True
    targets["hour_interval_convention"] = "pending_team_confirmation"
    # Retain original date/hour keys until their boundary convention is approved.
    targets["trade_date"] = targets.trade_date.dt.strftime("%Y-%m-%d")
    daily_solar, solar_report = audit_daily(jec_solar, "solar")
    daily_wind, wind_report = audit_daily(jec_wind, "wind")
    api_report: dict = {"status": "not_collected", "source_url": SOURCES["kpx_api"]}
    api_comparison = pd.DataFrame()
    api_conflicts = pd.DataFrame()
    if api_csv:
        api, api_report = normalize_kpx(api_csv, api_export=True)
        api_conflicts = api_report.pop("_conflict_rows")
        comparison = main[main.fuel.eq("solar")].merge(
            api, on=KEYS, how="outer", suffixes=("_file", "_api"), indicator=True,
        )
        comparable = (
            comparison.traded_energy_mwh_file.notna()
            & comparison.traded_energy_mwh_api.notna()
        )
        comparison["both_values_available"] = comparable
        comparison["difference_mwh"] = (
            comparison.traded_energy_mwh_api - comparison.traded_energy_mwh_file
        )
        api_comparison = comparison
        api_report.update({
            "status": "compared_not_used_for_filling",
            "overlap_keys": int(comparison._merge.eq("both").sum()),
            "comparable_keys": int(comparable.sum()),
            "different_values": int((comparable & comparison.difference_mwh.abs().gt(1e-6)).sum()),
        })
    per_fuel = {}
    for fuel, rows in targets.groupby("fuel"):
        valid = rows.loc[rows.target_available, "traded_energy_mwh"]
        per_fuel[fuel] = {
            "expected_hours": len(rows), "available_hours": int(rows.target_available.sum()),
            "missing_hours": int(rows.quality_flag.eq("missing_hour").sum()),
            "quality_flags": rows.quality_flag.value_counts().to_dict(),
            "min_mwh": float(valid.min()) if len(valid) else None,
            "median_mwh": float(valid.median()) if len(valid) else None,
            "max_mwh": float(valid.max()) if len(valid) else None,
        }
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "main_target": "kpx_file", "target_name": "traded_energy_mwh",
        "decision": "KPX hourly market trade is the candidate label; JEC daily remains separate",
        "is_shared_contract_approved": False,
        "time_boundary_status": "Original date and hour 1..24 retained; timestamp mapping pending",
        "start_date": lower.date().isoformat(), "end_date": upper.date().isoformat(),
        "target_quality": per_fuel,
        "sources": [file_report, api_report, solar_report, wind_report],
        "jec_date_overlap_days": {
            fuel: len(set(targets.trade_date) & set(daily.trade_date))
            for fuel, daily in [("solar", daily_solar), ("wind", daily_wind)]
        },
        "fill_policy": "No interpolation or substitution; only observed labels are eligible",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "renewable_targets_hourly.csv": targets.drop(columns="conflicting_duplicate"),
        "target_issues.csv": targets[~targets.target_available],
        "kpx_conflicting_rows.csv": conflicts,
        "jec_solar_daily_reference.csv": daily_solar,
        "jec_wind_daily_reference.csv": daily_wind,
    }.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
    if api_csv:
        api_comparison.to_csv(output_dir / "kpx_api_comparison.csv", index=False)
        api_conflicts.to_csv(output_dir / "kpx_api_conflicting_rows.csv", index=False)
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return targets


def main() -> None:
    """CLI for repeatable preprocessing of the four source candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kpx-file", required=True, type=Path)
    parser.add_argument("--jec-solar", required=True, type=Path)
    parser.add_argument("--jec-wind", required=True, type=Path)
    parser.add_argument("--kpx-api-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/targets"))
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    try:
        frame = build_targets(
            args.kpx_file, args.jec_solar, args.jec_wind, args.output_dir,
            start=args.start, end=args.end, api_csv=args.kpx_api_csv,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Target preparation failed: {exc}\n")
    print(f"Saved {len(frame)} hourly labels to {args.output_dir}")
    print(f"Available observed labels: {int(frame.target_available.sum())}")


if __name__ == "__main__":
    main()
