"""Audit forecast handoff files without changing models, dispatch, or shared schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp.portfolio_data import KST, SOURCES, build_portfolio_hourly

KEY = ["planning_date", "ts"]


def file_record(path: Path) -> dict[str, Any]:
    """Record an input's identity without publishing its absolute local path."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"empty input or missing columns: {missing}")


def _numeric(values: pd.Series, *, nonnegative: bool = False) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite or missing numeric values: {values.name}")
    if nonnegative and result.lt(0).any():
        raise ValueError(f"negative generation: {values.name}")
    return result


def _time(values: pd.Series, *, utc_column: bool = False) -> pd.Series:
    # A *_utc column explicitly declares its zone. Other columns must carry a zone.
    if not utc_column and any(pd.Timestamp(value).tzinfo is None for value in values):
        raise ValueError(f"timezone must be explicit: {values.name}")
    result = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    if result.isna().any():
        raise ValueError(f"invalid time: {values.name}")
    result = result.dt.tz_convert(KST)
    if result.ne(result.dt.floor("h")).any():
        raise ValueError(f"time is not an hourly interval start: {values.name}")
    return result


def _add_time_keys(frame: pd.DataFrame, issue: str, target: str) -> pd.DataFrame:
    out = frame.copy()
    out["issued_at"] = _time(out[issue], utc_column=issue.endswith("_utc"))
    out["ts"] = _time(out[target], utc_column=target.endswith("_utc"))
    out["planning_date"] = out["issued_at"].dt.normalize()
    cutoff = out["planning_date"] + pd.Timedelta(hours=11)
    if out["issued_at"].gt(cutoff).any():
        raise ValueError("forecast issued after the declared D-1 11:00 KST cutoff")
    if "lead_hour" in out:
        lead = (out["ts"] - out["issued_at"]) / pd.Timedelta(hours=1)
        if not np.allclose(_numeric(out["lead_hour"]), lead, atol=1e-8, rtol=0):
            raise ValueError("lead_hour disagrees with issue/target timestamps")
    return out


def _generation_rows(
    frame: pd.DataFrame, plants: pd.DataFrame, energy_type: str
) -> tuple[pd.DataFrame, set[str]]:
    """Validate rows even when an explicit policy later excludes incomplete windows."""
    _require(frame, ["asset_id", "issue_time_utc", "valid_time_utc",
                     "lightgbm_generation_kwh"])
    _require(plants, ["asset_id", "energy_type", "capacity_mw"])
    selected = plants[plants["energy_type"].eq(energy_type)]
    expected = set(selected["asset_id"])
    if not expected or selected["asset_id"].duplicated().any():
        raise ValueError("invalid plant master")
    out = _add_time_keys(frame, "issue_time_utc", "valid_time_utc")
    out["prediction_kwh"] = _numeric(out["lightgbm_generation_kwh"], nonnegative=True)
    if "energy_type" in out and not out["energy_type"].eq(energy_type).all():
        raise ValueError("unexpected energy_type")
    if out.duplicated([*KEY, "asset_id"]).any():
        raise ValueError("duplicate plant/planning-date/target key")
    if not set(out["asset_id"]) <= expected:
        raise ValueError(f"plant set differs from {energy_type} master")
    if out.groupby("planning_date")["issued_at"].nunique().gt(1).any():
        raise ValueError("multiple forecast issues on one planning date")
    caps = selected.set_index("asset_id")["capacity_mw"] * 1000.0
    if out["prediction_kwh"].gt(out["asset_id"].map(caps) + 1e-6).any():
        raise ValueError("prediction exceeds one-hour plant capacity")
    return out, expected


def generation_windows(
    frame: pd.DataFrame, plants: pd.DataFrame, energy_type: str
) -> pd.DataFrame:
    """Sum only complete, unique plant sets; never turn missing plants into zero."""
    out, expected = _generation_rows(frame, plants, energy_type)
    counts = out.groupby(KEY)["asset_id"].agg(set)
    if not counts.map(lambda value: value == expected).all():
        raise ValueError("incomplete plant set at one or more forecast targets")
    # Actual labels in the validation artifacts are deliberately not carried forward.
    grouped = out.groupby(KEY, as_index=False)["prediction_kwh"].sum()
    grouped[f"{energy_type}_mwh"] = grouped.pop("prediction_kwh") / 1000.0
    validate_windows(grouped)
    return grouped


def _price_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate prices before any window exclusion, including negative and zero SMP."""
    _require(frame, ["issue_time", "target_time", "smp_hat", "smp_hat_krw_per_mwh"])
    out = _add_time_keys(frame, "issue_time", "target_time")
    kwh = _numeric(out["smp_hat"])
    mwh = _numeric(out["smp_hat_krw_per_mwh"])
    # Published outputs round KRW/kWh to .001 and KRW/MWh to .1.
    if not np.allclose(kwh * 1000, mwh, atol=0.6, rtol=0):
        raise ValueError("SMP kWh/MWh conversion mismatch")
    if out.groupby("planning_date")["issued_at"].nunique().gt(1).any():
        raise ValueError("multiple SMP issues on one planning date")
    if out.duplicated(KEY).any():
        raise ValueError("duplicate SMP forecast key")
    out["price_krw_per_mwh"] = mwh
    return out


def price_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate SMP units, keeping real negative and zero prices."""
    out = _price_rows(frame)
    result = out[[*KEY, "price_krw_per_mwh"]].copy()
    validate_windows(result)
    return result


def select_complete_windows(
    frame: pd.DataFrame, plants: pd.DataFrame, source: str
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Exclude whole incomplete windows; malformed rows remain hard errors."""
    if source == "smp":
        out, assets = _price_rows(frame), None
    else:
        out, assets = _generation_rows(frame, plants, source)
    keep, excluded = [], []
    for day, group in out.groupby("planning_date"):
        expected_times = set(pd.date_range(day + pd.Timedelta(days=1), periods=48, freq="h"))
        current_times = set(group["ts"])
        if current_times - expected_times:
            raise ValueError(f"target outside declared 48-hour window: {day.date()}")
        reasons = []
        if expected_times - current_times:
            reasons.append("missing_target_hours")
        if assets is not None:
            sets = group.groupby("ts")["asset_id"].agg(set)
            if not sets.map(lambda values: values == assets).all():
                reasons.append("missing_asset")
        if reasons:
            excluded.append({"source": source, "planning_date": str(day.date()),
                             "rows": len(group), "reasons": ";".join(reasons)})
        else:
            keep.append(day)
    inventory = {
        "rows": len(frame), "target_start": str(out["ts"].min()),
        "target_end": str(out["ts"].max()), "complete_48h_windows": len(keep),
        "excluded_windows": len(excluded),
    }
    if not keep:
        raise ValueError(f"no complete 48-hour windows remain: {source}")
    return out[out["planning_date"].isin(keep)].copy(), excluded, inventory


def validate_windows(frame: pd.DataFrame) -> None:
    """Require exactly D 00:00 through D+1 23:00 for each D-1 issue date."""
    if frame.empty or frame.duplicated(KEY).any():
        raise ValueError("empty or duplicate forecast window")
    for day, group in frame.groupby("planning_date"):
        expected = pd.date_range(day + pd.Timedelta(days=1), periods=48, freq="h")
        actual = pd.DatetimeIndex(group["ts"].sort_values())
        if not actual.equals(expected):
            raise ValueError(f"incomplete or shifted 48-hour window: {day.date()}")


def align_forecasts(
    solar: pd.DataFrame, wind: pd.DataFrame, smp: pd.DataFrame, plants: pd.DataFrame,
    *, incomplete_windows: str = "reject",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join complete windows by planning date, not unequal GFS/SMP issue timestamps."""
    if incomplete_windows not in {"reject", "exclude"}:
        raise ValueError("incomplete_windows must be reject or exclude")
    excluded, inventory = [], {}
    if incomplete_windows == "exclude":
        selected = []
        for source, frame in [("solar", solar), ("wind", wind), ("smp", smp)]:
            complete, removed, info = select_complete_windows(frame, plants, source)
            selected.append(complete)
            excluded.extend(removed)
            inventory[source] = info
        solar, wind, smp = selected
    parts = {
        "solar": generation_windows(solar, plants, "solar"),
        "wind": generation_windows(wind, plants, "wind"),
        "smp": price_windows(smp),
    }
    date_sets = {key: set(value["planning_date"]) for key, value in parts.items()}
    common = set.intersection(*date_sets.values())
    if not common:
        raise ValueError("no common complete forecast planning dates")
    output = parts["solar"].merge(parts["wind"], on=KEY, validate="one_to_one")
    output = output.merge(parts["smp"], on=KEY, validate="one_to_one")
    output["generation_mwh"] = output["solar_mwh"] + output["wind_mwh"]
    output["dt_hours"] = 1.0
    output = output.sort_values(KEY).reset_index(drop=True)
    validate_windows(output)
    report = {
        "status": "PASS", "common_windows": len(common), "rows": len(output),
        "incomplete_window_policy": incomplete_windows,
        "incomplete_forecast_windows": excluded,
        "source_inventory": inventory,
        "planning_start": str(min(common).date()), "planning_end": str(max(common).date()),
        "excluded_planning_dates": {
            key: sorted(str(day.date()) for day in values - common)
            for key, values in date_sets.items()
        },
        "availability_check": "issue timestamps only; source publication latency not certified",
        "scope": "input alignment only; no dispatch feasibility or realized-profit certification",
    }
    return output, report


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def compare_generation_loaders(
    raw_dir: Path, plants_path: Path, start: str, end: str, actual: pd.DataFrame
) -> dict[str, Any]:
    """Compare forecast-training labels with ESS input; retain distinct quality policies."""
    from vpp.generation_forecast.dataset import (
        load_plants,
        load_solar_generation,
        load_wind_generation,
    )
    from vpp.portfolio_data import PV_FILE

    plants = load_plants(plants_path)
    first, last = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    solar = load_solar_generation(raw_dir / PV_FILE, plants, first, last)
    wind = load_wind_generation(raw_dir, plants, first, last)
    labels = pd.concat([solar, wind], ignore_index=True)
    grouped = labels.groupby("valid_time_kst").agg(
        training_generation_kwh=("generation_kwh", lambda s: s.sum(skipna=False)),
        asset_count=("asset_id", "nunique"),
    ).reset_index().rename(columns={"valid_time_kst": "ts"})
    merged = actual[["ts", "generation_mwh"]].merge(
        grouped, on="ts", how="left", validate="one_to_one"
    )
    complete = merged["asset_count"].eq(len(plants))
    diff = (merged["generation_mwh"] * 1000 - merged["training_generation_kwh"]).abs()
    mismatch = ~complete | diff.isna() | diff.gt(1e-5)
    return {
        "status": "PASS" if len(merged) and not mismatch.any() else "FAIL",
        "compared_retained_hours": len(merged),
        "mismatched_hours": int(mismatch.sum()),
        "max_abs_diff_kwh": float(diff.max()) if diff.notna().any() else None,
        "training_label_quality_counts": labels["generation_quality"].value_counts().to_dict(),
        "note": "Value comparison only. Training rejects invalid plant labels; ESS v0 retains "
                "capacity flags and excludes incomplete days. These policies are not unified here.",
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Write reproducible raw-data and optional forecast checks into a local folder."""
    args.out.mkdir(parents=True, exist_ok=True)
    # Do not leave a previous successful forecast artifact after a failed/pending rerun.
    for filename in ("forecast_windows.csv", "incomplete_forecast_windows.csv", "audit.json"):
        (args.out / filename).unlink(missing_ok=True)
    actual, checks = build_portfolio_hourly(args.raw_dir, args.start, args.end, market_path=None)
    actual.to_csv(args.out / "portfolio_actual_hourly.csv", index=False, encoding="utf-8-sig")
    excluded = pd.DataFrame(checks["excluded_days"], columns=["date", "reasons"])
    excluded.to_csv(args.out / "excluded_days.csv", index=False, encoding="utf-8-sig")
    expected = pd.date_range(args.start, args.end, freq="D", tz=KST)
    actual_times = set(actual["ts"])
    windows = [day for day in expected if set(pd.date_range(day, periods=48, freq="h"))
               <= actual_times]
    # The final day alone cannot form a 48-hour window inside the requested range.
    report: dict[str, Any] = {
        "base_commit": args.base_commit,
        "inputs": [file_record(args.raw_dir / source.file) for source in SOURCES]
                  + [file_record(args.plants)],
        "raw_data": checks,
        "complete_actual_48h_windows": len(windows),
        "generation_loader_comparison": compare_generation_loaders(
            args.raw_dir, args.plants, args.start, args.end, actual
        ),
        "forecast_alignment": {"status": "PENDING", "reason": "forecast files not supplied"},
        "realized_profit": {"status": "NOT_EVALUATED"},
    }
    # Exclude local directory details from the portable report.
    checks.pop("raw_dir", None)
    if actual.empty or report["generation_loader_comparison"]["status"] == "FAIL":
        report["status"] = "FAIL"
    else:
        report["status"] = "RAW_CHECKED_FORECAST_PENDING"
    paths = [args.solar, args.wind, args.smp]
    if any(paths):
        if not all(paths):
            report["forecast_alignment"] = {"status": "FAIL", "reason": "supply all 3 forecasts"}
        else:
            try:
                aligned, forecast_report = align_forecasts(
                    *[_read(path) for path in paths], pd.read_csv(args.plants),
                    incomplete_windows=args.incomplete_windows,
                )
                aligned.to_csv(args.out / "forecast_windows.csv", index=False, encoding="utf-8-sig")
                report["forecast_alignment"] = forecast_report
                pd.DataFrame(forecast_report["incomplete_forecast_windows"],
                             columns=["source", "planning_date", "rows", "reasons"]).to_csv(
                    args.out / "incomplete_forecast_windows.csv", index=False, encoding="utf-8-sig"
                )
                report["inputs"].extend(file_record(path) for path in paths)
                # A validated actual input has gaps by design: count matched targets explicitly.
                joined = aligned.merge(actual[["ts", "generation_mwh"]], on="ts", how="left",
                                       suffixes=("_forecast", "_actual"), validate="many_to_one")
                report["forecast_alignment"]["missing_actual_target_rows"] = int(
                    joined["generation_mwh_actual"].isna().sum()
                )
                missing_dates = joined.loc[
                    joined["generation_mwh_actual"].isna(), "planning_date"
                ].nunique()
                report["forecast_alignment"]["complete_actual_comparison_windows"] = int(
                    aligned["planning_date"].nunique() - missing_dates
                )
            except (ValueError, OSError, KeyError) as error:
                report["forecast_alignment"] = {"status": "FAIL", "reason": str(error)}
        report["status"] = (
            "INPUT_ALIGNMENT_CHECKED" if not actual.empty
            and report["generation_loader_comparison"]["status"] == "PASS"
            and report["forecast_alignment"]["status"] == "PASS" else "FAIL"
        )
    (args.out / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/portfolio"))
    parser.add_argument("--start", default="2024-06-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--solar", type=Path)
    parser.add_argument("--wind", type=Path)
    parser.add_argument("--smp", type=Path)
    parser.add_argument("--incomplete-windows", choices=["reject", "exclude"], default="reject",
                        help="검증용 파일의 불완전한 창 전체를 제외하려면 exclude를 명시")
    parser.add_argument("--plants", type=Path,
                        default=Path("data/reference/plant_location_master_21.csv"))
    parser.add_argument("--base-commit", default="unspecified")
    parser.add_argument("--out", type=Path, default=Path("data/processed/handoff_audit"))
    report = run_audit(parser.parse_args(argv))
    brief = {key: value for key, value in report["forecast_alignment"].items()
             if key not in {"excluded_planning_dates", "incomplete_forecast_windows"}}
    print(json.dumps({"status": report["status"], "forecast": brief},
                     ensure_ascii=False, indent=2))
    return {"FAIL": 1, "RAW_CHECKED_FORECAST_PENDING": 2}.get(report["status"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
