"""Audit 2023-2024 generation labels without reading the locked 2025 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from vpp.generation_forecast.paths import PROCESSED_DIR, RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def audit_one(path: Path, energy_type: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(path, filters=[("dataset_split", "in", ["train", "validation"])])
    labels = raw.sort_values("issue_time_utc").drop_duplicates(
        ["asset_id", "valid_time_utc"], keep="last"
    )
    labels["missing_label"] = labels["capacity_factor"].isna()
    labels["negative_label"] = labels["capacity_factor"].lt(0).fillna(False)
    labels["above_capacity"] = labels["capacity_factor"].gt(1.02).fillna(False)
    if energy_type == "solar":
        labels["weather_output_conflict"] = (
            labels["solar_elevation_deg"].gt(10)
            & labels["shortwave_radiation_w_m2"].gt(200)
            & labels["capacity_factor"].lt(0.005)
        ).fillna(False)
    else:
        labels["weather_output_conflict"] = (
            labels["wind_speed_80m_ms"].ge(8) & labels["capacity_factor"].lt(0.01)
        ).fillna(False)
    flags = [
        "missing_label",
        "negative_label",
        "above_capacity",
        "weather_output_conflict",
    ]
    rows = []
    for asset, part in labels.groupby("asset_id", observed=True):
        row = {"energy_type": energy_type, "asset_id": asset, "hours": len(part)}
        for flag in flags:
            row[f"{flag}_hours"] = int(part[flag].sum())
            row[f"{flag}_pct"] = float(100 * part[flag].mean())
        rows.append(row)
    report = pd.DataFrame(rows)
    summary = {
        "energy_type": energy_type,
        "development_years": [2023, 2024],
        "locked_2025_loaded": False,
        "unique_label_hours": int(len(labels)),
        **{f"{flag}_hours": int(labels[flag].sum()) for flag in flags},
    }
    return report, summary


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    reports = []
    summaries = {}
    for energy_type in ("solar", "wind"):
        report, summary = audit_one(args.data / f"{energy_type}_training_mart.parquet", energy_type)
        reports.append(report)
        summaries[energy_type] = summary
    combined = pd.concat(reports, ignore_index=True)
    combined.to_csv(args.reports / "generation_label_audit.csv", index=False, encoding="utf-8-sig")
    (args.reports / "generation_label_audit.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
