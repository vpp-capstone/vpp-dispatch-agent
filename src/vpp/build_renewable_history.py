"""Build team-ready Jeju renewable tables from collected public snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from vpp.renewable_targets import KEYS, normalize_kpx, numeric, read_source

PLANTS = {
    "교래태양광": "gyorae", "교래태양광발전소": "gyorae",
    "종합경기장태양광": "stadium", "제주종합경기장태양광발전소": "stadium",
    "행원태양광": "haengwon", "행원태양광발전소": "haengwon",
    "홍보관주차장태양광": "promotion_parking",
    "신재생에너지홍보관주차장태양광": "promotion_parking", "수산태양광": "susan",
    "가시": "gasi", "가시리": "gasi", "가시리풍력": "gasi",
    "김녕": "gimnyeong", "김녕리": "gimnyeong", "김녕풍력": "gimnyeong",
    "김녕풍력발전기": "gimnyeong",
    "동복": "dongbok", "동복리": "dongbok", "동복풍력": "dongbok",
    "동복북촌풍력": "dongbok",
    "신창": "sinchang", "신창리": "sinchang", "신창풍력": "sinchang",
    "행원": "haengwon", "행원리": "haengwon", "행원풍력": "haengwon",
    "전체": "reported_total", "태양광전체": "reported_total",
    "풍력전체": "reported_total",
}


def resolve_duplicates(frame: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Keep identical observations; quarantine conflicting values without averaging."""
    counts = frame.groupby(keys)[value].nunique(dropna=False)
    conflict_keys = counts[counts.gt(1)].index
    frame = frame.copy()
    frame["cross_file_conflict"] = pd.MultiIndex.from_frame(frame[keys]).isin(conflict_keys)
    provenance = frame.groupby(keys).source_file.agg(lambda s: "|".join(sorted(set(s))))
    result = frame.drop_duplicates(keys).drop(columns="source_file")
    result = result.merge(provenance.rename("source_file").reset_index(), on=keys)
    result.loc[result.cross_file_conflict, value] = float("nan")
    return result


def daily_observations(path: Path, fuel: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Normalize plant aliases, retaining unverified daily source units and bad rows."""
    frame = read_source(path)
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    valid = dates.notna() & dates.eq(dates.dt.normalize())
    rejected = frame.loc[~valid].assign(source_file=path.name)
    data = frame.loc[valid].copy()
    data["date"] = dates.loc[valid].dt.strftime("%Y-%m-%d")
    mapping = {}
    ignored_columns = []
    for column in frame.columns[1:]:
        compact = column.replace(" ", "")
        if compact in PLANTS:
            mapping[column] = PLANTS[compact]
        elif data[column].astype("string").str.strip().replace("", pd.NA).isna().all():
            ignored_columns.append(column)
        else:
            raise ValueError(f"{path.name}: unmapped populated column {column}")
    if not mapping:
        raise ValueError(f"{path.name}: no recognized plant columns")
    long = data.melt(
        id_vars="date", value_vars=list(mapping), var_name="plant_name", value_name="raw_value",
    )
    long["plant_id"] = long.plant_name.map(mapping)
    long["fuel"] = fuel
    long["value_source_unit"] = numeric(long.raw_value)
    long["unit_status"] = "unverified_daily_source_unit"
    long["source_file"] = path.name
    return long, rejected, {
        "file": path.name, "source": f"jec_{fuel}", "original_rows": len(frame),
        "rejected_date_rows": int((~valid).sum()), "ignored_empty_columns": ignored_columns,
        "start_date": data.date.min(), "end_date": data.date.max(),
        "has_reported_total": "reported_total" in mapping.values(),
    }


def build(
    raw: Path, output: Path, start: str, end: str, api_report: Path | None = None,
) -> dict:
    """Build hourly regional trades and separate daily plant references."""
    output.mkdir(parents=True, exist_ok=True)
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
        raise ValueError("Invalid start/end")
    if start_date != start_date.normalize() or end_date != end_date.normalize():
        raise ValueError("start/end must be dates without a time")
    files, kpx_frames, daily_frames, rejected_frames, skipped = [], [], [], [], []
    manifests = {}
    for path in sorted(raw.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict) and "file" in record:
            manifests[record["file"]] = record
    for path in sorted(raw.glob("*.csv")):
        if path.name in manifests:
            expected_hash = manifests[path.name]["sha256"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"{path.name}: raw file differs from downloaded checksum")
        if path.name.startswith("kpx_"):
            # This old solar-only file has no declared unit in its header and is redundant.
            if "20211231" in path.name:
                skipped.append({
                    "file": path.name,
                    "reason": "Legacy unit unverified; later file covers dates",
                })
                continue
            frame, info = normalize_kpx(path)
            info.pop("_conflict_rows")
            frame = frame[frame.trade_date.between(start_date, end_date)].copy()
            kpx_frames.append(frame)
        elif path.name.startswith(("jec_solar_", "jec_wind_")):
            fuel = "solar" if path.name.startswith("jec_solar_") else "wind"
            frame, rejected, info = daily_observations(path, fuel)
            daily_frames.append(frame)
            rejected_frames.append(rejected)
        else:
            continue
        if path.name in manifests:
            info.update({k: manifests[path.name][k] for k in
                         ("sha256", "source_url", "download_url", "downloaded_utc")})
        files.append(info)
    if not kpx_frames or not daily_frames:
        raise ValueError("Download KPX and JEC snapshots before building")
    observations = pd.concat(kpx_frames, ignore_index=True)
    main = resolve_duplicates(observations, KEYS, "traded_energy_mwh")
    calendar = pd.MultiIndex.from_product(
        [pd.date_range(start_date, end_date), range(1, 25), ["solar", "wind"]], names=KEYS,
    ).to_frame(index=False)
    hourly = calendar.merge(main, on=KEYS, how="left", validate="one_to_one")
    hourly["quality_flag"] = "observed"
    hourly.loc[hourly.source_file.isna(), "quality_flag"] = "missing_hour"
    invalid_value = hourly.source_file.notna() & hourly.traded_energy_mwh.isna()
    hourly.loc[invalid_value, "quality_flag"] = "missing_value"
    conflict = hourly.cross_file_conflict.eq(True) | hourly.conflicting_duplicate.eq(True)
    hourly.loc[conflict, "quality_flag"] = "conflicting_duplicate"
    hourly.loc[hourly.traded_energy_mwh.lt(0), "quality_flag"] = "negative_review"
    hourly["target_available"] = hourly.quality_flag.eq("observed")
    hourly["ts"] = (
        hourly.trade_date + pd.to_timedelta(hourly.hour_label - 1, unit="h")
    ).dt.tz_localize("Asia/Seoul")
    hourly["time_basis"] = "interval_start_team_convention_source_confirmation_pending"
    hourly["is_imputed"] = False
    columns = ["ts", *KEYS, "traded_energy_mwh", "quality_flag", "target_available",
               "source_file", "raw_value", "time_basis", "is_imputed"]
    hourly[columns].to_csv(output / "renewable_hourly_audit.csv", index=False, encoding="utf-8-sig")
    issues = hourly[~hourly.target_available]
    issues[columns].to_csv(
        output / "renewable_hourly_issues.csv", index=False, encoding="utf-8-sig",
    )
    wide = pd.DataFrame({"ts": sorted(hourly.ts.unique())})
    for fuel in ("solar", "wind"):
        part = hourly[hourly.fuel.eq(fuel)].copy()
        part.loc[~part.target_available, "traded_energy_mwh"] = float("nan")
        part = part[["ts", "traded_energy_mwh", "quality_flag", "source_file"]].rename(columns={
            "traded_energy_mwh": f"{fuel}_traded_mwh", "quality_flag": f"{fuel}_quality",
            "source_file": f"{fuel}_source",
        })
        wide = wide.merge(part, on="ts", validate="one_to_one")
    wide.to_csv(output / "renewables_hourly.csv", index=False, encoding="utf-8-sig")
    raw_daily = pd.concat(daily_frames, ignore_index=True)
    daily = resolve_duplicates(raw_daily, ["date", "fuel", "plant_id"], "value_source_unit")
    daily["quality_flag"] = "observed_unit_unverified"
    daily.loc[daily.value_source_unit.isna(), "quality_flag"] = "missing_value"
    daily.loc[daily.cross_file_conflict, "quality_flag"] = "conflicting_duplicate"
    daily.loc[daily.value_source_unit.lt(0), "quality_flag"] = "negative_review"
    daily["is_reported_total"] = daily.plant_id.eq("reported_total")
    daily.to_csv(output / "jec_daily_plants.csv", index=False, encoding="utf-8-sig")
    pd.concat(rejected_frames, ignore_index=True).to_csv(
        output / "jec_rejected_rows.csv", index=False, encoding="utf-8-sig",
    )
    daily[daily.cross_file_conflict].to_csv(output / "jec_conflict_keys.csv", index=False)
    raw_daily.to_csv(output / "jec_all_observations.csv", index=False, encoding="utf-8-sig")
    daily_coverage = {}
    coverage_rows = []
    for fuel, part in daily.groupby("fuel"):
        dates = pd.to_datetime(part.date.unique())
        expected = pd.date_range(dates.min(), dates.max())
        missing_dates = expected.difference(dates)
        coverage_rows.extend({"date": date.date().isoformat(), "fuel": fuel,
                              "has_any_source_row": date in dates}
                             for date in expected)
        daily_coverage[fuel] = {
            "start": dates.min().date().isoformat(), "end": dates.max().date().isoformat(),
            "available_dates": len(dates), "missing_dates": len(missing_dates),
            "missing_date_list": [d.date().isoformat() for d in missing_dates],
            "conflicting_plant_date_keys": int(part.cross_file_conflict.sum()),
            "quality_flags": part.quality_flag.value_counts().to_dict(),
            "overlap_dates_with_kpx": len(set(dates) & set(hourly.trade_date)),
        }
    pd.DataFrame(coverage_rows).to_csv(output / "jec_daily_coverage.csv", index=False)
    report = {
        "start": start, "end": end, "hourly_rows": len(wide),
        "time_basis": "KST interval start; follows market-team convention, not source verified",
        "hourly_population": "Jeju market-participant aggregate including existing ESS",
        "daily_population": "JEC operated plants; units unverified; cannot hourly interpolate",
        "hourly_quality": {
            fuel: part.quality_flag.value_counts().to_dict()
            for fuel, part in hourly.groupby("fuel")
        },
        "daily_coverage": daily_coverage, "files": files, "skipped_files": skipped,
        "kpx_solar_api": "not_collected", "shared_contract_approved": False,
    }
    if api_report is not None:
        api = json.loads(api_report.read_text(encoding="utf-8"))
        report["kpx_solar_api"] = {
            "status": "collected_compared" if "comparison" in api else "collected",
            "complete": api["complete"], "scope": api["scope"],
            "jeju_rows": api["jeju_rows"], "comparison": api.get("comparison"),
            "source_url": api["source_url"], "collected_utc": api["collected_utc"],
        }
    (output / "history_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main() -> None:
    """CLI after public snapshot collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/renewable_sources"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/renewable_history"))
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--api-report", type=Path)
    args = parser.parse_args()
    report = build(args.raw_dir, args.output_dir, args.start, args.end, args.api_report)
    print(json.dumps({k: report[k] for k in ("hourly_rows", "hourly_quality")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
