"""Download and validate one leakage-safe NOAA GFS day-ahead sample.

The bidding rule is represented by the D-2 18 UTC GFS cycle.  For Korean
calendar day D, forecast hours 21..44 cover D 00:00..23:00 KST and the cycle
is available before the D-1 11:00 KST bidding deadline.

Only selected GRIB records are downloaded with HTTP byte-range requests; the
500+ MB global files are never downloaded in full.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from vpp.generation_forecast.paths import PLANT_MASTER_PATH, SAMPLE_DIR

DEFAULT_LOCATIONS = PLANT_MASTER_PATH
OUTPUT_ROOT = SAMPLE_DIR
KST = timezone(timedelta(hours=9))
S3_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


@dataclass(frozen=True)
class IndexRecord:
    start: int
    end: int
    description: str


VARIABLE_PATTERNS = {
    "temperature_2m": lambda line: ":TMP:2 m above ground:" in line,
    "relative_humidity_2m": lambda line: ":RH:2 m above ground:" in line,
    "wind_u_80m": lambda line: ":UGRD:80 m above ground:" in line,
    "wind_v_80m": lambda line: ":VGRD:80 m above ground:" in line,
    "total_cloud_cover": lambda line: (
        ":TCDC:entire atmosphere:" in line and " ave fcst:" not in line
    ),
    "shortwave_radiation_interval_average": lambda line: ":DSWRF:surface:" in line,
    "precipitation_accumulated": lambda line: (
        ":APCP:surface:0-" in line and " acc fcst:" in line
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-day", type=date.fromisoformat, default=date(2025, 1, 3))
    parser.add_argument("--locations", type=Path, default=DEFAULT_LOCATIONS)
    # ecCodes' Windows wheel is not reliably thread-safe while decoding GRIB.
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser.parse_args()


def issue_for_valid_day(valid_day: date) -> datetime:
    return datetime.combine(
        valid_day - timedelta(days=2), datetime.min.time(), timezone.utc
    ).replace(hour=18)


def object_url(issue_utc: datetime, forecast_hour: int) -> str:
    return (
        f"{S3_BASE}/gfs.{issue_utc:%Y%m%d}/{issue_utc:%H}/atmos/"
        f"gfs.t{issue_utc:%H}z.pgrb2.0p25.f{forecast_hour:03d}"
    )


def load_grid_points(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    plants = pd.read_csv(path)
    required = {
        "asset_id",
        "energy_type",
        "latitude",
        "longitude",
        "gfs_grid_latitude",
        "gfs_grid_longitude",
        "gfs_grid_id",
    }
    missing = required - set(plants.columns)
    if missing:
        raise ValueError(f"location master fields missing: {sorted(missing)}")
    if plants["asset_id"].duplicated().any():
        raise ValueError("asset_id contains duplicates")
    if len(plants) != 21 or plants["energy_type"].value_counts().to_dict() != {
        "solar": 18,
        "wind": 3,
    }:
        raise ValueError("location master must contain 18 solar and 3 wind assets")

    grids = (
        plants[["gfs_grid_id", "gfs_grid_latitude", "gfs_grid_longitude"]]
        .drop_duplicates()
        .sort_values("gfs_grid_id")
        .reset_index(drop=True)
        .rename(
            columns={
                "gfs_grid_latitude": "latitude",
                "gfs_grid_longitude": "longitude",
            }
        )
    )
    return plants, grids


def parse_index(text: str, object_size: int) -> list[IndexRecord]:
    parsed: list[tuple[int, str]] = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        parsed.append((int(parts[1]), line))
    records: list[IndexRecord] = []
    for index, (start, description) in enumerate(parsed):
        end = parsed[index + 1][0] - 1 if index + 1 < len(parsed) else object_size - 1
        records.append(IndexRecord(start=start, end=end, description=description))
    return records


def select_records(records: list[IndexRecord], forecast_hour: int) -> dict[str, IndexRecord]:
    wanted = dict(VARIABLE_PATTERNS)
    # Hour 20 is needed only to de-accumulate the first valid hour.
    if forecast_hour == 20:
        wanted = {
            key: matcher
            for key, matcher in wanted.items()
            if key in {
                "shortwave_radiation_interval_average",
                "precipitation_accumulated",
            }
        }
    selected: dict[str, IndexRecord] = {}
    for variable, matcher in wanted.items():
        matches = [record for record in records if matcher(record.description)]
        if len(matches) != 1:
            raise ValueError(
                f"record match failure: f={forecast_hour}, variable={variable}, "
                f"matches={len(matches)}"
            )
        selected[variable] = matches[0]
    return selected


def request_bytes(
    session: requests.Session, url: str, start: int, end: int, timeout: float
) -> requests.Response:
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=(15, timeout),
    )
    if response.status_code != 206:
        raise RuntimeError(f"range request failed: HTTP {response.status_code}, url={url}")
    if len(response.content) != end - start + 1:
        raise RuntimeError("range response byte length mismatch")
    return response


def decode_nearest_values(
    message: bytes, grids: pd.DataFrame
) -> tuple[dict[str, float], str]:
    try:
        import eccodes
    except ImportError as exc:
        raise RuntimeError(
            "GRIB decoding requires the optional 'eccodes' package"
        ) from exc

    handle = eccodes.codes_new_from_message(message)
    try:
        units = str(eccodes.codes_get(handle, "units"))
        values: dict[str, float] = {}
        for grid in grids.itertuples(index=False):
            result = eccodes.codes_grib_find_nearest(
                handle, float(grid.latitude), float(grid.longitude)
            )[0]
            if abs(float(result["lat"]) - float(grid.latitude)) > 1e-9:
                raise ValueError(f"GFS latitude mismatch for {grid.gfs_grid_id}: {result}")
            result_lon = float(result["lon"]) % 360
            expected_lon = float(grid.longitude) % 360
            if abs(result_lon - expected_lon) > 1e-9:
                raise ValueError(f"GFS longitude mismatch for {grid.gfs_grid_id}: {result}")
            values[str(grid.gfs_grid_id)] = float(result["value"])
        return values, units
    finally:
        eccodes.codes_release(handle)


def fetch_forecast_hour(
    issue_utc: datetime,
    forecast_hour: int,
    grids: pd.DataFrame,
    timeout: float,
) -> dict:
    url = object_url(issue_utc, forecast_hour)
    with requests.Session() as session:
        index_response = session.get(url + ".idx", timeout=(15, timeout))
        index_response.raise_for_status()
        head = session.head(url, timeout=(15, timeout))
        head.raise_for_status()
        object_size = int(head.headers["Content-Length"])
        records = parse_index(index_response.text, object_size)
        selected = select_records(records, forecast_hour)

        output: dict[str, object] = {
            "forecast_hour": forecast_hour,
            "object_url": url,
            "last_modified_utc": parsedate_to_datetime(head.headers["Last-Modified"]),
            "values": {},
            "units": {},
            "descriptions": {},
        }
        for variable, record in selected.items():
            response = request_bytes(session, url, record.start, record.end, timeout)
            values, units = decode_nearest_values(response.content, grids)
            output["values"][variable] = values
            output["units"][variable] = units
            output["descriptions"][variable] = record.description
        return output


INTERVAL_RE = re.compile(r":(\d+)-(\d+) hour ave fcst:")


def radiation_interval(description: str) -> tuple[int, int]:
    match = INTERVAL_RE.search(description)
    if not match:
        raise ValueError(f"cannot parse radiation interval: {description}")
    return int(match.group(1)), int(match.group(2))


def to_hourly_frame(
    results: list[dict], grids: pd.DataFrame, issue_utc: datetime, valid_day: date
) -> pd.DataFrame:
    by_hour = {int(item["forecast_hour"]): item for item in results}
    rows: list[dict] = []
    bid_deadline_kst = pd.Timestamp(
        datetime.combine(valid_day - timedelta(days=1), datetime.min.time(), KST).replace(
            hour=11
        )
    )

    for forecast_hour in range(21, 45):
        current = by_hour[forecast_hour]
        previous = by_hour[forecast_hour - 1]
        valid_utc = pd.Timestamp(issue_utc + timedelta(hours=forecast_hour))
        valid_kst = valid_utc.tz_convert("Asia/Seoul")
        current_modified = pd.Timestamp(current["last_modified_utc"])

        current_interval = radiation_interval(
            current["descriptions"]["shortwave_radiation_interval_average"]
        )
        previous_interval = radiation_interval(
            previous["descriptions"]["shortwave_radiation_interval_average"]
        )
        current_duration = current_interval[1] - current_interval[0]
        previous_duration = previous_interval[1] - previous_interval[0]

        for grid in grids.itertuples(index=False):
            grid_id = str(grid.gfs_grid_id)
            values = current["values"]
            previous_values = previous["values"]

            radiation_average = values["shortwave_radiation_interval_average"][grid_id]
            if current_duration == 1:
                radiation_hourly = radiation_average
            elif (
                current_interval[0] == previous_interval[0]
                and current_interval[1] == previous_interval[1] + 1
            ):
                radiation_hourly = (
                    radiation_average * current_duration
                    - previous_values["shortwave_radiation_interval_average"][grid_id]
                    * previous_duration
                )
            else:
                raise ValueError(
                    f"unsupported radiation interval transition: "
                    f"{previous_interval} -> {current_interval}"
                )

            precipitation = (
                values["precipitation_accumulated"][grid_id]
                - previous_values["precipitation_accumulated"][grid_id]
            )
            u80 = values["wind_u_80m"][grid_id]
            v80 = values["wind_v_80m"][grid_id]
            speed80 = math.hypot(u80, v80)
            direction80 = math.degrees(math.atan2(-u80, -v80)) % 360
            temperature = values["temperature_2m"][grid_id]
            temperature_c = temperature - 273.15 if temperature > 150 else temperature

            rows.append(
                {
                    "gfs_grid_id": grid_id,
                    "gfs_grid_latitude": float(grid.latitude),
                    "gfs_grid_longitude": float(grid.longitude),
                    "model": "NOAA_GFS_0p25",
                    "issue_time_utc": pd.Timestamp(issue_utc),
                    "issue_time_kst": pd.Timestamp(issue_utc).tz_convert("Asia/Seoul"),
                    "source_file_last_modified_utc": current_modified,
                    "bid_deadline_kst": bid_deadline_kst,
                    "available_before_bid_deadline": (
                        current_modified.tz_convert("Asia/Seoul") <= bid_deadline_kst
                    ),
                    "valid_time_utc": valid_utc,
                    "valid_time_kst": valid_kst,
                    "lead_hour": forecast_hour,
                    "shortwave_radiation_w_m2": max(0.0, radiation_hourly),
                    "shortwave_radiation_raw_derived_w_m2": radiation_hourly,
                    "total_cloud_cover_pct": values["total_cloud_cover"][grid_id],
                    "temperature_2m_c": temperature_c,
                    "relative_humidity_2m_pct": values["relative_humidity_2m"][grid_id],
                    "precipitation_1h_mm": max(0.0, precipitation),
                    "precipitation_raw_derived_mm": precipitation,
                    "wind_u_80m_ms": u80,
                    "wind_v_80m_ms": v80,
                    "wind_speed_80m_ms": speed80,
                    "wind_direction_80m_deg": direction80,
                }
            )
    return pd.DataFrame(rows)


def validate_sample(
    frame: pd.DataFrame, plants: pd.DataFrame, grids: pd.DataFrame, valid_day: date
) -> dict:
    expected_rows = len(grids) * 24
    duplicate_count = int(frame.duplicated(["gfs_grid_id", "valid_time_utc"]).sum())
    missing_count = int(frame.isna().sum().sum())
    valid_dates = set(frame["valid_time_kst"].dt.date)
    per_grid_counts = frame.groupby("gfs_grid_id")["valid_time_utc"].nunique().to_dict()
    checks = {
        "asset_count": len(plants) == 21,
        "solar_asset_count": int((plants["energy_type"] == "solar").sum()) == 18,
        "wind_asset_count": int((plants["energy_type"] == "wind").sum()) == 3,
        "unique_gfs_grid_count": len(grids) == 5,
        "row_count": len(frame) == expected_rows,
        "no_duplicates": duplicate_count == 0,
        "no_missing_values": missing_count == 0,
        "lead_hours_21_to_44": frame["lead_hour"].tolist() != []
        and frame["lead_hour"].min() == 21
        and frame["lead_hour"].max() == 44,
        "valid_kst_day_exact": valid_dates == {valid_day},
        "24_hours_per_grid": set(per_grid_counts.values()) == {24},
        "all_available_before_bid_deadline": bool(
            frame["available_before_bid_deadline"].all()
        ),
        "solar_radiation_nonnegative": bool(
            (frame["shortwave_radiation_w_m2"] >= 0).all()
        ),
        "humidity_in_range": bool(
            frame["relative_humidity_2m_pct"].between(0, 100).all()
        ),
        "cloud_cover_in_range": bool(frame["total_cloud_cover_pct"].between(0, 100).all()),
        "wind_speed_nonnegative": bool((frame["wind_speed_80m_ms"] >= 0).all()),
    }
    return {
        "valid_day_kst": valid_day.isoformat(),
        "issue_time_utc": frame["issue_time_utc"].iloc[0].isoformat(),
        "issue_time_kst": frame["issue_time_kst"].iloc[0].isoformat(),
        "bid_deadline_kst": frame["bid_deadline_kst"].iloc[0].isoformat(),
        "latest_source_file_last_modified_utc": frame[
            "source_file_last_modified_utc"
        ].max().isoformat(),
        "asset_count": int(len(plants)),
        "solar_assets": int((plants["energy_type"] == "solar").sum()),
        "wind_assets": int((plants["energy_type"] == "wind").sum()),
        "unique_gfs_grids": int(len(grids)),
        "sample_rows": int(len(frame)),
        "expected_rows": int(expected_rows),
        "duplicate_count": duplicate_count,
        "missing_cell_count": missing_count,
        "per_grid_hour_counts": {str(k): int(v) for k, v in per_grid_counts.items()},
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_report(report: dict, output_path: Path) -> None:
    lines = [
        "# NOAA GFS 하루 샘플 검증 결과",
        "",
        f"- 유효일(KST): `{report['valid_day_kst']}`",
        f"- 발표시각: `{report['issue_time_utc']}` / `{report['issue_time_kst']}`",
        f"- 입찰 마감: `{report['bid_deadline_kst']}`",
        f"- 발전소: 태양광 {report['solar_assets']}개 + 풍력 {report['wind_assets']}개",
        f"- 중복 제거 후 GFS 격자: {report['unique_gfs_grids']}개",
        f"- 샘플 행: {report['sample_rows']} / 기대 {report['expected_rows']}",
        f"- 종합 결과: `{'PASS' if report['passed'] else 'FAIL'}`",
        "",
        "## 검사 항목",
        "",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    plants, grids = load_grid_points(args.locations)
    issue_utc = issue_for_valid_day(args.valid_day)
    forecast_hours = list(range(20, 45))
    print(
        f"GFS sample: valid_day={args.valid_day}, issue={issue_utc.isoformat()}, "
        f"assets={len(plants)}, unique_grids={len(grids)}"
    )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                fetch_forecast_hour, issue_utc, hour, grids, args.timeout
            ): hour
            for hour in forecast_hours
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            hour = futures[future]
            results.append(future.result())
            print(f"[{completed:02d}/{len(forecast_hours)}] forecast hour {hour:03d} complete")

    frame = to_hourly_frame(results, grids, issue_utc, args.valid_day)
    report = validate_sample(frame, plants, grids, args.valid_day)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    date_tag = args.valid_day.strftime("%Y%m%d")
    sample_path = OUTPUT_ROOT / f"gfs_day_ahead_{date_tag}.parquet"
    mapping_path = OUTPUT_ROOT / "plant_to_gfs_grid_mapping.parquet"
    json_path = OUTPUT_ROOT / f"gfs_day_ahead_{date_tag}_validation.json"
    markdown_path = OUTPUT_ROOT / f"gfs_day_ahead_{date_tag}_validation.md"
    frame.to_parquet(sample_path, index=False, compression="zstd")
    plants.to_parquet(mapping_path, index=False, compression="zstd")
    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else str(value),
        ),
        encoding="utf-8",
    )
    write_report(report, markdown_path)
    print(f"sample: {sample_path}")
    print(f"mapping: {mapping_path}")
    print(f"validation: {markdown_path}")
    if not report["passed"]:
        raise SystemExit("GFS sample validation failed")


if __name__ == "__main__":
    main()
