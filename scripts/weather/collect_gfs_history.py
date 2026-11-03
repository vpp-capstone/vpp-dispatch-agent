"""Collect leakage-safe NOAA GFS 48-hour weather forecasts for 2023-2025.

For Korean planning day D this collector uses the D-2 18 UTC GFS cycle. Leads
21..68 cover D 00:00 through D+1 23:00 KST and are checked against the D-1
11:00 KST bidding deadline. Only selected GRIB messages are fetched with HTTP
Range.

The job is restartable: each completed valid day is written atomically to a
ZSTD Parquet checkpoint.  Re-running the same command skips completed days.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts.weather.collect_gfs_dayahead_sample import (
    IndexRecord,
    decode_nearest_values,
    issue_for_valid_day,
    load_grid_points,
    object_url,
    parse_index,
    radiation_interval,
)
from vpp.generation_forecast.paths import (
    GFS_GRID_MASTER_PATH,
    PLANT_MASTER_PATH,
    RAW_GFS_DIR,
)


DEFAULT_LOCATIONS = PLANT_MASTER_PATH
DEFAULT_GRIDS = GFS_GRID_MASTER_PATH
DEFAULT_OUTPUT = RAW_GFS_DIR
KST = timezone(timedelta(hours=9))
FIRST_LEAD_HOUR = 21
LAST_LEAD_HOUR = 68
FORECAST_HORIZON_HOURS = LAST_LEAD_HOUR - FIRST_LEAD_HOUR + 1
RADIATION_PREVIOUS_HOUR = FIRST_LEAD_HOUR - 1
# State fields are sampled every three hours. f069 is needed to interpolate
# f067 and f068, but wind and radiation at f069 are not downloaded.
LAST_STATE_SAMPLE_HOUR = 69

PATTERNS = {
    "temperature_2m": lambda line: ":TMP:2 m above ground:" in line,
    "wind_u_80m": lambda line: ":UGRD:80 m above ground:" in line,
    "wind_v_80m": lambda line: ":VGRD:80 m above ground:" in line,
    "total_cloud_cover": lambda line: (
        ":TCDC:entire atmosphere:" in line and " ave fcst:" not in line
    ),
    "shortwave_radiation_interval_average": lambda line: ":DSWRF:surface:" in line,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--locations", type=Path, default=DEFAULT_LOCATIONS)
    parser.add_argument(
        "--grids",
        type=Path,
        default=DEFAULT_GRIDS,
        help="Jeju-wide GFS extraction points used by generation and SMP models",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--day-attempts", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=30.0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="process dates currently listed in failed_dates.json only",
    )
    return parser.parse_args()


def make_session(retries: int) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET", "HEAD")),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "jeju-vpp-gfs-history/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2))
    return session


def load_collection_grids(path: Path) -> pd.DataFrame:
    grids = pd.read_csv(path)
    required = {"gfs_grid_id", "latitude", "longitude", "jeju_region"}
    missing = required - set(grids.columns)
    if missing:
        raise ValueError(f"grid fields missing: {sorted(missing)}")
    if len(grids) != 6 or grids["gfs_grid_id"].nunique() != 6:
        raise ValueError("Jeju GFS grid master must contain exactly 6 unique grids")
    expected = {
        (33.25, 126.25),
        (33.25, 126.50),
        (33.25, 126.75),
        (33.50, 126.25),
        (33.50, 126.50),
        (33.50, 126.75),
    }
    actual = set(zip(grids["latitude"].astype(float), grids["longitude"].astype(float)))
    if actual != expected:
        raise ValueError(f"unexpected Jeju grid coordinates: {sorted(actual)}")
    return grids.sort_values("gfs_grid_id").reset_index(drop=True)


def variables_for_hour(hour: int) -> tuple[str, ...]:
    variables: list[str] = []
    if RADIATION_PREVIOUS_HOUR <= hour <= LAST_LEAD_HOUR:
        variables.append("shortwave_radiation_interval_average")
    if FIRST_LEAD_HOUR <= hour <= LAST_LEAD_HOUR:
        variables.extend(("wind_u_80m", "wind_v_80m"))
    if hour in range(FIRST_LEAD_HOUR, LAST_STATE_SAMPLE_HOUR + 1, 3):
        variables.extend(("temperature_2m", "total_cloud_cover"))
    return tuple(variables)


def select_records(
    records: list[IndexRecord], forecast_hour: int
) -> dict[str, IndexRecord]:
    selected: dict[str, IndexRecord] = {}
    for variable in variables_for_hour(forecast_hour):
        matches = [record for record in records if PATTERNS[variable](record.description)]
        if len(matches) != 1:
            raise ValueError(
                f"record match failure: f={forecast_hour}, variable={variable}, "
                f"matches={len(matches)}"
            )
        selected[variable] = matches[0]
    return selected


def request_range(
    session: requests.Session, url: str, record: IndexRecord, timeout: float
) -> bytes:
    response = session.get(
        url,
        headers={"Range": f"bytes={record.start}-{record.end}"},
        timeout=(20, timeout),
    )
    if response.status_code != 206:
        raise RuntimeError(f"range request HTTP {response.status_code}: {url}")
    expected = record.end - record.start + 1
    if len(response.content) != expected:
        raise RuntimeError(
            f"range byte mismatch: expected={expected}, got={len(response.content)}"
        )
    return response.content


def fetch_hour(
    session: requests.Session,
    issue_utc: datetime,
    hour: int,
    grids: pd.DataFrame,
    timeout: float,
) -> tuple[dict, int]:
    url = object_url(issue_utc, hour)
    index_response = session.get(url + ".idx", timeout=(20, timeout))
    index_response.raise_for_status()
    head = session.head(url, timeout=(20, timeout))
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    selected = select_records(parse_index(index_response.text, size), hour)
    downloaded = 0
    output: dict[str, object] = {
        "forecast_hour": hour,
        "object_url": url,
        "last_modified_utc": pd.Timestamp(head.headers["Last-Modified"]),
        "values": {},
        "descriptions": {},
    }
    for variable, record in selected.items():
        message = request_range(session, url, record, timeout)
        downloaded += len(message)
        values, _units = decode_nearest_values(message, grids)
        output["values"][variable] = values
        output["descriptions"][variable] = record.description
    return output, downloaded


def interpolate_state(results: dict[int, dict], variable: str, hour: int, grid_id: str) -> float:
    lower = FIRST_LEAD_HOUR + ((hour - FIRST_LEAD_HOUR) // 3) * 3
    upper = lower if hour == lower else lower + 3
    lower_value = float(results[lower]["values"][variable][grid_id])
    if upper == lower:
        return lower_value
    upper_value = float(results[upper]["values"][variable][grid_id])
    fraction = (hour - lower) / 3.0
    return lower_value + fraction * (upper_value - lower_value)


def make_day_frame(
    fetched: list[dict], grids: pd.DataFrame, issue_utc: datetime, valid_day: date
) -> pd.DataFrame:
    results = {int(item["forecast_hour"]): item for item in fetched}
    bid_deadline = pd.Timestamp(
        datetime.combine(valid_day - timedelta(days=1), datetime.min.time(), KST).replace(hour=11)
    )
    rows: list[dict] = []
    for hour in range(FIRST_LEAD_HOUR, LAST_LEAD_HOUR + 1):
        current = results[hour]
        previous = results[hour - 1]
        current_interval = radiation_interval(
            current["descriptions"]["shortwave_radiation_interval_average"]
        )
        previous_interval = radiation_interval(
            previous["descriptions"]["shortwave_radiation_interval_average"]
        )
        current_duration = current_interval[1] - current_interval[0]
        previous_duration = previous_interval[1] - previous_interval[0]
        valid_utc = pd.Timestamp(issue_utc + timedelta(hours=hour))
        last_modified = pd.Timestamp(current["last_modified_utc"])
        for grid in grids.itertuples(index=False):
            grid_id = str(grid.gfs_grid_id)
            radiation_avg = float(
                current["values"]["shortwave_radiation_interval_average"][grid_id]
            )
            if current_duration == 1:
                radiation_raw = radiation_avg
            elif (
                current_interval[0] == previous_interval[0]
                and current_interval[1] == previous_interval[1] + 1
            ):
                previous_avg = float(
                    previous["values"]["shortwave_radiation_interval_average"][grid_id]
                )
                radiation_raw = (
                    radiation_avg * current_duration - previous_avg * previous_duration
                )
            else:
                raise ValueError(
                    f"unsupported radiation interval {previous_interval}->{current_interval}"
                )
            u80 = float(current["values"]["wind_u_80m"][grid_id])
            v80 = float(current["values"]["wind_v_80m"][grid_id])
            temperature = interpolate_state(results, "temperature_2m", hour, grid_id)
            temperature_c = temperature - 273.15 if temperature > 150 else temperature
            cloud = interpolate_state(results, "total_cloud_cover", hour, grid_id)
            rows.append(
                {
                    "gfs_grid_id": grid_id,
                    "gfs_grid_latitude": float(grid.latitude),
                    "gfs_grid_longitude": float(grid.longitude),
                    "model": "NOAA_GFS_0p25",
                    "issue_time_utc": pd.Timestamp(issue_utc),
                    "issue_time_kst": pd.Timestamp(issue_utc).tz_convert("Asia/Seoul"),
                    "source_file_last_modified_utc": last_modified,
                    "bid_deadline_kst": bid_deadline,
                    "available_before_bid_deadline": bool(
                        last_modified.tz_convert("Asia/Seoul") <= bid_deadline
                    ),
                    "valid_time_utc": valid_utc,
                    "valid_time_kst": valid_utc.tz_convert("Asia/Seoul"),
                    "lead_hour": hour,
                    "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
                    "shortwave_radiation_w_m2": max(0.0, radiation_raw),
                    "shortwave_radiation_raw_derived_w_m2": radiation_raw,
                    "temperature_2m_c": temperature_c,
                    "temperature_is_interpolated": hour
                    not in range(FIRST_LEAD_HOUR, LAST_STATE_SAMPLE_HOUR + 1, 3),
                    "total_cloud_cover_pct": min(100.0, max(0.0, cloud)),
                    "cloud_cover_is_interpolated": hour
                    not in range(FIRST_LEAD_HOUR, LAST_STATE_SAMPLE_HOUR + 1, 3),
                    "wind_u_80m_ms": u80,
                    "wind_v_80m_ms": v80,
                    "wind_speed_80m_ms": math.hypot(u80, v80),
                    "wind_direction_80m_deg": math.degrees(math.atan2(-u80, -v80)) % 360,
                }
            )
    return pd.DataFrame(rows)


def validate_day(frame: pd.DataFrame, grid_count: int, valid_day: date) -> None:
    errors: list[str] = []
    if len(frame) != grid_count * FORECAST_HORIZON_HOURS:
        errors.append(
            f"rows={len(frame)} expected={grid_count * FORECAST_HORIZON_HOURS}"
        )
    if frame.duplicated(["gfs_grid_id", "valid_time_utc"]).any():
        errors.append("duplicate grid/time")
    if frame.isna().any().any():
        errors.append("missing values")
    expected_dates = {valid_day, valid_day + timedelta(days=1)}
    if set(frame["valid_time_kst"].dt.date) != expected_dates:
        errors.append("48-hour valid KST date range mismatch")
    expected_start = pd.Timestamp(
        datetime.combine(valid_day, datetime.min.time(), KST)
    )
    expected_end = expected_start + timedelta(hours=FORECAST_HORIZON_HOURS - 1)
    if frame["valid_time_kst"].min() != expected_start:
        errors.append("48-hour valid start mismatch")
    if frame["valid_time_kst"].max() != expected_end:
        errors.append("48-hour valid end mismatch")
    if frame["lead_hour"].min() != FIRST_LEAD_HOUR or frame["lead_hour"].max() != LAST_LEAD_HOUR:
        errors.append("lead-hour range mismatch")
    per_grid_counts = frame.groupby("gfs_grid_id")["valid_time_utc"].nunique()
    if not (per_grid_counts == FORECAST_HORIZON_HOURS).all():
        errors.append("not every grid has 48 unique valid hours")
    if not frame["available_before_bid_deadline"].all():
        errors.append("source was not available by bid deadline")
    if not frame["total_cloud_cover_pct"].between(0, 100).all():
        errors.append("cloud cover out of range")
    if (frame["shortwave_radiation_w_m2"] < 0).any():
        errors.append("negative radiation")
    if errors:
        raise ValueError("; ".join(errors))


def day_path(output: Path, valid_day: date) -> Path:
    return output / "daily" / f"year={valid_day.year}" / f"month={valid_day.month:02d}" / f"gfs_{valid_day:%Y%m%d}.parquet"


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end must be on or after --start")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def select_candidate_dates(
    start: date,
    end: date,
    failed: set[date],
    retry_failed_only: bool,
) -> tuple[list[date], int]:
    """Skip known failures on normal resumes unless retry is explicit."""
    all_dates = date_range(start, end)
    failed_in_range = {value for value in failed if start <= value <= end}
    if retry_failed_only:
        return [value for value in all_dates if value in failed_in_range], 0
    return [value for value in all_dates if value not in failed_in_range], len(
        failed_in_range
    )


def load_failed_dates(path: Path) -> set[date]:
    if not path.exists():
        return set()
    return {date.fromisoformat(value) for value in json.loads(path.read_text(encoding="utf-8"))}


def save_failed_dates(path: Path, values: set[date]) -> None:
    path.write_text(
        json.dumps(sorted(value.isoformat() for value in values), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    plants, _asset_grids = load_grid_points(args.locations)
    grids = load_collection_grids(args.grids)
    args.output.mkdir(parents=True, exist_ok=True)
    plants.to_parquet(
        args.output / "plant_to_gfs_grid_mapping.parquet", index=False, compression="zstd"
    )
    grids.to_parquet(
        args.output / "jeju_gfs_grid_master.parquet", index=False, compression="zstd"
    )
    failed_path = args.output / "failed_dates.json"
    failed = load_failed_dates(failed_path)
    all_dates = date_range(args.start, args.end)
    candidates, skipped_known_failures = select_candidate_dates(
        args.start, args.end, failed, args.retry_failed_only
    )
    pending_all = [value for value in candidates if not day_path(args.output, value).exists()]
    already_complete = sum(day_path(args.output, value).exists() for value in all_dates)
    pending = pending_all
    if args.max_days is not None:
        pending = pending[: args.max_days]

    print(
        f"GFS history {args.start}..{args.end}: total={len(all_dates)}, "
        f"complete={already_complete}, pending={len(pending)}"
        f", known_failed_skipped={skipped_known_failures}"
        f"{' (limited run)' if args.max_days is not None else ''}, grids={len(grids)}",
        flush=True,
    )
    manifest = args.output / "download_manifest.jsonl"
    errors = args.output / "errors.jsonl"
    session = make_session(args.request_retries)
    run_bytes = 0
    run_started = time.monotonic()
    for position, valid_day in enumerate(pending, start=1):
        target = day_path(args.output, valid_day)
        succeeded = False
        for attempt in range(1, args.day_attempts + 1):
            try:
                issue = issue_for_valid_day(valid_day)
                fetched: list[dict] = []
                downloaded = 0
                for hour in range(
                    RADIATION_PREVIOUS_HOUR, LAST_STATE_SAMPLE_HOUR + 1
                ):
                    result, byte_count = fetch_hour(
                        session, issue, hour, grids, args.timeout
                    )
                    fetched.append(result)
                    downloaded += byte_count
                frame = make_day_frame(fetched, grids, issue, valid_day)
                validate_day(frame, len(grids), valid_day)
                atomic_parquet(frame, target)
                run_bytes += downloaded
                failed.discard(valid_day)
                save_failed_dates(failed_path, failed)
                elapsed = time.monotonic() - run_started
                rate = position / elapsed if elapsed else 0.0
                eta_hours = (len(pending) - position) / rate / 3600 if rate else float("nan")
                append_jsonl(
                    manifest,
                    {
                        "valid_day_kst": valid_day,
                        "issue_time_utc": issue,
                        "path": str(target.relative_to(args.output)),
                        "rows": len(frame),
                        "selected_bytes": downloaded,
                        "completed_at_utc": datetime.now(timezone.utc),
                    },
                )
                print(
                    f"[{position}/{len(pending)}] PASS {valid_day} "
                    f"{downloaded / 2**20:.1f} MiB | run {run_bytes / 2**30:.2f} GiB "
                    f"| ETA {eta_hours:.1f}h",
                    flush=True,
                )
                succeeded = True
                break
            except Exception as exc:
                append_jsonl(
                    errors,
                    {
                        "valid_day_kst": valid_day,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "failed_at_utc": datetime.now(timezone.utc),
                    },
                )
                print(
                    f"[{position}/{len(pending)}] RETRY {valid_day} "
                    f"attempt={attempt}/{args.day_attempts}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                session.close()
                session = make_session(args.request_retries)
                if attempt < args.day_attempts:
                    time.sleep(args.retry_wait * attempt)
        if not succeeded:
            failed.add(valid_day)
            save_failed_dates(failed_path, failed)
            print(f"[{position}/{len(pending)}] FAIL {valid_day}; continuing", flush=True)
    session.close()
    print(
        f"Run finished. Remaining failed dates: {len(failed)}. "
        f"Output: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
