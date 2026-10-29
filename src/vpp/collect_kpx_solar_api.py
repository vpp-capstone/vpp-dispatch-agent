"""Collect the documented KPX solar API and compare Jeju with hourly file targets.

Credentials are loaded locally and never included in stored metadata or errors.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from vpp.renewable_targets import numeric

ENDPOINT = "https://apis.data.go.kr/B552115/PvAmountByLocHr/getPvAmountByLocHr"
SOURCE_URL = "https://www.data.go.kr/data/15103243/openapi.do"
API_KEYS = ("DATA_GO_KR_API_KEY", "DATA_GO_KR_KEY")
JEJU_REGIONS = ("제주", "제주도", "제주특별자치도")


class ApiError(RuntimeError):
    """A sanitized public API error without request URLs or credentials."""


def load_key(env_file: Path) -> str:
    """Read the two team naming conventions without printing credential values."""
    local = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.removeprefix("export ").split("=", 1)
            if name.strip() in API_KEYS:
                local[name.strip()] = value.strip().strip("\"'")
    for name in API_KEYS:
        if value := os.environ.get(name) or local.get(name):
            return urllib.parse.unquote(value.strip())
    raise ApiError(
        "Missing DATA_GO_KR_API_KEY. Apply for dataset 15103243 and save its key in .env."
    )


def parse_response(raw: bytes) -> tuple[list[dict], int, int]:
    """Read the official JSON body; also recognize gateway XML error envelopes."""
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError):
        try:
            root = ET.fromstring(raw)
            code = root.findtext(".//returnReasonCode") or root.findtext(".//resultCode")
            if code:
                raise ApiError(f"API gateway rejected the request (code {code}).")
        except ET.ParseError:
            pass
        raise ApiError("API returned a non-JSON response; check service status and dataType.")
    if not isinstance(payload, dict):
        raise ApiError("Unexpected API root structure.")
    response = payload.get("response", payload)
    if not isinstance(response, dict):
        raise ApiError("Unexpected API response structure.")
    header = response.get("header", {})
    code = str(header.get("resultCode", ""))
    if code not in ("00", "0", "0000"):
        if code in ("03", "3"):
            return [], 0, 0
        # Do not print resultMsg: upstream text may contain the submitted URL.
        raise ApiError(f"API rejected the request (code {code or 'missing'}).")
    body = response.get("body") or {}
    items = body.get("items") or {}
    if isinstance(items, dict):
        items = items.get("item") or []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ApiError("Unexpected API items structure; raw response kept for inspection.")
    try:
        return items, int(body.get("totalCount", 0)), int(body.get("pageNo", 0))
    except (ValueError, TypeError):
        raise ApiError("Invalid pagination metadata from API.") from None


def fetch_page(key: str, page: int, rows: int, date: str | None) -> bytes:
    """Call only the documented HTTPS gateway, sanitizing network exceptions."""
    params = {"serviceKey": key, "pageNo": page, "numOfRows": rows, "dataType": "json"}
    if date:
        params["tradeYmd"] = date
    request = urllib.request.Request(
        ENDPOINT + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": "vpp-data/1.0", "Accept-Encoding": "gzip"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                return gzip.decompress(raw)
            return raw
    except urllib.error.HTTPError as exc:
        raise ApiError(f"API HTTP {exc.code}; check authorization or request limits.") from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        raise ApiError(
            "API connection failed or timed out; cached pages remain available."
        ) from None


def normalize(items: list[dict]) -> pd.DataFrame:
    """Map official tradeYmd/tradeNo/regionNm/amgo fields to the existing adapter."""
    columns = ["trade_date", "hour_label", "region", "fuel", "traded_energy_mwh"]
    if not items:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(items)
    required = {"tradeYmd", "tradeNo", "regionNm", "amgo"}
    if missing := required.difference(frame.columns):
        raise ApiError(f"API response missing documented fields: {sorted(missing)}")
    frame = frame[frame.regionNm.astype("string").str.strip().isin(JEJU_REGIONS)].copy()
    dates = pd.to_datetime(
        frame.tradeYmd.astype("string").str.strip(), format="%Y%m%d", errors="coerce",
    )
    hours = numeric(frame.tradeNo)
    if (dates.isna() | hours.isna() | ~hours.between(1, 24) | hours.mod(1).ne(0)).any():
        raise ApiError("Jeju API response has invalid trade date/hour keys.")
    frame["trade_date"] = dates.dt.strftime("%Y-%m-%d")
    frame["hour_label"] = hours.astype(int)
    frame["region"] = frame.regionNm.str.strip()
    frame["fuel"] = "solar"
    # Preserve blanks and malformed numeric values for quality reporting.
    frame["traded_energy_mwh"] = frame.amgo
    return frame[columns]


def compare(api_csv: Path, file_csv: Path, output: Path) -> dict:
    """Compare observed overlaps; conflicts are excluded, never filled automatically."""
    api = pd.read_csv(api_csv, dtype=str, encoding="utf-8-sig")
    main = pd.read_csv(file_csv, dtype=str, encoding="utf-8-sig")
    keys = ["trade_date", "hour_label"]
    if "solar_traded_mwh" in main:
        timestamps = pd.to_datetime(main.ts, utc=True).dt.tz_convert("Asia/Seoul")
        main["trade_date"] = timestamps.dt.strftime("%Y-%m-%d")
        main["hour_label"] = timestamps.dt.hour + 1
        main["traded_energy_mwh"] = main.solar_traded_mwh
        available = main.solar_quality.eq("observed")
    else:
        main = main[main.fuel.eq("solar")].copy()
        available = main.target_available.astype("string").str.lower().eq("true")
    main.loc[~available, "traded_energy_mwh"] = float("nan")
    for frame in (api, main):
        frame["hour_label"] = pd.to_numeric(frame.hour_label, errors="raise").astype(int)
        frame["traded_energy_mwh"] = numeric(frame.traded_energy_mwh)
    conflicts = api.groupby(keys).traded_energy_mwh.nunique(dropna=False).gt(1)
    api["conflicting_duplicate"] = pd.MultiIndex.from_frame(api[keys]).isin(
        conflicts[conflicts].index,
    )
    api.to_csv(output / "api_jeju_observations.csv", index=False, encoding="utf-8-sig")
    api = api.drop_duplicates(keys).copy()
    api.loc[api.conflicting_duplicate, "traded_energy_mwh"] = float("nan")
    joined = main[keys + ["traded_energy_mwh"]].merge(
        api[keys + ["traded_energy_mwh"]], on=keys, how="outer",
        suffixes=("_file", "_api"), indicator=True, validate="one_to_one",
    )
    comparable = (
        joined.traded_energy_mwh_file.notna() & joined.traded_energy_mwh_api.notna()
        & joined.traded_energy_mwh_file.ge(0) & joined.traded_energy_mwh_api.ge(0)
    )
    joined["difference_mwh"] = joined.traded_energy_mwh_api - joined.traded_energy_mwh_file
    joined["comparable"] = comparable
    joined.to_csv(output / "api_file_comparison.csv", index=False, encoding="utf-8-sig")
    differences = joined.loc[comparable, "difference_mwh"].abs()
    observed = joined[comparable].copy()
    observed["year"] = observed.trade_date.str[:4]
    observed["different"] = observed.difference_mwh.abs().gt(1e-6)
    observed["abs_difference"] = observed.difference_mwh.abs()
    annual = observed.groupby("year").agg(
        compared=("different", "size"), different=("different", "sum"),
        max_abs_mwh=("abs_difference", "max"),
    )
    annual.to_csv(output / "comparison_by_year.csv")
    daily = observed.groupby("trade_date").agg(
        api_max=("traded_energy_mwh_api", "max"),
        file_max=("traded_energy_mwh_file", "max"), hours=("hour_label", "size"),
    )
    zero_review = daily[daily.api_max.eq(0) & daily.file_max.gt(1) & daily.hours.eq(24)]
    zero_review.to_csv(output / "api_zero_day_review.csv")
    return {
        "overlap_keys": int(joined._merge.eq("both").sum()),
        "comparable_keys": int(comparable.sum()), "conflicting_api_keys": int(conflicts.sum()),
        "different_values": int(differences.gt(1e-6).sum()),
        "max_abs_difference_mwh": float(differences.max()) if len(differences) else None,
        "all_zero_api_days_with_positive_file": len(zero_review),
        "policy": "Compare only; no replacement or interpolation",
    }


def collect(args: argparse.Namespace) -> dict:
    """Collect within a call budget; cached runs require explicit refresh to refetch."""
    key = load_key(args.env_file)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    scope = args.date or "all_history"
    if args.date:
        try:
            datetime.strptime(args.date, "%Y%m%d")
        except ValueError:
            raise ApiError("--date must be YYYYMMDD") from None
    directory = output / f"{scope}_rows{args.rows}"
    directory.mkdir(parents=True, exist_ok=True)
    items, calls, page, total = [], 0, 1, None
    complete = False
    page_fingerprints = set()
    page_size = None
    stop_reason = None

    def retrieve(number: int, cache: Path) -> bytes:
        if cache.exists() and not args.refresh:
            return cache.read_bytes()
        raw = fetch_page(key, number, args.rows, args.date)
        _, _, returned = parse_response(raw)
        if returned not in (0, number):
            raise ApiError("API returned a different page than requested.")
        cache.write_bytes(raw)
        return raw

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while not complete and stop_reason is None:
            final_page = math.ceil(total / page_size) if total and page_size else page
            pending = []
            for number in range(page, min(page + args.workers, final_page + 1)):
                cache = directory / f"page_{number:06d}.json"
                needs_call = not cache.exists() or args.refresh
                if needs_call and calls >= args.max_calls:
                    break
                if needs_call:
                    calls += 1
                pending.append((number, executor.submit(retrieve, number, cache)))
            if not pending:
                break
            for number, future in pending:
                try:
                    raw = future.result()
                except ApiError as exc:
                    stop_reason = str(exc)
                    break
                batch, count, returned_page = parse_response(raw)
                if returned_page not in (0, number):
                    raise ApiError("API returned a different page than requested.")
                if total is not None and total != count:
                    raise ApiError("API totalCount changed; restart with --refresh.")
                total = count
                if not batch and len(items) < total:
                    raise ApiError("API returned an empty page before totalCount was reached.")
                fingerprint = hashlib.sha256(json.dumps(batch, sort_keys=True).encode()).hexdigest()
                if batch and fingerprint in page_fingerprints:
                    raise ApiError("API repeated a page; stopping to avoid false completeness.")
                page_fingerprints.add(fingerprint)
                if page_size is None:
                    page_size = len(batch)
                items.extend(batch)
                print(f"Page {number}: {len(batch)} rows, {len(items)}/{total}", flush=True)
                page = number + 1
                if len(items) >= total:
                    complete = len(items) == total
                    if not complete:
                        raise ApiError("Returned item count exceeds totalCount.")
                    break
    frame = normalize(items)
    canonical = output / "jeju_solar_api.csv"
    frame.to_csv(canonical, index=False, encoding="utf-8-sig")
    frame.to_csv(directory / "jeju_solar_api.csv", index=False, encoding="utf-8-sig")
    api_quality = None
    if len(frame):
        expected = pd.MultiIndex.from_product([
            pd.date_range(frame.trade_date.min(), frame.trade_date.max()).strftime("%Y-%m-%d"),
            range(1, 25),
        ], names=["trade_date", "hour_label"])
        observed_keys = pd.MultiIndex.from_frame(frame[["trade_date", "hour_label"]])
        missing_keys = expected.difference(observed_keys)
        missing_keys.to_frame(index=False).to_csv(output / "api_missing_hours.csv", index=False)
        values = numeric(frame.traded_energy_mwh)
        api_quality = {
            "expected_hours_in_date_span": len(expected), "missing_hours": len(missing_keys),
            "duplicate_key_rows": int(observed_keys.duplicated().sum()),
            "missing_or_invalid_values": int(values.isna().sum()),
            "negative_values": int(values.lt(0).sum()),
        }
    report = {
        "source_url": SOURCE_URL, "endpoint": ENDPOINT,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope, "requested_rows_per_page": args.rows,
        "complete": complete, "new_calls": calls, "next_page": None if complete else page,
        "stop_reason": stop_reason,
        "reported_nationwide_rows": total, "collected_nationwide_rows": len(items),
        "jeju_rows": len(frame),
        "canonical_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
        "jeju_start_date": frame.trade_date.min() if len(frame) else None,
        "jeju_end_date": frame.trade_date.max() if len(frame) else None,
        "time_basis": "Original tradeYmd/tradeNo retained",
        "unit": "MWh (official service description)",
        "api_quality": api_quality,
    }
    if args.compare_file:
        report["comparison"] = compare(canonical, args.compare_file, output)
    (output / "collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (directory / "collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main() -> None:
    """CLI using local credentials and official API field names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/kpx_solar_api"))
    parser.add_argument("--date", help="YYYYMMDD; omit only after confirming bulk-history support")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--max-calls", type=int, default=90)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--compare-file", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.max_calls < 1 or args.max_calls > 90:
        parser.error("rows must be positive and max-calls must be 1..90")
    try:
        report = collect(args)
    except (ApiError, ValueError, OSError) as exc:
        # File-system/value errors cannot include the private request URL.
        parser.exit(2, f"Collection stopped: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        parser.exit(3, "Collection incomplete; partial data saved. Rerun later to resume.\n")


if __name__ == "__main__":
    main()
