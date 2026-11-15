"""KPX 예비력 엑셀에서 주파수제어예비력 가격만 1시간 평균으로 변환한다.

원본 01시의 1~4구간 → KST 00:00. 네 값 중 하나라도 없으면 시간 가격은 결측이다.
정산용 원/kW-h 단가를 기존 1시간 모델의 fcr_price_krw_per_kwh로 전달한다.
가격을 합산하거나 0.25/1,000을 곱하지 않는다. MWh 환산은 정산 입력에서 한 번 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vpp.market import PROCESSED_DIR
from vpp.market._util import print_coverage, save_csv

PRICE_COLUMN = "fcr_price_krw_per_kwh"
SOURCE = "kpx_reserve_xlsx_hourly_mean"
SOURCE_URL = "https://www.data.go.kr/data/15145101/fileData.do"
KINDS = {"주파수제어예비력", "1차예비력", "3차운전예비력"}


def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """2행 머리글 + 날짜별 예비력 3행. 날짜는 3행 중 가운데 셀에 있을 수 있다."""
    if raw.shape[1] != 98 or len(raw) < 5:
        raise ValueError("예비력 엑셀은 날짜·구분과 96구간 가격 열이 필요합니다.")
    if raw.iloc[0, :2].tolist() != ["거래일자", "구분"]:
        raise ValueError("예비력 엑셀 머리글이 예상과 다릅니다.")
    hours = raw.iloc[0, 2:].ffill().tolist()
    quarters = pd.to_numeric(raw.iloc[1, 2:], errors="raise").tolist()
    if hours != [f"{h:02d}시" for h in range(1, 25) for _ in range(4)] \
            or quarters != [1, 2, 3, 4] * 24:
        raise ValueError("예비력 가격은 01~24시, 시간별 1~4구간 순서여야 합니다.")
    body = raw.iloc[2:].reset_index(drop=True)
    if len(body) % 3:
        raise ValueError("날짜별 예비력 3행이 완전하지 않습니다.")
    blocks = []
    for start in range(0, len(body), 3):
        group = body.iloc[start:start + 3]
        dates = group.iloc[:, 0].dropna().astype(str).str.strip().unique()
        if len(dates) != 1 or set(group.iloc[:, 1]) != KINDS:
            raise ValueError(f"엑셀 {start + 3}행: 날짜별 예비력 구분을 확인하세요.")
        day = pd.to_datetime(dates[0], format="%Y%m%d", errors="raise")
        fcr = group.loc[group.iloc[:, 1].eq("주파수제어예비력")].iloc[0, 2:]
        values = pd.to_numeric(fcr, errors="raise").to_numpy(dtype=float).reshape(24, 4)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(f"{day.date()}: FCR 가격은 0 이상의 유한한 값 또는 결측이어야 합니다.")
        blocks.append(pd.DataFrame({
            "ts": pd.date_range(day, periods=24, freq="h", tz="Asia/Seoul"),
            PRICE_COLUMN: values.mean(axis=1),  # NaN을 제외한 부분 평균을 만들지 않는다.
            "fcr_quarter_count": np.isfinite(values).sum(axis=1),
            "fcr_price_source": SOURCE,
        }))
    result = pd.concat(blocks, ignore_index=True).sort_values("ts").reset_index(drop=True)
    if result["ts"].duplicated().any():
        raise ValueError("예비력 엑셀에 중복 거래일자가 있습니다.")
    return result


def read_hourly(path: str | Path) -> pd.DataFrame:
    return to_hourly(pd.read_excel(path, sheet_name="실시간예비력가격", header=None, dtype=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "fcr_hourly.csv")
    args = parser.parse_args(argv)
    if args.out.resolve() == args.input.resolve():
        parser.error("원본 엑셀을 출력 파일로 덮어쓸 수 없습니다.")
    result = read_hourly(args.input)
    print_coverage(result, "h")
    save_csv(result, args.out)
    metadata = {
        "source_path": str(args.input.resolve()),
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "price_type": "주파수제어예비력",
        "aggregation": "arithmetic_mean_of_all_four_quarters; missing_if_any_quarter_missing",
        "timestamp": "Asia/Seoul hour start; original 01시 -> 00:00",
        "unit": "원/kW-h; 1시간 모델의 fcr_price_krw_per_kwh에 동일 수치 사용",
        "hours": len(result),
        "start": result.ts.min().isoformat(), "end": result.ts.max().isoformat(),
        "missing_price_hours": int(result[PRICE_COLUMN].isna().sum()),
    }
    args.out.with_suffix(".source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
