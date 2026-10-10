"""하루전 SMP·수요예측 OpenAPI 전체 이력을 받아 시간별 표로 만든다.

원본: 한국전력거래소_계통한계가격 및 수요예측(하루전 발전계획용)
    https://www.data.go.kr/data/15131225/openapi.do

실행:
    python -m vpp.market.day_ahead               (수집 + 표 만들기)
    python -m vpp.market.day_ahead --build-only  (받아둔 원본으로 표만 다시 만들기)
인증키는 .env 의 DATA_GO_KR_API_KEY 에서 읽는다.

원본은 한 줄이 (날짜, 시, 지역)이고 지역은 육지·제주 두 가지다.
hour 표기가 '1' 과 '01' 두 가지로 섞여 있고, 일부 날짜는 같은 시각이 두 표기로
모두 들어 있다(값은 같음). 숫자로 바꾼 뒤 하나만 남긴다.

출력 (smp_forecast_hourly.csv):
    ts          구간 시작 시각 (원본 hour 01 = 00:00~01:00 -> 00:00)
    smp_land    육지 하루전 SMP (원/kWh)
    smp_jeju    제주 하루전 SMP (원/kWh)
    fcst_jeju   제주 수요예측   (원본 jlfd)
    fcst_land   육지 수요예측   (원본 mlfd)
    fcst_total  합계 수요예측   (원본 slfd = mlfd + jlfd)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vpp.market import PROCESSED_DIR, RAW_DIR
from vpp.market._openapi import collect, load_key, make_fetch, read_pages
from vpp.market._util import print_coverage, save_csv

ENDPOINT = "https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand"
AREA_NAMES = {"육지": "land", "제주": "jeju"}
VALUE_COLS = ["smp", "jlfd", "mlfd", "slfd"]


def build(rows: list[dict]) -> pd.DataFrame:
    """API 행 목록을 시간별 표로 바꾼다."""
    df = pd.DataFrame(rows)
    df["padded"] = df["hour"].astype(str).str.startswith("0")
    df["hour"] = df["hour"].astype(int)
    key = ["date", "hour", "areaName"]

    dup = df[df.duplicated(subset=key, keep=False)]
    if len(dup):
        spread = dup.groupby(key)[VALUE_COLS].agg(lambda s: s.max() - s.min())
        differing = spread[(spread > 0).any(axis=1)]
        print(
            f"같은 시각이 두 번 들어 있는 묶음 {len(spread)}개 "
            f"({dup['date'].min()} ~ {dup['date'].max()}), "
            f"그중 값이 서로 다른 묶음 {len(differing)}개"
        )
    # 겹치면 '01' 표기(최근 방식) 행을 남긴다.
    df = df.sort_values("padded", kind="stable").drop_duplicates(subset=key, keep="last")

    df["ts"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") + pd.to_timedelta(
        df["hour"] - 1, unit="h"
    )
    smp = df.pivot(index="ts", columns="areaName", values="smp")
    smp.columns = [f"smp_{AREA_NAMES.get(c, c)}" for c in smp.columns]
    fcst = (
        df.groupby("ts")[["jlfd", "mlfd", "slfd"]]
        .first()
        .rename(columns={"jlfd": "fcst_jeju", "mlfd": "fcst_land", "slfd": "fcst_total"})
    )
    return smp.join(fcst).sort_index().reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR / "day_ahead_api")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "smp_forecast_hourly.csv")
    parser.add_argument("--build-only", action="store_true", help="API를 부르지 않는다")
    args = parser.parse_args()

    if not args.build_only:
        collect(make_fetch(ENDPOINT, load_key()), args.raw_dir)
    table = build(read_pages(args.raw_dir))
    print_coverage(table, "h")
    save_csv(table, args.out)


if __name__ == "__main__":
    main()
