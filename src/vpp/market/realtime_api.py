"""제주시범사업정보 OpenAPI 전체 이력을 받아 15분 단위 표로 만든다.

원본: 한국전력거래소_제주시범사업정보
    https://www.data.go.kr/data/15133951/openapi.do

실행:
    python -m vpp.market.realtime_api               (수집 + 표 만들기)
    python -m vpp.market.realtime_api --build-only  (받아둔 원본으로 표만 다시 만들기)
인증키는 .env 의 DATA_GO_KR_API_KEY 에서 읽는다.

원본은 한 줄이 (날짜, 시, 구간)이고 한 시간이 1~4구간(15분)으로 나뉜다.

출력 (jeju_pilot_15min.csv):
    ts        구간 시작 시각 (hour 1, 1구간 = 00:00~00:15 -> 00:00)
    smp_da    제주 하루전 SMP    (원본 jsmp,   원/kWh, 1시간 값이 4구간에 반복됨)
    smp_rt    제주 실시간 SMP    (원본 jsmpRt, 원/kWh)
    fcst_da   제주 하루전 수요예측 (원본 jlfd,   1시간 값이 4구간에 반복됨)
    fcst_rt   제주 실시간 수요예측 (원본 jlfdRt)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vpp.market import PROCESSED_DIR, RAW_DIR
from vpp.market._openapi import collect, load_key, make_fetch, read_pages
from vpp.market._util import print_coverage, save_csv

ENDPOINT = "https://apis.data.go.kr/B552115/JejuSmpLfd2/getJejuSmpLfd2"
RENAME = {"jsmp": "smp_da", "jsmpRt": "smp_rt", "jlfd": "fcst_da", "jlfdRt": "fcst_rt"}


def build(rows: list[dict]) -> pd.DataFrame:
    """API 행 목록을 15분 단위 표로 바꾼다."""
    df = pd.DataFrame(rows)
    df["hour"] = df["hour"].astype(int)  # 1~24
    df["slot"] = df["gugan"].astype(str).str.extract(r"(\d)")[0].astype(int)  # 1~4구간
    # 날짜를 넘겨 이어 받으면 앞 페이지와 일부 겹칠 수 있다.
    df = df.drop_duplicates(subset=["date", "hour", "slot"], keep="first")

    df["ts"] = (
        pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        + pd.to_timedelta(df["hour"] - 1, unit="h")
        + pd.to_timedelta((df["slot"] - 1) * 15, unit="min")
    )
    out = df.rename(columns=RENAME)[["ts", *RENAME.values()]]
    return out.sort_values("ts").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR / "realtime_api")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "jeju_pilot_15min.csv")
    parser.add_argument("--build-only", action="store_true", help="API를 부르지 않는다")
    args = parser.parse_args()

    if not args.build_only:
        collect(make_fetch(ENDPOINT, load_key()), args.raw_dir)
    table = build(read_pages(args.raw_dir))
    print_coverage(table, "15min")
    save_csv(table, args.out)


if __name__ == "__main__":
    main()
