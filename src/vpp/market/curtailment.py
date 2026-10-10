"""KPX 제주 출력제어 실적 CSV 두 개를 시간별 표로 바꾼다.

원본: 한국전력거래소_전국 태양광 풍력 제어횟수 및 제주 풍력 제어량
    https://www.data.go.kr/data/15132422/fileData.do
    zip 을 풀어 '제주 풍력 출력제어횟수 및 제어량.csv', '제주 태양광 출력제어횟수.csv' 를
    data/raw/market/curtailment/ 에 넣는다. (육지 파일은 쓰지 않는다)

실행:
    python -m vpp.market.curtailment

원본 구조:
    출력제어가 있었던 날만 한 줄씩 있다 (일자, 제어지역, 누적 시행회차, 1시~24시).
    풍력은 칸마다 그 시간의 제어량(MWh), 태양광은 제어한 시간에 '출력제어' 글자만 있다.
    2024년 줄은 끝의 23시, 24시 칸이 비어 있다.

출력 (jeju_curtail_hourly.csv):
    ts                 구간 시작 시각 (1시 = 00:00~01:00 -> 00:00)
    wind_curtail_mwh   제주 풍력 제어량 (MWh). 제어가 없던 시간은 0
    solar_curtail      제주 태양광 제어 여부 (1 = 제어, 0 = 없음). 제어량은 원본에 없음
원본에 없는 날은 제어가 없었던 날로 보고 0으로 채운다. 채우는 범위는
2021-01-01 부터 각 파일의 마지막 기록일까지이고, 그 뒤는 빈 값이다.
기록은 2024년 6월 초에서 끝난다 (풍력 05-30, 태양광 06-03).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vpp.market import PROCESSED_DIR, RAW_DIR
from vpp.market._util import print_coverage, read_csv_any, save_csv

HOUR_COLS = [f"{h}시" for h in range(1, 25)]
START = pd.Timestamp("2021-01-01")  # 두 파일 모두 2021년 1회차부터 시작한다


def read_wide(path: Path) -> pd.DataFrame:
    df = read_csv_any(path, dtype=str)
    missing = [c for c in ["일자", *HOUR_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: 컬럼 없음 {missing}")
    df["일자"] = pd.to_datetime(df["일자"])
    return df[["일자", *HOUR_COLS]]


def to_hourly(wide: pd.DataFrame, name: str) -> pd.Series:
    """가로형(일자 x 24시간) 숫자 표를 시간별 시리즈로 펴고, 없는 날은 0으로 채운다."""
    long = wide.melt(id_vars="일자", var_name="hour", value_name=name)
    long["ts"] = long["일자"] + pd.to_timedelta(
        long["hour"].str.replace("시", "").astype(int) - 1, unit="h"
    )
    series = long.set_index("ts")[name]
    last = wide["일자"].max() + pd.Timedelta(hours=23)
    return series.reindex(pd.date_range(START, last, freq="h")).fillna(0)


def load(wind_csv: Path, solar_csv: Path) -> pd.DataFrame:
    wind = read_wide(wind_csv)
    wind[HOUR_COLS] = wind[HOUR_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

    solar = read_wide(solar_csv)
    solar[HOUR_COLS] = (solar[HOUR_COLS] == "출력제어").astype(int)

    out = pd.concat(
        [to_hourly(wind, "wind_curtail_mwh"), to_hourly(solar, "solar_curtail")], axis=1
    )
    return out.rename_axis("ts").reset_index()


def find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"{folder} 에서 '{pattern}' 파일을 하나로 찾지 못함: {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR / "curtailment")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "jeju_curtail_hourly.csv")
    args = parser.parse_args()

    result = load(
        find_one(args.raw_dir, "*제주*풍력*.csv"), find_one(args.raw_dir, "*제주*태양광*.csv")
    )
    print_coverage(result, "h")
    for col in ("wind_curtail_mwh", "solar_curtail"):
        active = result[result[col] > 0]
        by_year = active.groupby(active["ts"].dt.year)[col].agg(["size", "sum"])
        print(f"{col} (연도: 제어 시간 수, 합계)")
        print(by_year.to_string(header=False))
    save_csv(result, args.out)


if __name__ == "__main__":
    main()
