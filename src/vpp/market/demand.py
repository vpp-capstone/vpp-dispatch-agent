"""시간별 제주 전력수요 CSV 여러 개를 하나의 시계열로 합친다.

원본: 한국전력거래소_시간별 제주전력수요
    https://www.data.go.kr/data/15065239/fileData.do
    과거 버전 파일까지 전부 받아 data/raw/market/demand/ 에 넣는다.

실행:
    python -m vpp.market.demand

포털에 올라온 파일들은 시기마다 형식이 달라서 아래 차이를 여기서 맞춘다.
    - 인코딩     : cp949 가 대부분이고 일부는 utf-8
    - 날짜 컬럼명 : '거래일자' 또는 '날짜'
    - 단위       : 대부분 MWh, 일부 파일은 kWh (값이 1000배)
    - 숫자 표기   : 1,000 이상에 쉼표가 들어간 파일이 있음 ("1,019.690")
    - 기간 중복   : 여러 파일이 같은 날짜를 담고 있고, 값이 다른 경우도 있음
기간이 겹치면 나중에 올라온 파일의 값을 쓴다. 겹치는 구간을 비교해 보면
나중 파일에서 급락 구간이 메워져 있어 정정본으로 판단했다.

출력 (jeju_demand_hourly.csv):
    ts          구간 시작 시각 (원본 1시 = 00:00~01:00 -> 00:00)
    demand_mwh  시간별 수요 (MWh)
    source      값을 가져온 파일명
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from vpp.market import PROCESSED_DIR, RAW_DIR
from vpp.market._util import print_coverage, read_csv_any, save_csv

HOUR_COLS = [f"{h}시" for h in range(1, 25)]
KWH_THRESHOLD = 5000.0  # 파일 중앙값이 이보다 크면 kWh 단위로 본다


def read_one(path: Path) -> pd.DataFrame:
    """파일 하나를 읽어 MWh 단위의 가로형 표로 돌려준다."""
    df = read_csv_any(path, dtype=str)
    df = df.rename(columns={df.columns[0]: "date"})
    missing = [c for c in HOUR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: 컬럼 없음 {missing}")

    df = df[["date", *HOUR_COLS]].dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in HOUR_COLS:
        cleaned = df[col].str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(cleaned, errors="coerce")

    if df[HOUR_COLS].stack().median() > KWH_THRESHOLD:
        df[HOUR_COLS] = df[HOUR_COLS] / 1000.0
    df["source"] = path.name
    return df


def file_order(path: Path) -> str:
    """파일명 끝의 기준일(YYYYMMDD)로 정렬한다. 없으면 파일명 순."""
    match = re.search(r"(\d{8})(?=\.csv$)", path.name)
    return match.group(1) if match else path.name


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.melt(id_vars=["date", "source"], var_name="hour_end", value_name="demand_mwh")
    long["hour_end"] = long["hour_end"].str.replace("시", "").astype(int)
    long["ts"] = long["date"] + pd.to_timedelta(long["hour_end"] - 1, unit="h")
    return long[["ts", "demand_mwh", "source"]].sort_values("ts").reset_index(drop=True)


def load_dir(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"), key=file_order)
    if not files:
        raise FileNotFoundError(f"{folder} 에 CSV가 없음")
    wide = pd.concat([read_one(f) for f in files], ignore_index=True)
    # 같은 날짜가 여러 파일에 있으면 나중 파일 것만 남긴다.
    wide = wide.drop_duplicates(subset="date", keep="last")
    return to_long(wide)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR / "demand")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "jeju_demand_hourly.csv")
    args = parser.parse_args()

    result = load_dir(args.raw_dir)
    print_coverage(result, "h")
    save_csv(result, args.out)


if __name__ == "__main__":
    main()
