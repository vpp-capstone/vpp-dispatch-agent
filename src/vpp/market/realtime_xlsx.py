"""KPX 제주 실시간 SMP 연도별 엑셀을 15분 단위 시계열로 바꾼다.

원본: KPX > 제주시범사업 정보 > 계통한계가격(SMP) > 실시간
    https://www.kpx.or.kr/menu.es?mid=a10406020200
    '년도'를 고르고 엑셀다운로드 (smpDataRt_2024.xlsx 같은 이름으로 받아짐).
    받은 파일을 data/raw/market/realtime_xlsx/ 에 넣는다.

실행:
    python -m vpp.market.realtime_xlsx

OpenAPI(realtime_api)가 2024-07 부터만 있어서, 그 이전(2024-03 ~)은 이 엑셀로 채운다.

원본 구조:
    한 줄이 하루. 1h~24h 가 각각 1~4구간(15분)으로 나뉘어 96칸이고,
    그 뒤에 최대/최소/가중평균이 붙는다. 값은 문자열이다.

출력 (jeju_rt_smp_15min.csv):
    ts      구간 시작 시각 (1h 1구간 = 00:00~00:15 -> 00:00)
    smp_rt  실시간 SMP (원/kWh). 원본에 값이 없는 구간은 빈 값으로 둔다.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from vpp.market import PROCESSED_DIR, RAW_DIR
from vpp.market._util import print_coverage, save_csv

N_SLOTS = 96  # 24시간 x 4구간


def read_one(path: Path) -> pd.DataFrame:
    """엑셀 하나를 (날짜, 96개 값) 가로형 표로 읽는다."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 원본에 기본 스타일이 없다는 경고
        raw = pd.read_excel(path, header=None, dtype=str)

    if raw.shape[1] < 1 + N_SLOTS or str(raw.iloc[1, 1]).strip() != "1h":
        raise ValueError(f"{path.name}: 예상한 표 구조가 아님 (크기 {raw.shape})")

    # 날짜(YYYYMMDD)로 시작하는 줄만 데이터다. 제목, 머리글 줄은 버린다.
    body = raw[raw[0].str.fullmatch(r"\d{8}", na=False)]
    wide = body.iloc[:, 1 : 1 + N_SLOTS].apply(
        lambda col: pd.to_numeric(col.str.replace(",", "", regex=False), errors="coerce")
    )
    wide.columns = range(N_SLOTS)
    wide.insert(0, "date", pd.to_datetime(body[0], format="%Y%m%d"))
    return wide


def load_dir(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"{folder} 에 xlsx가 없음")
    wide = pd.concat([read_one(f) for f in files], ignore_index=True)
    wide = wide.drop_duplicates(subset="date", keep="last")

    long = wide.melt(id_vars="date", var_name="slot", value_name="smp_rt")
    long["ts"] = long["date"] + pd.to_timedelta(long["slot"].astype(int) * 15, unit="min")
    series = long.set_index("ts")["smp_rt"].sort_index()

    # 값이 있는 마지막 날까지를 15분 간격으로 빠짐없이 편다 (없는 구간은 빈 값).
    last_day = series.dropna().index.max().normalize()
    full = pd.date_range(series.index.min(), last_day + pd.Timedelta("23h45min"), freq="15min")
    return series.reindex(full).rename_axis("ts").reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR / "realtime_xlsx")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "jeju_rt_smp_15min.csv")
    args = parser.parse_args()

    result = load_dir(args.raw_dir)
    print_coverage(result, "15min")
    empty_days = result.set_index("ts")["smp_rt"].isna().groupby(lambda t: t.date()).all().sum()
    print(f"값이 통째로 없는 날 {int(empty_days)}일")
    save_csv(result, args.out)


if __name__ == "__main__":
    main()
