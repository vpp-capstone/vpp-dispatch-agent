"""정리된 자료들을 합쳐 실시간 15분 표와 시간별 표를 만든다.

실행 (앞의 다섯 모듈을 먼저 돌린 뒤):
    python -m vpp.market.build

입력 (data/processed/market/ 에서 읽는다):
    jeju_demand_hourly.csv    demand 결과         (ts, demand_mwh, source)
    smp_forecast_hourly.csv   day_ahead 결과      (ts, smp_jeju, fcst_jeju, ...)
    jeju_rt_smp_15min.csv     realtime_xlsx 결과  (ts, smp_rt)               2024-03 ~
    jeju_pilot_15min.csv      realtime_api 결과   (ts, smp_rt, fcst_rt, ...) 2024-07 ~
    jeju_curtail_hourly.csv   curtailment 결과    (ts, wind_curtail_mwh, solar_curtail)

실시간 가격은 두 출처가 있다. API 값을 기본으로 쓰고, API에 없는 구간(API 시작
이전과 최근 며칠)만 엑셀 값으로 채운다. 어느 쪽에서 왔는지는 source 에 남긴다.

출력 1: jeju_rt_15min.csv (15분 단위)
    ts        구간 시작 시각
    smp_rt    제주 실시간 SMP (원/kWh)
    fcst_rt   제주 실시간 수요예측 (API 구간에만 있음)
    source    smp_rt 의 출처: api 또는 xlsx

출력 2: market_hourly.csv (1시간 단위, 하루전 자료가 있는 2020-01-01 부터)
    ts                구간 시작 시각
    demand_mwh        제주 수요 실적 (MWh)
    fcst_jeju         제주 하루전 수요예측
    smp_da            제주 하루전 SMP (원/kWh)
    smp_rt_mean       제주 실시간 SMP, 그 시간 4개 구간의 단순평균
    smp_rt_min        제주 실시간 SMP, 그 시간 4개 구간 중 최저
    smp_rt_max        제주 실시간 SMP, 그 시간 4개 구간 중 최고
    fcst_rt_mean      제주 실시간 수요예측, 그 시간 4개 구간의 단순평균
    wind_curtail_mwh  제주 풍력 출력제어량 (MWh), 제어가 없던 시간은 0
    solar_curtail     제주 태양광 출력제어 여부 (1 = 제어, 0 = 없음)
자료가 없는 구간은 빈 값이다. 컬럼마다 값이 있는 기간이 다르다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vpp.market import PROCESSED_DIR
from vpp.market._util import print_coverage, save_csv

HOURLY_COLS = [
    "ts",
    "demand_mwh",
    "fcst_jeju",
    "smp_da",
    "smp_rt_mean",
    "smp_rt_min",
    "smp_rt_max",
    "fcst_rt_mean",
    "wind_curtail_mwh",
    "solar_curtail",
]


def build_rt(xlsx: pd.DataFrame, api: pd.DataFrame) -> pd.DataFrame:
    """두 출처의 실시간 가격을 15분 표 하나로 합친다. API 우선, 없는 구간만 엑셀."""
    xl = xlsx.set_index("ts")["smp_rt"]
    api = api.set_index("ts")[["smp_rt", "fcst_rt"]]

    both = api["smp_rt"].rename("api").to_frame().join(xl.rename("xl"), how="inner").dropna()
    differ = both[(both["api"] - both["xl"]).abs() > 0.005]
    days = sorted({t.date().isoformat() for t in differ.index})
    print(f"두 출처가 겹치는 구간 {len(both)}개 중 값이 다른 구간 {len(differ)}개, 날짜 {days}")

    start = min(xl.index.min(), api.index.min())
    end = max(xl.dropna().index.max(), api.index.max())
    index = pd.date_range(start, end, freq="15min")
    xl = xl.reindex(index)

    out = pd.DataFrame(index=index)
    out["smp_rt"] = api["smp_rt"].reindex(index)
    out["source"] = out["smp_rt"].notna().map({True: "api", False: None})
    fill = out["smp_rt"].isna() & xl.notna()
    out.loc[fill, "smp_rt"] = xl[fill]
    out.loc[fill, "source"] = "xlsx"
    out["fcst_rt"] = api["fcst_rt"].reindex(index)
    return out.rename_axis("ts").reset_index()[["ts", "smp_rt", "fcst_rt", "source"]]


def build_hourly(
    demand: pd.DataFrame, day_ahead: pd.DataFrame, rt: pd.DataFrame, curtail: pd.DataFrame
) -> pd.DataFrame:
    """하루전 자료의 시간축을 기준으로 나머지를 붙인다."""
    da = day_ahead[["ts", "fcst_jeju", "smp_jeju"]].rename(columns={"smp_jeju": "smp_da"})

    # 15분 값을 시간 단위로 묶는다. 4개 구간이 모두 비어 있으면 빈 값으로 남는다.
    rt_hourly = (
        rt.groupby(rt["ts"].dt.floor("h"))
        .agg(
            smp_rt_mean=("smp_rt", "mean"),
            smp_rt_min=("smp_rt", "min"),
            smp_rt_max=("smp_rt", "max"),
            fcst_rt_mean=("fcst_rt", "mean"),
        )
        .round(2)
        .reset_index()
    )
    out = (
        da.merge(demand[["ts", "demand_mwh"]], on="ts", how="left")
        .merge(rt_hourly, on="ts", how="left")
        .merge(curtail, on="ts", how="left")
    )
    return out[HOURLY_COLS].sort_values("ts").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=PROCESSED_DIR, help="입력과 출력 폴더")
    args = parser.parse_args()

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(args.dir / name, parse_dates=["ts"])

    rt = build_rt(read("jeju_rt_smp_15min.csv"), read("jeju_pilot_15min.csv"))
    print_coverage(rt, "15min")
    print("  출처별 구간 수:", rt["source"].value_counts().to_dict())
    save_csv(rt, args.dir / "jeju_rt_15min.csv")

    hourly = build_hourly(
        read("jeju_demand_hourly.csv"),
        read("smp_forecast_hourly.csv"),
        rt,
        read("jeju_curtail_hourly.csv"),
    )
    print_coverage(hourly, "h")
    save_csv(hourly, args.dir / "market_hourly.csv")


if __name__ == "__main__":
    main()
