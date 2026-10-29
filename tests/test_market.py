"""전력 시장·수요 데이터 정리(vpp.market) 테스트.

원본 파일은 Git 에 없으므로, 실제 원본에서 본 형식 차이를 작은 가짜 데이터로 재현한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vpp.market import _openapi, build, curtailment, day_ahead, demand, realtime_api, realtime_xlsx

HOURS = [f"{h}시" for h in range(1, 25)]


# ---------------------------------------------------------------- demand


def write_demand(path: Path, date_col: str, rows: dict[str, list], encoding: str) -> None:
    frame = pd.DataFrame(
        [[day, *values] for day, values in rows.items()], columns=[date_col, *HOURS]
    )
    frame.to_csv(path, index=False, encoding=encoding)


def test_demand_merges_files_with_different_formats(tmp_path):
    # 옛 파일: '거래일자', kWh 단위, cp949
    write_demand(
        tmp_path / "수요_20230228.csv",
        "거래일자",
        {"2023-02-27": [500000] * 24, "2023-02-28": [600000] * 24},
        "cp949",
    )
    # 새 파일: '날짜', MWh 단위, 1,000 이상 쉼표, utf-8. 02-28 은 정정된 값.
    write_demand(
        tmp_path / "수요_20230331.csv",
        "날짜",
        {"2023-02-28": [650.5] * 24, "2023-03-01": ["1,019.690"] * 24},
        "utf-8-sig",
    )

    out = demand.load_dir(tmp_path).set_index("ts")

    assert len(out) == 3 * 24
    assert out.index[0] == pd.Timestamp("2023-02-27 00:00")  # 1시 -> 00:00
    assert out.loc["2023-02-27 00:00", "demand_mwh"] == 500.0  # kWh -> MWh
    assert out.loc["2023-02-28 12:00", "demand_mwh"] == 650.5  # 겹치면 나중 파일
    assert out.loc["2023-02-28 12:00", "source"] == "수요_20230331.csv"
    assert out.loc["2023-03-01 23:00", "demand_mwh"] == 1019.69  # 쉼표 제거, 24시 -> 23:00


# ---------------------------------------------------------------- OpenAPI 공통


def make_response(rows: list[dict], total: int) -> str:
    body = {"totalCount": str(total), "items": {"item": rows} if rows else ""}
    header = {"resultCode": "00", "resultMsg": "OK"}
    return json.dumps({"response": {"header": header, "body": body}}, ensure_ascii=False)


class FakeServer:
    """페이지 크기 상한과 일일 한도가 있는 가짜 API."""

    def __init__(self, rows: list[dict], cap: int, quota: int):
        self.rows, self.cap, self.quota, self.calls = rows, cap, quota, 0

    def __call__(self, page: int, num_rows: int) -> str:
        self.calls += 1
        if self.calls > self.quota:
            return "<returnAuthMsg>LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR</returnAuthMsg>"
        size = min(num_rows, self.cap)
        return make_response(self.rows[(page - 1) * size : page * size], len(self.rows))


def test_collect_resumes_after_quota_and_skips_when_complete(tmp_path):
    rows = [{"n": i} for i in range(2500)]

    first = FakeServer(rows, cap=1000, quota=2)
    assert _openapi.collect(first, tmp_path, pause=0) is False  # 한도에 걸림
    assert len(list(tmp_path.glob("page_*.json"))) == 2

    second = FakeServer(rows, cap=1000, quota=100)
    assert _openapi.collect(second, tmp_path, pause=0) is True
    assert [r["n"] for r in _openapi.read_pages(tmp_path)] == list(range(2500))

    third = FakeServer(rows, cap=1000, quota=100)
    assert _openapi.collect(third, tmp_path, pause=0) is True
    assert third.calls == 0  # 이미 다 받았으면 다시 부르지 않는다


def test_parse_raises_on_api_error():
    bad = json.dumps({"response": {"header": {"resultCode": "30", "resultMsg": "KEY ERROR"}}})
    with pytest.raises(RuntimeError):
        _openapi.parse(bad)


def test_load_key_reads_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv(_openapi.ENV_KEY, raising=False)
    env = tmp_path / ".env"
    env.write_text("KMA_API_KEY=x\nDATA_GO_KR_API_KEY=abc%2F123\n", encoding="utf-8")
    assert _openapi.load_key(env) == "abc%2F123"
    with pytest.raises(RuntimeError):
        _openapi.load_key(tmp_path / "missing.env")


# ---------------------------------------------------------------- day_ahead


def test_day_ahead_handles_mixed_hour_format_and_duplicates():
    def row(hour, area, smp):
        return {
            "date": "20260729",
            "hour": hour,
            "areaName": area,
            "smp": smp,
            "jlfd": 600.0,
            "mlfd": 50000.0,
            "slfd": 50600.0,
        }

    rows = [
        row("1", "육지", 100.0),  # 옛 표기
        row("01", "육지", 100.0),  # 같은 시각이 새 표기로 한 번 더
        row("01", "제주", 90.0),
        row("10", "육지", 110.0),  # 제주 행이 없는 시각
    ]

    out = day_ahead.build(rows).set_index("ts")

    assert list(out.index) == [pd.Timestamp("2026-07-29 00:00"), pd.Timestamp("2026-07-29 09:00")]
    assert out.loc["2026-07-29 00:00", "smp_land"] == 100.0
    assert out.loc["2026-07-29 00:00", "smp_jeju"] == 90.0
    assert pd.isna(out.loc["2026-07-29 09:00", "smp_jeju"])
    assert out.loc["2026-07-29 09:00", "fcst_jeju"] == 600.0


# ---------------------------------------------------------------- realtime_api


def test_realtime_api_builds_15min_timestamps():
    rows = [
        {"date": "20240830", "hour": 1, "gugan": "1구간", "jsmp": 128.88, "jsmpRt": 128.88,
         "jlfd": 750, "jlfdRt": 779.82},
        {"date": "20240830", "hour": 1, "gugan": "4구간", "jsmp": 128.88, "jsmpRt": -5.0,
         "jlfd": 750, "jlfdRt": 730.58},
        {"date": "20240830", "hour": 24, "gugan": "4구간", "jsmp": 100.0, "jsmpRt": 99.0,
         "jlfd": 700, "jlfdRt": 701.0},
    ]  # fmt: skip

    out = realtime_api.build(rows)

    assert list(out["ts"]) == [
        pd.Timestamp("2024-08-30 00:00"),
        pd.Timestamp("2024-08-30 00:45"),
        pd.Timestamp("2024-08-30 23:45"),
    ]
    assert list(out.columns) == ["ts", "smp_da", "smp_rt", "fcst_da", "fcst_rt"]
    assert out.loc[1, "smp_rt"] == -5.0


# ---------------------------------------------------------------- realtime_xlsx


def test_realtime_xlsx_reads_wide_sheet(tmp_path):
    title = ["SMP(계통한계가격) (원/kWh)"] + [None] * 99
    hours = ["구분"] + [f"{h}h" if s == 0 else None for h in range(1, 25) for s in range(4)]
    hours += ["최대", "최소", "가중평균"]
    slots = [None] + ["1구간", "2구간", "3구간", "4구간"] * 24 + [None] * 3
    day1 = ["20240301"] + [f"{100 + i}.50" for i in range(96)] + ["195.50", "100.50", "150.00"]
    day2 = ["20240302"] + [None] * 99  # 값이 통째로 없는 날
    day3 = ["20240303"] + ["-70.94"] * 96 + ["-70.94", "-70.94", "-70.94"]
    notice = ["20240304", "확정가격은 D+1일 18시까지 공표 예정입니다."] + [None] * 98
    pd.DataFrame([title, hours, slots, day1, day2, day3, notice]).to_excel(
        tmp_path / "smpDataRt_2024.xlsx", header=False, index=False
    )

    out = realtime_xlsx.load_dir(tmp_path).set_index("ts")["smp_rt"]

    assert len(out) == 3 * 96  # 값이 있는 마지막 날(03-03)까지
    assert out["2024-03-01 00:00"] == 100.5
    assert out["2024-03-01 23:45"] == 195.5
    assert out["2024-03-02"].isna().all()
    assert (out["2024-03-03"] == -70.94).all()


# ---------------------------------------------------------------- curtailment


def test_curtailment_fills_days_without_records(tmp_path):
    def line(day, values):
        return ",".join([day, "제주", "1", *values])

    header = ",".join(["일자", "제어지역", "누적 시행회차", *HOURS])
    wind_values = ["0"] * 12 + ["12", "28"] + ["0"] * 8 + ["", ""]  # 끝 두 칸이 비어 있음
    solar_values = ["0"] * 11 + ["출력제어"] * 3 + ["0"] * 10
    wind = tmp_path / "제주 풍력 출력제어횟수 및 제어량.csv"
    solar = tmp_path / "제주 태양광 출력제어횟수.csv"
    wind.write_text("\n".join([header, line("2021-01-13", wind_values)]), encoding="cp949")
    solar.write_text("\n".join([header, line("2021-01-20", solar_values)]), encoding="utf-8-sig")

    out = curtailment.load(wind, solar).set_index("ts")

    assert out.index[0] == pd.Timestamp("2021-01-01 00:00")
    assert out.loc["2021-01-13 12:00", "wind_curtail_mwh"] == 12  # 13시 -> 12:00
    assert out.loc["2021-01-13 23:00", "wind_curtail_mwh"] == 0  # 빈 칸은 0
    assert out.loc["2021-01-05 12:00", "wind_curtail_mwh"] == 0  # 기록 없는 날은 0
    assert pd.isna(out.loc["2021-01-14 00:00", "wind_curtail_mwh"])  # 마지막 기록일 이후
    assert out.loc["2021-01-20 11:00":"2021-01-20 13:00", "solar_curtail"].tolist() == [1, 1, 1]
    assert out["solar_curtail"].sum() == 3


# ---------------------------------------------------------------- build


def test_build_prefers_api_and_fills_gaps_from_xlsx():
    ts = pd.date_range("2024-07-01 23:00", periods=8, freq="15min")  # 07-01 23시, 07-02 0시
    xlsx = pd.DataFrame({"ts": ts, "smp_rt": [10.0, 10, 10, 10, 20, 20, 20, 20]})
    api = pd.DataFrame(
        {"ts": ts[4:], "smp_rt": [21.0, 21, 21, 25], "fcst_rt": [600.0, 601, 602, 603]}
    )

    rt = build.build_rt(xlsx, api)

    assert rt["source"].tolist() == ["xlsx"] * 4 + ["api"] * 4
    assert rt["smp_rt"].tolist() == [10, 10, 10, 10, 21, 21, 21, 25]  # 겹치면 API 값
    assert rt["fcst_rt"].isna().tolist() == [True] * 4 + [False] * 4

    hours = pd.date_range("2024-07-01 22:00", periods=3, freq="h")
    demand_df = pd.DataFrame({"ts": hours, "demand_mwh": [700.0, 710, 720]})
    day_ahead_df = pd.DataFrame(
        {"ts": hours, "fcst_jeju": [690.0, 700, 710], "smp_jeju": [9.0, 10, 20]}
    )
    curtail_df = pd.DataFrame(
        {"ts": hours[:2], "wind_curtail_mwh": [0.0, 5], "solar_curtail": [0, 1]}
    )

    hourly = build.build_hourly(demand_df, day_ahead_df, rt, curtail_df).set_index("ts")

    assert list(hourly.columns) == build.HOURLY_COLS[1:]
    assert pd.isna(hourly.loc["2024-07-01 22:00", "smp_rt_mean"])  # 실시간 자료가 없는 시간
    assert hourly.loc["2024-07-01 23:00", "smp_rt_mean"] == 10
    row = hourly.loc["2024-07-02 00:00"]
    assert (row["smp_rt_mean"], row["smp_rt_min"], row["smp_rt_max"]) == (22, 21, 25)
    assert row["smp_da"] == 20
    assert row["fcst_rt_mean"] == 601.5
    assert pd.isna(row["wind_curtail_mwh"])  # 출력제어 기록이 끝난 뒤
