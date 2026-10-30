"""작은 가짜 CSV 4개로 포트폴리오 로더의 규칙(단위·합산·결측·시각·플래그)을 검증한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vpp.portfolio_data import (
    HANGYEONG_FILE,
    HOURS,
    PV_FILE,
    SANGMYEONG_FILE,
    SEONGSAN_FILE,
    build_portfolio_hourly,
)

DAYS = ["2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02"]


def _write_pv(path: Path, days: list[str], plants: int = 2) -> None:
    rows = []
    for d in days:
        for h in HOURS:
            for k in range(plants):
                rows.append({
                    "거래일": d, "지역": "제주", "발전원": "태양광", "발전기코드": f"P{k}",
                    "발전기명": f"발전소{k}", "설비용량": 1000.0, "시간": h,
                    "계량전력량": 100.0 * (k + 1), "단가": 100.0 + h, "전력판매매출": 0,
                })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_knp(path: Path, days: list[str], value: float, overrides: dict | None = None,
               extra_rows: list[dict] | None = None) -> None:
    rows = []
    for d in days:
        for stage in (1, 2):
            row = {"년월일": d, "단계": stage, "총량": 0, "평균": 0, "최대": 0, "최소": 0,
                   "최대(시간별)": 0, "최소(시간별)": 0}
            row.update({str(h): value * stage for h in HOURS})
            rows.append(row)
    for row in rows:
        for (d, stage, h), v in (overrides or {}).items():
            if row["년월일"] == d and row["단계"] == stage:
                row[str(h)] = v
    rows.extend(extra_rows or [])
    pd.DataFrame(rows).to_csv(path, index=False, encoding="cp949")


def _write_sangmyeong(path: Path, days: list[str], wh_value: float, kwh_value: float) -> None:
    rows = []
    for d in days:
        row = {"기준일": d, "호기": 1}
        v = wh_value if d <= "2025-01-31" else kwh_value
        row.update({f"{h}시": v for h in HOURS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="cp949")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    _write_pv(tmp_path / PV_FILE, DAYS)
    # 성산: 2025-02-02 1단계 3시에 '-' (결측 → 그 날 제외), 2025-01-30 1단계 5시는 용량 초과
    _write_knp(tmp_path / SEONGSAN_FILE, DAYS, 1000.0,
               overrides={("2025-02-02", 1, 3): "-", ("2025-01-30", 1, 5): 19390})
    _write_knp(tmp_path / HANGYEONG_FILE, DAYS, 500.0)
    # 상명: 1/31까지 Wh(5,000,000 → 5,000 kWh), 2/1부터 kWh(5,000)
    _write_sangmyeong(tmp_path / SANGMYEONG_FILE, DAYS, 5_000_000.0, 5_000.0)
    return tmp_path


def test_units_sums_and_time_rule(raw_dir: Path):
    frame, checks = build_portfolio_hourly(raw_dir, "2025-01-30", "2025-02-02",
                                           market_path=None, expected_pv_plants=2)
    # 결측('-')이 있는 2025-02-02는 제외, 나머지 3일 = 72시간
    assert checks["excluded_days"] == [{"date": "2025-02-02", "reasons": ["missing_seongsan"]}]
    assert len(frame) == 72 and checks["included_days"] == 3
    row0 = frame.iloc[0]
    # 시각: 원본 1시 → 00:00 KST
    assert str(row0["ts"]) == "2025-01-30 00:00:00+09:00" and row0["hour"] == 1
    assert str(frame.iloc[23]["ts"]) == "2025-01-30 23:00:00+09:00"
    # PV 2곳 합산 100+200, 단가는 시간별 1개 (100+hour)
    assert row0["pv_kwh"] == 300.0 and row0["price_krw_per_kwh"] == 101.0
    assert row0["price_krw_per_mwh"] == 101_000.0
    # 성산·한경 단계 1+2 합산
    assert frame.iloc[1]["seongsan_kwh"] == 3000.0 and frame.iloc[1]["hangyeong_kwh"] == 1500.0
    # 상명 배율: 1/31까지 ÷1000, 2/1부터 그대로 → 둘 다 5,000 kWh
    jan = frame[frame["date"].astype(str) == "2025-01-31"].iloc[0]
    feb = frame[frame["date"].astype(str) == "2025-02-01"].iloc[0]
    assert jan["sangmyeong_kwh"] == 5000.0 and feb["sangmyeong_kwh"] == 5000.0
    assert bool(jan["sangmyeong_scaled_from_wh"]) and not bool(feb["sangmyeong_scaled_from_wh"])
    # 합산 MWh = (300 + 3000 + 1500 + 5000) / 1000
    assert feb["generation_mwh"] == pytest.approx(9.8)
    assert checks["sources"]["seongsan"]["non_numeric_cells_to_nan"] == 1
    assert checks["price_cross_check"]["status"] == "skipped"


def test_over_capacity_is_flagged_not_fixed_unless_clipped(raw_dir: Path):
    frame, checks = build_portfolio_hourly(raw_dir, "2025-01-30", "2025-01-30",
                                           market_path=None, expected_pv_plants=2)
    hit = frame[frame["hour"] == 5].iloc[0]
    assert bool(hit["seongsan_over_capacity"]) and bool(hit["any_flag"])
    assert hit["seongsan_kwh"] == 19390 + 2000  # 그대로 유지
    assert checks["over_capacity_hours_in_range"]["seongsan"] == 1
    clipped, _ = build_portfolio_hourly(raw_dir, "2025-01-30", "2025-01-30", clip_to_capacity=True,
                                        market_path=None, expected_pv_plants=2)
    assert clipped[clipped["hour"] == 5].iloc[0]["seongsan_kwh"] == 12000 + 2000
    assert bool(clipped[clipped["hour"] == 5].iloc[0]["seongsan_over_capacity"])  # 플래그는 남김


def test_conflicting_duplicate_day_is_excluded(tmp_path: Path):
    _write_pv(tmp_path / PV_FILE, DAYS[:2])
    _write_knp(tmp_path / SEONGSAN_FILE, DAYS[:2], 1000.0)
    dup = {"년월일": "2025-01-30", "단계": 1, "총량": 0, "평균": 0, "최대": 0, "최소": 0,
           "최대(시간별)": 0, "최소(시간별)": 0, **{str(h): 1.0 for h in HOURS}}
    _write_knp(tmp_path / HANGYEONG_FILE, DAYS[:2], 500.0, extra_rows=[dup])
    _write_sangmyeong(tmp_path / SANGMYEONG_FILE, DAYS[:2], 5_000_000.0, 5_000.0)
    frame, checks = build_portfolio_hourly(tmp_path, "2025-01-30", "2025-01-31",
                                           market_path=None, expected_pv_plants=2)
    assert checks["sources"]["hangyeong"]["conflicting_duplicate_dates_excluded"] == ["2025-01-30"]
    assert [d["date"] for d in checks["excluded_days"]] == ["2025-01-30"]
    assert "duplicate_conflict_wind" in checks["excluded_days"][0]["reasons"]
    assert frame["date"].astype(str).unique().tolist() == ["2025-01-31"]


def test_missing_source_day_is_excluded_not_interpolated(raw_dir: Path):
    # 요청 기간이 원본보다 길면 원본에 없는 날은 결측으로 제외된다
    frame, checks = build_portfolio_hourly(raw_dir, "2025-01-29", "2025-01-31",
                                           market_path=None, expected_pv_plants=2)
    assert checks["excluded_days"][0]["date"] == "2025-01-29"
    assert "missing_pv" in checks["excluded_days"][0]["reasons"]
    assert frame["date"].astype(str).unique().tolist() == ["2025-01-30", "2025-01-31"]
    assert not frame["generation_mwh"].isna().any()
