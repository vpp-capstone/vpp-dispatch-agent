"""forecast_inputs: 발전량·SMP 예측을 계획일 48시간 창으로 결합하는 규칙을 합성 데이터로 검사."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from vpp import forecast_inputs as fi

KST = "Asia/Seoul"
SOLAR = [f"solar_{i:02d}" for i in range(18)]
WIND = ["wind_a", "wind_b", "wind_c"]
EXPECTED = {"solar": set(SOLAR), "wind": set(WIND)}


def make_predictions(days: list[date], kwh: float = 100.0) -> pd.DataFrame:
    """계획일마다 D-2 18 UTC 발표분으로 D 00:00~D+1 23:00 KST 48시간, 21개소. 실측은 예측의 2배."""
    rows = []
    for day in days:
        issue = pd.Timestamp(day - timedelta(days=2), tz="UTC") + pd.Timedelta(hours=18)
        for h in range(48):
            valid = pd.Timestamp(day, tz=KST) + pd.Timedelta(hours=h)
            for asset in SOLAR + WIND:
                rows.append({
                    "asset_id": asset,
                    "energy_type": "solar" if asset in SOLAR else "wind",
                    "issue_time_utc": issue,
                    "valid_time_utc": valid.tz_convert("UTC"),
                    "valid_time_kst": valid,
                    "forecast_day_offset": h // 24,
                    "lead_hour": 21 + h,
                    "lightgbm_generation_kwh": kwh,
                    "generation_kwh": kwh * 2,
                })
    return pd.DataFrame(rows)


def make_smp(days: list[date], price: float = 120.0) -> pd.DataFrame:
    """계획일 D 마다 발행일 D-1 의 48행. 예측 원/kWh=price, 실측=price+1."""
    rows = []
    for day in days:
        issue = day - timedelta(days=1)
        for h in range(48):
            target = pd.Timestamp(day, tz=KST) + pd.Timedelta(hours=h)
            rows.append({
                "issue_date": issue.isoformat(),
                "issue_time": f"{issue.isoformat()}T11:00:00+09:00",
                "horizon": 1 + h // 24,
                "lead_hour": 13 + h,
                "target_time": target.isoformat(),
                "smp_hat": price,
                "smp_hat_krw_per_mwh": price * 1000,
                "p_low": 0.0,
                "smp_b1": price,
                "smp_actual": price + 1,
                "trained_until": "2024-05-01",
                "train_rows": 1,
            })
    return pd.DataFrame(rows)


DAYS = [date(2024, 6, 1) + timedelta(days=i) for i in range(3)]


def test_aggregate_sums_21_assets_to_mwh():
    port = fi.aggregate_portfolio(make_predictions(DAYS, kwh=100.0), EXPECTED)
    assert len(port) == 3 * 48
    assert port["complete"].all()
    assert np.allclose(port["generation_forecast_mwh"], 21 * 100 / 1000)
    assert np.allclose(port["generation_actual_mwh"], 21 * 200 / 1000)
    assert set(port["planning_day"]) == set(DAYS)
    # 같은 발표분의 D+1 행도 planning_day 는 D
    d0 = port[port["issue_time_utc"] == port["issue_time_utc"].min()]
    assert d0["planning_day"].nunique() == 1
    assert sorted(d0["forecast_day_offset"].unique()) == [0, 1]


def test_duplicate_key_raises():
    pred = make_predictions(DAYS[:1])
    dup = pd.concat([pred, pred.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="중복 키"):
        fi.validate_generation_predictions(dup, EXPECTED)


def test_unknown_asset_raises():
    pred = make_predictions(DAYS[:1])
    pred.loc[0, "asset_id"] = "solar_zz"
    with pytest.raises(ValueError, match="기준정보에 없는"):
        fi.validate_generation_predictions(pred, EXPECTED)


def test_missing_asset_hour_excludes_window_with_reason():
    pred = make_predictions(DAYS)
    hour = pd.Timestamp(DAYS[1], tz=KST) + pd.Timedelta(hours=30)
    drop = (pred["asset_id"] == "wind_b") & (pred["valid_time_kst"] == hour)
    port = fi.aggregate_portfolio(pred[~drop], EXPECTED)
    gen, reason = fi.generation_window(port, DAYS[1])
    assert gen is None and reason.startswith("generation_incomplete_assets")
    assert fi.generation_window(port, DAYS[0])[1] is None


def test_plan_window_separates_forecast_and_actual():
    port = fi.aggregate_portfolio(make_predictions(DAYS, kwh=100.0), EXPECTED)
    window, reason = fi.build_plan_window(port, make_smp(DAYS, price=120.0), DAYS[1])
    assert reason is None
    f, a = window.forecast, window.actual
    assert len(f) == 48 and len(a) == 48
    assert f["ts"].iloc[0] == pd.Timestamp(DAYS[1], tz=KST)
    assert (f["ts"].diff().dropna() == pd.Timedelta(hours=1)).all()
    assert f["ts"].equals(a["ts"])
    assert np.allclose(f["generation_mwh"], 2.1) and np.allclose(a["generation_mwh"], 4.2)
    assert np.allclose(f["price_krw_per_mwh"], 120_000)
    assert np.allclose(a["price_krw_per_mwh"], 121_000)
    assert list(f["forecast_day_offset"]) == [0] * 24 + [1] * 24
    assert list(f["horizon"]) == [1] * 24 + [2] * 24
    assert not {"generation_actual_mwh", "price_actual_krw_per_mwh", "smp_actual"} & set(f.columns)
    assert window.sources["smp_issue_date"] == (DAYS[1] - timedelta(days=1)).isoformat()
    assert window.sources["generation_availability_verified"] is False


def test_missing_smp_issue_excluded():
    port = fi.aggregate_portfolio(make_predictions(DAYS), EXPECTED)
    smp = make_smp(DAYS[:2])  # 셋째 날 발행분 없음
    windows, excluded = fi.build_plan_windows(port, smp, DAYS[0], DAYS[-1])
    assert [w.planning_day for w in windows] == DAYS[:2]
    assert excluded.to_dict("records") == [{"planning_day": DAYS[2], "reason": "price_missing"}]


def test_short_price_window_excluded():
    port = fi.aggregate_portfolio(make_predictions(DAYS), EXPECTED)
    smp = make_smp(DAYS)
    first_issue = (DAYS[0] - timedelta(days=1)).isoformat()
    smp = smp.drop(smp[(smp["issue_date"] == first_issue) & (smp["lead_hour"] == 20)].index)
    _, reason = fi.build_plan_window(port, smp, DAYS[0])
    assert reason == "price_not_48h:47"


def test_nan_forecast_excluded_not_filled():
    pred = make_predictions(DAYS[:1])
    pred.loc[pred.index[5], "lightgbm_generation_kwh"] = np.nan
    port = fi.aggregate_portfolio(pred, EXPECTED)
    _, reason = fi.build_plan_window(port, make_smp(DAYS[:1]), DAYS[0])
    assert reason == "generation_nan"


def test_contiguous_segments_split_on_gap():
    days = [date(2024, 6, 1), date(2024, 6, 2), date(2024, 6, 4)]
    port = fi.aggregate_portfolio(make_predictions(days), EXPECTED)
    windows, _ = fi.build_plan_windows(port, make_smp(days), days[0], days[-1])
    segments = fi.contiguous_segments(windows)
    assert [[w.planning_day for w in s] for s in segments] == [days[:2], days[2:]]
