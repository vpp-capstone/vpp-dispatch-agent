"""forecast_metrics: 지표 정의가 기존 모델 평가 함수와 같고, 일차 분리·결측·제외 건수가 맞는지."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tests.test_forecast_inputs import EXPECTED, SOLAR, WIND, make_predictions, make_smp
from vpp import forecast_metrics as fm
from vpp import smp_forecast
from vpp.generation_forecast.model import metric_rows

DAYS = [date(2024, 6, 1) + timedelta(days=i) for i in range(3)]


def eval_predictions(days=DAYS) -> pd.DataFrame:
    pred = make_predictions(days, kwh=100.0)  # 예측 100, 실측 200
    pred["plant_name"] = pred["asset_id"]
    pred["capacity_kw"] = np.where(pred["energy_type"] == "solar", 1000.0, 3000.0)
    pred["baseline_generation_kwh"] = 150.0
    pred["issue_time_utc"] = pd.to_datetime(pred["issue_time_utc"], utc=True)
    return pred


def test_portfolio_metrics_match_generation_model_error_metrics():
    pred = eval_predictions()
    table = fm.generation_metrics(pred, EXPECTED, level="portfolio")
    mine = table[(table["forecast_day_offset"] == 0) & (table["model"] == "lightgbm")].iloc[0]
    theirs = pd.DataFrame(metric_rows(pred, "lightgbm_generation_kwh", "lightgbm", "all", 21))
    ref = theirs[(theirs["level"] == "portfolio") & (theirs["horizon"] == "day0_24h")].iloc[0]
    assert mine["mae_kwh"] == pytest.approx(ref["mae_kwh"])
    assert mine["rmse_kwh"] == pytest.approx(ref["rmse_kwh"])
    assert mine["nmae_pct"] == pytest.approx(ref["nmae_pct"])
    assert mine["n_evaluated"] == ref["complete_portfolio_windows"] == 3 * 24
    # 편향: 예측 100 − 실측 200 = −100 kWh × 21개소
    assert mine["bias_kwh"] == pytest.approx(-100.0 * 21)
    assert mine["bias_kwh"] < 0  # 과소예측이면 음수


def test_offsets_are_separate_and_counts_complete():
    table = fm.generation_metrics(eval_predictions(), EXPECTED, level="portfolio")
    assert set(table["forecast_day_offset"]) == {0, 1}
    assert set(table["model"]) == {"lightgbm", "baseline"}
    assert (table["n_target_hours"] == 72).all() and (table["n_evaluated"] == 72).all()
    base = table[table["model"] == "baseline"].iloc[0]
    assert base["bias_kwh"] == pytest.approx(-50.0 * 21)


def test_incomplete_and_missing_are_counted_not_filled():
    pred = eval_predictions()
    hour = pd.Timestamp(DAYS[0], tz="Asia/Seoul") + pd.Timedelta(hours=10)
    pred = pred[~((pred["asset_id"] == "wind_a") & (pred["valid_time_kst"] == hour))]
    hour2 = pd.Timestamp(DAYS[0], tz="Asia/Seoul") + pd.Timedelta(hours=11)
    pred.loc[(pred["asset_id"] == "solar_00") & (pred["valid_time_kst"] == hour2),
             "generation_kwh"] = np.nan
    table = fm.generation_metrics(pred, EXPECTED, level="portfolio")
    row = table[(table["forecast_day_offset"] == 0) & (table["model"] == "lightgbm")].iloc[0]
    assert row["n_excluded_incomplete_assets"] == 1
    assert row["n_missing_actual"] == 1
    assert row["n_evaluated"] == 72 - 2
    by_type = fm.generation_metrics(pred, EXPECTED, level="energy_type")
    solar = by_type[(by_type["group"] == "solar") & (by_type["forecast_day_offset"] == 0)
                    & (by_type["model"] == "lightgbm")].iloc[0]
    assert solar["n_excluded_incomplete_assets"] == 0 and solar["n_missing_actual"] == 1
    assets = fm.generation_metrics(pred, EXPECTED, level="asset")
    assert assets["group"].nunique() == 21
    ts = fm.generation_timeseries(pred, EXPECTED, level="portfolio", forecast_day_offset=0)
    assert len(ts) == 72 and ts["complete_assets"].sum() == 71
    assert ts["actual_kwh"].isna().sum() == 2  # 불완전 1 + 실측 결측 1


def test_file_without_actual_gives_zero_evaluated(tmp_path):
    pred = eval_predictions().drop(columns=["generation_kwh", "baseline_generation_kwh"])
    pred[pred["energy_type"] == "solar"].to_parquet(tmp_path / "s.parquet", index=False)
    pred[pred["energy_type"] == "wind"].to_parquet(tmp_path / "w.parquet", index=False)
    pd.DataFrame({"asset_id": SOLAR + WIND,
                  "energy_type": ["solar"] * len(SOLAR) + ["wind"] * len(WIND)}
                 ).to_csv(tmp_path / "plants.csv", index=False)
    loaded = fm.load_generation_eval_frame(tmp_path / "s.parquet", tmp_path / "w.parquet",
                                           tmp_path / "plants.csv")
    table = fm.generation_metrics(loaded, EXPECTED)
    assert (table["n_evaluated"] == 0).all() and table["mae_kwh"].isna().all()
    assert (table["n_missing_actual"] == 72).all()


def smp_eval(tmp_path, days=DAYS) -> pd.DataFrame:
    smp = make_smp(days, price=120.0)  # 예측 120, 실측 121, B1 = 120
    hour = pd.to_datetime(smp["target_time"]).dt.hour
    # 실측 고가 4시간 18~21시, 예측은 17~20시로 한 시간 어긋남 → 교집합 3 → 0.75
    smp.loc[hour.between(18, 21), "smp_actual"] = 200.0
    smp.loc[hour.between(17, 20), "smp_hat"] = 190.0
    smp["smp_b1"] = 120.0
    smp["smp_hat_krw_per_mwh"] = smp["smp_hat"] * 1000
    smp.to_csv(tmp_path / "smp.csv", index=False)
    return fm.load_smp_eval_frame(tmp_path / "smp.csv")


def test_smp_metrics_match_smp_forecast_metrics_and_hit_rate(tmp_path):
    frame = smp_eval(tmp_path)
    table = fm.smp_metrics(frame)
    model_h1 = table[(table["horizon"] == 1) & (table["model"] == "model")].iloc[0]
    # 같은 데이터로 SMP 담당의 metrics 를 돌려 MAE·RMSE 가 같은지
    theirs_in = pd.DataFrame({
        "target_ts": frame["target_time"].dt.tz_localize(None), "horizon": frame["horizon"],
        "smp_pred": frame["smp_hat"], "smp_b1": frame["smp_b1"],
        "smp_actual": frame["smp_actual"], "p_low": 0.0,
    })
    theirs = smp_forecast.metrics(theirs_in)
    ref = theirs[(theirs["대상"] == "내일") & (theirs["방법"] == "모델")].iloc[0]
    assert round(model_h1["mae"], 2) == ref["MAE"]
    assert round(model_h1["rmse"], 2) == ref["RMSE"]
    assert model_h1["n_evaluated"] == ref["시간수"] == 72
    assert model_h1["top4_hit_rate"] == pytest.approx(0.75)
    assert model_h1["n_days_scored"] == 3 and model_h1["n_days_excluded_incomplete"] == 0
    assert model_h1["unit"] == "원/kWh"
    assert model_h1["unit_check_max_abs_diff_krw_per_mwh"] == pytest.approx(0.0)
    b1 = table[(table["horizon"] == 1) & (table["model"] == "b1")].iloc[0]
    assert b1["bias"] < 0  # B1(120) 은 실측(121~200)보다 낮다


def test_top_hours_tie_breaks_to_earlier_hour_and_incomplete_day_excluded(tmp_path):
    frame = pd.DataFrame({"hour": range(24), "v": [100.0] * 24})
    assert fm.top_hours(frame, "v") == {0, 1, 2, 3}
    smp = smp_eval(tmp_path)
    drop = (smp["target_day"] == DAYS[0]) & (smp["hour"] == 5) & (smp["horizon"] == 1)
    table = fm.smp_metrics(smp[~drop])
    row = table[(table["horizon"] == 1) & (table["model"] == "model")].iloc[0]
    assert row["n_days_scored"] == 2 and row["n_days_excluded_incomplete"] == 1
    assert row["n_target_hours"] == 71
