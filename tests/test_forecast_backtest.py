"""forecast_backtest: 합성 입력 파일로 CLI 전체 흐름과 비교 조건 검사를 확인."""

import json
from datetime import date, timedelta

import pandas as pd
import yaml

from tests.test_forecast_inputs import SOLAR, WIND, make_predictions, make_smp
from vpp import forecast_backtest as fb
from vpp import forecast_rolling as fr
from vpp.forecast_inputs import PlanWindow
from vpp.optimization import EssParams

DAYS = [date(2024, 6, 5) + timedelta(days=i) for i in range(3)]
ESS_CFG = {
    "capacity_mwh": 16.0, "charge_power_mw": 4.0, "discharge_power_mw": 4.0,
    "soc_min": 0.1, "soc_max": 0.9, "soc_initial": 0.5,
    "charge_efficiency": 0.95, "discharge_efficiency": 0.95,
}


def write_inputs(tmp_path, days):
    """예측 parquet 2개, SMP csv, 발전소 기준정보, ESS yaml 을 임시 폴더에 만든다."""
    pred = make_predictions(days)
    # 시간대별로 값이 달라야 ESS 가 움직인다: 낮 발전, 저녁 고가
    hour = pred["valid_time_kst"].dt.hour
    pred["lightgbm_generation_kwh"] = ((hour >= 8) & (hour <= 16)) * 300.0
    pred["generation_kwh"] = ((hour >= 8) & (hour <= 16)) * 250.0
    pred[pred["energy_type"] == "solar"].to_parquet(tmp_path / "solar.parquet", index=False)
    pred[pred["energy_type"] == "wind"].to_parquet(tmp_path / "wind.parquet", index=False)
    smp = make_smp(days)
    evening = pd.to_datetime(smp["target_time"]).dt.hour.between(18, 21)
    smp.loc[evening, ["smp_hat", "smp_actual"]] = [[200.0, 180.0]]
    smp["smp_hat_krw_per_mwh"] = smp["smp_hat"] * 1000
    smp.to_csv(tmp_path / "smp.csv", index=False)
    pd.DataFrame({
        "asset_id": SOLAR + WIND,
        "energy_type": ["solar"] * len(SOLAR) + ["wind"] * len(WIND),
    }).to_csv(tmp_path / "plants.csv", index=False)
    (tmp_path / "ess.yaml").write_text(yaml.safe_dump(ESS_CFG), encoding="utf-8")


def test_cli_run_writes_outputs_and_passes_checks(tmp_path):
    write_inputs(tmp_path, DAYS)
    out = fb.run_forecast_backtest(
        DAYS[0], DAYS[-1],
        solar_path=tmp_path / "solar.parquet", wind_path=tmp_path / "wind.parquet",
        smp_path=tmp_path / "smp.csv", plants_path=tmp_path / "plants.csv",
        ess_path=tmp_path / "ess.yaml", out_dir=tmp_path / "out", run_id="t",
    )
    for name in ("hourly_forecast.csv", "hourly_actual.csv", "daily_comparison.csv",
                 "excluded_days.csv", "comparison.html", "summary.json", "validation.json"):
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    assert summary["mode"] == fb.MODE and summary["days"] == 3 and summary["excluded_days"] == 0
    assert validation["comparison_all_passed"] and validation["plan_checks_all_passed"]
    assert {c["item"] for c in validation["comparison_checks"]} >= {
        "same_planning_days_symmetric_difference", "same_hourly_timestamps",
        "same_initial_energy_each_segment",
    }
    assert validation["comparison_warnings"][0]["flagged"] is False
    hf = pd.read_csv(out / "hourly_forecast.csv")
    ha = pd.read_csv(out / "hourly_actual.csv")
    assert len(hf) == len(ha) == 24 * 3
    assert (hf["scenario"] == "forecast").all() and (ha["scenario"] == "actual").all()
    daily = pd.read_csv(out / "daily_comparison.csv")
    assert len(daily) == 3 and "planned_objective_diff_krw" in daily
    # 이월 확인 열: 첫날은 초기화, 이후는 전날 종료 = 오늘 시작
    assert daily["segment_start_forecast"].tolist() == [True, False, False]
    assert daily["carry_gap_mwh_forecast"].iloc[1:].abs().max() < 1e-9
    assert pd.isna(daily["carry_gap_mwh_forecast"].iloc[0])
    diff = summary["scenarios"]["forecast"]["planned_objective_krw"] - \
        summary["scenarios"]["actual"]["planned_objective_krw"]
    assert summary["planned_objective_diff_krw"] == diff


def test_days_without_actual_are_compared_out_not_planned_out(tmp_path):
    """실측이 빠진 날은 두 시나리오 모두 비교에서 빠지고(stage=comparison) 검사는 통과한다."""
    write_inputs(tmp_path, DAYS)
    smp = pd.read_csv(tmp_path / "smp.csv")
    smp.loc[smp["issue_date"] == (DAYS[1] - timedelta(days=1)).isoformat(), "smp_actual"] = None
    smp.to_csv(tmp_path / "smp.csv", index=False)
    out = fb.run_forecast_backtest(
        DAYS[0], DAYS[-1],
        solar_path=tmp_path / "solar.parquet", wind_path=tmp_path / "wind.parquet",
        smp_path=tmp_path / "smp.csv", plants_path=tmp_path / "plants.csv",
        ess_path=tmp_path / "ess.yaml", out_dir=tmp_path / "out", run_id="t2",
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    assert summary["days"] == 2 and summary["excluded_days"] == 0
    assert summary["comparison_excluded_days"] == 1
    assert validation["comparison_all_passed"] and validation["plan_checks_all_passed"]
    assert validation["data_checks"]["windows_built"] == 3
    assert validation["data_checks"]["comparable_windows"] == 2
    excluded = pd.read_csv(out / "excluded_days.csv")
    assert excluded.to_dict("records") == [
        {"planning_day": DAYS[1].isoformat(), "reason": "price_actual_missing",
         "stage": "comparison"},
    ]
    daily = pd.read_csv(out / "daily_comparison.csv")
    assert daily["planning_day"].tolist() == [DAYS[0].isoformat(), DAYS[2].isoformat()]
    # 가운데 날이 빠졌으니 구간 2개, 둘째 구간도 설정값 SOC 로 시작
    assert daily["segment_start_forecast"].tolist() == [True, True]


def test_cli_main_exits_nonzero_when_checks_fail(tmp_path, monkeypatch, capsys):
    write_inputs(tmp_path, DAYS)
    # 비교 검사 하나를 강제로 실패시켜 종료 코드와 메시지를 확인
    monkeypatch.setattr(fb, "comparison_checks",
                        lambda f, a, ess, tol=1e-6: [fb._item("forced_mismatch", 1.0, False)])
    argv = ["--start", DAYS[0].isoformat(), "--end", DAYS[-1].isoformat(),
            "--solar", str(tmp_path / "solar.parquet"), "--wind", str(tmp_path / "wind.parquet"),
            "--smp", str(tmp_path / "smp.csv"), "--plants", str(tmp_path / "plants.csv"),
            "--ess", str(tmp_path / "ess.yaml"), "--out", str(tmp_path / "out")]
    try:
        fb.main(argv)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("불일치인데 종료 코드 1 이 아님")
    assert "forced_mismatch" in capsys.readouterr().out


def _window(day, price_f, price_a):
    ts = pd.date_range(pd.Timestamp(day, tz="Asia/Seoul"), periods=48, freq="h")
    gen = [5.0 if 8 <= t.hour <= 16 else 0.0 for t in ts]
    base = {"ts": ts, "generation_mwh": gen, "dt_hours": 1.0}
    return PlanWindow(
        day,
        pd.DataFrame({**base, "price_krw_per_mwh": price_f}),
        pd.DataFrame({**base, "price_krw_per_mwh": price_a}),
        {},
    )


def test_comparison_checks_detect_mismatch():
    ess = EssParams(**ESS_CFG)
    windows = [_window(d, 150_000.0, 120_000.0) for d in DAYS]
    forecast = fr.run_scenario(windows, ess, "forecast")
    actual_short = fr.run_scenario(windows[:2], ess, "actual")
    checks_short = fb.comparison_checks(forecast, actual_short, ess)
    failed = {c["item"] for c in checks_short if not c["passed"]}
    assert "same_planning_days_symmetric_difference" in failed
    assert "same_hourly_timestamps" in failed

    # 예측 입력이 실측과 전부 같으면 실패가 아니라 확인용 경고로 표시한다
    leaked = [_window(d, 120_000.0, 120_000.0) for d in DAYS]
    lf, la = fr.run_scenario(leaked, ess, "forecast"), fr.run_scenario(leaked, ess, "actual")
    assert all(c["passed"] for c in fb.comparison_checks(lf, la, ess))
    warnings = fb.comparison_warnings(lf, la)
    assert warnings[0]["item"] == "forecast_inputs_identical_to_actual"
    assert warnings[0]["flagged"] is True
    assert fb.comparison_warnings(forecast, fr.run_scenario(windows, ess, "actual"))[0][
        "flagged"] is False
