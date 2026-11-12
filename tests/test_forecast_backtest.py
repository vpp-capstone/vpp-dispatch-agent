"""forecast_backtest: 합성 입력 파일로 CLI 전체 흐름과 비교 조건 검사를 확인."""

import json
from datetime import date, timedelta

import pandas as pd
import pytest
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


def test_rec_config_splits_revenue_and_applies_to_both_scenarios(tmp_path):
    """--rec: 두 시나리오 모두 REC 가산으로 풀고 요약에 판매(SMP)·REC·열화비·ESS 없음 대비 분리."""
    write_inputs(tmp_path, DAYS)
    rec_yaml = tmp_path / "rec.yaml"
    rec_yaml.write_text(yaml.safe_dump({"weight": 1.0, "price_krw_per_rec": 70000}),
                        encoding="utf-8")
    common = dict(solar_path=tmp_path / "solar.parquet", wind_path=tmp_path / "wind.parquet",
                  smp_path=tmp_path / "smp.csv", plants_path=tmp_path / "plants.csv",
                  ess_path=tmp_path / "ess.yaml", out_dir=tmp_path / "out")
    base = json.loads((fb.run_forecast_backtest(DAYS[0], DAYS[-1], run_id="base", **common)
                       / "summary.json").read_text(encoding="utf-8"))
    out = fb.run_forecast_backtest(DAYS[0], DAYS[-1], run_id="rec", rec_path=rec_yaml, **common)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    assert validation["comparison_all_passed"] and validation["plan_checks_all_passed"]
    assert summary["rec"]["enabled"] and summary["rec"]["pricing"]["price_krw_per_rec"] == 70000
    assert summary["inputs"]["rec_config"] == str(rec_yaml)
    assert base["rec"]["enabled"] is False and base["scenarios"]["actual"]["rec_revenue_krw"] == 0
    for key in ("forecast", "actual"):
        t = summary["scenarios"][key]
        assert t["rec_revenue_krw"] > 0
        assert t["planned_objective_krw"] == pytest.approx(
            t["sales_revenue_krw"] + t["rec_revenue_krw"] - t["throughput_cost_krw"])
        assert t["ess_gain_vs_no_ess_krw"] == pytest.approx(
            t["planned_objective_krw"] - t["no_ess_net_revenue_krw"])
        assert t["terminal_constraint_cost_krw"] >= 0
        hourly = pd.read_csv(out / f"hourly_{key}.csv")
        assert (hourly["rec_krw_per_mwh"] == 70000).all()
        assert t["rec_revenue_krw"] == pytest.approx(70000 * hourly["sell_mwh"].sum())
    assert summary["initial_energy_rec_upper_bound_krw"] > 0
    # REC 는 발전량에 붙는 돈: 합성 입력에선 감축이 없어 ESS 없음 수익이 정확히 REC 만큼 는다
    delta_no_ess = (summary["scenarios"]["actual"]["no_ess_net_revenue_krw"]
                    - base["scenarios"]["actual"]["no_ess_net_revenue_krw"])
    gen = summary["scenarios"]["actual"]["generation_mwh"]
    assert delta_no_ess == pytest.approx(70000 * gen)


def test_execution_outputs_keep_three_revenue_concepts_apart(tmp_path):
    """실측이 있으면 실행 운전·정산 결과가 붙고, 예상/실행/완전예측 수익이 따로 저장된다."""
    write_inputs(tmp_path, DAYS)
    rec_yaml = tmp_path / "rec.yaml"
    rec_yaml.write_text(yaml.safe_dump({"weight": 1.0, "price_krw_per_rec": 70000}),
                        encoding="utf-8")
    out = fb.run_forecast_backtest(
        DAYS[0], DAYS[-1], rec_path=rec_yaml, terminal_comparison=True,
        solar_path=tmp_path / "solar.parquet", wind_path=tmp_path / "wind.parquet",
        smp_path=tmp_path / "smp.csv", plants_path=tmp_path / "plants.csv",
        ess_path=tmp_path / "ess.yaml", out_dir=tmp_path / "out", run_id="x",
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    assert validation["execution_checks_all_passed"] and validation["plan_checks_all_passed"]
    exe = summary["execution"]
    assert exe["available"] and exe["basis"] == "assumed_actual_smp"
    assert set(exe["scenarios"]) == {"executed", "legacy_plan_executed", "no_ess_executed"}
    assert set(summary["revenue_concepts"]) == {"planned_forecast", "executed",
                                                "perfect_foresight"}
    planned = summary["scenarios"]["forecast"]["planned_objective_krw"]
    realized = exe["scenarios"]["executed"]["realized_net_krw"]
    perfect = summary["scenarios"]["actual"]["planned_objective_krw"]
    assert len({round(planned), round(realized), round(perfect)}) == 3
    hourly = pd.read_csv(out / "hourly_execution.csv")
    daily = pd.read_csv(out / "daily_execution.csv")
    assert set(hourly["scenario"]) == set(exe["scenarios"]) and len(hourly) == 3 * 72
    assert {"committed_sell_mwh", "supply_mwh", "shortfall_mwh", "surplus_mwh",
            "realized_net_krw", "settlement_basis"} <= set(hourly.columns)
    assert daily.loc[daily.scenario == "executed", "segment_start"].tolist() == [
        True, False, False,
    ]
    # 실행 종료 SOC 가 다음 날 시작 SOC 로 이어진다 (계획 종료 SOC 와 따로 저장)
    d = daily[daily.scenario == "executed"]
    assert d["exec_start_energy_mwh"].iloc[1:].tolist() == pytest.approx(
        d["executed_final_energy_mwh"].iloc[:-1].tolist())
    assert "planned_final_energy_mwh" in d
    assert exe["last_executed_day"] == str(DAYS[-1])
    table = pd.read_csv(out / "terminal_policy_comparison.csv")
    assert len(table) == 4 and "executed_end_shortfall_mwh" in table
    # ESS 미사용 기준은 배터리를 쓰지 않는다
    base = hourly[hourly.scenario == "no_ess_executed"]
    assert base[["charge_mwh", "discharge_mwh"]].abs().to_numpy().max() == 0


def test_period_without_any_actual_still_produces_forecast_plan(tmp_path):
    """실측이 하나도 없는 기간(운영자의 기본 상황): 비교 없이 예측 계획만 만들고 그렇게 표시한다."""
    write_inputs(tmp_path, DAYS)
    for kind in ("solar", "wind"):
        pred = pd.read_parquet(tmp_path / f"{kind}.parquet").drop(columns=["generation_kwh"])
        pred.to_parquet(tmp_path / f"{kind}.parquet", index=False)
    smp = pd.read_csv(tmp_path / "smp.csv")
    smp["smp_actual"] = None
    smp.to_csv(tmp_path / "smp.csv", index=False)
    out = fb.run_forecast_backtest(
        DAYS[0], DAYS[-1],
        solar_path=tmp_path / "solar.parquet", wind_path=tmp_path / "wind.parquet",
        smp_path=tmp_path / "smp.csv", plants_path=tmp_path / "plants.csv",
        ess_path=tmp_path / "ess.yaml", out_dir=tmp_path / "out", run_id="fo",
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    assert summary["actual_available"] is False
    assert summary["execution"]["available"] is False  # 예측값으로 대체해 정산하지 않는다
    assert not (out / "hourly_execution.csv").exists()
    assert summary["days"] == len(DAYS) and "actual" not in summary["scenarios"]
    assert summary["planned_objective_diff_krw"] is None
    assert validation["plan_checks_all_passed"] and validation["comparison_all_passed"]
    assert validation["data_checks"]["comparable_windows"] == 0
    assert (out / "hourly_forecast.csv").exists() and (out / "daily_forecast.csv").exists()
    assert not (out / "hourly_actual.csv").exists()
    hourly = pd.read_csv(out / "hourly_forecast.csv")
    assert len(hourly) == 24 * len(DAYS)
    # 대시보드 로더도 실측 없이 읽힌다
    pytest.importorskip("streamlit")
    from dashboard.overview_data import load_experiment
    data = load_experiment(out)
    assert data["actual"] is None and "actual" not in data["revenues"]
    assert data["metrics"]["days"] == len(DAYS) and data["metrics"]["actual_available"] is False


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
