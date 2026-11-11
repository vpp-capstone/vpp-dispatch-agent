"""대시보드가 실행 운전·정산 결과를 계획·완전예측과 섞지 않고 읽고 재생하는지 검사."""

from datetime import date

import numpy as np
import pytest

from tests.test_forecast_backtest import DAYS, write_inputs
from vpp import forecast_backtest as fb

pytest.importorskip("streamlit")
from dashboard.operator_simulation import replay_payload, simulator_html  # noqa: E402
from dashboard.overview_data import load_experiment  # noqa: E402


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exec")
    write_inputs(tmp, DAYS)
    return fb.run_forecast_backtest(
        DAYS[0], DAYS[-1],
        solar_path=tmp / "solar.parquet", wind_path=tmp / "wind.parquet",
        smp_path=tmp / "smp.csv", plants_path=tmp / "plants.csv",
        ess_path=tmp / "ess.yaml", out_dir=tmp / "out", run_id="dash",
    )


def test_loader_exposes_execution_and_plan_separately(experiment):
    data = load_experiment(experiment)
    assert data["metrics"]["execution_available"] is True
    assert set(data["execution"]) == {"executed", "legacy_plan_executed", "no_ess_executed"}
    executed = data["execution"]["executed"]
    plan = data["plan"]
    # 화면의 계획 = 실행 사슬의 확정 계획 (실행 SOC 에서 시작). 비교 쌍 forecast/actual 은 그대로.
    assert np.allclose(plan["sell_mwh"], executed["committed_sell_mwh"])
    assert np.allclose(plan["charge_mwh"], executed["planned_charge_mwh"])
    assert np.allclose(plan["generation_mwh"] + plan["discharge_mwh"],
                       plan["sell_mwh"] + plan["charge_mwh"] + plan["curtailment_mwh"])
    assert len(data["forecast"]) == len(data["actual"]) == len(plan) == 72
    assert data["realized_profit_krw"] == pytest.approx(executed["realized_net_krw"].sum())
    assert data["realized_profit_krw"] != pytest.approx(
        data["revenues"]["forecast"]["net_revenue_krw"])
    daily = data["execution_daily"]
    assert set(daily["scenario"]) == set(data["execution"])
    assert daily.loc[daily.scenario == "executed", "segment_start"].tolist() == [
        True, False, False,
    ]


def test_replay_adds_executed_mode_with_committed_and_supplied(experiment):
    data = load_experiment(experiment)
    day = date(2024, 6, 6)
    payload = replay_payload(data, day)
    assert payload["executionAvailable"] is True
    assert set(payload["series"]) == {"forecast", "actual", "executed"}
    rows = payload["series"]["executed"]
    executed = data["execution"]["executed"]
    block = executed[executed.ts.dt.date == day]
    assert len(rows) == 24
    assert [r["committed"] for r in rows] == pytest.approx(block["committed_sell_mwh"].tolist())
    assert [r["q"] for r in rows] == pytest.approx(block["supply_mwh"].tolist())
    assert [r["g"] for r in rows] == pytest.approx(block["actual_generation_mwh"].tolist())
    assert rows[-1]["cumulative"] == pytest.approx(block["realized_net_krw"].sum())
    # 전날 실행 종료 → 오늘 시작 (실행 SOC 이월)
    assert payload["dailySoc"]["executed"]["previous"]["status"] == "matched"
    assert payload["dailySoc"]["executed"]["start"] == pytest.approx(
        block["energy_start_mwh"].iloc[0] / data["ess"].capacity_mwh * 100)
    # forecast 모드는 화면의 계획(실행 사슬 확정 계획)을 재생한다
    forecast_rows = payload["series"]["forecast"]
    assert [r["q"] for r in forecast_rows] == pytest.approx(block["committed_sell_mwh"].tolist())
    html = simulator_html(data, day)
    assert "실행 결과" in html and "실시간 운전 아님" in html


def test_loader_without_execution_keeps_plan_none(tmp_path):
    from tests.test_overview_data import saved_experiment

    data = load_experiment(saved_experiment(tmp_path))
    assert data["metrics"]["execution_available"] is False
    assert data["plan"] is None and data["execution"] == {}
    assert data["realized_profit_krw"] is None
    payload = replay_payload(data, date(2024, 6, 1))
    assert payload["executionAvailable"] is False and "executed" not in payload["series"]
