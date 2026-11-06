"""열화비 집계 통일: 엔진·롤링·기준 계획·비교표가 모두 열화모델의 비용 함수와 같은 값을 낸다.

charge_plus_discharge 에서는 직접 계산식과 모델이 같아 차이가 안 보이므로 discharge_only 로
검사한다.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tests.test_forecast_rolling import make_window
from vpp import forecast_rolling as fr
from vpp.baselines import no_ess, plan_revenue_krw, rule_based
from vpp.evaluation import summarize_backtest
from vpp.optimization import EssParams, run_rolling
from vpp.validation import check_plan

DAYS = [date(2024, 6, 5) + timedelta(days=i) for i in range(3)]
DISCHARGE_ONLY = EssParams(
    capacity_mwh=16.0, charge_power_mw=4.0, discharge_power_mw=4.0,
    soc_min=0.1, soc_max=0.9, soc_initial=0.5,
    charge_efficiency=0.95, discharge_efficiency=0.95,
    throughput_cost_krw_per_mwh=20_000.0, degradation_model="fixed_throughput",
    throughput_basis="discharge_only",
)


def model_cost(plan: pd.DataFrame) -> float:
    return DISCHARGE_ONLY.degradation().cost_for_arrays(plan["charge_mwh"], plan["discharge_mwh"])


def test_forecast_rolling_daily_cost_matches_model_not_direct_formula():
    seg = fr.run_segment([make_window(d) for d in DAYS], DISCHARGE_ONLY, "forecast")
    hourly, daily = seg.result.hourly, seg.result.daily
    assert daily["throughput_cost_krw"].sum() == pytest.approx(model_cost(hourly))
    direct = DISCHARGE_ONLY.throughput_cost_krw_per_mwh * float(
        hourly["charge_mwh"].sum() + hourly["discharge_mwh"].sum()
    )
    assert hourly["charge_mwh"].sum() > 0  # 충전이 있어야 두 식이 갈린다
    assert daily["throughput_cost_krw"].sum() != pytest.approx(direct)
    sales = float(hourly["price_krw_per_mwh"] @ hourly["sell_mwh"])
    assert seg.result.objective_krw == pytest.approx(sales - model_cost(hourly))
    checks = check_plan(hourly, DISCHARGE_ONLY,
                        initial_energy_mwh=seg.result.period_initial_energy_mwh,
                        objective_krw=seg.result.objective_krw)
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]


def test_compare_daily_carries_sales_and_cost_columns():
    windows = [make_window(d) for d in DAYS]
    table = fr.compare_daily(fr.run_scenario(windows, DISCHARGE_ONLY, "forecast"),
                             fr.run_scenario(windows, DISCHARGE_ONLY, "actual"))
    for s in ("forecast", "actual"):
        assert np.allclose(table[f"objective_krw_{s}"],
                           table[f"sales_revenue_krw_{s}"] - table[f"throughput_cost_krw_{s}"])
        assert (table[f"throughput_cost_krw_{s}"] > 0).any()


def test_run_rolling_baselines_and_summary_use_model_cost():
    ts = pd.date_range("2024-06-05", periods=72, freq="h", tz="Asia/Seoul")
    hour = np.array(ts.hour)
    # 규칙 계획은 하루 중 가장 싼 4시간에 충전하므로, 그 시간(10~13시)에 발전이 있어야 한다.
    price = np.where((hour >= 18) & (hour <= 21), 150_000.0, 60_000.0)
    price = np.where((hour >= 10) & (hour <= 13), 30_000.0, price)
    frame = pd.DataFrame({
        "ts": ts,
        "generation_mwh": np.where((hour >= 8) & (hour <= 16), 6.0, 0.0),
        "price_krw_per_mwh": price,
        "dt_hours": 1.0,
    })
    plan = run_rolling(frame, DISCHARGE_ONLY)
    assert plan.throughput_cost_krw == pytest.approx(model_cost(plan.hourly))
    rule = rule_based(frame, DISCHARGE_ONLY)
    assert rule["charge_mwh"].sum() > 0
    expected_rule = float(rule["price_krw_per_mwh"] @ rule["sell_mwh"]) - model_cost(rule)
    assert plan_revenue_krw(rule, DISCHARGE_ONLY) == pytest.approx(expected_rule)
    summary = summarize_backtest(plan, {"no_ess": no_ess(frame), "rule_based": rule},
                                 DISCHARGE_ONLY)
    assert summary["throughput_cost_krw"] == pytest.approx(model_cost(plan.hourly))
    assert summary["baselines"]["rule_based"]["net_revenue_krw"] == pytest.approx(expected_rule)
    assert summary["throughput_basis"] == "discharge_only"
