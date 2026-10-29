"""검산 모듈: 올바른 계획은 전부 ✅, 일부러 틀린 계획은 해당 항목만 ❌."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vpp.optimization import EssParams, run_rolling, solve_window
from vpp.validation import check_plan, check_properties


def ess() -> EssParams:
    return EssParams(capacity_mwh=2, charge_power_mw=1, discharge_power_mw=1, soc_min=0,
                     soc_max=1, soc_initial=0.5, charge_efficiency=0.9, discharge_efficiency=0.9)


def good_plan() -> pd.DataFrame:
    g = np.array([1.0, 0.5, 0.0, 0.0])
    p = np.array([10.0, 20.0, 100.0, 90.0])
    r = solve_window(g, p, np.ones(4), ess(), ess().initial_energy_mwh)
    return r.to_frame(g, p, np.ones(4))


def failed_items(checks: list[dict]) -> list[str]:
    return [c["item"] for c in checks if not c["passed"]]


def test_correct_plan_passes_everything():
    plan = good_plan()
    checks = check_plan(plan, ess(), initial_energy_mwh=1.0)
    assert checks and failed_items(checks) == []
    assert {"power_balance_max_abs_error_mwh", "energy_balance_max_abs_error_mwh",
            "soc_bounds_max_violation_mwh", "simultaneous_charge_discharge_max_mwh"} <= {
        c["item"] for c in checks
    }


def test_wrong_soc_is_flagged():
    plan = good_plan()
    plan.loc[1, "energy_end_mwh"] += 0.3  # SOC 재계산과 어긋나게
    failed = failed_items(check_plan(plan, ess(), initial_energy_mwh=1.0))
    assert "energy_balance_max_abs_error_mwh" in failed
    assert "power_balance_max_abs_error_mwh" not in failed


def test_soc_out_of_range_and_simultaneous_are_flagged():
    plan = good_plan()
    plan.loc[2, "energy_end_mwh"] = 2.5  # 용량 2 MWh 초과
    plan.loc[0, "charge_mwh"] = 0.2
    plan.loc[0, "discharge_mwh"] = 0.2
    failed = failed_items(check_plan(plan, ess(), initial_energy_mwh=1.0))
    assert "soc_bounds_max_violation_mwh" in failed
    assert "simultaneous_charge_discharge_max_mwh" in failed


def test_wrong_objective_is_flagged():
    plan = good_plan()
    failed = failed_items(check_plan(plan, ess(), initial_energy_mwh=1.0, objective_krw=123.0))
    assert failed == ["objective_recomputed_abs_error_krw"]


def test_rolling_hourly_passes_with_window_continuity_item():
    frame = pd.DataFrame({
        "ts": pd.date_range("2025-01-01", periods=48, freq="h"),
        "generation_mwh": np.tile([1.0, 0.0], 24),
        "price_krw_per_mwh": np.tile([10.0, 100.0], 24),
    })
    res = run_rolling(frame, ess(), policy="rolling48")
    checks = check_plan(res.hourly, ess(), objective_krw=res.objective_krw)
    assert failed_items(checks) == []
    assert "rolling_energy_continuity_max_abs_error_mwh" in {c["item"] for c in checks}


def test_check_properties():
    summary = {
        "baselines": {"no_ess": {"milp_gain_krw": 5.0}, "rule_based": {"milp_gain_krw": -2.0}},
        "size_sweep": [{"capacity_mwh": 8, "net_revenue_krw": 100},
                       {"capacity_mwh": 16, "net_revenue_krw": 90}],
    }
    result = {c["item"]: c["passed"] for c in check_properties(summary)}
    assert result == {
        "milp_ge_no_ess_gain_krw": True,
        "milp_ge_rule_based_gain_krw": False,
        "revenue_nondecreasing_in_capacity_min_step_krw": False,
    }
