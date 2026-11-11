"""execution_rolling: 실행 SOC 이월 사슬·종료 정책·결측 구간·인과 검사."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tests.test_forecast_rolling import DAYS, ESS, make_window, profile
from vpp import execution_rolling as xr
from vpp import forecast_rolling as fr
from vpp.forecast_inputs import PlanWindow
from vpp.optimization import EssParams, plan_window
from vpp.rec import RecParams

WORN = EssParams(**{**ESS.to_dict(), "throughput_cost_krw_per_mwh": 1000.0})


def chain(windows, ess=ESS, **kw):
    return xr.run_execution_segment(windows, ess, **kw)


def test_plan_starts_from_executed_soc_and_both_finals_are_stored():
    windows = [make_window(d, gen_f=6.0, gen_a=4.0) for d in DAYS]  # 실측이 예측보다 적다
    seg = chain(windows)
    d = seg.daily
    assert np.allclose(d["plan_start_energy_mwh"], d["exec_start_energy_mwh"])
    assert np.allclose(d["exec_start_energy_mwh"].iloc[1:],
                       d["executed_final_energy_mwh"].iloc[:-1])
    assert (d["planned_final_energy_mwh"] != d["executed_final_energy_mwh"]).any()
    assert seg.executed_final_energy_mwh == pytest.approx(d["executed_final_energy_mwh"].iloc[-1])
    assert seg.planned_final_energy_mwh == pytest.approx(d["planned_final_energy_mwh"].iloc[-1])
    assert all(c["passed"] for c in seg.checks)
    # 판매계획은 실행이 바꾸지 않는다
    assert np.allclose(seg.exec_hourly["committed_sell_mwh"], seg.plan_hourly["sell_mwh"])


def test_identical_forecast_and_actual_gives_plan_equal_execution_and_realized_equal_planned():
    windows = [make_window(d, gen_f=6.0, price_f=150_000.0, gen_a=6.0, price_a=150_000.0)
               for d in DAYS]
    seg = chain(windows, ess=WORN, rec=RecParams(weight=1.0, price_krw_per_rec=70_000))
    h, p = seg.exec_hourly, seg.plan_hourly
    assert np.allclose(h["charge_mwh"], p["charge_mwh"])
    assert np.allclose(h["discharge_mwh"], p["discharge_mwh"])
    assert np.allclose(h["supply_mwh"], p["sell_mwh"])
    assert h[["shortfall_mwh", "surplus_mwh"]].to_numpy().max() == 0
    assert np.allclose(h["energy_end_mwh"], p["energy_end_mwh"])
    d = seg.daily
    assert np.allclose(d["realized_net_krw"], d["planned_objective_krw"])
    assert np.allclose(d["rec_krw"], d["planned_rec_revenue_krw"])
    assert np.allclose(d["wear_krw"], d["planned_throughput_cost_krw"])
    assert np.allclose(d["deviation_krw"], 0.0)


def test_legacy_planned_soc_source_reproduces_forecast_scenario_plan():
    windows = [make_window(d, gen_f=6.0, gen_a=4.0) for d in DAYS]
    legacy = fr.run_segment(windows, ESS, "forecast").result.hourly
    seg = chain(windows, plan_soc_source="planned", terminal_mode="last_day_only")
    assert np.allclose(seg.plan_hourly["sell_mwh"], legacy["sell_mwh"])
    assert np.allclose(seg.plan_hourly["energy_end_mwh"], legacy["energy_end_mwh"])
    # 계획은 계획상 SOC 를 잇고, 실행은 실제 잔량을 따른다 (낮 동안 둘이 벌어진다)
    d = seg.daily
    assert np.allclose(d["plan_start_energy_mwh"].iloc[1:], d["planned_final_energy_mwh"].iloc[:-1])
    h = seg.exec_hourly
    assert (np.abs(h["energy_end_mwh"] - h["planned_energy_end_mwh"]) > 1e-6).any()


def test_horizon_aware_applies_end_condition_one_day_early():
    windows = [make_window(d) for d in DAYS]
    old = chain(windows, terminal_mode="last_day_only").daily
    new = chain(windows, terminal_mode="horizon_aware").daily
    assert old["terminal_constraint"].tolist() == ["none", "none", "period_end_ge_initial"]
    assert new["terminal_constraint"].tolist() == [
        "none", "period_end_ge_initial", "period_end_ge_initial",
    ]
    assert new["solved_intervals"].tolist() == [48, 48, 24]
    assert new["planned_final_energy_mwh"].iloc[-1] >= ESS.initial_energy_mwh - 1e-6
    single = chain(windows[:1], terminal_mode="horizon_aware").daily
    assert single["terminal_constraint"].tolist() == ["period_end_ge_initial"]
    with pytest.raises(ValueError, match="terminal_mode"):
        chain(windows, terminal_mode="whenever")


def test_terminal_value_is_plan_only_and_excluded_from_revenue():
    windows = [make_window(d) for d in DAYS]
    value = fr.TerminalValue(fraction=1.0)
    seg = chain(windows, terminal_mode="horizon_aware", terminal_value=value)
    d = seg.daily
    assert d["terminal_constraint"].tolist()[0] == "valued"
    assert d["terminal_value_krw"].iloc[0] > 0 and d["terminal_value_krw"].iloc[1:].eq(0).all()
    frame = windows[0].forecast
    expected = ESS.discharge_efficiency * np.maximum(frame["price_krw_per_mwh"], 0).mean()
    assert value.value_krw_per_mwh(frame, ESS) == pytest.approx(expected)
    # objective_krw 는 판매 + REC − 열화비 그대로 (평가액 제외): 창을 직접 풀어 대조
    plan = plan_window(frame, ESS, ESS.initial_energy_mwh,
                       terminal=fr.window_terminal(frame, 0, 3, ESS.initial_energy_mwh, ESS,
                                                   terminal_mode="horizon_aware",
                                                   terminal_value=value)[1])
    t = plan.totals(committed_only=True)
    assert t["objective_krw"] == pytest.approx(t["sales_revenue_krw"] + t["rec_revenue_krw"]
                                               - t["throughput_cost_krw"])
    assert plan.terminal_value_krw == pytest.approx(
        value.value_krw_per_mwh(frame, ESS) * plan.final_solved_energy_mwh)
    with pytest.raises(ValueError, match="fraction"):
        fr.TerminalValue(fraction=2.0)


def test_gap_restarts_segment_with_reason_and_config_soc():
    days = [date(2024, 6, 1), date(2024, 6, 2), date(2024, 6, 4)]
    windows = [make_window(d, gen_f=6.0, gen_a=3.0) for d in days]
    results = xr.run_execution_scenario(
        windows, ESS, gap_reasons={date(2024, 6, 3): "generation_actual_missing"},
    )
    assert [(r.segment_id, r.start_day, r.end_day) for r in results] == [
        (0, days[0], days[1]), (1, days[2], days[2]),
    ]
    assert results[0].segment_start_reason == "period_start"
    assert results[1].segment_start_reason == "restart_after_gap:generation_actual_missing"
    assert results[1].daily["exec_start_energy_mwh"].iloc[0] == pytest.approx(
        ESS.initial_energy_mwh)
    daily = xr.stack_exec_daily(results)
    assert daily["segment_start"].tolist() == [True, False, True]
    hourly = xr.stack_exec_hourly(results)
    gaps = pd.to_datetime(hourly["ts"]).diff().gt(pd.Timedelta(hours=1)).sum()
    assert gaps == 1  # 빠진 날을 이어 붙이지 않았다
    with pytest.raises(ValueError, match="연속"):
        chain(windows)


def test_day_without_actual_is_refused_not_filled():
    w = make_window(DAYS[0])
    plan_only = PlanWindow(w.planning_day, w.forecast, None, w.sources,
                           comparison_reason="generation_actual_missing")
    with pytest.raises(ValueError, match="generation_actual_missing"):
        chain([plan_only])


def test_future_actual_change_does_not_alter_earlier_days():
    windows = [make_window(d, gen_f=6.0, gen_a=4.0) for d in DAYS]
    base = chain(windows).exec_hourly
    third = windows[2]
    changed_actual = profile(DAYS[2], 1.0, 50_000.0)
    windows[2] = PlanWindow(third.planning_day, third.forecast, changed_actual, third.sources)
    changed = chain(windows).exec_hourly
    cols = ["charge_mwh", "discharge_mwh", "supply_mwh", "energy_end_mwh", "realized_net_krw"]
    pd.testing.assert_frame_equal(base.loc[:47, cols], changed.loc[:47, cols])
    assert not np.allclose(base.loc[48:, "supply_mwh"], changed.loc[48:, "supply_mwh"])


def test_no_ess_baseline_never_touches_battery_and_sells_actual_generation():
    windows = [make_window(d, gen_f=6.0, gen_a=4.0) for d in DAYS]
    res = xr.run_execution_scenario(windows, ESS, scenario="no_ess_executed")[0]
    h = res.exec_hourly
    assert h[["charge_mwh", "discharge_mwh"]].to_numpy().max() == 0
    assert np.allclose(h["energy_end_mwh"], ESS.initial_energy_mwh)
    assert np.allclose(h["supply_mwh"], h["actual_generation_mwh"])  # 양수 가격: 전량 공급
    assert np.allclose(h["committed_sell_mwh"], h["forecast_generation_mwh"])
    assert np.allclose(h["committed_sales_krw"] + h["deviation_krw"],
                       h["supply_mwh"] * h["settlement_price_krw_per_mwh"])
    totals = xr.execution_scenario_totals([res])
    assert totals["computable"] and totals["realized_net_krw"] == pytest.approx(
        h["realized_net_krw"].sum())
    assert totals["executed_end_shortfall_mwh"] == 0


def test_scenario_totals_report_end_soc_gap_and_segments():
    days = [DAYS[0], DAYS[1], DAYS[2] + timedelta(days=1)]
    windows = [make_window(d, gen_f=6.0, gen_a=2.0) for d in days]
    results = xr.run_execution_scenario(windows, ESS)
    totals = xr.execution_scenario_totals(results)
    assert totals["days"] == 3 and len(totals["segments"]) == 2
    assert totals["segments"][1]["start_reason"].startswith("restart_after_gap")
    assert totals["realized_net_krw"] == pytest.approx(
        totals["committed_sales_krw"] + totals["deviation_krw"] + totals["rec_krw"]
        - totals["wear_krw"])
    assert totals["executed_end_shortfall_mwh"] == pytest.approx(sum(
        s["executed_end_shortfall_mwh"] for s in totals["segments"]))
    assert totals["settings"]["plan_soc_source"] == "executed"
