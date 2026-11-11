"""REC 수익 가정 v0: 설정 읽기, 단가 가산, REC 0 = 기존 결과, 손계산 예제, 기준 방식 일치."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_optimization import frame, ideal
from vpp.baselines import no_ess, plan_revenue_breakdown, plan_revenue_krw, rule_based
from vpp.evaluation import summarize_backtest
from vpp.optimization import EssParams, plan_window, run_rolling, solve_window
from vpp.rec import REC_COLUMN, RecParams, load_rec, rec_revenue_krw
from vpp.validation import check_plan, check_properties


def write_rec(tmp_path, **overrides):
    cfg = {"status": "experiment_assumption_not_contract", "eligible_basis": "sold_mwh_once",
           "weight": 1.0, "price_krw_per_rec": 70000}
    cfg.update(overrides)
    path = tmp_path / "rec.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# ---------- 설정 ----------

def test_yaml_flat_price_and_apply(tmp_path):
    rec = RecParams.from_yaml(write_rec(tmp_path, weight=1.2))
    out = rec.apply(frame([1, 1], [10, 20]))
    assert out[REC_COLUMN].tolist() == [84000.0, 84000.0]
    meta = rec.metadata()
    assert meta["enabled"] and meta["pricing"]["type"] == "flat_scenario"
    assert meta["status"] == "experiment_assumption_not_contract"
    assert load_rec(None) is None


def test_monthly_table_maps_by_month_and_rejects_missing_month(tmp_path):
    table = tmp_path / "rec.csv"
    pd.DataFrame({"month": ["2024-06", "2024-07"], "price_krw_per_rec": [60000, 80000],
                  "note": ["", ""]}).to_csv(table, index=False)
    cfg = write_rec(tmp_path, price_krw_per_rec=None, price_table=str(table))
    rec = RecParams.from_yaml(cfg)
    ts = pd.Series(pd.to_datetime(["2024-06-30 23:00", "2024-07-01 00:00"]))
    assert rec.price_for(ts).tolist() == [60000.0, 80000.0]
    with pytest.raises(ValueError, match="2024-08"):
        rec.price_for(pd.Series(pd.to_datetime(["2024-08-01"])))
    # 비어 있는 월은 0 으로 채우지 않고 읽을 때 거부
    pd.DataFrame({"month": ["2024-06"], "price_krw_per_rec": [None]}).to_csv(table, index=False)
    with pytest.raises(ValueError, match="비었거나"):
        RecParams.from_yaml(cfg)


def test_config_validation():
    with pytest.raises(ValueError, match="하나만"):
        RecParams(weight=1.0)
    with pytest.raises(ValueError, match="하나만"):
        RecParams(weight=1.0, price_krw_per_rec=1.0,
                  price_table=pd.DataFrame({"month": ["2024-06"], "price_krw_per_rec": [1.0]}))
    with pytest.raises(ValueError, match="가중치"):
        RecParams(weight=-1.0, price_krw_per_rec=1.0)
    with pytest.raises(ValueError, match="eligible_basis"):
        RecParams(weight=1.0, price_krw_per_rec=1.0, eligible_basis="generation")


# ---------- REC 0 = 기존 결과 ----------

def test_zero_rec_is_identical_to_no_rec():
    ess = ideal(charge_efficiency=0.9, discharge_efficiency=0.9)
    g, p, dt = [1, 0, 1, 0], [10, 100, -5, 50], [1, 1, 1, 1]
    base = solve_window(g, p, dt, ess, 0.0)
    zero = solve_window(g, p, dt, ess, 0.0, rec_krw_per_mwh=np.zeros(4))
    for col in ("sell_mwh", "charge_mwh", "discharge_mwh", "curtailment_mwh", "energy_end_mwh"):
        assert np.array_equal(getattr(base, col), getattr(zero, col))
    assert zero.objective_krw == base.objective_krw and zero.rec_revenue_krw == 0.0
    f = frame(g, p)
    with_col = plan_window(f.assign(**{REC_COLUMN: 0.0}), ess, 0.0, commit_hours=4)
    without = plan_window(f, ess, 0.0, commit_hours=4)
    pd.testing.assert_frame_equal(with_col.hourly.drop(columns=REC_COLUMN), without.hourly)
    assert with_col.totals() == without.totals()


# ---------- 손계산 예제 ----------

def test_negative_smp_sells_when_smp_plus_rec_is_positive():
    """SMP −5,000 원/MWh, REC 70,000: REC 없으면 감축, 있으면 판매. 수익 분리와 기준 방식 일치."""
    ess = ideal()  # 시작 0, 종료 0 으로 고정해 ESS 는 움직이지 않는다
    g, p = [1.0], [-5000.0]
    without = solve_window(g, p, [1], ess, 0.0, final_energy_min_mwh=0, final_energy_max_mwh=0)
    assert without.curtailment_mwh[0] == pytest.approx(1) and without.objective_krw == 0
    with_rec = solve_window(g, p, [1], ess, 0.0, final_energy_min_mwh=0, final_energy_max_mwh=0,
                            rec_krw_per_mwh=[70000.0])
    assert with_rec.sell_mwh[0] == pytest.approx(1)
    assert with_rec.sales_revenue_krw == pytest.approx(-5000)
    assert with_rec.rec_revenue_krw == pytest.approx(70000)
    assert with_rec.objective_krw == pytest.approx(65000)
    assert with_rec.no_ess_same_curtailment_policy_krw == pytest.approx(65000)

    f = frame(g, p).assign(**{REC_COLUMN: 70000.0})
    base = no_ess(f)
    assert base["sell_mwh"].iloc[0] == pytest.approx(1)  # 기준 방식도 같은 가산으로 판단
    assert plan_revenue_breakdown(base, ess) == pytest.approx(
        {"sales_revenue_krw": -5000, "rec_revenue_krw": 70000, "throughput_cost_krw": 0,
         "net_revenue_krw": 65000})
    assert no_ess(frame(g, p))["curtailment_mwh"].iloc[0] == pytest.approx(1)


def test_rec_is_counted_once_at_sale_and_shrinks_with_round_trip_loss():
    """충전 1 → 방전 0.81: REC 는 팔린 0.81 MWh 에만 붙는다 (충전분엔 안 붙음)."""
    ess = ideal(charge_efficiency=0.9, discharge_efficiency=0.9)
    r = solve_window([1, 0], [10, 100], [1, 1], ess, 0.0, rec_krw_per_mwh=[50.0, 50.0])
    assert r.charge_mwh[0] == pytest.approx(1) and r.discharge_mwh[1] == pytest.approx(0.81)
    assert r.sales_revenue_krw == pytest.approx(81)
    assert r.rec_revenue_krw == pytest.approx(50 * 0.81)
    assert r.objective_krw == pytest.approx(81 + 40.5)
    # 가산이 커서 저장 손실이 아까우면 바로 판다: 가치 (10+900) vs 0.81×(100+900)=810 → 즉시 판매
    direct = solve_window([1, 0], [10, 100], [1, 1], ess, 0.0, rec_krw_per_mwh=[900.0, 900.0])
    assert direct.sell_mwh[0] == pytest.approx(1) and direct.charge_mwh[0] == pytest.approx(0)


def test_check_plan_recomputes_objective_with_rec_column():
    ess = ideal()
    f = frame([1, 1], [-5000, 100]).assign(**{REC_COLUMN: 70000.0})
    plan = plan_window(f, ess, 0.0, commit_hours=2)
    obj = plan.totals()["objective_krw"]
    ok = {c["item"]: c["passed"] for c in check_plan(plan.hourly, ess, objective_krw=obj)}
    assert ok["objective_recomputed_abs_error_krw"]
    # REC 열을 떼고 같은 목적함수를 대면 재계산이 맞지 않아야 한다 (REC 가 목적함수에 들어간 증거)
    bad = check_plan(plan.hourly.drop(columns=REC_COLUMN), ess, objective_krw=obj)
    assert not {c["item"]: c["passed"] for c in bad}["objective_recomputed_abs_error_krw"]
    assert rec_revenue_krw(plan.hourly) == pytest.approx(70000 * plan.hourly["sell_mwh"].sum())


# ---------- 기준 방식·요약 ----------

def test_rolling_summary_splits_rec_and_baselines_use_same_adder():
    # 시작 0 MWh: 종료 제약(E ≥ 시작)이 비어 있어 규칙 계획과 같은 조건으로 비교된다
    ess = EssParams(capacity_mwh=2, charge_power_mw=1, discharge_power_mw=1, soc_min=0,
                    soc_max=1, soc_initial=0.0, charge_efficiency=0.95,
                    discharge_efficiency=0.95, throughput_cost_krw_per_mwh=1000)
    hours = 48
    price = np.tile([-2000.0] * 6 + [30000.0] * 6 + [90000.0] * 6 + [40000.0] * 6, 2)
    gen = np.tile([0.5] * 12 + [0.0] * 12, 2)
    f = frame(gen, price).assign(**{REC_COLUMN: 70000.0})
    plan = run_rolling(f, ess)
    assert plan.rec_revenue_krw == pytest.approx(70000 * plan.hourly["sell_mwh"].sum())
    assert plan.objective_krw == pytest.approx(
        plan.sales_revenue_krw + plan.rec_revenue_krw - plan.throughput_cost_krw)
    assert plan.terminal_constraint_cost_krw is not None and plan.terminal_constraint_cost_krw >= 0
    summary = summarize_backtest(plan, {"no_ess": no_ess(f), "rule_based": rule_based(f, ess)}, ess)
    assert summary["rec_revenue_krw"] == pytest.approx(plan.rec_revenue_krw)
    assert summary["baselines"]["no_ess"]["rec_revenue_krw"] == pytest.approx(70000 * gen.sum())
    assert summary["baselines"]["no_ess"]["net_revenue_krw"] == pytest.approx(
        plan_revenue_krw(no_ess(f), ess))
    assert all(c["passed"] for c in check_properties(summary)), check_properties(summary)
    assert len(plan.hourly) == hours


def test_initial_energy_rec_upper_bound_uses_actual_start_and_period_max_price(tmp_path):
    """상한 = 실제 시작 잔량(하한 초과분) × 방전 효율 × 기간 최고 가산. 첫 달 단가가 아니다."""
    table = tmp_path / "rec.csv"
    pd.DataFrame({"month": ["2024-06", "2024-07"], "price_krw_per_rec": [10000, 100000]}
                 ).to_csv(table, index=False)
    rec = RecParams.from_yaml(write_rec(tmp_path, price_krw_per_rec=None, price_table=str(table)))
    ess = ideal(soc_initial=0.5, discharge_efficiency=0.9)  # 설정 시작 1 MWh
    ts = pd.Series(pd.date_range("2024-06-30 00:00", periods=48, freq="h"))  # 6월→7월 걸침
    assert rec.initial_energy_rec_upper_bound_krw(ess, ts) == pytest.approx(1 * 0.9 * 100000)
    june_only = ts.iloc[:24]
    assert rec.initial_energy_rec_upper_bound_krw(ess, june_only) == pytest.approx(0.9 * 10000)
    # --start-soc / --from-plan 으로 정한 실제 시작 잔량을 반영한다 (설정값 1 MWh 가 아니라 2 MWh)
    assert rec.initial_energy_rec_upper_bound_krw(
        ess, ts, initial_energy_mwh=2.0) == pytest.approx(2 * 0.9 * 100000)
    assert rec.initial_energy_rec_upper_bound_krw(ess, ts, initial_energy_mwh=0.0) == 0.0
    # 첫 시각 하나만 줘도 된다 (예전 호출 방식)
    assert rec.initial_energy_rec_upper_bound_krw(ess, ts.iloc[0]) == pytest.approx(0.9 * 10000)


def test_dashboard_baseline_day_keeps_rec_column_for_fair_comparison():
    """대시보드 하루 비교: ESS 없음 기준 수익에도 REC 가 들어가야 MILP 와 공정하다.

    예제: SMP −5,000 + REC 70,000 = 65,000 (REC 열을 버리면 −5,000 으로 보인다)."""
    pytest.importorskip("streamlit")
    from dashboard.ui import baseline_day, net_revenue

    ess = ideal()
    base = frame([1.0], [-5000.0], start="2024-06-01").assign(**{REC_COLUMN: 70000.0})
    base["date"] = base["ts"].dt.date
    for col, val in (("sell_mwh", 1.0), ("charge_mwh", 0.0), ("discharge_mwh", 0.0),
                     ("curtailment_mwh", 0.0), ("energy_end_mwh", 0.0)):
        base[f"no_ess_{col}"] = val
    day = baseline_day({"baselines": base}, base["date"].iloc[0], "no_ess")
    assert REC_COLUMN in day.columns
    assert net_revenue(day, ess) == pytest.approx(65000)
    # REC 열이 없는 예전 결과 폴더는 그대로 읽힌다
    legacy = baseline_day({"baselines": base.drop(columns=REC_COLUMN)}, base["date"].iloc[0],
                          "no_ess")
    assert REC_COLUMN not in legacy.columns and net_revenue(legacy, ess) == pytest.approx(-5000)


def test_terminal_constraint_cost_is_the_forgone_sale_of_initial_energy():
    """발전 0, 가격 100 인 하루: 충·방전이 전혀 없어도 종료 제약이 초기 1 MWh 판매를 막아 100.

    즉 이 값은 '못 판 잔량의 기회비용'이지 충전 비용이 아니다."""
    ess = ideal(soc_initial=0.5)
    plan = run_rolling(frame([0] * 24, [100] * 24), ess)
    assert plan.objective_krw == pytest.approx(0)
    assert plan.terminal_constraint_cost_krw == pytest.approx(100)
    free = run_rolling(frame([0] * 24, [100] * 24), ess, policy="free")
    assert free.terminal_constraint_cost_krw is None and free.objective_krw == pytest.approx(100)
