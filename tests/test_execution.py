"""execution: 판매계획 고정 ESS 실행 운전의 물리·약속·인과 검사."""

import numpy as np
import pandas as pd
import pytest
import yaml

from vpp import execution as ex
from vpp.optimization import EssParams
from vpp.rec import REC_COLUMN

ESS = EssParams(
    capacity_mwh=16.0, charge_power_mw=4.0, discharge_power_mw=4.0,
    soc_min=0.1, soc_max=0.9, soc_initial=0.5,
    charge_efficiency=0.95, discharge_efficiency=0.95, throughput_cost_krw_per_mwh=1000.0,
)
E0 = ESS.initial_energy_mwh  # 8.0


def plan(sell, charge=None, discharge=None, price=100_000.0, rec=None, gen=None):
    n = len(sell)
    ts = pd.date_range("2024-06-01", periods=n, freq="h", tz="Asia/Seoul")
    out = pd.DataFrame({
        "ts": ts, "sell_mwh": np.asarray(sell, dtype=float),
        "charge_mwh": np.zeros(n) if charge is None else np.asarray(charge, dtype=float),
        "discharge_mwh": np.zeros(n) if discharge is None else np.asarray(discharge, dtype=float),
        "price_krw_per_mwh": np.full(n, price, dtype=float), "dt_hours": 1.0,
    })
    if gen is not None:
        out["generation_mwh"] = np.asarray(gen, dtype=float)
    if rec is not None:
        out[REC_COLUMN] = rec
    # 계획상 잔량: 계획 충방전을 그대로 적용했을 때
    e = E0
    ends = []
    for c, d in zip(out["charge_mwh"], out["discharge_mwh"], strict=True):
        e = e + ESS.charge_efficiency * c - d / ESS.discharge_efficiency
        ends.append(e)
    out["energy_end_mwh"] = ends
    return out


def run(p, actual, policy=None, start=E0):
    return ex.execute_plan(p, actual, start, ESS, policy)


def test_identical_forecast_and_actual_reproduces_plan_exactly():
    gen = [10, 10, 2, 2]
    p = plan(sell=[7, 10, 5, 2], charge=[3, 0, 0, 0], discharge=[0, 0, 3, 0], gen=gen)
    res = run(p, gen)
    h = res.hourly
    assert np.allclose(h["charge_mwh"], p["charge_mwh"])
    assert np.allclose(h["discharge_mwh"], p["discharge_mwh"])
    assert np.allclose(h["supply_mwh"], p["sell_mwh"])
    assert h[["shortfall_mwh", "surplus_mwh", "curtailment_mwh"]].abs().to_numpy().max() == 0
    assert np.allclose(h["energy_end_mwh"], p["energy_end_mwh"])
    assert res.energy_end_mwh == pytest.approx(p["energy_end_mwh"].iloc[-1])
    assert res.all_passed and (h["actions"] == "").all()


def test_shortage_cuts_charge_then_discharges_then_records_shortfall():
    # 계획: 발전 10 중 3 충전, 7 판매. 실측이 줄어들수록 규칙이 순서대로 작동한다.
    p = plan(sell=[7], charge=[3], gen=[10])
    light = run(p, [8]).hourly.iloc[0]  # 부족 2 → 충전 3 → 1
    assert light["charge_mwh"] == pytest.approx(1) and light["discharge_mwh"] == 0
    assert light["supply_mwh"] == 7 and light["shortfall_mwh"] == 0
    assert light["actions"] == "charge_limited|charge_cut" or light["actions"] == "charge_cut"
    deep = run(p, [4]).hourly.iloc[0]  # 부족 6 → 충전 0, 추가 방전 3
    assert deep["charge_mwh"] == 0 and deep["discharge_mwh"] == pytest.approx(3)
    assert deep["supply_mwh"] == pytest.approx(7) and deep["shortfall_mwh"] == 0
    assert "extra_discharge" in deep["actions"]
    none = run(p, [0]).hourly.iloc[0]  # 발전 0: 방전 상한 4 → 부족 3 기록
    assert none["discharge_mwh"] == pytest.approx(4)
    assert none["supply_mwh"] == pytest.approx(4) and none["shortfall_mwh"] == pytest.approx(3)
    assert none["committed_sell_mwh"] == 7  # 판매계획은 바뀌지 않는다
    assert "shortfall" in none["actions"]


def test_extra_discharge_can_be_disabled_by_policy():
    p = plan(sell=[7], gen=[7])
    res = run(p, [2], ex.RuleBasedPolicy(extra_discharge=False)).hourly.iloc[0]
    assert res["discharge_mwh"] == 0 and res["shortfall_mwh"] == pytest.approx(5)


def test_surplus_default_reduces_discharge_then_sells_and_charges_only_unsellable():
    # 계획: 발전 2 + 방전 3 = 판매 5. 실측 6 → 잉여 4: 방전 3 줄이고 1 은 초과 판매 (충전 아님).
    p = plan(sell=[5], discharge=[3], gen=[2])
    row = run(p, [6]).hourly.iloc[0]
    assert row["discharge_mwh"] == 0 and row["charge_mwh"] == 0
    assert row["supply_mwh"] == pytest.approx(6) and row["surplus_mwh"] == pytest.approx(1)
    # 판매 가치 ≥ 0: 잉여 전부 초과 판매, 계획에 없던 충전 없음
    big = run(plan(sell=[5], gen=[5]), [12]).hourly.iloc[0]
    assert big["charge_mwh"] == 0 and big["surplus_mwh"] == pytest.approx(7)
    assert big["supply_mwh"] == pytest.approx(12) and big["curtailment_mwh"] == 0
    # 판매 가치가 음수(예측 SMP −10원/kWh, REC 없음)면 팔지 않고 충전 한도(4)까지 충전, 나머지 감축
    neg = run(plan(sell=[5], gen=[5], price=-10_000.0), [12]).hourly.iloc[0]
    assert neg["surplus_mwh"] == 0 and neg["charge_mwh"] == pytest.approx(4)
    assert neg["curtailment_mwh"] == pytest.approx(3)
    assert "extra_charge" in neg["actions"] and "curtail" in neg["actions"]


def test_surplus_order_charge_first_is_an_explicit_policy_choice():
    charge_first = ex.RuleBasedPolicy(surplus_order=("reduce_discharge", "charge", "sell",
                                                     "curtail"))
    big = run(plan(sell=[5], gen=[5]), [12], charge_first).hourly.iloc[0]
    assert big["charge_mwh"] == pytest.approx(4) and big["surplus_mwh"] == pytest.approx(3)
    assert big["supply_mwh"] == pytest.approx(8)


def test_soc_and_power_limits_hold_when_plan_is_infeasible_from_actual_state():
    # 잔량이 하한 근처면 계획 방전을 못 지키고 부족분이 기록된다
    low = ESS.energy_min_mwh + 0.95  # 방전 가능 0.95 × 0.95 = 0.9025 MWh
    p = plan(sell=[4], discharge=[4], gen=[0])
    row = run(p, [0], start=low).hourly.iloc[0]
    assert row["discharge_mwh"] == pytest.approx(0.95 * 0.95)
    assert row["energy_end_mwh"] == pytest.approx(ESS.energy_min_mwh)
    assert row["shortfall_mwh"] == pytest.approx(4 - 0.95 * 0.95)
    # 상한 근처면 (팔 수 없는 전력의) 추가 충전이 남은 용량까지만
    high = ESS.energy_max_mwh - 0.95  # 충전 가능 0.95 / 0.95 = 1.0 MWh
    row = run(plan(sell=[1], gen=[1], price=-1.0), [6], start=high).hourly.iloc[0]
    assert row["charge_mwh"] == pytest.approx(1.0)
    assert row["energy_end_mwh"] == pytest.approx(ESS.energy_max_mwh)
    assert row["curtailment_mwh"] == pytest.approx(4.0) and row["surplus_mwh"] == 0


def test_energy_balance_includes_efficiencies_every_hour():
    gen = [10, 3, 0, 8]
    p = plan(sell=[6, 6, 2, 6], charge=[4, 0, 0, 2], discharge=[0, 3, 2, 0], gen=gen)
    h = run(p, [9, 1, 0, 12]).hourly
    recomputed = h["energy_start_mwh"] + 0.95 * h["charge_mwh"] - h["discharge_mwh"] / 0.95
    assert np.allclose(h["energy_end_mwh"], recomputed)
    assert np.allclose(h["actual_generation_mwh"] + h["discharge_mwh"],
                       h["supply_mwh"] + h["charge_mwh"] + h["curtailment_mwh"])
    assert np.allclose(h["energy_start_mwh"].iloc[1:], h["energy_end_mwh"].iloc[:-1])
    names = {c["item"] for c in ex.check_execution(h, ESS)}
    assert {"power_balance_max_abs_error_mwh", "simultaneous_charge_discharge_max_mwh",
            "supply_equals_committed_minus_shortfall_plus_surplus_mwh"} <= names


def test_future_actual_changes_do_not_alter_earlier_hours():
    gen = [10, 10, 10, 10]
    p = plan(sell=[8, 8, 8, 8], charge=[2, 2, 0, 0], discharge=[0, 0, 2, 2], gen=gen)
    base = run(p, [9, 7, 5, 10]).hourly
    changed = run(p, [9, 7, 5, 0]).hourly
    cols = list(ex.EXECUTION_COLUMNS)
    pd.testing.assert_frame_equal(base.loc[:2, cols], changed.loc[:2, cols])
    assert base.loc[3, "shortfall_mwh"] != changed.loc[3, "shortfall_mwh"]


def test_missing_actual_is_not_interpolated_and_sales_plan_is_never_rewritten():
    p = plan(sell=[5, 5], gen=[5, 5])
    with pytest.raises(ValueError, match="결측"):
        run(p, [5, float("nan")])
    with pytest.raises(ValueError, match="길이"):
        run(p, [5])
    h = run(p, [1, 9]).hourly
    assert h["committed_sell_mwh"].tolist() == [5, 5]


def test_policy_yaml_round_trip(tmp_path):
    cfg = tmp_path / "execution.yaml"
    cfg.write_text(yaml.safe_dump({"policy": {
        "extra_discharge": False, "surplus_order": ["charge", "curtail"],
    }}), encoding="utf-8")
    policy = ex.load_policy(cfg)
    assert policy.extra_discharge is False and policy.surplus_order == ("charge", "curtail")
    assert policy.metadata()["status"] == "rule_based_not_profit_optimal"
    assert ex.load_policy(None) == ex.RuleBasedPolicy()
    with pytest.raises(ValueError, match="surplus_order"):
        ex.RuleBasedPolicy(surplus_order=("charge", "charge"))
    # 감축을 빼놓아도 남는 전력은 버린다 (물리 균형)
    row = run(plan(sell=[1], gen=[1], price=-1.0), [10],
              ex.RuleBasedPolicy(surplus_order=("sell",))).hourly.iloc[0]
    assert row["curtailment_mwh"] == pytest.approx(9) and row["surplus_mwh"] == 0
