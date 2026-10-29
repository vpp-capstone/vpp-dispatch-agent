"""ESS 운영계획 v0: 작은 문제에서 기대하는 선택/수익을 직접 검증한다 (Codex 8개 이식 + 추가 6개)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vpp.optimization import EssParams, run_rolling, solve_window

ROOT = Path(__file__).resolve().parents[1]


def ideal(**overrides) -> EssParams:
    """손실 없는 2 MWh/1 MW 배터리, SOC 0~100%, 시작 0 (Codex 테스트 setUp과 동일)."""
    base = dict(
        capacity_mwh=2, charge_power_mw=1, discharge_power_mw=1,
        soc_min=0, soc_max=1, soc_initial=0,
        charge_efficiency=1, discharge_efficiency=1, throughput_cost_krw_per_mwh=0,
    )
    base.update(overrides)
    return EssParams(**base)


def solve(g, p, dt, ess, start=None, **kw):
    start = ess.initial_energy_mwh if start is None else start
    return solve_window(g, p, dt, ess, start, **kw)


def frame(generation, price, start="2025-04-01"):
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=len(generation), freq="h"),
        "generation_mwh": generation,
        "price_krw_per_mwh": price,
        "dt_hours": 1.0,
    })


# ---------- Codex 테스트 8개 이식 ----------

def test_store_then_sell():
    r = solve([1, 0], [10, 100], [1, 1], ideal())
    assert r.charge_mwh[0] == pytest.approx(1)
    assert r.discharge_mwh[1] == pytest.approx(1)
    assert r.objective_krw == pytest.approx(100)


def test_efficiency_losses():
    r = solve([1, 0], [10, 100], [1, 1], ideal(charge_efficiency=0.9, discharge_efficiency=0.9))
    assert r.discharge_mwh[1] == pytest.approx(0.81)
    assert r.objective_krw == pytest.approx(81)


def test_initial_energy_is_not_free_revenue():
    ess = ideal(soc_initial=0.5)
    start = ess.initial_energy_mwh
    r = solve([0, 0], [10, 100], [1, 1], ess,
              final_energy_min_mwh=start, final_energy_max_mwh=start)
    assert r.objective_krw == pytest.approx(0)
    assert r.energy_end_mwh[-1] == pytest.approx(1)


def test_negative_price_can_curtail():
    r = solve([1], [-10], [1], ideal(), final_energy_min_mwh=0, final_energy_max_mwh=0)
    assert r.curtailment_mwh[0] == pytest.approx(1)
    assert r.objective_krw == pytest.approx(0)


def test_nonhourly_power_limit():
    r = solve([1, 0], [10, 100], [0.25, 0.25], ideal())
    assert r.charge_mwh[0] == pytest.approx(0.25)
    assert r.objective_krw == pytest.approx(32.5)


def test_bad_input_rejected():
    with pytest.raises(ValueError):
        solve([float("nan")], [10], [1], ideal())
    with pytest.raises(ValueError):
        EssParams(**{**ideal().to_dict(), "soc_min": 0.5, "soc_max": 0.4})
    with pytest.raises(ValueError):
        EssParams(**{**ideal().to_dict(), "grid_charging_allowed": True})


def test_only_first_day_initializes_at_half():
    ess = ideal(soc_initial=0.5, soc_min=0.1)
    day1 = solve([0], [100], [1], ess)
    assert day1.initial_energy_mwh == pytest.approx(1)
    assert day1.energy_end_mwh[-1] == pytest.approx(0.2)
    day2 = solve([0], [100], [1], ess, day1.energy_end_mwh[-1])
    assert day2.initial_energy_mwh == pytest.approx(0.2)
    assert day2.objective_krw == pytest.approx(0)


def test_carryover_can_exceed_initial_half():
    ess = ideal(soc_initial=0.5, throughput_cost_krw_per_mwh=1)
    # 음수 가격에는 잔량을 유지하고 다음 날 양수 가격에 판매한다.
    day1 = solve([0], [-100], [1], ess, 1.5)
    assert day1.energy_end_mwh[-1] == pytest.approx(1.5)
    day2 = solve([0], [100], [1], ess, day1.energy_end_mwh[-1])
    assert day2.initial_energy_mwh == pytest.approx(1.5)
    assert day2.energy_end_mwh[-1] == pytest.approx(0.5)


# ---------- 추가 테스트 ----------

def test_efficiency_threshold_decides_charging():
    """왕복 효율 0.938² 과 열화비 2만원/MWh에서 가격 차가 손익분기를 넘어야만 충전한다."""
    ess = ideal(charge_efficiency=0.938, discharge_efficiency=0.938,
                throughput_cost_krw_per_mwh=20_000)
    below = solve([1, 0], [100_000, 155_000], [1, 1], ess)
    assert below.charge_mwh.sum() == pytest.approx(0, abs=1e-6)
    assert below.objective_krw == pytest.approx(100_000)
    above = solve([1, 0], [100_000, 158_000], [1, 1], ess)
    assert above.charge_mwh.sum() > 1e-3


def test_tiny_ess_matches_no_ess_baseline():
    ess = ideal(capacity_mwh=1e-6)
    g = [3.0, 2.0, 0.5, 4.0]
    p = [50_000, -1_000, 120_000, 90_000]
    r = solve(g, p, [1] * 4, ess)
    assert r.objective_krw == pytest.approx(r.no_ess_same_curtailment_policy_krw, rel=1e-6)
    expected = 3 * 50_000 + 0.5 * 120_000 + 4 * 90_000
    assert r.no_ess_same_curtailment_policy_krw == pytest.approx(expected)


def test_no_simultaneous_charge_discharge_under_negative_prices():
    ess = EssParams.from_yaml(ROOT / "config/ess_v0.yaml")
    g = [20.0, 25.0, 30.0, 10.0, 0.0, 0.0]
    p = [-5_000, -20_000, -1_000, 50_000, 150_000, 120_000]
    r = solve(g, p, [1] * 6, ess)
    assert np.minimum(r.charge_mwh, r.discharge_mwh).max() <= 1e-6
    assert (r.sell_mwh[:3] <= 1e-6).all()  # 음수 가격에는 판매하지 않음
    assert r.sell_mwh[4] > 0  # 저장한 전기를 고가에 판다


def test_rolling_energy_continuity_between_days():
    ess = EssParams.from_yaml(ROOT / "config/ess_v0.yaml")
    rng = np.random.default_rng(0)
    n = 24 * 4
    f = frame(rng.uniform(0, 12, n), rng.uniform(60_000, 160_000, n))
    res = run_rolling(f, ess, policy="rolling48")
    d = res.daily
    assert len(d) == 4
    np.testing.assert_allclose(d["initial_energy_mwh"].to_numpy()[1:],
                               d["final_energy_mwh"].to_numpy()[:-1], atol=1e-6)
    assert d["initial_energy_mwh"].iloc[0] == pytest.approx(ess.initial_energy_mwh)
    # 시간별 SOC 재계산도 창 경계를 넘어 이어져야 한다
    h = res.hourly
    prev = np.r_[ess.initial_energy_mwh, h["energy_end_mwh"].to_numpy()[:-1]]
    recomputed = (prev + ess.charge_efficiency * h["charge_mwh"]
                  - h["discharge_mwh"] / ess.discharge_efficiency)
    np.testing.assert_allclose(recomputed, h["energy_end_mwh"], atol=1e-6)


def test_rolling48_commits_only_first_24_hours():
    """48시간을 보되 앞 24시간만 확정: 뒤 24시간의 값은 결과에 들어가지 않는다."""
    ess = ideal(soc_initial=0.5)
    # 1일차: 발전 있고 가격 낮음. 2일차: 발전 없고 가격 아주 높음
    # → 1일차에 저장해 2일차에 팔고 싶어진다.
    g = [1.0] * 24 + [0.0] * 24
    p = [10.0] * 24 + [1_000.0] * 24
    res = run_rolling(frame(g, p), ess, horizon_hours=48, commit_hours=24, policy="rolling48")
    assert len(res.hourly) == 48 and res.hourly["ts"].is_unique
    d0 = res.daily.iloc[0]
    assert d0["solved_intervals"] == 48 and d0["committed_intervals"] == 24
    first = res.hourly.iloc[:24]
    # 1일차 수익은 확정 24시간의 판매만 반영 (2일차 고가 판매는 포함되지 않음)
    committed_sales = float(first["price_krw_per_mwh"] @ first["sell_mwh"])
    assert d0["sales_revenue_krw"] == pytest.approx(committed_sales)
    assert d0["sales_revenue_krw"] < 1_000 * 1
    # 1일차 끝에는 2일차를 위해 가득 채워 둔다 (free 일별 풀이라면 비웠을 것)
    assert d0["final_energy_mwh"] == pytest.approx(ess.energy_max_mwh)
    free = run_rolling(frame(g, p), ess, policy="free")
    assert free.daily.iloc[0]["final_energy_mwh"] < d0["final_energy_mwh"]
    # 짧은 꼬리: 36시간이면 2번째 창은 남은 12시간만 푼다
    tail = run_rolling(frame(g[:36], p[:36]), ess, policy="rolling48")
    assert tail.daily["solved_intervals"].tolist() == [36, 12]
    assert tail.daily["committed_intervals"].tolist() == [24, 12]
    assert len(tail.hourly) == 36


def test_period_final_energy_not_below_initial():
    ess = ideal(soc_initial=0.5)
    # 가격이 계속 높아 비우고 싶지만, 마지막 창에서 시작 에너지(1 MWh) 이상으로 되돌려야 한다.
    g = [0.5] * 48
    p = [100.0] * 48
    res = run_rolling(frame(g, p), ess, policy="rolling48")
    assert res.period_final_energy_mwh >= res.period_initial_energy_mwh - 1e-6
    assert res.daily.iloc[-1]["terminal_constraint"] == "period_end_ge_initial"
    # 같은 데이터를 free로 풀면 초기 저장분을 팔아 더 많이 번다
    free = run_rolling(frame(g, p), ess, policy="free")
    assert free.period_final_energy_mwh < res.period_final_energy_mwh
    assert free.objective_krw > res.objective_krw
    # equal_initial은 창마다 시작 에너지로 돌아온다
    eq = run_rolling(frame(g, p), ess, policy="equal_initial")
    np.testing.assert_allclose(eq.daily["final_energy_mwh"], eq.daily["initial_energy_mwh"],
                               atol=1e-6)


def test_yaml_params_are_marked_as_assumption():
    ess = EssParams.from_yaml(ROOT / "config/ess_v0.yaml")
    assert ess.status == "assumption_not_equipment_spec"
    assert (ess.capacity_mwh, ess.charge_power_mw, ess.soc_min, ess.soc_max) == (16, 4, 0.1, 0.9)
    assert ess.initial_energy_mwh == pytest.approx(8)
    assert ess.usable_capacity_mwh == pytest.approx(12.8)
