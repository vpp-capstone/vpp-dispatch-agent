import numpy as np
import pandas as pd
import pytest

from vpp import distribution_backtest as db
from vpp import project_economics as pe
from vpp.optimization import EssParams

CFG = pe.EconomicsConfig.from_yaml("config/project_economics_v0.yaml")
BASE = EssParams.from_yaml("config/ess_v0_1.yaml")
DAYS = list(pd.date_range("2025-06-01", periods=3, freq="D"))


def _inputs(price_scale: float = 1.0) -> db.Inputs:
    ts = pd.date_range("2025-05-31", periods=24 * 6, freq="h")
    hour = ts.hour.to_numpy()
    solar = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 5.7  # 낮에 최대 5.7MW
    smp = np.where((hour >= 10) & (hour < 15), 40_000.0, 160_000.0) * price_scale
    hourly = pd.DataFrame(
        {"solar_mwh": solar, "smp_krw_per_mwh": smp, "rec_krw_per_mwh": 70_000.0}, index=ts
    )
    fc = {}
    for d in pd.date_range("2025-05-30", periods=6, freq="D"):
        idx = pd.date_range(d + pd.Timedelta(days=1), periods=48, freq="h")
        fc[d.strftime("%Y-%m-%d")] = hourly["smp_krw_per_mwh"].reindex(idx).ffill().to_numpy()
    return db.Inputs(hourly, fc, 5.7)


def test_line_cap_profiles():
    ts = pd.date_range("2025-06-01", periods=24, freq="h")
    fixed = db.line_cap_mw(db.LineCase("fixed", 1.7), ts, np.zeros(24), 5.7)
    assert fixed[12] == 1.7 and fixed[2] > 1e5 and fixed[18] > 1e5
    dyn = db.line_cap_mw(db.LineCase("dynamic", 1.5, 5.0), ts, np.array([5.7] + [0.0] * 23), 5.7)
    assert dyn[0] == pytest.approx(1.5)  # 해가 가장 강할 때 c
    assert dyn[1] == pytest.approx(6.5)  # 해가 없으면 c + 기존 태양광
    with pytest.raises(ValueError):
        db.LineCase("hourly", 1.0)


def test_ess_for_uses_battery_values():
    b = CFG.batteries["second_life_example"]
    ess = db.ess_for(BASE, CFG, b, db.LineCase("fixed", 1.7))
    assert ess.capacity_mwh == pytest.approx(16.0)
    assert ess.charge_efficiency == pytest.approx(0.93)
    assert ess.throughput_cost_krw_per_mwh == pytest.approx(
        b.throughput_cost_krw_per_mwh(20.0, CFG.fx)
    )
    small = db.ess_for(BASE, CFG, b, db.LineCase("fixed", 1.7, ess_power_mw=2, ess_energy_mwh=10))
    assert small.charge_power_mw == 2 and small.capacity_mwh == pytest.approx(8.0)


def test_run_case_respects_cap_and_settles_at_actual_price():
    inp = _inputs()
    case = db.LineCase("fixed", 1.5)
    h = db.run_case(inp, CFG, BASE, case, "new_2025_bnef", "forecast", DAYS)
    assert len(h) == 72
    day = (h["ts"].dt.hour >= 8) & (h["ts"].dt.hour < 18)
    assert (h.loc[day, "sell_mwh"] <= 1.5 + 1e-6).all()
    actual = inp.hourly["smp_krw_per_mwh"].reindex(h["ts"]).to_numpy()
    assert np.allclose(h["price_krw_per_mwh"], actual)
    assert h["charge_mwh"].sum() > 0  # 상한 때문에 남는 태양광을 담는다
    base = db.run_case(inp, CFG, BASE, case, db.BASELINE, "forecast", DAYS)
    assert (base["charge_mwh"] == 0).all()
    assert (base.loc[day, "sell_mwh"] <= 1.5 + 1e-9).all()


def test_operating_rows_feed_economics():
    inp = _inputs()
    jobs = db.build_jobs(
        ["new_2025_bnef"], fixed_caps=[1.5], dynamic_caps=[], existing_pv_mw=5.0,
        operations=["forecast", "perfect"],
    )  # fmt: skip
    assert {(c.name, b, o) for c, b, o in jobs} == {
        ("cap_1.50MW", "new_2025_bnef", "forecast"),
        ("cap_1.50MW", "none", "forecast"),
        ("cap_1.50MW", "new_2025_bnef", "perfect"),
        ("cap_1.50MW", "none", "perfect"),
    }
    op = db.run_all(inp, CFG, BASE, jobs, DAYS, workers=1)
    assert set(pe.OPERATING_COLUMNS) <= set(op.columns)
    assert (op["days"] == 3).all()
    ess = op[op.battery == "new_2025_bnef"].set_index("operation")
    base = op[op.battery == "none"].set_index("operation")
    assert (ess["market_revenue_krw"] > base["market_revenue_krw"]).all()
    summary, _ = pe.summarize(CFG, op)
    assert {"standard", "total", "incremental"} <= set(summary["basis"])


def test_size_cases_pair_with_one_baseline():
    jobs = db.build_jobs(
        ["new_2025_bnef"], fixed_caps=[1.5], dynamic_caps=[], existing_pv_mw=5.0,
        operations=["forecast"], sizes=[(2.0, 10.0), (4.0, 20.0)],
    )  # fmt: skip
    names = sorted(c.name for c, b, _ in jobs if b != "none")
    assert names == ["cap_1.50MW_ess2MW10MWh", "cap_1.50MW_ess4MW20MWh"]
    assert sum(b == "none" for _, b, _ in jobs) == 1
    op = db.run_all(_inputs(), CFG, BASE, jobs, DAYS[:1], workers=1)
    summary, _ = pe.summarize(CFG, op)
    inc = summary[(summary.basis == "incremental") & (summary.revenue_scenario == "base")]
    assert set(inc["ess_energy_mwh"]) == {10.0, 20.0}
    small = inc[inc.ess_energy_mwh == 10.0].iloc[0]
    assert small["capex_krw"] == pytest.approx(
        CFG.batteries["new_2025_bnef"].capex_krw(10.0, CFG.fx)
    )
