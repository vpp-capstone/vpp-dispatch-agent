import dataclasses

import pandas as pd
import pytest

from vpp import project_economics as pe

CFG = pe.EconomicsConfig.from_yaml("config/distribution_ess_v0.yaml")
OP = pe.Operating(market_revenue_krw=15e8, solar_to_vpp_mwh=8000.0, ess_throughput_mwh=6000.0)


def test_shipped_config_matches_degradation_stage1_cost():
    """2021 PNNL 열화비 = degradation_stage1 의 64,096원/MWh (16MWh 기준)."""
    b = CFG.batteries["new_2021_pnnl"]
    assert b.throughput_cost_krw_per_mwh(16.0, CFG.fx) == pytest.approx(64096.38, rel=1e-4)
    assert set(CFG.batteries) >= {"new_2021_pnnl", "new_2025_bnef", "second_life_example"}


def test_battery_life_is_shorter_of_throughput_and_calendar():
    b = CFG.batteries["second_life_example"]
    lifetime = b.lifetime_throughput_mwh(20.0)  # 2 x 3000 x 16 x 0.8
    assert lifetime == pytest.approx(76800.0)
    assert b.life_years(20.0, 7680.0) == pytest.approx(10.0)  # 처리량 기준 10년 = 달력 10년
    assert b.life_years(20.0, 15360.0) == pytest.approx(5.0)
    assert b.life_years(20.0, 0.0) == b.calendar_life_years


@pytest.mark.parametrize(
    "life, years",
    [(13.7, [14]), (7.0, [7, 14]), (20.0, []), (25.0, []), (6.5, [7, 13])],
)
def test_replacement_years(life, years):
    assert pe.replacement_years(life, 20) == years


def test_cashflow_year0_and_subsidy_cap():
    b = CFG.batteries["new_2025_bnef"]
    cf = pe.cashflow(CFG, b, OP, ppa_krw_per_kwh=150)
    capex = b.capex_krw(20.0, CFG.fx)
    assert cf.loc[0, "net"] == pytest.approx(-capex + 0.5 * capex)
    assert cf.loc[1, "ppa_payment"] == pytest.approx(-150 * 1000 * 8000)
    assert len(cf) == 21
    capped = dataclasses.replace(CFG, subsidy_cap_krw_per_mwh=1e7)  # 20MWh x 1천만 = 2억 상한
    assert pe.cashflow(capped, b, OP).attrs["subsidy"] == pytest.approx(2e8)


def test_npv_irr_payback_simple_case():
    net = pd.Series([-100.0, 60.0, 60.0], index=[0, 1, 2])
    assert pe.npv(net, 0.0) == pytest.approx(20.0)
    assert pe.npv(net, pe.irr(net)) == pytest.approx(0.0, abs=1e-6)
    assert pe.payback_year(net.cumsum()) == pytest.approx(1 + 40 / 60)
    assert pd.isna(pe.payback_year(pd.Series([-100.0, 10.0]).cumsum()))


def test_breakevens_put_npv_at_zero():
    b = CFG.batteries["new_2021_pnnl"]
    op = pe.Operating(market_revenue_krw=8e8, solar_to_vpp_mwh=5000.0, ess_throughput_mwh=5000.0)
    m = pe.evaluate(CFG, b, op, ppa_krw_per_kwh=130)
    r = CFG.discount_rate
    if not pd.isna(m["breakeven_subsidy_rate"]) and 0 < m["breakeven_subsidy_rate"] < 1:
        cf = pe.cashflow(CFG, b, op, ppa_krw_per_kwh=130, subsidy_rate=m["breakeven_subsidy_rate"])
        assert pe.npv(cf["net"], r) == pytest.approx(0.0, abs=1e3)
    cf = pe.cashflow(CFG, b, op, ppa_krw_per_kwh=130, capex_krw=m["breakeven_capex_krw"])
    assert pe.npv(cf["net"], r) == pytest.approx(0.0, abs=1e3)
    scale = m["min_annual_market_revenue_krw"] / op.market_revenue_krw
    cf = pe.cashflow(CFG, b, op, ppa_krw_per_kwh=130, revenue_scale=scale)
    assert pe.npv(cf["net"], r) == pytest.approx(0.0, abs=1e3)


def test_load_operating_annualizes_partial_year(tmp_path):
    p = tmp_path / "op.csv"
    pd.DataFrame(
        [
            {
                "case": "c",
                "battery": "new_2025_bnef",
                "operation": "forecast",
                "market_revenue_krw": 1e8,
                "solar_to_vpp_mwh": 100.0,
                "ess_charge_mwh": 50.0,
                "ess_discharge_mwh": 45.0,
                "days": 73,
            }
        ]
    ).to_csv(p, index=False)
    df = pe.load_operating(p)
    assert df.loc[0, "market_revenue_krw"] == pytest.approx(5e8)
    assert df.loc[0, "ess_charge_mwh"] == pytest.approx(250.0)
    with pytest.raises(ValueError):
        pd.DataFrame([{"case": "c"}]).to_csv(p, index=False)
        pe.load_operating(p)


def test_summarize_decision_follows_npv_sign():
    op = pd.DataFrame(
        [
            {
                "case": "c",
                "battery": b,
                "operation": "forecast",
                "market_revenue_krw": rev,
                "solar_to_vpp_mwh": 8000.0,
                "ess_charge_mwh": 3000.0,
                "ess_discharge_mwh": 2700.0,
            }
            for b in ("new_2025_bnef", "second_life_example")
            for rev in (5e8, 30e8)
        ]
    )
    summary, flows = pe.summarize(CFG, op)
    assert len(summary) == len(op) * len(CFG.revenue_scenarios) * len(CFG.ppa_prices)
    assert ((summary["npv_krw"] > 0) == (summary["decision"] == "참여 검토")).all()
    assert set(flows["year"]) == set(range(21))
    with pytest.raises(ValueError):
        pe.summarize(CFG, op.assign(battery="unknown"))


def test_integrated_mode_requires_solar_capex():
    with pytest.raises(ValueError):
        dataclasses.replace(CFG, split_mode="integrated", solar_capex_krw=None).validate()
    cfg = dataclasses.replace(CFG, split_mode="integrated", solar_capex_krw=6e9)
    cf = pe.cashflow(cfg, CFG.batteries["new_2025_bnef"], OP)
    assert (cf["ppa_payment"] == 0).all()
    assert cf.loc[0, "capex"] == pytest.approx(
        -CFG.batteries["new_2025_bnef"].capex_krw(20.0, CFG.fx) - 6e9
    )


def _with_baseline():
    rows = [
        {
            "case": "c",
            "battery": "new_2025_bnef",
            "operation": "forecast",
            "market_revenue_krw": 15e8,
            "solar_to_vpp_mwh": 8000.0,
            "ess_charge_mwh": 3000.0,
            "ess_discharge_mwh": 2700.0,
        },
        {
            "case": "c",
            "battery": "none",
            "operation": "forecast",
            "market_revenue_krw": 10e8,
            "solar_to_vpp_mwh": 5000.0,
            "ess_charge_mwh": 0.0,
            "ess_discharge_mwh": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def test_incremental_basis_uses_baseline_difference():
    summary, flows = pe.summarize(CFG, _with_baseline())
    inc = summary[(summary["basis"] == "incremental") & (summary["revenue_scenario"] == "base")]
    assert len(inc) == len(CFG.ppa_prices)
    assert (inc["annual_market_revenue_krw"] == 5e8).all()
    assert (inc["annual_solar_to_vpp_mwh"] == 3000.0).all()
    row = inc[inc["ppa_krw_per_kwh"] == 150].iloc[0]
    assert row["solar_owner_gain_krw_per_year"] == pytest.approx(150 * 1000 * 3000)
    # 기준선 행 자체는 결과에 나오지 않는다
    assert "none" not in set(summary["battery"])
    assert set(flows["basis"]) == {"total", "incremental"}


def test_breakeven_ppa_puts_npv_at_zero():
    b = CFG.batteries["new_2025_bnef"]
    op = pe.Operating(market_revenue_krw=5e8, solar_to_vpp_mwh=3000.0, ess_throughput_mwh=5700.0)
    m = pe.evaluate(CFG, b, op, ppa_krw_per_kwh=0)
    assert 0 < m["breakeven_ppa_krw_per_kwh"] < 500
    cf = pe.cashflow(CFG, b, op, ppa_krw_per_kwh=m["breakeven_ppa_krw_per_kwh"])
    assert pe.npv(cf["net"], CFG.discount_rate) == pytest.approx(0.0, abs=1e3)
    poor = pe.Operating(market_revenue_krw=1e7, solar_to_vpp_mwh=3000.0, ess_throughput_mwh=5700.0)
    assert pd.isna(pe.evaluate(CFG, b, poor, ppa_krw_per_kwh=0)["breakeven_ppa_krw_per_kwh"])
