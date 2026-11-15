import pandas as pd
import pytest

from vpp.distribution_scenario import (
    DistributionESSConfig,
    attach_line_export_cap,
    prepare_distribution_v0_frame,
    run_distribution_v0,
    scale_solar_to_capacity,
    summarize_distribution_outputs,
)
from vpp.optimization import EssParams


def _ess() -> EssParams:
    return EssParams(
        capacity_mwh=20,
        charge_power_mw=4,
        discharge_power_mw=4,
        soc_min=0.1,
        soc_max=0.9,
        soc_initial=0.5,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        degradation_model="none",
        throughput_cost_krw_per_mwh=0,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=4, freq="h"),
            "dt_hours": 1.0,
            "pv_kwh": [1000.0, 2000.0, 3000.0, 4000.0],
            "wind_kwh": [1000.0, 1000.0, 1000.0, 1000.0],
            "generation_mwh": [2.0, 3.0, 4.0, 5.0],
            "price_krw_per_mwh": [100.0, 300.0, 100.0, 300.0],
        }
    )


def test_scale_solar_preserves_wind_and_targets_capacity():
    out = scale_solar_to_capacity(
        _frame(), source_solar_capacity_mw=10.0, target_solar_capacity_mw=5.0
    )
    assert out["pv_kwh"].tolist() == [500.0, 1000.0, 1500.0, 2000.0]
    assert out["wind_kwh"].tolist() == [1000.0] * 4
    assert out["generation_mwh"].tolist() == [1.5, 2.0, 2.5, 3.0]
    assert out["solar_target_capacity_mw"].iloc[0] == 5.0


def test_line_cap_converts_mw_to_mwh():
    out = attach_line_export_cap(_frame(), cap_mw=2.5)
    assert out["line_export_cap_mw"].tolist() == [2.5] * 4
    assert out["export_cap_mwh"].tolist() == [2.5] * 4


def test_distribution_plan_respects_line_cap_and_summary_has_handoff_values():
    config = DistributionESSConfig(
        source_solar_capacity_mw=10.0,
        target_solar_capacity_mw=5.0,
        line_export_cap_mw=2.5,
        ess=_ess(),
    )
    prepared = prepare_distribution_v0_frame(_frame(), config)
    plan = run_distribution_v0(prepared, config, horizon_hours=4, commit_hours=4)
    assert (plan.hourly["sell_mwh"] <= plan.hourly["export_cap_mwh"] + 1e-6).all()
    summary = summarize_distribution_outputs(plan, config.ess)
    assert summary["period"]["discharge_mwh"] >= 0
    assert summary["period"]["throughput_cost_krw"] == pytest.approx(0)
    assert "annualized_net_revenue_krw" in summary["annualized"]
    assert summary["line_export_cap_violation_mwh"] == pytest.approx(0)
