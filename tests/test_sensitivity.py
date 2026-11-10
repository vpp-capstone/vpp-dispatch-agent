from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpp.optimization import EssParams
from vpp.sensitivity import config_from_base, run_sensitivity


def ess() -> EssParams:
    return EssParams(
        capacity_mwh=4.0,
        charge_power_mw=2.0,
        discharge_power_mw=2.0,
        soc_min=0.1,
        soc_max=0.9,
        soc_initial=0.5,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        throughput_cost_krw_per_mwh=0.0,
        degradation_model="fixed_throughput",
    )


def frame() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=48, freq="h")
    hour = ts.hour.to_numpy()
    generation = np.where((hour >= 8) & (hour <= 16), 5.0, 0.0)
    price = np.where((hour >= 18) & (hour <= 21), 200_000.0, 50_000.0)
    return pd.DataFrame({
        "ts": ts,
        "generation_mwh": generation,
        "price_krw_per_mwh": price,
        "dt_hours": 1.0,
    })


def test_sensitivity_runs_cartesian_product_and_returns_metrics():
    base = ess()
    config = config_from_base(
        base,
        capacities_mwh=[4.0, 8.0],
        charge_powers_mw=[2.0],
        discharge_powers_mw=[2.0],
        throughput_costs_krw_per_mwh=[0.0, 100.0],
    )
    results = run_sensitivity(frame(), base, config, policy="rolling48")

    assert len(results) == 4
    assert set(results["capacity_mwh"]) == {4.0, 8.0}
    assert set(results["throughput_cost_krw_per_mwh"]) == {0.0, 100.0}
    required = {
        "net_revenue_krw", "throughput_cost_krw", "gain_vs_no_ess_krw",
        "charge_mwh", "discharge_mwh", "curtailment_mwh", "cycles_per_day",
        "solver_checks_passed",
    }
    assert required <= set(results.columns)
    assert results["solver_checks_passed"].all()


def test_sensitivity_rejects_too_many_scenarios():
    base = ess()
    config = config_from_base(
        base,
        capacities_mwh=range(1, 22),
        charge_powers_mw=range(1, 6),
        discharge_powers_mw=range(1, 6),
        throughput_costs_krw_per_mwh=[0.0],
    )
    with pytest.raises(ValueError, match="500"):
        run_sensitivity(frame(), base, config)
