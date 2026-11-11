"""ESS 중앙계약시장식 계약가격 정산 계산 테스트."""

import pandas as pd
import pytest

from vpp.ess_contract_settlement import (
    EssCentralContractConfig,
    annualized_ess_contract_revenue,
    calculate_ess_contract_performance,
    calculate_ess_contract_revenue,
    summarize_ess_contract_revenue,
    supply_capacity_mw_from_storage,
)


def config(**overrides) -> EssCentralContractConfig:
    values = {"bid_price_krw_per_kw_h": 10.0}
    values.update(overrides)
    return EssCentralContractConfig(**values)


def test_supply_capacity_uses_six_hour_rule_by_default():
    assert supply_capacity_mw_from_storage(16.0) == pytest.approx(16 / 6)
    assert config().effective_contracted_power_mw == pytest.approx(16 / 6)


def test_explicit_contract_power_overrides_storage_conversion():
    assert config(contracted_power_mw=4.0).effective_contracted_power_mw == 4.0


def test_contract_price_uses_bid_price_ratio():
    assert config().contract_price_krw_per_kw_h == pytest.approx(6.0)


def test_hourly_revenue_uses_power_time_and_performance():
    frame = pd.DataFrame({"dt_hours": [1.0, 2.0]})
    result = calculate_ess_contract_revenue(
        frame,
        config(
            contracted_power_mw=4.0,
            availability_rate=0.9,
            performance_rate=0.8,
        ),
    )
    # 6 원/kW·h × 4,000 kW × 3 h × 0.9 × 0.8
    assert result["ess_contract_revenue_krw"].sum() == pytest.approx(51_840.0)
    assert summarize_ess_contract_revenue(result)["contract_hours"] == pytest.approx(3.0)


def test_hourly_rates_override_config_defaults():
    frame = pd.DataFrame(
        {
            "dt_hours": [1.0],
            "availability_rate": [0.5],
            "performance_rate": [0.8],
        }
    )
    result = calculate_ess_contract_revenue(frame, config(contracted_power_mw=4.0))
    assert result["ess_contract_revenue_krw"].iloc[0] == pytest.approx(9_600.0)


def test_official_performance_formula_uses_dispatch_and_metered_energy():
    performance = calculate_ess_contract_performance(
        pd.Series([10.0]),
        pd.Series([10.0]),
        pd.Series([9.0]),
        pd.Series([8.0]),
    )
    # alpha=0.5, non-performance=(0.1*0.5)+(0.2*0.5), performance=0.85.
    assert performance.iloc[0] == pytest.approx(0.85)


def test_available_capacity_column_overrides_constant_availability():
    frame = pd.DataFrame({"available_capacity_mw": [2.0]})
    result = calculate_ess_contract_revenue(
        frame,
        config(contracted_power_mw=4.0, availability_rate=1.0),
    )
    # 6 원/kW·h × 2,000 kW × 1 h
    assert result["ess_contract_revenue_krw"].iloc[0] == pytest.approx(12_000.0)


def test_dispatch_columns_take_priority_over_fallback_performance_rate():
    frame = pd.DataFrame(
        {
            "charge_dispatch_mwh": [10.0],
            "discharge_dispatch_mwh": [10.0],
            "charge_metered_mwh": [9.0],
            "discharge_metered_mwh": [8.0],
            "performance_rate": [0.1],
        }
    )
    result = calculate_ess_contract_revenue(
        frame,
        config(contracted_power_mw=4.0),
    )
    assert result["ess_contract_performance_rate"].iloc[0] == pytest.approx(0.85)
    assert result["ess_contract_revenue_krw"].iloc[0] == pytest.approx(20_400.0)


def test_annualized_example_uses_six_hour_supply_capacity():
    # 5 원/kW·h × (16/6) MW × 8,760 h = 116,800,000원
    assert annualized_ess_contract_revenue(
        config(bid_price_krw_per_kw_h=5 / 0.6)
    ) == pytest.approx(116_800_000.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("bid_price_krw_per_kw_h", -1),
        ("storage_energy_mwh", 0),
        ("supply_duration_hours", 0),
        ("contracted_power_mw", 0),
        ("contract_price_ratio", 1.1),
        ("availability_rate", -0.1),
        ("performance_rate", 1.1),
        ("bid_price_krw_per_kw_h", float("inf")),
    ],
)
def test_invalid_contract_inputs_rejected(field, value):
    with pytest.raises(ValueError):
        config(**{field: value})


def test_invalid_dt_rejected():
    with pytest.raises(ValueError):
        calculate_ess_contract_revenue(
            pd.DataFrame({"dt_hours": [0.0]}), config()
        )


def test_nonfinite_dt_rejected():
    with pytest.raises(ValueError):
        calculate_ess_contract_revenue(
            pd.DataFrame({"dt_hours": [float("inf")]}), config()
        )


def test_invalid_hourly_rate_rejected():
    with pytest.raises(ValueError):
        calculate_ess_contract_revenue(
            pd.DataFrame({"availability_rate": [1.1]}), config()
        )


def test_partial_dispatch_columns_rejected():
    with pytest.raises(ValueError, match="공식 이행률"):
        calculate_ess_contract_revenue(
            pd.DataFrame({"charge_dispatch_mwh": [1.0]}), config()
        )
