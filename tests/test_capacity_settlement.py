from pathlib import Path

import pandas as pd
import pytest

from vpp.capacity_settlement import (
    RenewableCapacityConfig,
    calculate_renewable_capacity_settlement,
    kpx_capacity_quantity_mwh,
    summarize_capacity_settlement,
)


def _reference_tables(tmp_path: Path) -> tuple[Path, Path]:
    rcp_path = tmp_path / "rcp.csv"
    pd.DataFrame(
        [
            {
                "period_start": "2024-07-01",
                "period_end": "2025-06-30",
                "month": 7,
                "application_label": "2024/2025",
                "rcp_krw_per_kwh": 22.05,
            }
        ]
    ).to_csv(rcp_path, index=False)
    ratio_path = tmp_path / "ratio.csv"
    pd.DataFrame(
        [
            {
                "period_start": "2024-07-01",
                "period_end": "2025-06-30",
                "month": 7,
                "application_label": "2024/2025",
                "solar_pct": 5.0,
                "wind_pct": 10.0,
            }
        ]
    ).to_csv(ratio_path, index=False)
    return rcp_path, ratio_path


def test_uses_min_generation_bid_and_effective_capacity(tmp_path: Path):
    rcp_path, ratio_path = _reference_tables(tmp_path)
    hourly = pd.DataFrame(
        {
            "ts": ["2024-07-01 00:00:00+09:00"],
            "dt_hours": [1.0],
            "pv_kwh": [100.0],  # 0.1 MWh; effective capacity is 0.05 MW
            "wind_kwh": [0.0],
        }
    )
    result = calculate_renewable_capacity_settlement(
        hourly,
        config=RenewableCapacityConfig(pv_capacity_mw=1.0, wind_capacity_mw=1.0),
        rcp_path=rcp_path,
        ratio_path=ratio_path,
    )

    # PV: min(0.1 MWh, 1 MWh bid, 0.05 MWh effective) × 1000 × 22.05.
    assert result.loc[0, "pv_capacity_settlement_quantity_mwh"] == pytest.approx(0.05)
    assert result.loc[0, "pv_capacity_settlement_krw"] == pytest.approx(1102.5)
    assert result.loc[0, "capacity_settlement_krw"] == pytest.approx(1102.5)


def test_acceptance_rate_limits_bid_quantity(tmp_path: Path):
    rcp_path, ratio_path = _reference_tables(tmp_path)
    hourly = pd.DataFrame(
        {
            "ts": ["2024-07-01 00:00:00+09:00"],
            "dt_hours": [1.0],
            "pv_kwh": [900.0],
            "wind_kwh": [0.0],
        }
    )
    result = calculate_renewable_capacity_settlement(
        hourly,
        config=RenewableCapacityConfig(
            pv_capacity_mw=1.0,
            wind_capacity_mw=1.0,
            acceptance_rate=0.8,
            bid_basis="generation",
        ),
        rcp_path=rcp_path,
        ratio_path=ratio_path,
    )

    # The project scenario treats 80% as 80% of the actual sales plan (0.9 MWh).
    assert result.loc[0, "pv_capacity_bid_mw"] == pytest.approx(0.72)
    assert result.loc[0, "pv_capacity_settlement_quantity_mwh"] == pytest.approx(0.05)


def test_missing_period_is_not_silently_zero(tmp_path: Path):
    rcp_path, ratio_path = _reference_tables(tmp_path)
    hourly = pd.DataFrame(
        {
            "ts": ["2024-06-30 00:00:00+09:00"],
            "pv_kwh": [100.0],
            "wind_kwh": [100.0],
        }
    )
    with pytest.raises(ValueError, match="RCP 또는 실효용량비율"):
        calculate_renewable_capacity_settlement(
            hourly, rcp_path=rcp_path, ratio_path=ratio_path
        )

    result = calculate_renewable_capacity_settlement(
        hourly, rcp_path=rcp_path, ratio_path=ratio_path, strict=False
    )
    assert bool(result.loc[0, "capacity_missing_reference"])
    assert pd.isna(result.loc[0, "capacity_settlement_krw"])
    with pytest.raises(ValueError, match="누락된 기간"):
        summarize_capacity_settlement(result)


def test_kpx_set_point_branch_excludes_metered_generation_when_curtailed():
    values = dict(
        available_mwh=pd.Series([10.0]),
        bid_mwh=pd.Series([8.0]),
        metered_mwh=pd.Series([3.0]),
        effective_mwh=pd.Series([6.0]),
    )
    assert kpx_capacity_quantity_mwh(
        **values, set_point_mwh=pd.Series([8.0])
    ).iloc[0] == pytest.approx(3.0)
    assert kpx_capacity_quantity_mwh(
        **values, set_point_mwh=pd.Series([5.0])
    ).iloc[0] == pytest.approx(6.0)


def test_kpx_capacity_quantity_rejects_missing_or_negative_inputs():
    with pytest.raises(ValueError, match="음수·결측"):
        kpx_capacity_quantity_mwh(
            pd.Series([1.0]), pd.Series([1.0]), pd.Series([float("nan")]), pd.Series([1.0])
        )
