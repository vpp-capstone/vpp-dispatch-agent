"""Hourly replay must retain the existing execution/settlement accounting and SOC."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dashboard import distribution_replay as replay
from dashboard.distribution_display import load_display

from vpp import distribution_backtest as db
from vpp import project_economics as pe
from vpp.optimization import EssParams

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(
    scope="module", params=[("fixed", "new_2025_bnef"), ("dynamic", "second_life_example")]
)
def result(request):
    # Small synthetic test fixture, never exported or used by the dashboard.
    ts = pd.date_range("2025-06-01", periods=96, freq="h")
    hour = ts.hour.to_numpy()
    generation = np.maximum(0, np.sin((hour - 6) / 12 * np.pi)) * 5.7
    prices = np.where((hour >= 10) & (hour < 15), 40000.0, 160000.0)
    h = pd.DataFrame(
        {
            "solar_mwh": generation,
            "smp_krw_per_mwh": prices,
            "smp_rt_krw_per_mwh": prices * 1.1,
            "rec_krw_per_mwh": 70000.0,
            "fcr_price_krw_per_kwh": 12.0,
        },
        index=ts,
    )
    days = list(pd.date_range("2025-06-01", periods=3))
    gen = {
        str(d.date()): h.solar_mwh.loc[d : d + pd.Timedelta(hours=47)].to_numpy() * 1.2
        for d in days
    }
    smp = {
        str((d - pd.Timedelta(days=1)).date()): h.smp_krw_per_mwh.loc[
            d : d + pd.Timedelta(hours=47)
        ].to_numpy()
        for d in days
    }
    # An unforecast daytime outage exercises shortage penalties as well as energy revenue.
    h.loc["2025-06-02 08:00":"2025-06-02 16:00", "solar_mwh"] = 0
    inputs = db.Inputs(h, smp, 5.7, gen)
    cfg = pe.EconomicsConfig.from_yaml(ROOT / "config/project_economics_v0.yaml")
    base = EssParams.from_yaml(ROOT / "config/ess_v0_1.yaml")
    kind, battery = request.param
    case = db.LineCase(kind, 1.5, ess_power_mw=2, ess_energy_mwh=8)
    hourly, annual, capacity = replay.execute(inputs, cfg, base, case, battery, days)
    return hourly, annual, capacity, cfg


def test_hourly_income_sums_to_existing_standard_allocation(result):
    h, a, capacity, cfg = result
    expected = pe._standard_view(cfg, pd.Series(a), a["ess_charge_mwh"] + a["ess_discharge_mwh"])
    assert h.vpp_income_krw.sum() == pytest.approx(expected.op.market_revenue_krw, abs=1e-6)
    assert h.charge_mwh.sum() == pytest.approx(a["ess_charge_mwh"])
    assert h.discharge_mwh.sum() == pytest.approx(a["ess_discharge_mwh"])
    assert np.allclose(h.vpp_income_krw, h.vpp_ess_krw + h.vpp_solar_cp_krw + h.vpp_penalty_krw)
    assert h.vpp_penalty_krw.sum() < 0
    assert np.allclose(h.soc_end, h.energy_end_mwh / capacity)
    assert np.allclose(h.energy_end_mwh.iloc[:-1], h.energy_start_mwh.iloc[1:])
    assert h.energy_start_mwh.iloc[24] == pytest.approx(h.energy_end_mwh.iloc[23])
    assert h.charge_mwh.sum() > 0 and h.discharge_mwh.sum() > 0


def test_day_selection_keeps_soc_but_resets_daily_income(result):
    h, _, capacity, _ = result
    before = h.copy(deep=True)
    p = replay.day_payload(h, {"candidate": "selected", "capacity_mwh": capacity}, "2025-06-02")
    assert len(p["hours"]) == 24
    assert p["operation"] == "forecast_exec" and p["basis"] == "standard"
    assert p["initial_energy"] == pytest.approx(h.energy_end_mwh.iloc[23])
    assert p["hours"][0]["cumulative_vpp_income_krw"] == p["hours"][0]["vpp_income_krw"]
    assert p["hours"][-1]["cumulative_vpp_income_krw"] == pytest.approx(
        h.vpp_income_krw.iloc[24:48].sum()
    )
    assert p["hours"][-1]["cumulative_charge_mwh"] == pytest.approx(h.charge_mwh.iloc[24:48].sum())
    json.dumps(p, allow_nan=False)
    pd.testing.assert_frame_equal(h, before)
    with pytest.raises(ValueError, match="24시간"):
        replay.day_payload(h, {}, "2025-06-10")


def test_missing_settlement_hours_and_broken_soc_are_rejected(result):
    h, _, capacity, _ = result
    expected = pd.date_range("2025-06-01", periods=72, freq="h")
    with pytest.raises(ValueError, match="누락"):
        replay.validate_hours(h.iloc[1:], capacity, expected)
    broken = h.copy()
    broken.loc[24, "energy_start_mwh"] += 0.5
    with pytest.raises(ValueError, match="SOC"):
        replay.validate_hours(broken, capacity, expected)
    broken = h.copy()
    broken.loc[5, "vpp_income_krw"] = np.nan
    with pytest.raises(ValueError, match="결측"):
        replay.validate_hours(broken, capacity, expected)


def test_reconciliation_accepts_rounding_but_rejects_other_inputs(result):
    _, annual, _, _ = result
    reported = pd.Series({k: round(annual[k], 2) for k in replay.TOTALS})
    replay.reconcile(annual, reported)
    reported["ess_charge_mwh"] += 0.1
    with pytest.raises(ValueError, match="연간 원본 결과와 일치하지"):
        replay.reconcile(annual, reported)


def test_cache_isolated_by_candidate_source_and_annual_result(tmp_path, monkeypatch):
    source = tmp_path / "forecast.csv"
    source.write_text("first")
    monkeypatch.setattr(replay, "sources", lambda root: {"dependencies": [source]})
    rows = load_display(ROOT)[0]
    row = rows.iloc[0].copy()
    first = replay.cache_path(tmp_path, replay.context(tmp_path, row))
    source.write_text("updated")
    second = replay.cache_path(tmp_path, replay.context(tmp_path, row))
    assert first != second
    row["id"] = "different battery and size"
    third = replay.cache_path(tmp_path, replay.context(tmp_path, row))
    assert third != second
    row["imbalance_penalty_krw"] += 10
    assert replay.cache_path(tmp_path, replay.context(tmp_path, row)) != third


def test_modified_cached_file_is_not_displayed(tmp_path, monkeypatch):
    row = load_display(ROOT)[0].iloc[0]
    stamp = {"candidate": str(row.id), "basis": "standard"}
    monkeypatch.setattr(replay, "context", lambda root, selected: stamp)
    folder = replay.cache_path(tmp_path, stamp)
    folder.mkdir(parents=True)
    (folder / "hourly.parquet").write_bytes(b"tampered")
    (folder / "manifest.json").write_text(
        json.dumps({**stamp, "file": "hourly.parquet", "sha256": "original"})
    )
    with pytest.raises(ValueError, match="변경"):
        replay.load(tmp_path, row)
