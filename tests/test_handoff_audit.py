"""Exercise missing plants, time windows, unit mistakes and unsafe handoff inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpp.handoff_audit import align_forecasts, generation_windows, main, price_windows


@pytest.fixture
def inputs():
    plants = pd.DataFrame({
        "asset_id": ["solar_a", "solar_b", "wind_a"],
        "energy_type": ["solar", "solar", "wind"],
        "capacity_mw": [1.0, 2.0, 5.0],
    })
    targets = pd.date_range("2024-06-02", periods=48, freq="h", tz="Asia/Seoul")

    def generation(ids):
        rows = [
            {"asset_id": asset, "issue_time_utc": pd.Timestamp("2024-06-01T03:00+09:00"),
             "valid_time_utc": ts.tz_convert("UTC"), "lightgbm_generation_kwh": 100.0,
             "generation_kwh": 99999.0}
            for ts in targets for asset in ids
        ]
        return pd.DataFrame(rows)

    smp = pd.DataFrame({
        "issue_time": pd.Timestamp("2024-06-01T11:00+09:00"), "target_time": targets,
        "smp_hat": [-10.0, 0.0] + [100.0] * 46,
        "smp_hat_krw_per_mwh": [-10000.0, 0.0] + [100000.0] * 46,
        "smp_actual": 999999.0,
    })
    return generation(["solar_a", "solar_b"]), generation(["wind_a"]), smp, plants


def test_different_issue_times_join_by_planning_date_and_ignore_actuals(inputs):
    aligned, report = align_forecasts(*inputs)
    assert report["common_windows"] == 1
    assert len(aligned) == 48
    np.testing.assert_allclose(aligned["generation_mwh"], 0.3, atol=1e-12)
    assert aligned["price_krw_per_mwh"].iloc[:2].tolist() == [-10000.0, 0.0]
    assert "generation_kwh" not in aligned and "smp_actual" not in aligned


def test_missing_one_plant_at_one_hour_is_not_silently_summed(inputs):
    solar, _, _, plants = inputs
    with pytest.raises(ValueError, match="incomplete plant"):
        generation_windows(solar.iloc[1:], plants, "solar")


def test_duplicate_plant_is_rejected(inputs):
    solar, _, _, plants = inputs
    with pytest.raises(ValueError, match="duplicate plant"):
        generation_windows(pd.concat([solar, solar.iloc[:1]]), plants, "solar")


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0, 2001.0])
def test_invalid_generation_is_rejected(inputs, value):
    solar, _, _, plants = inputs
    solar.loc[1, "lightgbm_generation_kwh"] = value
    with pytest.raises(ValueError):
        generation_windows(solar, plants, "solar")


def test_unknown_plant_is_rejected(inputs):
    solar, _, _, plants = inputs
    solar.loc[0, "asset_id"] = "unknown"
    with pytest.raises(ValueError, match="plant set"):
        generation_windows(solar, plants, "solar")


@pytest.mark.parametrize("change", ["missing", "shift", "duplicate", "unit", "late", "naive"])
def test_invalid_price_windows(inputs, change):
    _, _, smp, _ = inputs
    if change == "missing":
        smp = smp.iloc[:-1]
    elif change == "shift":
        smp["target_time"] += pd.Timedelta(hours=1)
    elif change == "duplicate":
        smp = pd.concat([smp, smp.iloc[:1]])
    elif change == "unit":
        smp["smp_hat_krw_per_mwh"] /= 1000
    elif change == "late":
        smp["issue_time"] += pd.Timedelta(hours=1)
    else:
        smp["issue_time"] = smp["issue_time"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        price_windows(smp)


def test_bad_lead_is_rejected(inputs):
    _, _, smp, _ = inputs
    smp["lead_hour"] = 1
    with pytest.raises(ValueError, match="lead_hour"):
        price_windows(smp)


def test_unshared_dates_are_reported(inputs):
    solar, wind, smp, plants = inputs
    extra = smp.copy()
    extra["issue_time"] += pd.Timedelta(days=1)
    extra["target_time"] += pd.Timedelta(days=1)
    aligned, report = align_forecasts(solar, wind, pd.concat([smp, extra]), plants)
    assert len(aligned) == 48
    assert report["excluded_planning_dates"]["smp"] == ["2024-06-02"]


def test_no_common_dates_is_an_error(inputs):
    solar, wind, smp, plants = inputs
    smp["issue_time"] += pd.Timedelta(days=1)
    smp["target_time"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="no common"):
        align_forecasts(solar, wind, smp, plants)


def test_exclusion_is_explicit_and_auditable(inputs):
    solar, wind, smp, plants = inputs
    partial = solar.iloc[:20].copy()
    for column in ("issue_time_utc", "valid_time_utc"):
        partial[column] += pd.Timedelta(days=1)
    received = pd.concat([solar, partial], ignore_index=True)
    with pytest.raises(ValueError, match="48-hour"):
        align_forecasts(received, wind, smp, plants)
    result, report = align_forecasts(received, wind, smp, plants, incomplete_windows="exclude")
    assert len(result) == 48
    assert report["source_inventory"]["solar"]["excluded_windows"] == 1
    assert report["incomplete_forecast_windows"] == [{
        "source": "solar", "planning_date": "2024-06-02", "rows": 20,
        "reasons": "missing_target_hours",
    }]


@pytest.mark.parametrize("corruption", ["duplicate", "negative", "unit", "late", "unknown"])
def test_exclude_mode_does_not_hide_bad_rows(inputs, corruption):
    solar, wind, smp, plants = inputs
    if corruption == "duplicate":
        solar = pd.concat([solar, solar.iloc[:1]], ignore_index=True)
    elif corruption == "negative":
        solar.loc[0, "lightgbm_generation_kwh"] = -1.0
    elif corruption == "unit":
        smp["smp_hat_krw_per_mwh"] /= 1000
    elif corruption == "late":
        smp["issue_time"] += pd.Timedelta(hours=1)
    else:
        solar.loc[0, "asset_id"] = "unknown"
    with pytest.raises(ValueError):
        align_forecasts(solar, wind, smp, plants, incomplete_windows="exclude")


def test_exclude_mode_rejects_when_nothing_complete_remains(inputs):
    solar, wind, smp, plants = inputs
    with pytest.raises(ValueError, match="no complete"):
        align_forecasts(solar.iloc[1:], wind, smp, plants, incomplete_windows="exclude")


def test_cli_exports_aligned_input_and_clears_stale_input_on_failure(inputs, tmp_path, monkeypatch):
    import vpp.handoff_audit as audit
    from vpp.portfolio_data import SOURCES

    solar, wind, smp, plants = inputs
    for source in SOURCES:
        (tmp_path / source.file).write_text("fixture", encoding="utf-8")
    plant_path = tmp_path / "plants.csv"
    plants.to_csv(plant_path, index=False)
    paths = [tmp_path / name for name in ("solar.parquet", "wind.parquet", "smp.csv")]
    partial = solar.iloc[:20].copy()
    for column in ("issue_time_utc", "valid_time_utc"):
        partial[column] += pd.Timedelta(days=1)
    pd.concat([solar, partial], ignore_index=True).to_parquet(paths[0], index=False)
    wind.to_parquet(paths[1], index=False)
    smp.to_csv(paths[2], index=False)
    actual = pd.DataFrame({
        "ts": smp["target_time"], "generation_mwh": 0.3,
    })
    monkeypatch.setattr(audit, "build_portfolio_hourly", lambda *a, **kw: (
        actual.copy(), {"excluded_days": []}
    ))
    monkeypatch.setattr(audit, "compare_generation_loaders", lambda *a: {"status": "PASS"})
    out = tmp_path / "audit"
    arguments = ["--raw-dir", str(tmp_path), "--plants", str(plant_path),
                 "--start", "2024-06-02", "--end", "2024-06-03", "--out", str(out),
                 "--solar", str(paths[0]), "--wind", str(paths[1]), "--smp", str(paths[2])]
    assert main([*arguments, "--incomplete-windows", "exclude"]) == 0
    result = pd.read_csv(out / "forecast_windows.csv")
    assert len(result) == 48
    assert "smp_actual" not in result
    assert len(pd.read_csv(out / "incomplete_forecast_windows.csv")) == 1
    # Reject mode fails on the same partial source and must remove the former output.
    assert main(arguments) == 1
    assert not (out / "forecast_windows.csv").exists()
    assert not (out / "incomplete_forecast_windows.csv").exists()
