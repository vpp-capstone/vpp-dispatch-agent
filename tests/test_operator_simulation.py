"""Replay must display saved plans without inventing execution or resetting SOC."""

from datetime import date

import pandas as pd
import pytest
from dashboard.operator_simulation import replay_payload, simulator_html
from dashboard.overview_data import load_experiment

from tests.test_optimization import ideal
from tests.test_overview_data import saved_experiment

DAY = date(2024, 6, 1)


def test_replay_preserves_forecast_actual_and_revenue(tmp_path):
    data = load_experiment(saved_experiment(tmp_path, rec=True))
    payload = replay_payload(data, DAY)
    for mode in ("forecast", "actual"):
        rows = payload["series"][mode]
        assert len(rows) == 24
        assert rows[-1]["cumulative"] == pytest.approx(data["revenues"][mode]["net_revenue_krw"])
    first = payload["series"]["forecast"][0]
    assert first["p"] == 0.1
    assert first["actualP"] == 0.01
    assert first["rec"] == 70
    assert data["realized_profit_krw"] is None
    html = simulator_html(data, DAY)
    assert "__REPLAY_DATA__" not in html
    assert "실시간 운전 아님" in html


def test_soc_uses_saved_energy_and_losses_not_initial_setting(tmp_path):
    data = load_experiment(saved_experiment(tmp_path, rec=True))
    data["ess"] = ideal(
        charge_efficiency=0.9, discharge_efficiency=0.9, throughput_cost_krw_per_mwh=100
    )
    for mode in ("forecast", "actual"):
        frame = data[mode]
        frame.loc[0, ["charge_mwh", "discharge_mwh", "energy_end_mwh", "soc_end"]] = [
            1,
            0,
            1.5,
            0.75,
        ]
        frame.loc[1, ["charge_mwh", "discharge_mwh", "energy_end_mwh", "soc_end"]] = [
            0,
            0.9,
            0.5,
            0.25,
        ]
    rows = replay_payload(data, DAY)["series"]["forecast"]
    assert rows[0]["startSoc"] == pytest.approx(30)
    assert rows[0]["soc"] == 75
    assert rows[1]["startSoc"] == 75
    assert rows[1]["soc"] == 25
    assert rows[0]["wear"] == data["ess"].degradation_cost_krw([1], [0])


@pytest.mark.parametrize("problem", ["missing_hour", "wrong_time", "absent_day"])
def test_incomplete_replay_is_blocked(tmp_path, problem):
    data = load_experiment(saved_experiment(tmp_path))
    if problem == "missing_hour":
        for mode in ("forecast", "actual"):
            data[mode] = data[mode].iloc[1:]
    elif problem == "wrong_time":
        data["forecast"].loc[0, "ts"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        replay_payload(data, date(2024, 6, 2) if problem == "absent_day" else DAY)


@pytest.mark.parametrize("case", ["continuous", "missing", "reset", "mismatch"])
def test_daily_soc_checks_adjacent_boundaries(tmp_path, case):
    data = load_experiment(saved_experiment(tmp_path))
    capacity = data["ess"].capacity_mwh
    for mode in ("forecast", "actual"):
        day = data[mode].copy()
        day["segment_id"] = 0
        day["energy_end_mwh"] = capacity * 0.1
        day["soc_end"] = 0.1
        # First hour starts at 50%, then discharges to 10%.
        day.loc[0, "discharge_mwh"] = capacity * 0.4 * data["ess"].discharge_efficiency
        prev = day.iloc[[-1]].copy()
        prev["ts"] = day.iloc[0].ts - pd.Timedelta(hours=1)
        prev["energy_end_mwh"] = capacity * 0.5
        prev["soc_end"] = 0.5
        nxt = day.iloc[[-1]].copy()
        nxt["ts"] = day.iloc[-1].ts + pd.Timedelta(hours=1)
        if case == "missing":
            prev["ts"] -= pd.Timedelta(days=1)
            nxt["ts"] += pd.Timedelta(days=1)
        elif case == "reset":
            prev["segment_id"] = -1
            nxt["segment_id"] = 1
        elif case == "mismatch":
            prev["energy_end_mwh"] = capacity * 0.8
            nxt["energy_end_mwh"] = capacity * 0.8
        data[mode] = pd.concat([prev, day, nxt], ignore_index=True)
    payload = replay_payload(data, DAY)
    for mode in ("forecast", "actual"):
        daily = payload["dailySoc"][mode]
        assert daily["start"] == pytest.approx(50)
        assert daily["end"] == pytest.approx(10)
        expected = "matched" if case == "continuous" else case
        assert daily["previous"]["status"] == expected
        assert daily["next"]["status"] == expected
        if case == "missing":
            assert daily["previous"]["soc"] is None
            assert daily["next"]["soc"] is None
