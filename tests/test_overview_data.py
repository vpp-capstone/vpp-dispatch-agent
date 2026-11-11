"""Overview must never relabel hypothetical profit as realized profit."""

import json

import pandas as pd
import pytest
from dashboard.overview_data import load_experiment

from tests.test_optimization import ideal
from vpp.baselines import no_ess
from vpp.rec import REC_COLUMN


def saved_experiment(tmp_path, *, rec=False):
    summary = {
        "mode": "forecast_vs_perfect_foresight_plan",
        "ess": ideal().to_dict(),
        "rec": {"enabled": rec},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    for scenario, price in (("forecast", 100), ("actual", 10)):
        frame = pd.DataFrame(
            {
                "ts": pd.date_range("2024-06-01", periods=24, freq="h", tz="Asia/Seoul"),
                "generation_mwh": 1.0,
                "price_krw_per_mwh": price,
                "dt_hours": 1.0,
            }
        )
        if rec:
            frame[REC_COLUMN] = 70000.0
        plan = no_ess(frame)
        plan["energy_end_mwh"] = 0.0
        plan["soc_end"] = 0.0
        plan.to_csv(tmp_path / f"hourly_{scenario}.csv", index=False)
    return tmp_path


def test_two_hypothetical_revenues_are_not_realized_profit(tmp_path):
    report = load_experiment(saved_experiment(tmp_path))
    assert report["metrics"]["smp_mae_krw_per_kwh"] == pytest.approx(0.09)
    assert report["revenues"]["forecast"]["net_revenue_krw"] == 2400
    assert report["revenues"]["actual"]["net_revenue_krw"] == 240
    assert report["realized_profit_krw"] is None
    assert report["revenues"]["actual"]["ess_gain_krw"] == 0


def test_rec_is_applied_to_no_ess_too(tmp_path):
    report = load_experiment(saved_experiment(tmp_path, rec=True))
    assert report["revenues"]["actual"]["rec_revenue_krw"] == 24 * 70000
    assert report["revenues"]["actual"]["ess_gain_krw"] == 0


@pytest.mark.parametrize("problem", ["missing_hour", "duplicate", "missing_rec", "wrong_summary"])
def test_invalid_comparisons_are_blocked(tmp_path, problem):
    saved_experiment(tmp_path, rec=True)
    path = tmp_path / "hourly_actual.csv"
    frame = pd.read_csv(path)
    if problem == "missing_hour":
        frame = frame.iloc[1:]
    elif problem == "duplicate":
        frame.loc[1, "ts"] = frame.loc[0, "ts"]
    elif problem == "missing_rec":
        frame = frame.drop(columns=REC_COLUMN)
    else:
        summary_path = tmp_path / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["scenarios"] = {"actual": {"planned_objective_krw": 999}}
        summary_path.write_text(json.dumps(summary))
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_experiment(tmp_path)


def test_gap_is_visible_not_presented_as_continuous(tmp_path):
    saved_experiment(tmp_path)
    for scenario in ("forecast", "actual"):
        path = tmp_path / f"hourly_{scenario}.csv"
        frame = pd.read_csv(path)
        frame["ts"] = pd.to_datetime(frame.ts)
        frame.loc[12:, "ts"] += pd.Timedelta(days=1)
        frame.to_csv(path, index=False)
    assert load_experiment(tmp_path)["metrics"]["segments"] == 2
