"""forecast_rolling: 계획일별 예측 창 롤링(24시간 확정·SOC 이월·구간 분리·종료 제약) 검사."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from vpp import forecast_rolling as fr
from vpp.forecast_inputs import PlanWindow
from vpp.optimization import EssParams
from vpp.validation import check_plan

KST = "Asia/Seoul"
ESS = EssParams(
    capacity_mwh=16.0, charge_power_mw=4.0, discharge_power_mw=4.0,
    soc_min=0.1, soc_max=0.9, soc_initial=0.5,
    charge_efficiency=0.95, discharge_efficiency=0.95,
)


def profile(day: date, gen_peak: float, price_peak: float) -> pd.DataFrame:
    """낮에 발전, 저녁에 비싼 가격인 48시간 (D, D+1 같은 모양)."""
    ts = pd.date_range(pd.Timestamp(day, tz=KST), periods=48, freq="h")
    hour = np.array(ts.hour)
    gen = np.where((hour >= 8) & (hour <= 16), gen_peak, 0.0)
    price = np.where((hour >= 18) & (hour <= 21), price_peak, 60_000.0)
    return pd.DataFrame(
        {"ts": ts, "generation_mwh": gen, "price_krw_per_mwh": price, "dt_hours": 1.0}
    )


def make_window(
    day: date, *, gen_f=6.0, price_f=150_000.0, gen_a=5.0, price_a=120_000.0
) -> PlanWindow:
    forecast = profile(day, gen_f, price_f)
    forecast["forecast_day_offset"] = [0] * 24 + [1] * 24
    forecast["horizon"] = [1] * 24 + [2] * 24
    return PlanWindow(day, forecast, profile(day, gen_a, price_a), {"smp_issue_date": "x"})


DAYS = [date(2024, 6, 5) + timedelta(days=i) for i in range(3)]


def test_segment_commits_24h_and_carries_planned_soc():
    seg = fr.run_segment([make_window(d) for d in DAYS], ESS, "forecast")
    hourly, daily = seg.result.hourly, seg.result.daily
    assert len(hourly) == 72 and list(daily["window_id"]) == [0, 1, 2]
    assert list(daily["solved_intervals"]) == [48, 48, 24]
    assert list(daily["committed_intervals"]) == [24, 24, 24]
    assert (hourly["ts"].diff().dropna() == pd.Timedelta(hours=1)).all()
    assert hourly.groupby("planning_day").size().tolist() == [24, 24, 24]
    assert daily["initial_energy_mwh"].iloc[0] == pytest.approx(ESS.initial_energy_mwh)
    assert np.allclose(daily["initial_energy_mwh"].iloc[1:], daily["final_energy_mwh"].iloc[:-1])
    assert daily["final_energy_mwh"].iloc[-1] >= ESS.initial_energy_mwh - 1e-6
    assert list(daily["terminal_constraint"]) == ["none", "none", "period_end_ge_initial"]
    assert seg.start_day == DAYS[0] and seg.end_day == DAYS[-1]


def test_scenario_uses_only_its_own_inputs():
    windows = [make_window(d) for d in DAYS]
    f = fr.run_segment(windows, ESS, "forecast").result
    a = fr.run_segment(windows, ESS, "actual").result
    assert (f.hourly["scenario"] == "forecast").all() and (a.hourly["scenario"] == "actual").all()
    f_price = np.concatenate([w.forecast["price_krw_per_mwh"][:24] for w in windows])
    a_gen = np.concatenate([w.actual["generation_mwh"][:24] for w in windows])
    assert np.allclose(f.hourly["price_krw_per_mwh"], f_price)
    assert np.allclose(a.hourly["generation_mwh"], a_gen)
    assert f.objective_krw != pytest.approx(a.objective_krw)
    with pytest.raises(ValueError, match="scenario"):
        fr.run_segment(windows, ESS, "mixed")


def test_committed_totals_only_count_first_24h():
    windows = [make_window(d) for d in DAYS]
    daily = fr.run_segment(windows, ESS, "forecast").result.daily
    assert daily["committed_hours"].sum() == 72
    expected_gen = sum(w.forecast["generation_mwh"][:24].sum() for w in windows)
    assert daily["generation_mwh"].sum() == pytest.approx(expected_gen)


def test_gap_splits_segments_and_resets_initial_soc():
    days = [date(2024, 6, 1), date(2024, 6, 2), date(2024, 6, 4)]
    results = fr.run_scenario([make_window(d) for d in days], ESS, "forecast")
    assert [(r.segment_id, r.start_day, r.end_day) for r in results] == [
        (0, days[0], days[1]), (1, days[2], days[2]),
    ]
    for r in results:
        assert r.result.daily["initial_energy_mwh"].iloc[0] == pytest.approx(ESS.initial_energy_mwh)
    with pytest.raises(ValueError, match="연속"):
        fr.run_segment([make_window(d) for d in days], ESS, "forecast")


def test_committed_plan_passes_constraint_checks():
    seg = fr.run_segment([make_window(d) for d in DAYS], ESS, "forecast")
    checks = check_plan(
        seg.result.hourly, ESS, initial_energy_mwh=seg.result.period_initial_energy_mwh
    )
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
    assert all(c["passed"] for c in seg.result.checks)


def test_soc_trace_puts_start_and_end_energy_at_true_times():
    seg = fr.run_segment([make_window(d) for d in DAYS[:2]], ESS, "forecast")
    hourly = seg.result.hourly.assign(segment_id=0)
    trace = fr.soc_trace(hourly, {0: seg.result.period_initial_energy_mwh})
    start = trace[trace["point"] == "start"].iloc[0]
    assert start["ts"] == hourly["ts"].iloc[0]
    assert start["energy_mwh"] == pytest.approx(ESS.initial_energy_mwh)
    ends = trace[trace["point"] == "end"]
    assert len(ends) == 48
    assert ends["ts"].iloc[0] == hourly["ts"].iloc[0] + pd.Timedelta(hours=1)
    assert np.allclose(ends["energy_mwh"], hourly["energy_end_mwh"])
    assert trace["point"].iloc[-1] == "gap" and np.isnan(trace["energy_mwh"].iloc[-1])


def test_compare_daily_marks_segment_start_and_carry_gap():
    days = [date(2024, 6, 1), date(2024, 6, 2), date(2024, 6, 4)]
    windows = [make_window(d) for d in days]
    table = fr.compare_daily(fr.run_scenario(windows, ESS, "forecast"),
                             fr.run_scenario(windows, ESS, "actual"))
    assert table["segment_start_forecast"].tolist() == [True, False, True]
    assert table["segment_id_forecast"].tolist() == [0, 0, 1]
    assert pd.isna(table["carry_gap_mwh_forecast"].iloc[0])
    assert table["carry_gap_mwh_forecast"].iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert pd.isna(table["carry_gap_mwh_actual"].iloc[2])


def test_compare_daily_joins_on_planning_day():
    windows = [make_window(d) for d in DAYS]
    table = fr.compare_daily(
        fr.run_scenario(windows, ESS, "forecast"), fr.run_scenario(windows[:2], ESS, "actual")
    )
    assert list(table["planning_day"]) == DAYS[:2]
    assert np.allclose(
        table["planned_objective_diff_krw"],
        table["objective_krw_forecast"] - table["objective_krw_actual"],
    )
