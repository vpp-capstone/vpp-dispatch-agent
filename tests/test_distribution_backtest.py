import numpy as np
import pandas as pd
import pytest

from vpp import distribution_backtest as db
from vpp import project_economics as pe
from vpp.optimization import EssParams

CFG = pe.EconomicsConfig.from_yaml("config/project_economics_v0.yaml")
BASE = EssParams.from_yaml("config/ess_v0_1.yaml")
DAYS = list(pd.date_range("2025-06-01", periods=3, freq="D"))


def _inputs(price_scale: float = 1.0) -> db.Inputs:
    ts = pd.date_range("2025-05-31", periods=24 * 6, freq="h")
    hour = ts.hour.to_numpy()
    solar = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 5.7  # 낮에 최대 5.7MW
    smp = np.where((hour >= 10) & (hour < 15), 40_000.0, 160_000.0) * price_scale
    hourly = pd.DataFrame(
        {"solar_mwh": solar, "smp_krw_per_mwh": smp, "rec_krw_per_mwh": 70_000.0}, index=ts
    )
    fc = {}
    for d in pd.date_range("2025-05-30", periods=6, freq="D"):
        idx = pd.date_range(d + pd.Timedelta(days=1), periods=48, freq="h")
        fc[d.strftime("%Y-%m-%d")] = hourly["smp_krw_per_mwh"].reindex(idx).ffill().to_numpy()
    return db.Inputs(hourly, fc, 5.7)


def test_line_cap_profiles():
    ts = pd.date_range("2025-06-01", periods=24, freq="h")
    fixed = db.line_cap_mw(db.LineCase("fixed", 1.7), ts, np.zeros(24), 5.7)
    assert fixed[12] == 1.7 and fixed[2] > 1e5 and fixed[18] > 1e5
    dyn = db.line_cap_mw(db.LineCase("dynamic", 1.5, 5.0), ts, np.array([5.7] + [0.0] * 23), 5.7)
    assert dyn[0] == pytest.approx(1.5)  # 해가 가장 강할 때 c
    assert dyn[1] == pytest.approx(6.5)  # 해가 없으면 c + 기존 태양광
    with pytest.raises(ValueError):
        db.LineCase("hourly", 1.0)


def test_ess_for_uses_battery_values():
    b = CFG.batteries["second_life_example"]
    ess = db.ess_for(BASE, CFG, b, db.LineCase("fixed", 1.7))
    assert ess.capacity_mwh == pytest.approx(16.0)
    assert ess.charge_efficiency == pytest.approx(0.93)
    assert ess.throughput_cost_krw_per_mwh == pytest.approx(
        b.throughput_cost_krw_per_mwh(20.0, CFG.fx)
    )
    small = db.ess_for(BASE, CFG, b, db.LineCase("fixed", 1.7, ess_power_mw=2, ess_energy_mwh=10))
    assert small.charge_power_mw == 2 and small.capacity_mwh == pytest.approx(8.0)


def test_run_case_respects_cap_and_settles_at_actual_price():
    inp = _inputs()
    case = db.LineCase("fixed", 1.5)
    h = db.run_case(inp, CFG, BASE, case, "new_2025_bnef", "forecast", DAYS)
    assert len(h) == 72
    day = (h["ts"].dt.hour >= 8) & (h["ts"].dt.hour < 18)
    assert (h.loc[day, "sell_mwh"] <= 1.5 + 1e-6).all()
    actual = inp.hourly["smp_krw_per_mwh"].reindex(h["ts"]).to_numpy()
    assert np.allclose(h["price_krw_per_mwh"], actual)
    assert h["charge_mwh"].sum() > 0  # 상한 때문에 남는 태양광을 담는다
    base = db.run_case(inp, CFG, BASE, case, db.BASELINE, "forecast", DAYS)
    assert (base["charge_mwh"] == 0).all()
    assert (base.loc[day, "sell_mwh"] <= 1.5 + 1e-9).all()


def test_operating_rows_feed_economics():
    inp = _inputs()
    jobs = db.build_jobs(
        ["new_2025_bnef"], fixed_caps=[1.5], dynamic_caps=[], existing_pv_mw=5.0,
        operations=["forecast", "perfect"],
    )  # fmt: skip
    assert {(c.name, b, o) for c, b, o in jobs} == {
        ("cap_1.50MW", "new_2025_bnef", "forecast"),
        ("cap_1.50MW", "none", "forecast"),
        ("cap_1.50MW", "new_2025_bnef", "perfect"),
        ("cap_1.50MW", "none", "perfect"),
    }
    op = db.run_all(inp, CFG, BASE, jobs, DAYS, workers=1)
    assert set(pe.OPERATING_COLUMNS) <= set(op.columns)
    assert (op["days"] == 3).all()
    ess = op[op.battery == "new_2025_bnef"].set_index("operation")
    base = op[op.battery == "none"].set_index("operation")
    assert (ess["market_revenue_krw"] > base["market_revenue_krw"]).all()
    summary, _ = pe.summarize(CFG, op)
    assert {"standard", "total", "incremental"} <= set(summary["basis"])


def test_size_cases_pair_with_one_baseline():
    jobs = db.build_jobs(
        ["new_2025_bnef"], fixed_caps=[1.5], dynamic_caps=[], existing_pv_mw=5.0,
        operations=["forecast"], sizes=[(2.0, 10.0), (4.0, 20.0)],
    )  # fmt: skip
    names = sorted(c.name for c, b, _ in jobs if b != "none")
    assert names == ["cap_1.50MW_ess2MW10MWh", "cap_1.50MW_ess4MW20MWh"]
    assert sum(b == "none" for _, b, _ in jobs) == 1
    op = db.run_all(_inputs(), CFG, BASE, jobs, DAYS[:1], workers=1)
    summary, _ = pe.summarize(CFG, op)
    inc = summary[(summary.basis == "incremental") & (summary.revenue_scenario == "base")]
    assert set(inc["ess_energy_mwh"]) == {10.0, 20.0}
    small = inc[inc.ess_energy_mwh == 10.0].iloc[0]
    assert small["capex_krw"] == pytest.approx(
        CFG.batteries["new_2025_bnef"].capex_krw(10.0, CFG.fx)
    )


def _exec_inputs(over: float = 1.2) -> db.Inputs:
    """발전량 예측이 실측보다 over 배 많은 입력 (공급 부족이 생기게)."""
    inp = _inputs()
    h = inp.hourly.copy()
    h["smp_rt_krw_per_mwh"] = h["smp_krw_per_mwh"] * 1.1
    h["fcr_price_krw_per_kwh"] = 12.0
    gen_fc = {}
    for d in pd.date_range("2025-05-31", periods=5, freq="D"):
        idx = pd.date_range(d, periods=48, freq="h")
        gen_fc[d.strftime("%Y-%m-%d")] = h["solar_mwh"].reindex(idx).fillna(0).to_numpy() * over
    return db.Inputs(h, inp.forecast, 5.7, gen_fc)


def test_load_fcr_csv_hourly_mean(tmp_path):
    cols = [f"{h:02d}시_{q}" for h in range(1, 25) for q in range(1, 5)]
    rows = []
    for kind, base in (("주파수제어예비력", 10.0), ("1차예비력", 99.0), ("3차운전예비력", 0.0)):
        vals = [base + (i % 4) for i in range(96)]
        rows.append(["2025-07-21", kind, *vals])
    rows[0][2 + 5] = np.nan  # 02시_2 결측 → 01:00 결측
    path = tmp_path / "reserve.csv"
    pd.DataFrame(rows, columns=["거래일자", "예비력 구분", *cols]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    s = db.load_fcr_csv([path])
    assert len(s) == 24
    assert s[pd.Timestamp("2025-07-21 00:00")] == pytest.approx(11.5)  # 01시 1~4구간 평균
    assert np.isnan(s[pd.Timestamp("2025-07-21 01:00")])


def test_load_generation_scales_portfolio(tmp_path):
    rows = []
    for asset, cap in (("a", 1000.0), ("b", 3000.0)):
        for day_offset in (0, 1):
            for h in range(24):
                t = pd.Timestamp("2025-06-01", tz="Asia/Seoul") + pd.Timedelta(
                    days=day_offset, hours=h
                )
                rows.append(
                    {
                        "asset_id": asset,
                        "capacity_kw": cap,
                        "planning_day_kst": "2025-06-01",
                        "valid_time_kst": t,
                        "forecast_day_offset": day_offset,
                        "improved_generation_kwh": cap * 0.5,
                        "generation_kwh": cap * 0.4,
                        "available_before_bid_deadline": True,
                    }
                )
    path = tmp_path / "solar.parquet"
    pd.DataFrame(rows).to_parquet(path)
    actual, fc, src = db.load_generation(path, 5.7)
    assert src == pytest.approx(4.0)
    assert len(actual) == 24  # 당일(offset 0) 실측만
    assert actual.iloc[0] == pytest.approx(4000 * 0.4 / 1000 * 5.7 / 4.0)
    assert fc["2025-06-01"].shape == (48,)
    assert fc["2025-06-01"][0] == pytest.approx(4000 * 0.5 / 1000 * 5.7 / 4.0)


def test_forecast_exec_never_oversells_and_settles():
    inp = _exec_inputs(1.2)
    case = db.LineCase("fixed", 1.5)
    for battery in ("new_2025_bnef", db.BASELINE):
        h = db.run_case(inp, CFG, BASE, case, battery, db.EXEC, DAYS)
        assert len(h) == 72
        assert (h["supply_mwh"] <= h["committed_sell_mwh"] + 1e-6).all()  # 초과 판매 없음
        day = (h["ts"].dt.hour >= 8) & (h["ts"].dt.hour < 18)
        assert (h.loc[day, "supply_mwh"] <= 1.5 + 1e-6).all()
        ess = (
            BASE if battery == db.BASELINE else db.ess_for(BASE, CFG, CFG.batteries[battery], case)
        )
        if battery != db.BASELINE:
            assert {
                "fixed_plan_supply_mwh",
                "fixed_plan_charge_mwh",
                "fixed_plan_discharge_mwh",
                "fixed_plan_energy_start_mwh",
                "fixed_plan_energy_end_mwh",
            } <= set(h)
            assert np.allclose(
                h.fixed_plan_energy_end_mwh.iloc[:-1],
                h.fixed_plan_energy_start_mwh.iloc[1:],
            )
            expected = db.income_scenario_hourly(h, "expected_plan")
            fixed = db.income_scenario_hourly(h, "fixed_plan")
            assert np.allclose(expected.supply_mwh, h.committed_sell_mwh)
            assert np.allclose(expected.price_krw_per_mwh, h.plan_price_krw_per_mwh)
            assert np.allclose(fixed.supply_mwh, h.fixed_plan_supply_mwh)
        row = db.operating_row_from_settlement(
            db.settle_exec(h, ess, 5.7),
            case=case.name, battery=battery, operation=db.EXEC,
            solar_capacity_credit_mw=0.2, ess_capacity_credit_mw=2.6, rcp_krw_per_kwh_h=22.05,
        )  # fmt: skip
        assert row["shortfall_mwh"] > 0  # 예측이 실측보다 많아서
        assert row["imbalance_penalty_krw"] >= 0 and row["penalty_missing_hours"] == 0
        assert row["market_revenue_krw"] == pytest.approx(
            row["solar_direct_revenue_krw"]
            + row["ess_discharge_revenue_krw"]
            + row["solar_capacity_payment_krw"]
            - row["imbalance_penalty_krw"]
        )


def test_forecast_exec_rows_feed_economics():
    inp = _exec_inputs(1.0)
    jobs = db.build_jobs(
        ["new_2025_bnef"], fixed_caps=[1.5], dynamic_caps=[], existing_pv_mw=5.0,
        operations=[db.EXEC],
    )  # fmt: skip
    op = db.run_all(inp, CFG, BASE, jobs, DAYS, workers=1)
    assert set(op["operation"]) == {db.EXEC}
    summary, _ = pe.summarize(CFG, op)
    assert {"standard", "total", "incremental"} <= set(summary["basis"])
    with pytest.raises(ValueError):
        db.run_case(_inputs(), CFG, BASE, db.LineCase("fixed", 1.5), "none", db.EXEC, DAYS)


def test_load_fcr_accepts_hourly_csv_and_later_file_wins(tmp_path):
    hourly = tmp_path / "fcr_hourly.csv"
    pd.DataFrame(
        {
            "ts": ["2025-07-20 23:00:00+09:00", "2025-07-21 00:00:00+09:00"],
            "fcr_price_krw_per_kwh": [13.0, 99.0],
            "fcr_price_source": "kpx_reserve_xlsx_hourly_mean",
        }
    ).to_csv(hourly, index=False)
    cols = [f"{h:02d}시_{q}" for h in range(1, 25) for q in range(1, 5)]
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        [["2025-07-21", "주파수제어예비력", *([20.0] * 96)]],
        columns=["거래일자", "예비력 구분", *cols],
    ).to_csv(raw, index=False, encoding="utf-8-sig")
    s = db.load_fcr_csv([hourly, raw])
    assert s[pd.Timestamp("2025-07-20 23:00")] == 13.0
    assert s[pd.Timestamp("2025-07-21 00:00")] == 20.0  # 뒤 파일 우선
    assert len(s) == 25
