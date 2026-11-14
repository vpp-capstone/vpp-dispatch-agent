"""현행 규칙의 시간 근사: 손계산 금액, 누락/단위/시각, 독립 CLI 회귀 검증."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tests.test_execution import ESS
from tests.test_settlement import executed
from vpp.hourly_settlement import (
    hourly_settlement_daily,
    hourly_settlement_totals,
    main,
    settle_hourly,
)
from vpp.imbalance import hourly_imbalance
from vpp.settlement import settle
from vpp.settlement_inputs import HourlySettlementParams

PARAMS = HourlySettlementParams(resource_id="test_resource")


def inputs(supply=(8.4,), rt=(150.0,)):
    ts = pd.date_range("2024-06-01", periods=len(supply), freq="h", tz="Asia/Seoul")
    execution = pd.DataFrame({
        "ts": ts, "dt_hours": 1, "committed_sell_mwh": 10.0, "supply_mwh": supply,
        "charge_mwh": 0.0, "discharge_mwh": 0.0,
        "rec_krw_per_mwh": 0.0,
    })
    market = pd.DataFrame({"ts": ts, "smp_da": 100.0, "smp_rt_mean": rt})
    dispatch = pd.DataFrame({
        "ts": ts, "resource_capacity_mw": 10.0, "set_point_mw": 8.0,
        "available_capacity_mw": 10.0, "fcr_price_krw_per_kwh": 0.0,
        "min_offer_price_krw_per_kwh": -10.0, "offer_floor_krw_per_kwh": -80.0,
        "dispatch_source": "scenario", "dispatch_note": "손계산용 가상 급전·입찰 입력",
    })
    return execution, market, dispatch


def calculate(execution, market, dispatch=None, params=PARAMS):
    return settle_hourly(execution, market, ESS, params=params, dispatch=dispatch)


def test_dual_price_and_dispatch_penalty_are_separate_and_not_double_counted():
    # 판매 편차 -1.6 MWh와 급전 편차 +0.4 MWh는 부호부터 다르다.
    row = calculate(*inputs()).iloc[0]
    assert row["committed_sales_krw"] == 1_000_000
    assert row["sales_deviation_mwh"] == pytest.approx(-1.6)
    assert row["deviation_krw"] == pytest.approx(-240_000)
    assert row["energy_revenue_krw"] == pytest.approx(760_000)
    assert row["dispatch_deviation_mwh"] == pytest.approx(0.4)
    assert row["allowed_deviation_mwh"] == pytest.approx(0.15)
    assert row["penalized_mwh"] == pytest.approx(0.25)
    assert row["penalty_price_krw_per_mwh"] == 160_000
    assert row["imbalance_penalty_krw"] == pytest.approx(40_000)
    assert row["realized_net_krw"] == pytest.approx(720_000)
    assert row["rule_application"] == "current_rule_counterfactual"
    assert row["dispatch_source"] == "scenario"


@pytest.mark.parametrize(("rt", "available", "supply", "tolerance", "branch", "cost"), [
    (150_000, 8, 8.8, 0.06, "normal", 1_000),
    (150_000, 8, 7.2, 0.06, "normal", 1_000),
    (150_000, 10, 8.4, 0.015, "dispatch_reduction", 41_250),
    (150_000, 10, 7, 0.015, "dispatch_reduction", 0),
    (-20_000, 10, 8.4, 0.015, "nonpositive_rt", 51_250),
    (-20_000, 10, 7, 0.015, "nonpositive_rt", 0),
    (0, 8, 8.4, 0.015, "nonpositive_rt", 51_250),
])
def test_all_rule_table_branches(rt, available, supply, tolerance, branch, cost):
    result = hourly_imbalance(
        supply_mwh=supply, resource_capacity_mw=10, set_point_mw=8,
        available_capacity_mw=available, realtime_price_krw_per_mwh=rt,
        fcr_price_krw_per_mwh=5_000, min_offer_price_krw_per_mwh=-10_000,
        offer_floor_krw_per_mwh=-80_000,
    )
    assert result["imbalance_status"] == "computed"
    assert result["tolerance_ratio"] == tolerance
    assert result["imbalance_branch"] == branch
    assert result["imbalance_penalty_krw"] == pytest.approx(cost)


@pytest.mark.parametrize(("available", "supply", "cost"), [
    (10, 8.15, 0), (10, 8.1499, 0), (10, 8.1501, 16),
    (8, 8.6, 0), (8, 7.4, 0), (8, 8.6001, 0.5), (8, 7.3999, 0.5),
    (8, 8.0, 0),
])
def test_tolerance_boundaries(available, supply, cost):
    result = hourly_imbalance(
        supply_mwh=supply, resource_capacity_mw=10, set_point_mw=8,
        available_capacity_mw=available, realtime_price_krw_per_mwh=150_000,
        fcr_price_krw_per_mwh=0 if available == 10 else 5_000,
        min_offer_price_krw_per_mwh=-10_000,
    )
    assert result["imbalance_penalty_krw"] == pytest.approx(cost)


def test_price_maximum_and_known_zero_branches_do_not_need_irrelevant_inputs():
    common = dict(supply_mwh=8.4, resource_capacity_mw=10, set_point_mw=8,
                  available_capacity_mw=10, realtime_price_krw_per_mwh=150_000)
    row = hourly_imbalance(**common, min_offer_price_krw_per_mwh=200_000,
                           fcr_price_krw_per_mwh=5_000)
    assert row["imbalance_penalty_krw"] == pytest.approx(1_250)
    for changes in ({"supply_mwh": 8.15}, {"supply_mwh": 7.0},
                    {"supply_mwh": 7.0, "realtime_price_krw_per_mwh": -20_000,
                     "available_capacity_mw": np.nan}):
        row = hourly_imbalance(**(common | changes))
        assert row["imbalance_status"] == "computed"
        assert row["imbalance_penalty_krw"] == 0


def test_missing_dispatch_preserves_energy_revenue_without_zero_penalty():
    execution, market, _ = inputs(supply=(8.0,))
    result = calculate(execution, market)
    row = result.iloc[0]
    assert row["energy_revenue_krw"] == 700_000
    assert np.isnan(row["imbalance_penalty_krw"])
    assert np.isnan(row["realized_net_krw"])
    assert "set_point_mw" in row["imbalance_missing_inputs"]
    total = hourly_settlement_totals(result)
    assert total["energy_revenue_krw"] == 700_000
    assert total["imbalance_penalty_krw"] is None
    assert total["realized_net_krw"] is None
    assert total["penalty_missing_hours"] == 1
    assert total["computable"] is False


@pytest.mark.parametrize("column", [
    "resource_capacity_mw", "set_point_mw", "available_capacity_mw",
    "fcr_price_krw_per_kwh", "min_offer_price_krw_per_kwh",
])
def test_missing_required_dispatch_input_is_not_zero(column):
    execution, market, dispatch = inputs()
    row = calculate(execution, market, dispatch.drop(columns=column)).iloc[0]
    assert row["imbalance_status"] == "missing_inputs"
    assert column.replace("per_kwh", "per_mwh") in row["imbalance_missing_inputs"]
    assert np.isnan(row["realized_net_krw"])


def test_negative_rt_requires_floor_and_shortfall_revenue_has_correct_sign():
    execution, market, dispatch = inputs(supply=(8.4,), rt=(-20.0,))
    row = calculate(execution, market, dispatch.drop(columns="offer_floor_krw_per_kwh")).iloc[0]
    assert row["deviation_krw"] == pytest.approx(32_000)
    assert row["imbalance_missing_inputs"] == "offer_floor_krw_per_mwh"
    assert np.isnan(row["realized_net_krw"])


@pytest.mark.parametrize("column", ["smp_da", "smp_rt_mean"])
def test_missing_price_is_never_filled_with_forecast_or_other_market(column):
    execution, market, dispatch = inputs((8.4, 8.4), (150, 150))
    market.loc[1, column] = np.nan
    result = calculate(execution, market, dispatch)
    total = hourly_settlement_totals(result)
    assert total["price_missing_hours"] == 1
    assert total["energy_revenue_krw"] is None
    assert total["realized_net_krw"] is None
    assert result.iloc[0]["realized_net_krw"] == pytest.approx(720_000)
    assert np.isnan(result.iloc[1]["realized_net_krw"])


def test_rec_wear_and_loss_factors_use_explicit_configuration_once():
    execution, market, dispatch = inputs()
    execution["charge_mwh"] = 2
    execution["discharge_mwh"] = 1
    execution["rec_krw_per_mwh"] = 70_000
    params = replace(PARAMS, day_ahead_loss_factor=0.9, realtime_loss_factor=0.8,
                     rec_mode="supplied_mwh_once")
    row = calculate(execution, market, dispatch, params).iloc[0]
    assert row["day_ahead_price_krw_per_mwh"] == 90_000
    assert row["realtime_price_krw_per_mwh"] == 120_000
    assert row["energy_revenue_krw"] == pytest.approx(708_000)
    assert row["rec_krw"] == pytest.approx(588_000)
    assert row["wear_krw"] == 3_000
    assert row["imbalance_penalty_krw"] == pytest.approx(32_500)
    assert row["realized_net_krw"] == pytest.approx(1_260_500)
    with pytest.raises(ValueError, match="rec_krw_per_mwh"):
        calculate(execution.drop(columns="rec_krw_per_mwh"), market, dispatch, params)


def test_matching_uses_kst_timestamp_not_row_position_and_records_gaps():
    execution, market, dispatch = inputs((8.0, 8.4, 8.0), (150, 200, 150))
    market["ts"] = market["ts"].dt.tz_convert("UTC")
    dispatch["ts"] = dispatch["ts"].dt.tz_localize(None)
    result = calculate(execution.iloc[[2, 0]], market.iloc[::-1], dispatch.iloc[::-1])
    assert result["energy_revenue_krw"].tolist() == [700_000, 700_000]
    assert str(result["ts"].dt.tz) == "Asia/Seoul"
    assert hourly_settlement_totals(result)["missing_execution_hours_between_start_end"] == 1
    row = hourly_settlement_daily(result).iloc[0]
    assert row["date"] == "2024-06-01"
    assert not row["full_day_coverage"]
    # 가격 행이 없는 시간에 다른 시간의 값을 가져오지 않는다.
    result = calculate(execution, market.iloc[[0, 2]], dispatch.iloc[[0, 2]])
    assert not result.iloc[1]["settlement_computable"]


@pytest.mark.parametrize("frame_index", [0, 1, 2])
@pytest.mark.parametrize("problem", ["duplicate", "quarter_hour", "dt_hours", "resource_id"])
def test_invalid_time_interval_and_mixed_resource_inputs_fail(frame_index, problem):
    frames = list(inputs())
    frame = frames[frame_index].copy()
    if problem == "duplicate":
        frame = pd.concat([frame, frame])
    elif problem == "quarter_hour":
        frame["ts"] += pd.Timedelta(minutes=15)
    elif problem == "dt_hours":
        frame["dt_hours"] = 0.25
    else:
        frame["resource_id"] = "another_resource"
    frames[frame_index] = frame
    with pytest.raises(ValueError):
        calculate(*frames)


@pytest.mark.parametrize(("frame_index", "column", "value"), [
    (0, "supply_mwh", -1), (0, "committed_sell_mwh", np.nan),
    (0, "charge_mwh", np.inf), (1, "smp_da", np.inf), (1, "smp_rt_mean", "bad"),
    (2, "resource_capacity_mw", 0), (2, "resource_capacity_mw", -10),
    (2, "fcr_price_krw_per_kwh", -1), (2, "dispatch_note", ""),
    (2, "dispatch_source", "default_actual"),
])
def test_invalid_values_are_rejected(frame_index, column, value):
    frames = list(inputs())
    frames[frame_index][column] = value
    with pytest.raises(ValueError):
        calculate(*frames)


def test_rule_date_is_explicit_and_does_not_switch_to_historical_tolerance():
    execution, market, dispatch = inputs((8.4,) * 2, (150,) * 2)
    times = pd.to_datetime(["2026-03-31T23:00+09:00", "2026-04-01T00:00+09:00"])
    for frame in (execution, market, dispatch):
        frame["ts"] = times
    result = calculate(execution, market, dispatch)
    assert result["rule_application"].tolist() == [
        "current_rule_counterfactual", "effective_rule_hourly_approximation",
    ]
    assert result["tolerance_ratio"].tolist() == [0.015, 0.015]
    assert hourly_settlement_daily(result)["date"].tolist() == ["2026-03-31", "2026-04-01"]
    assert hourly_settlement_totals(result)["counterfactual_hours"] == 1


def test_cli_writes_auditable_outputs_and_refuses_overwrite(tmp_path):
    frames = inputs((8.4, 8.15), (150, 150))
    paths = {}
    for name, frame in zip(("execution", "market", "dispatch"), frames, strict=True):
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    config = tmp_path / "settlement.yaml"
    config.write_text("resource_id: test_resource\n", encoding="utf-8")
    outdir = tmp_path / "result"
    args = ["--config", str(config), "--outdir", str(outdir), "--ess", "config/ess_v0.yaml"]
    for name, path in paths.items():
        args += [f"--{name}", str(path)]
    assert main(args) == 0
    hourly = pd.read_csv(outdir / "hourly_settlement.csv")
    daily = pd.read_csv(outdir / "daily_settlement.csv")
    summary = json.loads((outdir / "summary.json").read_text())
    assert hourly["imbalance_penalty_krw"].tolist() == pytest.approx([40_000, 0])
    assert daily.iloc[0]["realized_net_krw"] == pytest.approx(1_442_500)
    assert summary["totals"]["realized_net_krw"] == pytest.approx(1_442_500)
    assert summary["parameters"]["rule_effective_date"] == "2026-04-01"
    assert summary["parameters"]["loss_factor_source"] == "assumption"
    with pytest.raises(SystemExit):
        main(args)


def test_cli_without_dispatch_reports_unavailable_net_as_json_null(tmp_path):
    execution, market, _ = inputs()
    execution.to_csv(tmp_path / "execution.csv", index=False)
    market.to_csv(tmp_path / "market.csv", index=False)
    outdir = tmp_path / "missing"
    main(["--execution", str(tmp_path / "execution.csv"),
          "--market", str(tmp_path / "market.csv"), "--outdir", str(outdir),
          "--ess", "config/ess_v0.yaml"])
    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["totals"]["realized_net_krw"] is None
    assert summary["totals"]["penalty_missing_hours"] == 1
    assert summary["totals"]["energy_revenue_krw"] == pytest.approx(760_000)


@pytest.mark.parametrize("kwargs", [
    {"resource_id": ""}, {"rule_version": "historical"}, {"day_ahead_loss_factor": 0},
    {"realtime_loss_factor": np.inf}, {"rec_mode": "unknown"}, {"loss_factor_note": ""},
])
def test_config_rejects_invalid_assumptions(kwargs):
    with pytest.raises(ValueError):
        HourlySettlementParams(**({"resource_id": "test_resource"} | kwargs))


def test_existing_execution_keeps_rec_wear_and_matches_legacy_when_prices_equal():
    legacy = settle(executed(rec=70_000), [100_000] * 4, ESS)
    market = pd.DataFrame({"ts": legacy.ts, "smp_da": 100, "smp_rt_mean": 100})
    # 급전 이행 오차 0인 검증 시나리오. 실제 급전값으로 간주하지 않는다.
    dispatch = pd.DataFrame({
        "ts": legacy.ts, "set_point_mw": legacy.supply_mwh,
        "available_capacity_mw": legacy.supply_mwh, "resource_capacity_mw": 10,
    })
    result = calculate(legacy, market, dispatch)
    for col in ("rec_krw", "wear_krw", "realized_net_krw"):
        np.testing.assert_allclose(result[col], legacy[col])
    assert "settlement_price_krw_per_mwh" not in result
    assert result["legacy_settlement_price_krw_per_mwh"].eq(100_000).all()
    assert result["rec_basis"].eq("supplied_mwh_once").all()


def test_cli_reuses_ess_snapshot_and_execution_rec_instead_of_zero_defaults(tmp_path):
    execution, market, _ = inputs()
    execution["charge_mwh"] = 1.0
    execution["rec_krw_per_mwh"] = 70_000.0
    execution.to_csv(tmp_path / "hourly_execution.csv", index=False)
    market.to_csv(tmp_path / "market.csv", index=False)
    (tmp_path / "summary.json").write_text(
        json.dumps({"ess": ESS.to_dict(), "rec": {"enabled": True}}), encoding="utf-8",
    )
    outdir = tmp_path / "result"
    main(["--execution", str(tmp_path / "hourly_execution.csv"),
          "--market", str(tmp_path / "market.csv"), "--outdir", str(outdir)])
    result = pd.read_csv(outdir / "hourly_settlement.csv")
    summary = json.loads((outdir / "summary.json").read_text())
    assert result.loc[0, "wear_krw"] == 1_000
    assert result.loc[0, "rec_krw"] == 588_000
    assert summary["ess_source"].endswith("summary.json#ess")
    assert summary["source_rec_metadata"] == {"enabled": True}


def test_cli_requires_original_ess_settings_when_no_snapshot_exists(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        main(["--execution", str(tmp_path / "hourly_execution.csv"),
              "--market", str(tmp_path / "market.csv"), "--outdir", str(tmp_path / "result")])
    assert not (tmp_path / "result").exists()


def test_default_requires_rec_input_and_exclusion_must_be_explicit():
    execution, market, dispatch = inputs()
    execution = execution.drop(columns="rec_krw_per_mwh")
    with pytest.raises(ValueError, match="rec_krw_per_mwh"):
        calculate(execution, market, dispatch)
    result = calculate(execution, market, dispatch, replace(PARAMS, rec_mode="excluded"))
    assert result["rec_krw"].eq(0).all()
