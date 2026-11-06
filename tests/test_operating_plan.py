"""operating_plan: 실측 없는 예측만으로 48시간 계산·24시간 확정, 시작 SOC 출처, 덮어쓰기 금지."""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tests.test_forecast_backtest import write_inputs
from vpp import operating_plan as op
from vpp.optimization import EssParams

DAYS = [date(2024, 6, 5) + timedelta(days=i) for i in range(3)]


def write_forecast_only_inputs(tmp_path, days):
    """test_forecast_backtest 의 합성 입력에서 실측 열을 아예 떼어낸다."""
    write_inputs(tmp_path, days)
    for name in ("solar", "wind"):
        pred = pd.read_parquet(tmp_path / f"{name}.parquet").drop(columns=["generation_kwh"])
        pred.to_parquet(tmp_path / f"{name}.parquet", index=False)
    smp = pd.read_csv(tmp_path / "smp.csv").drop(columns=["smp_actual"])
    smp.to_csv(tmp_path / "smp.csv", index=False)


def paths(tmp_path):
    return dict(
        solar_path=tmp_path / "solar.parquet", wind_path=tmp_path / "wind.parquet",
        smp_path=tmp_path / "smp.csv", plants_path=tmp_path / "plants.csv",
        ess_path=tmp_path / "ess.yaml", out_dir=tmp_path / "out",
    )


def test_plan_from_forecast_only_solves_48h_and_commits_24h(tmp_path):
    write_forecast_only_inputs(tmp_path, DAYS)
    out = op.run_operating_plan(DAYS[1], run_id="d1", **paths(tmp_path))
    full = pd.read_csv(out / "plan_48h.csv")
    committed = pd.read_csv(out / "committed_24h.csv")
    assert len(full) == 48 and len(committed) == 24
    assert full["status"].tolist() == ["committed"] * 24 + ["provisional"] * 24
    assert "generation_actual_mwh" not in full.columns
    assert pd.to_datetime(full["ts"]).iloc[0].strftime("%Y-%m-%d %H") == f"{DAYS[1]} 00"

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    assert summary["mode"] == op.MODE and summary["planning_day"] == DAYS[1].isoformat()
    assert summary["start_energy"]["source"] == "config_soc_initial"
    assert summary["start_energy"]["energy_mwh"] == pytest.approx(8.0)
    assert summary["terminal"] == "none"
    assert summary["window"]["solved_hours"] == 48 and summary["window"]["commit_hours"] == 24
    assert summary["inputs"]["actual_used"] is False
    assert {"sha256", "size_bytes", "modified_at"} <= set(summary["inputs"]["solar_predictions"])
    assert "package_version" in summary["code_version"]
    # 테스트 yaml 은 degradation_model 미지정 → 기본 fixed_throughput, 비용 0
    assert summary["ess"]["degradation"]["throughput_cost_krw_per_mwh"] == 0.0
    c = summary["committed"]
    assert c["objective_krw"] == pytest.approx(c["sales_revenue_krw"] - c["throughput_cost_krw"])
    assert c["final_energy_mwh"] == pytest.approx(committed["energy_end_mwh"].iloc[-1])
    assert validation["all_passed"]
    assert {x["item"] for x in validation["structure_checks"]} == {
        "solved_hours_48", "committed_hours_24", "committed_is_first_24h",
    }


def test_previous_plan_soc_carries_only_for_consecutive_day(tmp_path):
    write_forecast_only_inputs(tmp_path, DAYS)
    p = paths(tmp_path)
    first = op.run_operating_plan(DAYS[0], run_id="d0", **p)
    prev_final = json.loads((first / "summary.json").read_text())["committed"]["final_energy_mwh"]
    second = op.run_operating_plan(DAYS[1], run_id="d1", from_plan=first, **p)
    start = json.loads((second / "summary.json").read_text())["start_energy"]
    assert start["source"] == "previous_plan"
    assert start["energy_mwh"] == pytest.approx(prev_final)
    assert "계획상" in start["note"]
    with pytest.raises(ValueError, match="D-1"):
        op.run_operating_plan(DAYS[2], run_id="d2", from_plan=first, **p)


def test_start_soc_input_and_range(tmp_path):
    ess = EssParams.from_yaml("config/ess_v0.yaml")
    s = op.resolve_start_energy(ess, DAYS[0], start_soc=0.25)
    assert s.source == "user_input" and s.energy_mwh == pytest.approx(4.0)
    with pytest.raises(ValueError, match="범위"):
        op.resolve_start_energy(ess, DAYS[0], start_energy_mwh=15.0)
    with pytest.raises(ValueError, match="하나만"):
        op.resolve_start_energy(ess, DAYS[0], start_soc=0.5, start_energy_mwh=8.0)


def test_cli_matches_function_and_refuses_overwrite(tmp_path, capsys):
    write_forecast_only_inputs(tmp_path, DAYS)
    p = paths(tmp_path)
    by_function = op.run_operating_plan(DAYS[1], run_id="f", **p)
    argv = ["--day", DAYS[1].isoformat(), "--ess", str(p["ess_path"]),
            "--solar", str(p["solar_path"]), "--wind", str(p["wind_path"]),
            "--smp", str(p["smp_path"]), "--plants", str(p["plants_path"]),
            "--out", str(p["out_dir"]), "--run-id", "c"]
    op.main(argv)
    assert "✅" in capsys.readouterr().out
    a = pd.read_csv(by_function / "committed_24h.csv")
    b = pd.read_csv(p["out_dir"] / "c" / "committed_24h.csv")
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(FileExistsError, match="overwrite"):
        op.main(argv)
    op.main(argv + ["--overwrite"])


def test_degradation_config_overlays_only_cost_fields(tmp_path):
    """화면의 열화비 2택: ESS 파일은 그대로, 열화 3필드만 다른 YAML 에서. 어느 파일도 안 고친다."""
    before = {p: p.read_text() for p in (Path("config/ess_v0.yaml"), Path("config/ess_v0_1.yaml"))}
    ess = op.resolve_ess("config/ess_v0.yaml", "config/ess_v0_1.yaml")
    base = EssParams.from_yaml("config/ess_v0.yaml")
    v01 = EssParams.from_yaml("config/ess_v0_1.yaml")
    assert ess.capacity_mwh == base.capacity_mwh and ess.soc_initial == base.soc_initial
    assert (ess.degradation_model, ess.throughput_basis, ess.throughput_cost_krw_per_mwh) == (
        v01.degradation_model, v01.throughput_basis, v01.throughput_cost_krw_per_mwh)
    assert op.resolve_ess("config/ess_v0.yaml") == base
    assert {p: p.read_text() for p in before} == before
    write_forecast_only_inputs(tmp_path, DAYS)
    out = op.run_operating_plan(DAYS[1], run_id="deg", degradation_path="config/ess_v0_1.yaml",
                                **paths(tmp_path))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["ess"]["degradation_source"] == "config/ess_v0_1.yaml"
    assert summary["ess"]["throughput_cost_krw_per_mwh"] == v01.throughput_cost_krw_per_mwh


def test_missing_forecast_day_is_rejected_with_reason(tmp_path):
    write_forecast_only_inputs(tmp_path, DAYS)
    with pytest.raises(ValueError, match="generation_missing"):
        op.run_operating_plan(DAYS[-1] + timedelta(days=5), run_id="x", **paths(tmp_path))
