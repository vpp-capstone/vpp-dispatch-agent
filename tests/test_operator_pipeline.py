"""Content changes, improved predictions and blocked producers must be visible."""

from pathlib import Path

import pandas as pd
import pytest

from tests.test_forecast_backtest import DAYS, write_inputs
from vpp import forecast_inputs as fi
from vpp import operator_pipeline as op
from vpp.generation_forecast.infer_improved import prepare_inference


def test_improved_values_override_old_column_but_not_fill_gaps():
    frame = pd.DataFrame(
        {"lightgbm_generation_kwh": [1, 2], "improved_generation_kwh": [10, float("nan")]}
    )
    result = fi.normalize_generation_model(frame)
    assert result.lightgbm_generation_kwh.iloc[0] == 10
    assert pd.isna(result.lightgbm_generation_kwh.iloc[1])
    assert result.forecast_source_column.eq("improved_generation_kwh").all()
    assert frame.lightgbm_generation_kwh.tolist() == [1, 2]


def test_generation_loader_uses_improved_per_file(tmp_path):
    write_inputs(tmp_path, DAYS)
    solar = pd.read_parquet(tmp_path / "solar.parquet")
    solar["improved_generation_kwh"] = 50.0
    solar.to_parquet(tmp_path / "solar.parquet")
    pred = fi.load_generation_predictions(
        tmp_path / "solar.parquet", tmp_path / "wind.parquet", tmp_path / "plants.csv"
    )
    assert pred.loc[pred.energy_type == "solar", "lightgbm_generation_kwh"].eq(50).all()
    assert set(pred.forecast_source_column) == {
        "improved_generation_kwh",
        "lightgbm_generation_kwh",
    }


def test_same_inputs_reuse_but_price_change_recalculates(tmp_path):
    write_inputs(tmp_path, DAYS)
    sources = {
        k: tmp_path / f"{k}{'.csv' if k == 'smp' else '.parquet'}" for k in ("solar", "wind", "smp")
    }
    cfg = {
        "output": str(tmp_path / "output"),
        "plants": str(tmp_path / "plants.csv"),
        "ess": str(tmp_path / "ess.yaml"),
        "rec": None,
    }
    first = op.refresh_dispatch(tmp_path, cfg, sources, label="test")
    again = op.refresh_dispatch(tmp_path, cfg, sources, label="test")
    assert again == first
    original_snapshot = Path(first["inputs"]["smp"]).read_bytes()
    smp = pd.read_csv(sources["smp"])
    smp.smp_hat += 1
    smp.smp_hat_krw_per_mwh += 1000
    smp.to_csv(sources["smp"], index=False)
    second = op.refresh_dispatch(tmp_path, cfg, sources, label="test")
    assert second["id"] != first["id"]
    assert Path(first["inputs"]["smp"]).read_bytes() == original_snapshot
    assert Path(second["plan"], "plan_48h.csv").exists()
    assert second["realized_profit"] is None


def test_invalid_new_input_does_not_publish_release(tmp_path):
    write_inputs(tmp_path, DAYS)
    sources = {
        k: tmp_path / f"{k}{'.csv' if k == 'smp' else '.parquet'}" for k in ("solar", "wind", "smp")
    }
    cfg = {
        "output": str(tmp_path / "output"),
        "plants": str(tmp_path / "plants.csv"),
        "ess": str(tmp_path / "ess.yaml"),
        "rec": None,
    }
    original = op.refresh_dispatch(tmp_path, cfg, sources, label="test")
    pd.read_csv(sources["smp"]).iloc[:1].to_csv(sources["smp"], index=False)
    with pytest.raises(ValueError, match="48시간"):
        op.refresh_dispatch(tmp_path, cfg, sources, label="broken")
    assert len(list((tmp_path / "output/releases").glob("*.json"))) == 1
    assert Path(original["backtest"], "summary.json").exists()


def test_inference_keeps_missing_labels_and_uses_training_category_map():
    raw = pd.DataFrame(
        {
            "valid_time_kst": ["2025-01-01T00:00:00+09:00"],
            "asset_id": ["b"],
            "gfs_grid_id": ["g"],
            "label_is_valid": [False],
            "generation_kwh": [999.0],
        }
    )
    bundle = {
        "training_years": [2023, 2024],
        "category_maps": {"asset_id": {"a": 0, "b": 7}, "gfs_grid_id": {"g": 4}},
    }
    frame = prepare_inference(raw, bundle, "2025-01-01", "2025-01-01")
    assert len(frame) == 1
    assert frame.asset_code.iloc[0] == 7
    assert pd.isna(frame.generation_kwh.iloc[0])
    with pytest.raises(ValueError, match="학습"):
        prepare_inference(raw, bundle, "2024-01-01", "2024-12-31")


def test_run_uses_shared_files_and_blocks_when_one_is_missing(tmp_path, monkeypatch):
    """공유 파일이 다 있으면 MILP 한 단계로 끝나고, 하나라도 없으면 그 경로를 적고 멈춘다."""
    write_inputs(tmp_path, DAYS)
    cfg = {
        "output": str(tmp_path / "output"),
        "plants": str(tmp_path / "plants.csv"),
        "ess": str(tmp_path / "ess.yaml"),
        "rec": None,
        **{k: str(tmp_path / f"{k}.parquet") for k in ("solar", "wind")},
        "smp": str(tmp_path / "smp.csv"),
    }
    monkeypatch.setattr(op, "config", lambda root: cfg)
    result = op.run(tmp_path)
    assert result["state"] == "complete"
    assert [s["name"] for s in result["stages"]] == ["MILP 재계산·검산"]
    assert result["active"]["label"] == "Drive 공유 예측 파일"
    assert result["active"]["forecast_columns"] == ["lightgbm_generation_kwh"]
    assert all(r["ready"] for r in op.requirements(cfg))
    first_revision = result["revision"]

    (tmp_path / "smp.csv").rename(tmp_path / "smp_moved.csv")
    blocked = op.run(tmp_path)
    assert blocked["state"] == "needs_data"
    assert blocked["stages"][-1]["state"] == "blocked"
    assert blocked["stages"][-1]["missing"] == [str(tmp_path / "smp.csv")]
    assert blocked["active"]["id"] == result["active"]["id"]  # 이전 검산 결과 유지
    assert blocked["revision"] != first_revision
    assert [r["name"] for r in op.requirements(cfg) if not r["ready"]] == ["SMP 예측"]


def test_monthly_rec_change_with_same_extremes_invalidates_result(tmp_path):
    write_inputs(tmp_path, DAYS)
    table = tmp_path / "monthly.csv"
    table.write_text("month,price_krw_per_rec\n2024-05,100\n2024-06,200\n2024-07,900\n")
    rec = tmp_path / "rec.yaml"
    rec.write_text(f"weight: 1\nprice_table: {table}\n")
    cfg = {
        "output": str(tmp_path / "output"),
        "plants": str(tmp_path / "plants.csv"),
        "ess": str(tmp_path / "ess.yaml"),
        "rec": str(rec),
    }
    sources = {
        k: tmp_path / f"{k}{'.csv' if k == 'smp' else '.parquet'}" for k in ("solar", "wind", "smp")
    }
    first = op.refresh_dispatch(tmp_path, cfg, sources, label="test")
    table.write_text("month,price_krw_per_rec\n2024-05,100\n2024-06,300\n2024-07,900\n")
    second = op.refresh_dispatch(tmp_path, cfg, sources, label="test")
    assert second["id"] != first["id"]
    assert first["hashes"]["rec_table"] != second["hashes"]["rec_table"]
