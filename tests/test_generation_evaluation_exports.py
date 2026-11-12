from __future__ import annotations

import pandas as pd
import pytest

from vpp.generation_forecast.export_evaluation_predictions import (
    attach_frozen_actuals,
    evaluation_frame,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "energy_type": ["wind", "wind"],
            "capacity_kw": [10.0, 10.0],
            "planning_day_kst": ["2025-01-01", "2025-01-01"],
            "issue_time_utc": pd.to_datetime(["2024-12-29T18:00Z"] * 2),
            "available_before_bid_deadline": [True, True],
            "valid_time_utc": pd.to_datetime(["2024-12-31T15:00Z"] * 2),
            "valid_time_kst": pd.to_datetime(["2025-01-01T00:00+09:00"] * 2),
            "lead_hour": [45, 45],
            "forecast_day_offset": [0, 0],
            "improved_generation_kwh": [4.0, 6.0],
            "model_version": ["frozen", "frozen"],
            "trained_until": ["2024-12-31", "2024-12-31"],
        }
    )


def test_attach_frozen_actuals_preserves_predictions_and_masks_invalid_label():
    predictions = _predictions()
    labels = predictions[["asset_id", "issue_time_utc", "valid_time_utc"]].copy()
    labels["generation_kwh"] = [5.0, 999.0]
    labels["label_is_valid"] = [True, False]
    result = attach_frozen_actuals(predictions, labels)
    assert result.improved_generation_kwh.tolist() == [4.0, 6.0]
    assert result.generation_kwh.iloc[0] == 5.0
    assert pd.isna(result.generation_kwh.iloc[1])


def test_attach_frozen_actuals_rejects_duplicate_labels():
    predictions = _predictions()
    labels = predictions[["asset_id", "issue_time_utc", "valid_time_utc"]].copy()
    labels["generation_kwh"] = [5.0, 7.0]
    labels["label_is_valid"] = True
    labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="중복 키"):
        attach_frozen_actuals(predictions, labels)


def test_evaluation_frame_keeps_actual_but_excludes_raw_features():
    result = _predictions()
    result["generation_kwh"] = [5.0, 7.0]
    result["wind_speed_80m_ms"] = [3.0, 4.0]
    exported = evaluation_frame(result)
    assert exported.generation_kwh.tolist() == [5.0, 7.0]
    assert "wind_speed_80m_ms" not in exported
    assert "improved_generation_kwh" in exported
