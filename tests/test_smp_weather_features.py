from datetime import date

import numpy as np
import pandas as pd

from vpp import smp_forecast as sf
from vpp.smp_weather_features import (
    aggregate_daily_frame,
    capacity_weights,
    dates_between,
)


def test_capacity_weights_sum_duplicated_asset_grids() -> None:
    plants = pd.DataFrame(
        {
            "energy_type": ["solar", "solar", "solar", "wind"],
            "gfs_grid_id": ["g1", "g1", "g2", "g2"],
            "capacity_mw": [1.0, 2.0, 1.0, 4.0],
        }
    )
    weights = capacity_weights(plants, "solar")
    assert weights.to_dict() == {"g1": 0.75, "g2": 0.25}


def test_aggregate_uses_latest_component_availability_and_capacity_weights() -> None:
    issue = pd.Timestamp("2024-01-01 18:00", tz="UTC")
    target = pd.Timestamp("2024-01-02 15:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "gfs_grid_id": ["g1", "g2"],
            "issue_time_utc": [issue, issue],
            "issue_time_kst": [issue.tz_convert("Asia/Seoul")] * 2,
            "source_file_last_modified_utc": [
                issue + pd.Timedelta(hours=2),
                issue + pd.Timedelta(hours=3),
            ],
            "bid_deadline_kst": [pd.Timestamp("2024-01-02 11:00", tz="Asia/Seoul")] * 2,
            "available_before_bid_deadline": [True, True],
            "valid_time_utc": [target, target],
            "valid_time_kst": [target.tz_convert("Asia/Seoul")] * 2,
            "lead_hour": [21, 21],
            "shortwave_radiation_w_m2": [100.0, 300.0],
            "temperature_2m_c": [10.0, 20.0],
            "temperature_is_interpolated": [False, True],
            "total_cloud_cover_pct": [20.0, 80.0],
            "cloud_cover_is_interpolated": [False, True],
            "wind_u_80m_ms": [3.0, 0.0],
            "wind_v_80m_ms": [4.0, 2.0],
            "wind_speed_80m_ms": [5.0, 2.0],
        }
    )
    result_frame = aggregate_daily_frame(
        frame,
        pd.Series({"g1": 0.75, "g2": 0.25}),
        pd.Series({"g1": 0.25, "g2": 0.75}),
    )
    result = result_frame.iloc[0]

    assert result["available_at"] == (issue + pd.Timedelta(hours=3)).tz_convert("Asia/Seoul")
    assert result["target_ts"] == target.tz_convert("Asia/Seoul")
    assert result["solar_shortwave_w_m2_fleet_weighted"] == 150.0
    assert result["wind_80m_wind_speed_ms_fleet_weighted"] == 2.75
    assert result["jeju_shortwave_mean_w_m2"] == 200.0
    assert result["weather_grid_coverage"] == 2
    assert np.isclose(result["jeju_wind_vector_speed_ms"], np.hypot(1.5, 3.0))

    model_features = sf.weather_feature_columns(result_frame)
    assert "solar_shortwave_w_m2_fleet_weighted" in model_features
    assert "weather_grid_coverage" not in model_features
    assert "quality_temperature_interpolation_ratio" not in model_features


def test_dates_between_is_inclusive() -> None:
    assert dates_between(date(2025, 1, 1), date(2025, 1, 3)) == [
        date(2025, 1, 1),
        date(2025, 1, 2),
        date(2025, 1, 3),
    ]
