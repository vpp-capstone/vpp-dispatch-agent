import unittest

import numpy as np
import pandas as pd

from vpp.generation_forecast.aws_features import (
    AWS_SOLAR_FEATURES,
    AWS_WIND_FEATURES,
    fit_aws_solar_calibrator,
    fit_aws_wind_calibrator,
)
from vpp.generation_forecast.ensemble import optimize_nonnegative_weights
from vpp.generation_forecast.improvement import (
    SOLAR_IMPROVEMENT_FEATURES,
    WIND_IMPROVEMENT_FEATURES,
    add_solar_improvement_features,
    add_wind_improvement_features,
    enforce_portfolio_physics,
    portfolio_frame,
)
from vpp.generation_forecast.model import (
    add_wind_change_features,
    error_metrics,
    prepare_frame,
    training_sample_weights,
)
from vpp.generation_forecast.tune_lightgbm import (
    TimeFold,
    aggregate_results,
    select_jointly_stable_candidates,
    split_fold,
)
from vpp.generation_forecast.wind_direction_features import add_wind_direction_interactions
from vpp.generation_forecast.wind_hybrid import DirectionalWindPowerCurve, WindPowerCurve
from vpp.generation_forecast.wind_postprocessing import smooth_wind_prediction
from vpp.generation_forecast.wind_regime import wind_speed_regime
from vpp.generation_forecast.wind_uncertainty import (
    WindResidualQuantiles,
    weighted_quantile,
)


class TrainGenerationModelsTests(unittest.TestCase):
    def test_wind_smoothing_stays_inside_issue_trajectory(self) -> None:
        frame = pd.DataFrame(
            {
                "asset_id": ["a", "a", "a", "a"],
                "issue_time_utc": pd.to_datetime(
                    ["2024-01-01"] * 3 + ["2024-01-02"], utc=True
                ),
                "lead_hour": [1, 2, 3, 1],
            }
        )
        result = smooth_wind_prediction(
            frame, np.array([0.0, 1.0, 0.0, 0.8]), "mean3", 0.25
        )
        np.testing.assert_allclose(result[:3], [0.125, 5.0 / 6.0, 0.125])
        self.assertAlmostEqual(float(result[3]), 0.8)

    def test_wind_uncertainty_is_ordered_and_contains_point(self) -> None:
        calibrator = WindResidualQuantiles(
            grouped={(0, 0): np.array([-0.2, -0.01, 0.1])},
            by_offset={},
            global_quantiles=np.array([-0.3, 0.0, 0.3]),
        )
        result = calibrator.predict(
            np.array([0.15, 0.9]), np.array([0, 3]), np.array([2.0, 8.0])
        )
        np.testing.assert_allclose(result[:, 1], [0.15, 0.9])
        self.assertTrue((result[:, 0] <= result[:, 1]).all())
        self.assertTrue((result[:, 1] <= result[:, 2]).all())
        self.assertTrue((result >= 0.0).all() and (result <= 1.0).all())

    def test_weighted_quantile_respects_heavy_observation(self) -> None:
        result = weighted_quantile(
            np.array([0.0, 1.0]), (0.5,), np.array([100.0, 1.0])
        )
        self.assertLess(float(result[0]), 0.1)

    def test_wind_direction_interactions_create_expected_sector(self) -> None:
        frame = pd.DataFrame(
            {
                "aws_proxy_wind_direction_sin": [0.0, 1.0],
                "aws_proxy_wind_direction_cos": [1.0, 0.0],
                "aws_proxy_wind_speed_ms": [2.0, 3.0],
                "wind_direction_sin": [0.0, 1.0],
                "wind_direction_cos": [1.0, 0.0],
            }
        )
        result = add_wind_direction_interactions(frame)
        self.assertEqual(result["aws_proxy_wind_sector_8"].tolist(), [0, 2])
        self.assertTrue(np.allclose(result["gfs_aws_direction_alignment"], 1.0))

    def test_wind_speed_regime_respects_boundaries(self) -> None:
        result = wind_speed_regime(
            np.array([0.0, 2.99, 3.0, 6.99, 7.0, 10.0]), (3.0, 7.0)
        )
        self.assertEqual(result.tolist(), [0, 0, 1, 1, 2, 2])

    def test_error_metrics(self) -> None:
        result = error_metrics(
            np.array([10.0, 20.0]),
            np.array([12.0, 16.0]),
            np.array([100.0, 100.0]),
        )
        self.assertEqual(result["observations"], 2)
        self.assertAlmostEqual(result["mae_kwh"], 3.0)
        self.assertAlmostEqual(result["rmse_kwh"], np.sqrt(10.0))
        self.assertAlmostEqual(result["nmae_pct"], 3.0)
        self.assertAlmostEqual(result["mae_capacity_factor"], 0.03)

    def test_prepare_frame_keeps_locked_test_split(self) -> None:
        frame = pd.DataFrame(
            {
                "label_is_valid": [True, True, True, False],
                "dataset_split": ["train", "validation", "test", "test"],
                "asset_id": ["a", "a", "a", "a"],
                "gfs_grid_id": ["g", "g", "g", "g"],
            }
        )
        prepared, maps = prepare_frame(frame)
        self.assertEqual(prepared["dataset_split"].tolist(), ["train", "validation", "test"])
        self.assertEqual(maps["asset_id"], {"a": 0})

    def test_weighted_ensemble_weights_are_nonnegative_and_sum_to_one(self) -> None:
        predictions = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        actual = np.array([0.1, 0.2, 0.3])
        weights = optimize_nonnegative_weights(predictions, actual, np.ones(3))
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], 0.99)

    def test_tuning_fold_keeps_valid_times_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "valid_time_kst": pd.to_datetime(
                    ["2023-06-30 23:00", "2023-07-01 00:00", "2023-09-30 23:00"]
                ).tz_localize("Asia/Seoul"),
                "capacity_kw": [1.0, 1.0, 1.0],
            }
        )
        train, validation = split_fold(frame, TimeFold("test", "2023-07-01", "2023-10-01"))
        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 2)
        self.assertTrue(set(train["valid_time_kst"]).isdisjoint(validation["valid_time_kst"]))

    def test_capacity_sample_weights(self) -> None:
        frame = pd.DataFrame({"capacity_kw": [4.0, 9.0]})
        np.testing.assert_array_equal(training_sample_weights(frame, "capacity"), [4.0, 9.0])
        np.testing.assert_array_equal(training_sample_weights(frame, "sqrt_capacity"), [2.0, 3.0])
        self.assertIsNone(training_sample_weights(frame, "none"))

    def test_wind_changes_do_not_cross_forecast_issue_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "asset_id": ["a", "a", "a", "a"],
                "issue_time_utc": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                    utc=True,
                ),
                "lead_hour": [1, 2, 1, 2],
                "wind_speed_80m_ms": [3.0, 5.0, 10.0, 9.0],
                "wind_direction_80m_deg": [350.0, 10.0, 90.0, 80.0],
            }
        )
        featured = add_wind_change_features(frame)
        np.testing.assert_allclose(featured["wind_speed_change_1h_ms"], [0.0, 2.0, 0.0, -1.0])
        self.assertAlmostEqual(
            featured.loc[1, "wind_direction_change_1h_sin"],
            np.sin(np.deg2rad(20.0)),
        )
        self.assertEqual(featured.loc[2, "wind_direction_change_1h_sin"], 0.0)

    def test_wind_power_curve_is_monotonic_and_bounded(self) -> None:
        frame = pd.DataFrame(
            {
                "asset_id": ["a"] * 6,
                "wind_speed_80m_ms": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "capacity_factor": [0.0, 0.1, 0.3, 0.25, 0.8, 1.1],
            }
        )
        curve = WindPowerCurve.fit(frame)
        prediction = curve.predict(frame)
        self.assertTrue((np.diff(prediction) >= 0.0).all())
        self.assertGreaterEqual(float(prediction.min()), 0.0)
        self.assertLessEqual(float(prediction.max()), 1.0)

    def test_directional_power_curve_falls_back_for_sparse_sector(self) -> None:
        frame = pd.DataFrame(
            {
                "asset_id": ["a"] * 8,
                "wind_speed_80m_ms": np.arange(1.0, 9.0),
                "wind_direction_80m_deg": [0.0] * 7 + [180.0],
                "capacity_factor": np.linspace(0.0, 0.8, 8),
            }
        )
        curve = DirectionalWindPowerCurve.fit(frame, minimum_rows=3)
        prediction = curve.predict(frame)
        self.assertEqual(len(prediction), len(frame))
        self.assertTrue(np.isfinite(prediction).all())

    def test_solar_features_use_only_forecast_trajectory(self) -> None:
        frame = pd.DataFrame(
            {
                "asset_id": ["a", "a", "b", "b"],
                "gfs_grid_id": ["g1", "g1", "g2", "g2"],
                "issue_time_utc": pd.to_datetime(["2024-01-01"] * 4, utc=True),
                "valid_time_utc": pd.to_datetime(
                    ["2024-01-01 01:00", "2024-01-01 02:00"] * 2, utc=True
                ),
                "lead_hour": [1, 2, 1, 2],
                "solar_elevation_deg": [30.0, 40.0, 30.0, 40.0],
                "shortwave_radiation_w_m2": [100.0, 160.0, 200.0, 280.0],
                "total_cloud_cover_pct": [20.0, 30.0, 40.0, 50.0],
            }
        )
        featured = add_solar_improvement_features(frame)
        self.assertFalse(featured[SOLAR_IMPROVEMENT_FEATURES].isna().any().any())
        self.assertEqual(featured.loc[0, "shortwave_change_1h_w_m2"], 0.0)
        self.assertEqual(featured.loc[1, "shortwave_change_1h_w_m2"], 60.0)
        self.assertEqual(featured.loc[0, "regional_shortwave_mean_w_m2"], 150.0)

    def test_portfolio_frame_sums_generation_and_capacity(self) -> None:
        frame = pd.DataFrame(
            {
                "issue_time_utc": pd.to_datetime(["2024-01-01"] * 2, utc=True),
                "valid_time_utc": pd.to_datetime(["2024-01-01 01:00"] * 2, utc=True),
                "valid_time_kst": pd.to_datetime(["2024-01-01 10:00"] * 2).tz_localize(
                    "Asia/Seoul"
                ),
                "dataset_split": ["validation", "validation"],
                "asset_id": ["a", "b"],
                "generation_kwh": [10.0, 30.0],
                "capacity_kw": [100.0, 300.0],
                "lead_hour": [1, 1],
                "forecast_day_offset": [0, 0],
                "hour_sin": [0.1, 0.1],
                "hour_cos": [0.9, 0.9],
                "day_of_year_sin": [0.2, 0.2],
                "day_of_year_cos": [0.8, 0.8],
                "weather": [1.0, 3.0],
            }
        )
        portfolio, features = portfolio_frame(frame, ["weather"])
        self.assertEqual(portfolio.loc[0, "generation_kwh"], 40.0)
        self.assertEqual(portfolio.loc[0, "capacity_kw"], 400.0)
        self.assertAlmostEqual(portfolio.loc[0, "capacity_factor"], 0.1)
        self.assertIn("weather_portfolio_std", features)

    def test_advanced_wind_features_use_trajectory_and_unique_grids(self) -> None:
        frame = pd.DataFrame(
            {
                "asset_id": ["a", "a", "b", "b"],
                "gfs_grid_id": ["g1", "g1", "g2", "g2"],
                "issue_time_utc": pd.to_datetime(["2024-01-01"] * 4, utc=True),
                "valid_time_utc": pd.to_datetime(
                    ["2024-01-01 01:00", "2024-01-01 02:00"] * 2, utc=True
                ),
                "lead_hour": [1, 2, 1, 2],
                "wind_speed_80m_ms": [3.0, 5.0, 7.0, 8.0],
                "wind_u_80m_ms": [1.0, 2.0, 3.0, 4.0],
                "wind_v_80m_ms": [2.0, 3.0, 4.0, 5.0],
            }
        )
        featured = add_wind_improvement_features(frame)
        self.assertFalse(featured[WIND_IMPROVEMENT_FEATURES].isna().any().any())
        self.assertAlmostEqual(featured.loc[1, "wind_speed_neighbor_mean_3h_ms"], 13.0 / 3.0)
        self.assertEqual(featured.loc[0, "regional_wind_speed_mean_ms"], 5.0)
        self.assertEqual(featured.loc[0, "wind_speed_minus_regional_mean_ms"], -2.0)

    def test_portfolio_solar_physics_forces_night_to_zero(self) -> None:
        frame = pd.DataFrame(
            {
                "is_daylight_portfolio_mean": [0.0, 1.0],
                "shortwave_radiation_w_m2_portfolio_mean": [0.0, 300.0],
            }
        )
        prediction = enforce_portfolio_physics("solar", frame, np.array([0.2, 0.4]))
        np.testing.assert_allclose(prediction, [0.0, 0.4])

    def test_tuning_ranking_penalizes_instability(self) -> None:
        folds = pd.DataFrame(
            {
                "energy_type": ["solar"] * 4,
                "candidate": ["stable", "stable", "unstable", "unstable"],
                "weight_mode": ["none"] * 4,
                "fold": ["a", "b", "a", "b"],
                "best_iteration": [10, 12, 10, 12],
                "portfolio_nmae_pct": [4.0, 4.0, 3.0, 5.0],
            }
        )
        ranking = aggregate_results(folds)
        self.assertEqual(ranking.iloc[0]["candidate"], "stable")

    def test_joint_selection_rejects_candidate_that_worsens_full_holdout(self) -> None:
        ranking = pd.DataFrame(
            {
                "energy_type": ["solar", "solar", "solar"],
                "candidate": ["current", "rolling_only", "safe"],
                "selection_score": [4.0, 3.8, 3.9],
                "mean_nmae_pct": [3.8, 3.7, 3.75],
                "std_nmae_pct": [0.8, 0.4, 0.6],
                "worst_nmae_pct": [4.5, 4.3, 4.4],
                "median_best_iteration": [100, 120, 110],
                "folds": [4, 4, 4],
            }
        )
        holdout = pd.DataFrame(
            {
                "energy_type": ["solar", "solar", "solar"],
                "candidate": ["current", "rolling_only", "safe"],
                "portfolio_nmae_pct": [3.8, 3.9, 3.75],
            }
        )
        selected = select_jointly_stable_candidates(ranking, holdout)
        self.assertEqual(selected.iloc[0]["candidate"], "safe")

    def test_aws_calibrator_predicts_without_validation_observations(self) -> None:
        count = 24 * 31
        times = pd.date_range("2023-01-01", periods=count, freq="h", tz="Asia/Seoul")
        speed = 4.0 + np.sin(np.arange(count) / 24.0)
        frame = pd.DataFrame(
            {
                "asset_id": ["wind_hangyeong"] * count,
                "valid_time_kst": times,
                "wind_u_80m_ms": speed,
                "wind_v_80m_ms": speed / 2,
                "wind_speed_80m_ms": np.hypot(speed, speed / 2),
                "wind_direction_sin": np.full(count, -0.8),
                "wind_direction_cos": np.full(count, -0.4),
                "temperature_2m_c": np.full(count, 15.0),
                "total_cloud_cover_pct": np.full(count, 40.0),
                "hour_sin": np.sin(2 * np.pi * times.hour / 24),
                "hour_cos": np.cos(2 * np.pi * times.hour / 24),
                "day_of_year_sin": np.sin(2 * np.pi * times.day_of_year / 365.25),
                "day_of_year_cos": np.cos(2 * np.pi * times.day_of_year / 365.25),
            }
        )
        observations = pd.DataFrame(
            {
                "station_id": [990] * count,
                "valid_time_kst": times,
                "wind_speed_ms": speed * 0.7,
                "aws_wind_u_ms": speed * 0.6,
                "aws_wind_v_ms": speed * 0.3,
            }
        )
        calibrator = fit_aws_wind_calibrator(
            frame, observations, {"wind_hangyeong": (880, 990)}
        )
        self.assertEqual(calibrator.station_map["wind_hangyeong"], (990,))
        future = frame.tail(3).copy()
        future["valid_time_kst"] = future["valid_time_kst"] + pd.Timedelta(days=365)
        transformed = calibrator.transform(future)
        self.assertFalse(transformed[AWS_WIND_FEATURES].isna().any().any())
        self.assertTrue((transformed["aws_proxy_wind_speed_ms"] >= 0).all())

    def test_aws_solar_calibrator_predicts_without_future_observations(self) -> None:
        count = 24 * 31
        times = pd.date_range("2023-01-01", periods=count, freq="h", tz="Asia/Seoul")
        hour = times.hour.to_numpy()
        radiation = np.maximum(np.sin(np.pi * (hour - 6) / 12), 0.0) * 600.0
        frame = pd.DataFrame(
            {
                "asset_id": ["solar_pyoseon"] * count,
                "valid_time_kst": times,
                "shortwave_radiation_w_m2": radiation,
                "temperature_2m_c": 12.0 + radiation / 100.0,
                "total_cloud_cover_pct": np.full(count, 40.0),
                "solar_elevation_deg": np.maximum(
                    45.0 * np.sin(np.pi * (hour - 6) / 12), 0.0
                ),
                "is_daylight": (radiation > 0).astype("int8"),
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "day_of_year_sin": np.sin(2 * np.pi * times.day_of_year / 365.25),
                "day_of_year_cos": np.cos(2 * np.pi * times.day_of_year / 365.25),
                "lead_hour": np.resize(np.arange(1, 49), count),
            }
        )
        observations = pd.DataFrame(
            {
                "station_id": [792] * count,
                "valid_time_kst": times,
                "temperature_c": frame["temperature_2m_c"] - 1.5,
                "humidity_pct": 75.0 - radiation / 30.0,
            }
        )
        calibrator = fit_aws_solar_calibrator(
            frame, observations, {"solar_pyoseon": (792,)}
        )
        future = frame.tail(3).copy()
        future["valid_time_kst"] += pd.Timedelta(days=365)
        transformed = calibrator.transform(future)
        self.assertFalse(transformed[AWS_SOLAR_FEATURES].isna().any().any())
        self.assertTrue(transformed["aws_proxy_humidity_pct"].between(1, 100).all())


if __name__ == "__main__":
    unittest.main()
