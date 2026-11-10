"""Train leakage-safe solar and wind generation baselines and LightGBM models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp.generation_forecast.paths import MODEL_DIR, PREDICTION_DIR, PROCESSED_DIR, RESULTS_DIR

DEFAULT_DATA = PROCESSED_DIR
DEFAULT_MODELS = MODEL_DIR
DEFAULT_PREDICTIONS = PREDICTION_DIR
DEFAULT_REPORTS = RESULTS_DIR
TARGET = "capacity_factor"
RANDOM_SEED = 42

WIND_CHANGE_FEATURES = [
    "wind_speed_change_1h_ms",
    "wind_speed_change_3h_ms",
    "wind_direction_change_1h_sin",
    "wind_direction_change_1h_cos",
    "wind_direction_change_3h_sin",
    "wind_direction_change_3h_cos",
]

FEATURES = {
    "solar": [
        "asset_code",
        "grid_code",
        "capacity_kw",
        "lead_hour",
        "forecast_day_offset",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "shortwave_radiation_w_m2",
        "temperature_2m_c",
        "total_cloud_cover_pct",
        "solar_elevation_deg",
        "is_daylight",
    ],
    "wind": [
        "asset_code",
        "grid_code",
        "capacity_kw",
        "lead_hour",
        "forecast_day_offset",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "wind_u_80m_ms",
        "wind_v_80m_ms",
        "wind_speed_80m_ms",
        "wind_direction_sin",
        "wind_direction_cos",
        "temperature_2m_c",
        "total_cloud_cover_pct",
        *WIND_CHANGE_FEATURES,
    ],
}
CATEGORICAL_FEATURES = ["asset_code", "grid_code"]
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    # Selected using 2023-2024 walk-forward folds and the full 2024 validation period.
    "solar": {"weight_mode": "sqrt_capacity", "params": {}},
    "wind": {"weight_mode": "none", "params": {}},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--estimators", type=int, default=1200)
    parser.add_argument("--early-stopping", type=int, default=80)
    return parser.parse_args()


def add_wind_change_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add forecast-trajectory changes without crossing issue-time boundaries."""
    original_index = frame.index
    ordered = frame.sort_values(["asset_id", "issue_time_utc", "lead_hour"]).copy()
    group_keys = [ordered["asset_id"], ordered["issue_time_utc"]]
    groups = ordered.groupby(group_keys, observed=True, sort=False)
    direction = np.deg2rad(ordered["wind_direction_80m_deg"].to_numpy(dtype=float))
    for hours in (1, 3):
        previous_speed = groups["wind_speed_80m_ms"].shift(hours)
        previous_lead = groups["lead_hour"].shift(hours)
        exact_interval = ordered["lead_hour"].sub(previous_lead).eq(hours)
        speed_change = ordered["wind_speed_80m_ms"] - previous_speed
        ordered[f"wind_speed_change_{hours}h_ms"] = speed_change.where(exact_interval, 0.0)
        previous_direction = (
            pd.Series(direction, index=ordered.index)
            .groupby(group_keys, observed=True, sort=False)
            .shift(hours)
        )
        delta = np.arctan2(
            np.sin(direction - previous_direction.to_numpy(dtype=float)),
            np.cos(direction - previous_direction.to_numpy(dtype=float)),
        )
        missing = (~exact_interval | previous_direction.isna()).to_numpy()
        ordered[f"wind_direction_change_{hours}h_sin"] = np.where(missing, 0.0, np.sin(delta))
        ordered[f"wind_direction_change_{hours}h_cos"] = np.where(missing, 1.0, np.cos(delta))
    return ordered.loc[original_index]


def prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    result = frame[frame["label_is_valid"]].copy()
    result = result[result["dataset_split"].isin(["train", "validation", "test"])].copy()
    category_maps: dict[str, dict[str, int]] = {}
    for source, target in [("asset_id", "asset_code"), ("gfs_grid_id", "grid_code")]:
        values = sorted(result[source].astype(str).unique())
        mapping = {value: index for index, value in enumerate(values)}
        result[target] = result[source].astype(str).map(mapping).astype("int16")
        category_maps[source] = mapping
    if "wind_direction_80m_deg" in result:
        radians = np.deg2rad(result["wind_direction_80m_deg"])
        result["wind_direction_sin"] = np.sin(radians)
        result["wind_direction_cos"] = np.cos(radians)
        result = add_wind_change_features(result)
    if "is_daylight" in result:
        result["is_daylight"] = result["is_daylight"].astype("int8")
    return result, category_maps


def climatology_prediction(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    detailed_keys = ["asset_id", "month_kst", "hour_kst"]
    detailed = train.groupby(detailed_keys, observed=True)[TARGET].mean()
    detailed_index = pd.MultiIndex.from_frame(validation[detailed_keys])
    prediction = detailed.reindex(detailed_index).to_numpy(dtype=float)

    missing = np.isnan(prediction)
    if missing.any():
        fallback_keys = ["asset_id", "hour_kst"]
        fallback = train.groupby(fallback_keys, observed=True)[TARGET].mean()
        fallback_index = pd.MultiIndex.from_frame(validation.loc[missing, fallback_keys])
        prediction[missing] = fallback.reindex(fallback_index).to_numpy(dtype=float)
    missing = np.isnan(prediction)
    if missing.any():
        by_asset = train.groupby("asset_id", observed=True)[TARGET].mean()
        prediction[missing] = (
            validation.loc[missing, "asset_id"].map(by_asset).to_numpy(dtype=float)
        )
    if np.isnan(prediction).any():
        prediction[np.isnan(prediction)] = float(train[TARGET].mean())
    return np.clip(prediction, 0.0, 1.0)


def make_model(estimators: int, **overrides: Any) -> Any:
    import lightgbm as lgb

    params = {
        "objective": "regression_l1",
        "n_estimators": estimators,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    params.update(overrides)
    return lgb.LGBMRegressor(**params)


def training_sample_weights(frame: pd.DataFrame, mode: str) -> np.ndarray | None:
    """Return label-safe training weights aligned with the portfolio objective."""
    if mode == "none":
        return None
    capacity = frame["capacity_kw"].to_numpy(dtype=float)
    if mode == "capacity":
        return capacity
    if mode == "sqrt_capacity":
        return np.sqrt(capacity)
    raise ValueError(f"unknown weight mode: {mode}")


def enforce_physics(energy_type: str, frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    if energy_type == "solar":
        night = frame["is_daylight"].eq(0) | frame["shortwave_radiation_w_m2"].le(0)
        result[night.to_numpy()] = 0.0
    return result


def error_metrics(
    actual_kwh: np.ndarray, predicted_kwh: np.ndarray, capacity_kw: np.ndarray
) -> dict:
    error = predicted_kwh - actual_kwh
    return {
        "observations": int(len(error)),
        "mae_kwh": float(np.mean(np.abs(error))),
        "rmse_kwh": float(np.sqrt(np.mean(np.square(error)))),
        "nmae_pct": float(100.0 * np.sum(np.abs(error)) / np.sum(capacity_kw)),
        "mae_capacity_factor": float(np.mean(np.abs(error / capacity_kw))),
    }


def metric_rows(
    frame: pd.DataFrame,
    prediction_column: str,
    model_name: str,
    energy_type: str,
    expected_assets: int,
) -> list[dict]:
    rows: list[dict] = []
    horizons: list[tuple[str, pd.Series]] = [
        ("all_48h", pd.Series(True, index=frame.index)),
        ("day0_24h", frame["forecast_day_offset"].eq(0)),
        ("day1_24h", frame["forecast_day_offset"].eq(1)),
    ]
    for horizon, mask in horizons:
        selected = frame.loc[mask]
        metrics = error_metrics(
            selected["generation_kwh"].to_numpy(),
            selected[prediction_column].to_numpy(),
            selected["capacity_kw"].to_numpy(),
        )
        rows.append(
            {
                "energy_type": energy_type,
                "model": model_name,
                "level": "plant_row",
                "horizon": horizon,
                **metrics,
                "complete_portfolio_windows": None,
            }
        )

        grouped = (
            selected.groupby(["issue_time_utc", "valid_time_utc"], observed=True)
            .agg(
                actual_kwh=("generation_kwh", "sum"),
                predicted_kwh=(prediction_column, "sum"),
                capacity_kw=("capacity_kw", "sum"),
                asset_count=("asset_id", "nunique"),
            )
            .reset_index()
        )
        complete = grouped[grouped["asset_count"].eq(expected_assets)]
        portfolio_metrics = error_metrics(
            complete["actual_kwh"].to_numpy(),
            complete["predicted_kwh"].to_numpy(),
            complete["capacity_kw"].to_numpy(),
        )
        rows.append(
            {
                "energy_type": energy_type,
                "model": model_name,
                "level": "portfolio",
                "horizon": horizon,
                **portfolio_metrics,
                "complete_portfolio_windows": int(len(complete)),
            }
        )
    return rows


def asset_metrics(frame: pd.DataFrame, prediction_column: str, model_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    for asset_id, group in frame.groupby("asset_id", observed=True):
        metrics = error_metrics(
            group["generation_kwh"].to_numpy(),
            group[prediction_column].to_numpy(),
            group["capacity_kw"].to_numpy(),
        )
        rows.append(
            {
                "asset_id": asset_id,
                "plant_name": group["plant_name"].iloc[0],
                "energy_type": group["energy_type"].iloc[0],
                "model": model_name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def add_model_predictions(
    energy_type: str,
    frame: pd.DataFrame,
    model: Any,
    features: list[str],
    train_for_baseline: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["baseline_capacity_factor"] = enforce_physics(
        energy_type, result, climatology_prediction(train_for_baseline, result)
    )
    result["lightgbm_capacity_factor"] = enforce_physics(
        energy_type,
        result,
        model.predict(result[features], num_iteration=model.best_iteration_),
    )
    for name in ["baseline", "lightgbm"]:
        result[f"{name}_generation_kwh"] = result[f"{name}_capacity_factor"] * result["capacity_kw"]
    return result


def prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "asset_id",
        "plant_name",
        "energy_type",
        "capacity_kw",
        "gfs_grid_id",
        "issue_time_utc",
        "issue_time_kst",
        "valid_time_utc",
        "valid_time_kst",
        "lead_hour",
        "forecast_day_offset",
        "dataset_split",
        "generation_kwh",
        "capacity_factor",
        "baseline_capacity_factor",
        "baseline_generation_kwh",
        "lightgbm_capacity_factor",
        "lightgbm_generation_kwh",
    ]
    return frame[columns].sort_values(["issue_time_utc", "valid_time_utc", "asset_id"])


def train_one(
    energy_type: str,
    data_path: Path,
    model_dir: Path,
    prediction_dir: Path,
    report_dir: Path,
    estimators: int,
    early_stopping_rounds: int,
) -> tuple[list[dict], pd.DataFrame, dict]:
    import joblib
    import lightgbm as lgb

    raw = pd.read_parquet(data_path)
    frame, category_maps = prepare_frame(raw)
    train = frame[frame["dataset_split"].eq("train")].copy()
    validation = frame[frame["dataset_split"].eq("validation")].copy()
    test = frame[frame["dataset_split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"{energy_type}: train, validation, or test split is empty")
    features = FEATURES[energy_type]
    if any(part[features].isna().any().any() for part in (train, validation, test)):
        raise ValueError(f"{energy_type}: feature matrix contains missing values")

    config = MODEL_CONFIGS[energy_type]
    selection_model = make_model(estimators, **config["params"])
    selection_model.fit(
        train[features],
        train[TARGET],
        sample_weight=training_sample_weights(train, config["weight_mode"]),
        categorical_feature=CATEGORICAL_FEATURES,
        eval_X=validation[features],
        eval_y=validation[TARGET],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    validation = add_model_predictions(energy_type, validation, selection_model, features, train)
    # The 2025 test prediction is made before any 2025 result is used for model changes.
    test = add_model_predictions(energy_type, test, selection_model, features, train)

    expected_assets = int(validation["asset_id"].nunique())
    metrics: list[dict] = []
    assets: list[pd.DataFrame] = []
    for split_name, evaluated in [("validation", validation), ("test", test)]:
        for name in ["baseline", "lightgbm"]:
            prediction = f"{name}_generation_kwh"
            split_metrics = metric_rows(evaluated, prediction, name, energy_type, expected_assets)
            for row in split_metrics:
                row["dataset_split"] = split_name
            metrics.extend(split_metrics)
            asset_frame = asset_metrics(evaluated, prediction, name)
            asset_frame["dataset_split"] = split_name
            assets.append(asset_frame)
        prediction_frame(evaluated).to_parquet(
            prediction_dir / f"{energy_type}_{split_name}_predictions.parquet",
            index=False,
            compression="zstd",
        )

    importance = pd.DataFrame(
        {
            "feature": features,
            "gain": selection_model.booster_.feature_importance(importance_type="gain"),
            "split": selection_model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(
        report_dir / f"{energy_type}_feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    best_iteration = int(selection_model.best_iteration_)
    selection_bundle = {
        "model": selection_model,
        "energy_type": energy_type,
        "features": features,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_maps": category_maps,
        "target": TARGET,
        "clip_range": [0.0, 1.0],
        "solar_night_zero_rule": energy_type == "solar",
        "training_split": "2023",
        "selection_split": "2024",
        "final_test_split": "2025",
    }
    joblib.dump(selection_bundle, model_dir / f"{energy_type}_lightgbm_selection.joblib")

    # Only after the locked 2025 prediction has been produced, refit the operational model
    # on 2023-2024 with the iteration count selected exclusively on 2024.
    development = pd.concat([train, validation], ignore_index=True)
    production_model = make_model(best_iteration, **config["params"])
    production_model.fit(
        development[features],
        development[TARGET],
        sample_weight=training_sample_weights(development, config["weight_mode"]),
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    production_bundle = {
        **selection_bundle,
        "model": production_model,
        "best_iteration": best_iteration,
        "training_split": "2023-2024",
        "selection_model_path": f"{energy_type}_lightgbm_selection.joblib",
    }
    joblib.dump(production_bundle, model_dir / f"{energy_type}_lightgbm.joblib")

    metadata = {
        "energy_type": energy_type,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_valid_time_start": train["valid_time_kst"].min().isoformat(),
        "train_valid_time_end": train["valid_time_kst"].max().isoformat(),
        "validation_valid_time_start": validation["valid_time_kst"].min().isoformat(),
        "validation_valid_time_end": validation["valid_time_kst"].max().isoformat(),
        "test_valid_time_start": test["valid_time_kst"].min().isoformat(),
        "test_valid_time_end": test["valid_time_kst"].max().isoformat(),
        "assets": expected_assets,
        "features": features,
        "best_iteration": best_iteration,
        "best_validation_l1": float(selection_model.best_score_["valid_0"]["l1"]),
        "final_test_was_predicted_before_production_refit": True,
        "production_training_years": [2023, 2024],
        "weight_mode": config["weight_mode"],
        "tuned_params": config["params"],
        "tuning_years": [2023, 2024],
        "locked_test_used_for_tuning": False,
    }
    return metrics, pd.concat(assets, ignore_index=True), metadata


def main() -> None:
    args = parse_args()
    for output_dir in (args.models, args.predictions, args.reports):
        output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    all_asset_metrics: list[pd.DataFrame] = []
    metadata: dict[str, dict] = {}
    for energy_type in ["solar", "wind"]:
        metrics, asset_frame, details = train_one(
            energy_type,
            args.data / f"{energy_type}_training_mart.parquet",
            args.models,
            args.predictions,
            args.reports,
            args.estimators,
            args.early_stopping,
        )
        all_metrics.extend(metrics)
        all_asset_metrics.append(asset_frame)
        metadata[energy_type] = details
    metric_frame = pd.DataFrame(all_metrics)
    metric_frame.to_csv(
        args.reports / "generation_model_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_asset_metrics, ignore_index=True).to_csv(
        args.reports / "generation_asset_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "random_seed": RANDOM_SEED,
        "split_rule": "valid_time year: 2023=train, 2024=validation, 2025=test",
        "validation_is_for_model_selection_not_final_test": True,
        "test_is_locked_2025_evaluation": True,
        "production_refit_uses_2023_2024_only": True,
        "models": metadata,
        "metrics": metric_frame.to_dict("records"),
    }
    (args.reports / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metric_frame.to_string(index=False))


if __name__ == "__main__":
    main()
