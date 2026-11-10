"""Train leakage-safe weighted and stacked generation forecast ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from vpp.generation_forecast.model import (
    CATEGORICAL_FEATURES,
    FEATURES,
    RANDOM_SEED,
    TARGET,
    asset_metrics,
    enforce_physics,
    metric_rows,
    prepare_frame,
)
from vpp.generation_forecast.paths import MODEL_DIR, PREDICTION_DIR, PROCESSED_DIR, RESULTS_DIR

DEFAULT_DATA = PROCESSED_DIR
DEFAULT_MODELS = MODEL_DIR
DEFAULT_PREDICTIONS = PREDICTION_DIR
DEFAULT_REPORTS = RESULTS_DIR
BASE_MODEL_NAMES = ["lightgbm", "hist_gradient_boosting", "extra_trees"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--extra-trees", type=int, default=160)
    parser.add_argument("--hist-iterations", type=int, default=250)
    return parser.parse_args()


def make_hist_model(iterations: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.06,
        max_iter=iterations,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=0.2,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def make_extra_trees_model(n_estimators: int) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=50,
        max_features=0.8,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def optimize_nonnegative_weights(
    predictions: np.ndarray,
    actual_capacity_factor: np.ndarray,
    capacity_kw: np.ndarray,
) -> np.ndarray:
    """Minimize capacity-weighted MAE subject to nonnegative weights summing to one."""
    matrix = np.asarray(predictions, dtype=float)
    actual = np.asarray(actual_capacity_factor, dtype=float)
    capacity = np.asarray(capacity_kw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(actual):
        raise ValueError("predictions must be a rows-by-models matrix")
    if matrix.shape[1] < 2:
        raise ValueError("at least two base models are required")
    if len(capacity) != len(actual) or np.any(capacity <= 0):
        raise ValueError("capacity_kw must be positive and match prediction rows")

    def objective(weights: np.ndarray) -> float:
        return float(np.average(np.abs(matrix @ weights - actual), weights=capacity))

    initial = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"ensemble weight optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def base_prediction_matrix(
    energy_type: str,
    frame: pd.DataFrame,
    features: list[str],
    models: dict[str, Any],
) -> np.ndarray:
    columns = []
    for name in BASE_MODEL_NAMES:
        model = models[name]
        kwargs = {"num_iteration": model.best_iteration_} if name == "lightgbm" else {}
        prediction = model.predict(frame[features], **kwargs)
        columns.append(enforce_physics(energy_type, frame, prediction))
    return np.column_stack(columns)


def add_ensemble_predictions(
    energy_type: str,
    frame: pd.DataFrame,
    base_predictions: np.ndarray,
    weights: np.ndarray,
    stacker: Any,
) -> pd.DataFrame:
    result = frame.copy()
    for index, name in enumerate(BASE_MODEL_NAMES):
        result[f"{name}_capacity_factor"] = base_predictions[:, index]
    result["weighted_ensemble_capacity_factor"] = enforce_physics(
        energy_type, result, base_predictions @ weights
    )
    result["stacking_capacity_factor"] = enforce_physics(
        energy_type, result, stacker.predict(base_predictions)
    )
    for name in [*BASE_MODEL_NAMES, "weighted_ensemble", "stacking"]:
        result[f"{name}_generation_kwh"] = result[f"{name}_capacity_factor"] * result["capacity_kw"]
    return result


def output_predictions(frame: pd.DataFrame, path: Path) -> None:
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
        *[f"{name}_generation_kwh" for name in BASE_MODEL_NAMES],
        "weighted_ensemble_generation_kwh",
        "stacking_generation_kwh",
    ]
    frame[columns].sort_values(["issue_time_utc", "valid_time_utc", "asset_id"]).to_parquet(
        path, index=False, compression="zstd"
    )


def train_one(
    energy_type: str,
    data_path: Path,
    model_dir: Path,
    prediction_dir: Path,
    extra_trees: int,
    hist_iterations: int,
) -> tuple[list[dict], pd.DataFrame, dict]:
    import joblib
    from sklearn.linear_model import RidgeCV

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

    selection_bundle = joblib.load(model_dir / f"{energy_type}_lightgbm_selection.joblib")
    bundle_mismatch = (
        selection_bundle["features"] != features
        or selection_bundle["category_maps"] != category_maps
    )
    if bundle_mismatch:
        raise ValueError(f"{energy_type}: LightGBM bundle does not match current training mart")
    selection_models = {
        "lightgbm": selection_bundle["model"],
        "hist_gradient_boosting": make_hist_model(hist_iterations),
        "extra_trees": make_extra_trees_model(extra_trees),
    }
    for name in ["hist_gradient_boosting", "extra_trees"]:
        selection_models[name].fit(train[features], train[TARGET])

    validation_base = base_prediction_matrix(energy_type, validation, features, selection_models)
    test_base = base_prediction_matrix(energy_type, test, features, selection_models)
    weights = optimize_nonnegative_weights(
        validation_base,
        validation[TARGET].to_numpy(),
        validation["capacity_kw"].to_numpy(),
    )
    stacker = RidgeCV(alphas=np.logspace(-6, 2, 17))
    stacker.fit(
        validation_base,
        validation[TARGET],
        sample_weight=validation["capacity_kw"],
    )
    validation = add_ensemble_predictions(
        energy_type, validation, validation_base, weights, stacker
    )
    test = add_ensemble_predictions(energy_type, test, test_base, weights, stacker)

    expected_assets = int(validation["asset_id"].nunique())
    metrics: list[dict] = []
    assets: list[pd.DataFrame] = []
    evaluated_models = [*BASE_MODEL_NAMES, "weighted_ensemble", "stacking"]
    for split_name, evaluated in [("validation_meta_fit", validation), ("test", test)]:
        for name in evaluated_models:
            column = f"{name}_generation_kwh"
            rows = metric_rows(evaluated, column, name, energy_type, expected_assets)
            for row in rows:
                row["dataset_split"] = split_name
            metrics.extend(rows)
            per_asset = asset_metrics(evaluated, column, name)
            per_asset["dataset_split"] = split_name
            assets.append(per_asset)
        output_predictions(
            evaluated,
            prediction_dir / f"{energy_type}_ensemble_{split_name}_predictions.parquet",
        )

    # Keep the exact base models whose 2024 outputs trained the weights and stacker.
    # Replacing them with refitted models would change the meta-model input distribution.
    bundle = {
        "models": selection_models,
        "weighted_model_names": BASE_MODEL_NAMES,
        "weights": dict(zip(BASE_MODEL_NAMES, weights.tolist(), strict=True)),
        "stacker": stacker,
        "energy_type": energy_type,
        "features": features,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_maps": category_maps,
        "target": TARGET,
        "base_model_training_split": "2023",
        "meta_training_split": "2024 predictions from 2023 base models",
        "final_test_split": "2025",
        "clip_range": [0.0, 1.0],
        "solar_night_zero_rule": energy_type == "solar",
    }
    joblib.dump(bundle, model_dir / f"{energy_type}_ensemble.joblib")
    metadata = {
        "energy_type": energy_type,
        "base_models": BASE_MODEL_NAMES,
        "weights": bundle["weights"],
        "stacking_alpha": float(stacker.alpha_),
        "stacking_coefficients": dict(
            zip(BASE_MODEL_NAMES, stacker.coef_.tolist(), strict=True)
        ),
        "stacking_intercept": float(stacker.intercept_),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_was_not_used_for_weights_or_stacker": True,
    }
    return metrics, pd.concat(assets, ignore_index=True), metadata


def main() -> None:
    args = parse_args()
    for output_dir in (args.models, args.predictions, args.reports):
        output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict] = []
    assets: list[pd.DataFrame] = []
    metadata: dict[str, dict] = {}
    for energy_type in ["solar", "wind"]:
        rows, per_asset, details = train_one(
            energy_type,
            args.data / f"{energy_type}_training_mart.parquet",
            args.models,
            args.predictions,
            args.extra_trees,
            args.hist_iterations,
        )
        metrics.extend(rows)
        assets.append(per_asset)
        metadata[energy_type] = details
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(
        args.reports / "generation_ensemble_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(assets, ignore_index=True).to_csv(
        args.reports / "generation_ensemble_asset_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "random_seed": RANDOM_SEED,
        "base_training_split": "2023",
        "meta_training_split": "2024",
        "locked_final_test_split": "2025",
        "test_is_not_used_for_model_or_ensemble_selection": True,
        "models": metadata,
        "metrics": metric_frame.to_dict("records"),
    }
    (args.reports / "generation_ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected = metric_frame[
        metric_frame["level"].eq("portfolio") & metric_frame["horizon"].eq("all_48h")
    ]
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
