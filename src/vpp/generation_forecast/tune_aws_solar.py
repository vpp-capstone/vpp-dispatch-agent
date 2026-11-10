"""Evaluate leakage-safe AWS temperature and humidity proxies for solar models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vpp.generation_forecast.aws_features import (
    AWS_SOLAR_FEATURES,
    SOLAR_PRIMARY_STATIONS,
    fit_aws_solar_calibrator,
    load_aws_observations,
)
from vpp.generation_forecast.improvement import (
    SOLAR_IMPROVEMENT_FEATURES,
    add_solar_improvement_features,
    enforce_portfolio_physics,
    portfolio_frame,
)
from vpp.generation_forecast.model import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
    enforce_physics,
    make_model,
    training_sample_weights,
)
from vpp.generation_forecast.paths import PROCESSED_DIR, RESULTS_DIR
from vpp.generation_forecast.tune_lightgbm import (
    FOLDS,
    TimeFold,
    aggregate_results,
    development_frame,
    portfolio_nmae,
    split_fold,
)

ASSET_WEIGHT = 0.3503164036548981
DIRECT_WEIGHT = 1.0 - ASSET_WEIGHT
SOLAR_WEATHER_FEATURES = [
    "shortwave_radiation_w_m2", "temperature_2m_c", "total_cloud_cover_pct",
    "solar_elevation_deg", "is_daylight", *SOLAR_IMPROVEMENT_FEATURES,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--aws", type=Path, default=Path("data/raw/aws"))
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--estimators", type=int, default=1800)
    parser.add_argument("--early-stopping", type=int, default=120)
    return parser.parse_args()


def _fit_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    estimators: int,
    early_stopping_rounds: int,
    *,
    categorical: list[str] | None = None,
    weight_mode: str = "none",
):
    import lightgbm as lgb

    model = make_model(estimators)
    model.fit(
        train[features],
        train[TARGET],
        sample_weight=training_sample_weights(train, weight_mode),
        categorical_feature=categorical or [],
        eval_X=validation[features],
        eval_y=validation[TARGET],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _portfolio_totals(frame: pd.DataFrame, capacity_factor: np.ndarray) -> pd.DataFrame:
    values = frame[["issue_time_utc", "valid_time_utc", "generation_kwh", "capacity_kw"]].copy()
    values["predicted_kwh"] = capacity_factor * values["capacity_kw"]
    return values.groupby(["issue_time_utc", "valid_time_utc"], observed=True).agg(
        actual_kwh=("generation_kwh", "sum"),
        capacity_kw=("capacity_kw", "sum"),
        predicted_kwh=("predicted_kwh", "sum"),
    )


def _score_totals(actual: np.ndarray, predicted: np.ndarray, capacity: np.ndarray) -> float:
    return float(100.0 * np.abs(predicted - actual).sum() / capacity.sum())


def evaluate_fold(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    fold: TimeFold,
    estimators: int,
    early_stopping_rounds: int,
) -> list[dict]:
    expected_assets = int(frame["asset_id"].nunique())
    train, validation = split_fold(frame, fold)
    calibrator = fit_aws_solar_calibrator(train, observations, SOLAR_PRIMARY_STATIONS)
    aws_train = calibrator.transform(train)
    aws_validation = calibrator.transform(validation)
    rows: list[dict] = []
    asset_components: dict[str, tuple[pd.DataFrame, int]] = {}
    direct_components: dict[str, tuple[np.ndarray, int]] = {}

    for prefix, train_asset, validation_asset, aws_features in [
        ("current", train, validation, []),
        ("aws", aws_train, aws_validation, AWS_SOLAR_FEATURES),
    ]:
        asset_features = [*FEATURES["solar"], *aws_features]
        asset_model = _fit_model(
            train_asset,
            validation_asset,
            asset_features,
            estimators,
            early_stopping_rounds,
            categorical=CATEGORICAL_FEATURES,
            weight_mode="sqrt_capacity",
        )
        asset_prediction = enforce_physics(
            "solar",
            validation_asset,
            asset_model.predict(
                validation_asset[asset_features], num_iteration=asset_model.best_iteration_
            ),
        )
        asset_score = portfolio_nmae(
            validation_asset, asset_prediction, expected_assets
        )
        asset_iteration = int(asset_model.best_iteration_)
        rows.append(
            {
                "energy_type": "solar", "candidate": f"{prefix}_asset",
                "weight_mode": "sqrt_capacity", "fold": fold.name,
                "train_rows": len(train_asset), "validation_rows": len(validation_asset),
                "best_iteration": asset_iteration,
                "asset_best_iteration": asset_iteration,
                "direct_best_iteration": np.nan,
                "portfolio_nmae_pct": asset_score,
                "aws_calibration_rows": sum(calibrator.fit_rows.values()) if prefix == "aws" else 0,
            }
        )

        weather_features = [*SOLAR_WEATHER_FEATURES, *aws_features]
        train_portfolio, portfolio_features = portfolio_frame(train_asset, weather_features)
        validation_portfolio, _ = portfolio_frame(validation_asset, weather_features)
        train_portfolio = train_portfolio[train_portfolio["asset_count"].eq(expected_assets)]
        validation_portfolio = validation_portfolio[
            validation_portfolio["asset_count"].eq(expected_assets)
        ]
        direct_model = _fit_model(
            train_portfolio,
            validation_portfolio,
            portfolio_features,
            estimators,
            early_stopping_rounds,
        )
        direct_prediction = enforce_portfolio_physics(
            "solar",
            validation_portfolio,
            direct_model.predict(
                validation_portfolio[portfolio_features],
                num_iteration=direct_model.best_iteration_,
            ),
        )
        direct_score = _score_totals(
            validation_portfolio["generation_kwh"].to_numpy(dtype=float),
            direct_prediction * validation_portfolio["capacity_kw"].to_numpy(dtype=float),
            validation_portfolio["capacity_kw"].to_numpy(dtype=float),
        )
        direct_iteration = int(direct_model.best_iteration_)
        rows.append(
            {
                "energy_type": "solar", "candidate": f"{prefix}_direct",
                "weight_mode": "none", "fold": fold.name,
                "train_rows": len(train_portfolio),
                "validation_rows": len(validation_portfolio),
                "best_iteration": direct_iteration,
                "asset_best_iteration": np.nan,
                "direct_best_iteration": direct_iteration,
                "portfolio_nmae_pct": direct_score,
                "aws_calibration_rows": sum(calibrator.fit_rows.values()) if prefix == "aws" else 0,
            }
        )
        asset_totals = _portfolio_totals(validation_asset, asset_prediction)
        asset_components[prefix] = (asset_totals, asset_iteration)
        direct_components[prefix] = (direct_prediction, direct_iteration)
        direct_cf = direct_prediction
        direct_kwh = direct_cf * validation_portfolio["capacity_kw"].to_numpy(dtype=float)
        weighted = ASSET_WEIGHT * asset_totals["predicted_kwh"].to_numpy(dtype=float) + (
            DIRECT_WEIGHT * direct_kwh
        )
        weighted_score = _score_totals(
            asset_totals["actual_kwh"].to_numpy(dtype=float),
            weighted,
            asset_totals["capacity_kw"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "energy_type": "solar", "candidate": f"{prefix}_weighted",
                "weight_mode": "fixed_oof", "fold": fold.name,
                "train_rows": len(train_asset), "validation_rows": len(validation_asset),
                "best_iteration": int(round((asset_iteration + direct_iteration) / 2)),
                "asset_best_iteration": asset_iteration,
                "direct_best_iteration": direct_iteration,
                "portfolio_nmae_pct": weighted_score,
                "aws_calibration_rows": sum(calibrator.fit_rows.values()) if prefix == "aws" else 0,
            }
        )
    for candidate, asset_prefix, direct_prefix in [
        ("current_asset_aws_direct", "current", "aws"),
        ("aws_asset_current_direct", "aws", "current"),
    ]:
        asset_totals, asset_iteration = asset_components[asset_prefix]
        direct_cf, direct_iteration = direct_components[direct_prefix]
        direct_kwh = direct_cf * asset_totals["capacity_kw"].to_numpy(dtype=float)
        weighted = ASSET_WEIGHT * asset_totals["predicted_kwh"].to_numpy(dtype=float) + (
            DIRECT_WEIGHT * direct_kwh
        )
        rows.append(
            {
                "energy_type": "solar", "candidate": candidate,
                "weight_mode": "fixed_oof", "fold": fold.name,
                "train_rows": len(train), "validation_rows": len(validation),
                "best_iteration": int(round((asset_iteration + direct_iteration) / 2)),
                "asset_best_iteration": asset_iteration,
                "direct_best_iteration": direct_iteration,
                "portfolio_nmae_pct": _score_totals(
                    asset_totals["actual_kwh"].to_numpy(dtype=float),
                    weighted,
                    asset_totals["capacity_kw"].to_numpy(dtype=float),
                ),
                "aws_calibration_rows": sum(calibrator.fit_rows.values()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    frame, _ = development_frame(args.data / "solar_training_mart.parquet")
    frame = add_solar_improvement_features(frame)
    observations = load_aws_observations(args.aws)
    full_2024 = [TimeFold("2024_full", "2024-01-01", "2025-01-01")]
    fold_rows: list[dict] = []
    holdout_rows: list[dict] = []
    for fold in FOLDS:
        print(f"evaluating solar AWS fold: {fold.name}", flush=True)
        fold_rows.extend(
            evaluate_fold(frame, observations, fold, args.estimators, args.early_stopping)
        )
    for fold in full_2024:
        print("evaluating solar AWS 2024 holdout", flush=True)
        holdout_rows.extend(
            evaluate_fold(frame, observations, fold, args.estimators, args.early_stopping)
        )
    folds = pd.DataFrame(fold_rows)
    holdout = pd.DataFrame(holdout_rows)
    ranking = aggregate_results(folds)
    folds.to_csv(args.reports / "solar_aws_tuning_folds.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(args.reports / "solar_aws_2024_holdout.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(args.reports / "solar_aws_tuning_ranking.csv", index=False, encoding="utf-8-sig")
    print(ranking.to_string(index=False))
    columns = [
        "candidate", "portfolio_nmae_pct",
        "asset_best_iteration", "direct_best_iteration",
    ]
    print(holdout[columns].to_string(index=False))


if __name__ == "__main__":
    main()
