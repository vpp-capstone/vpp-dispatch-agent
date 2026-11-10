"""Evaluate solar physics/weather features and a direct portfolio model on 2023-2024."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

SOLAR_WEATHER_FEATURES = [
    "shortwave_radiation_w_m2",
    "temperature_2m_c",
    "total_cloud_cover_pct",
    "solar_elevation_deg",
    "is_daylight",
    *SOLAR_IMPROVEMENT_FEATURES,
]
CANDIDATES = ("current", "clear_sky_regional", "direct_portfolio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--estimators", type=int, default=1800)
    parser.add_argument("--early-stopping", type=int, default=120)
    return parser.parse_args()


def direct_portfolio_nmae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    predicted_kwh = np.asarray(prediction) * frame["capacity_kw"].to_numpy(dtype=float)
    error = np.abs(predicted_kwh - frame["generation_kwh"].to_numpy(dtype=float)).sum()
    return float(100.0 * error / frame["capacity_kw"].sum())


def evaluate(
    asset_frame: pd.DataFrame,
    portfolio: pd.DataFrame,
    portfolio_features: list[str],
    candidate: str,
    folds: list[TimeFold],
    estimators: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    import lightgbm as lgb

    rows: list[dict[str, Any]] = []
    for fold in folds:
        if candidate == "direct_portfolio":
            train, validation = split_fold(portfolio, fold)
            features = portfolio_features
            categorical: list[str] = []
            weights = None
        else:
            train, validation = split_fold(asset_frame, fold)
            features = FEATURES["solar"]
            if candidate == "clear_sky_regional":
                features = [*features, *SOLAR_IMPROVEMENT_FEATURES]
            categorical = CATEGORICAL_FEATURES
            weights = training_sample_weights(train, "sqrt_capacity")
        model = make_model(estimators)
        model.fit(
            train[features],
            train[TARGET],
            sample_weight=weights,
            categorical_feature=categorical,
            eval_X=validation[features],
            eval_y=validation[TARGET],
            eval_metric="l1",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        prediction = model.predict(validation[features], num_iteration=model.best_iteration_)
        if candidate == "direct_portfolio":
            prediction = enforce_portfolio_physics("solar", validation, prediction)
            score = direct_portfolio_nmae(validation, prediction)
        else:
            prediction = enforce_physics("solar", validation, prediction)
            score = portfolio_nmae(
                validation,
                prediction,
                expected_assets=int(asset_frame["asset_id"].nunique()),
            )
        rows.append(
            {
                "energy_type": "solar",
                "candidate": candidate,
                "weight_mode": "sqrt_capacity" if candidate != "direct_portfolio" else "none",
                "fold": fold.name,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "best_iteration": int(model.best_iteration_),
                "portfolio_nmae_pct": score,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    frame, _ = development_frame(args.data / "solar_training_mart.parquet")
    enhanced = add_solar_improvement_features(frame)
    portfolio, portfolio_features = portfolio_frame(enhanced, SOLAR_WEATHER_FEATURES)
    expected_assets = int(enhanced["asset_id"].nunique())
    portfolio = portfolio[portfolio["asset_count"].eq(expected_assets)].copy()
    full_2024 = [TimeFold("2024_full", "2024-01-01", "2025-01-01")]
    fold_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        print(f"evaluating solar: {candidate}", flush=True)
        fold_rows.extend(
            evaluate(
                enhanced,
                portfolio,
                portfolio_features,
                candidate,
                FOLDS,
                args.estimators,
                args.early_stopping,
            )
        )
        holdout_rows.extend(
            evaluate(
                enhanced,
                portfolio,
                portfolio_features,
                candidate,
                full_2024,
                args.estimators,
                args.early_stopping,
            )
        )
    folds = pd.DataFrame(fold_rows)
    holdout = pd.DataFrame(holdout_rows)
    ranking = aggregate_results(folds)
    folds.to_csv(
        args.reports / "solar_improvement_tuning_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout.to_csv(
        args.reports / "solar_improvement_2024_holdout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking.to_csv(
        args.reports / "solar_improvement_tuning_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(ranking.to_string(index=False))
    print(holdout[["candidate", "portfolio_nmae_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
