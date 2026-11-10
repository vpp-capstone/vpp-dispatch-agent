"""Evaluate wind trajectory, spatial and direct-portfolio features on 2023-2024."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp.generation_forecast.improvement import (
    WIND_IMPROVEMENT_FEATURES,
    add_wind_improvement_features,
    portfolio_frame,
)
from vpp.generation_forecast.model import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
    WIND_CHANGE_FEATURES,
    enforce_physics,
    make_model,
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

WIND_WEATHER_FEATURES = [
    "wind_u_80m_ms",
    "wind_v_80m_ms",
    "wind_speed_80m_ms",
    "wind_direction_sin",
    "wind_direction_cos",
    "temperature_2m_c",
    "total_cloud_cover_pct",
    *WIND_CHANGE_FEATURES,
    *WIND_IMPROVEMENT_FEATURES,
]
CANDIDATES = ("current", "trajectory_spatial", "direct_portfolio_advanced")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--estimators", type=int, default=1800)
    parser.add_argument("--early-stopping", type=int, default=120)
    return parser.parse_args()


def _direct_nmae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    error = np.abs(
        prediction * frame["capacity_kw"].to_numpy() - frame["generation_kwh"].to_numpy()
    )
    return float(100.0 * error.sum() / frame["capacity_kw"].sum())


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

    expected_assets = int(asset_frame["asset_id"].nunique())
    rows = []
    for fold in folds:
        if candidate == "direct_portfolio_advanced":
            train, validation = split_fold(portfolio, fold)
            features = portfolio_features
            categorical: list[str] = []
        else:
            train, validation = split_fold(asset_frame, fold)
            features = FEATURES["wind"]
            if candidate == "trajectory_spatial":
                features = [*features, *WIND_IMPROVEMENT_FEATURES]
            categorical = CATEGORICAL_FEATURES
        model = make_model(estimators)
        model.fit(
            train[features],
            train[TARGET],
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
        if candidate == "direct_portfolio_advanced":
            prediction = np.clip(prediction, 0.0, 1.0)
            score = _direct_nmae(validation, prediction)
        else:
            prediction = enforce_physics("wind", validation, prediction)
            score = portfolio_nmae(validation, prediction, expected_assets)
        rows.append(
            {
                "energy_type": "wind",
                "candidate": candidate,
                "weight_mode": "none",
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
    frame, _ = development_frame(args.data / "wind_training_mart.parquet")
    enhanced = add_wind_improvement_features(frame)
    portfolio, portfolio_features = portfolio_frame(enhanced, WIND_WEATHER_FEATURES)
    expected_assets = int(enhanced["asset_id"].nunique())
    portfolio = portfolio[portfolio["asset_count"].eq(expected_assets)].copy()
    full_2024 = [TimeFold("2024_full", "2024-01-01", "2025-01-01")]
    fold_rows = []
    holdout_rows = []
    for candidate in CANDIDATES:
        print(f"evaluating wind: {candidate}", flush=True)
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
        args.reports / "wind_improvement_tuning_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout.to_csv(
        args.reports / "wind_improvement_2024_holdout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking.to_csv(
        args.reports / "wind_improvement_tuning_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(ranking.to_string(index=False))
    print(holdout[["candidate", "portfolio_nmae_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
