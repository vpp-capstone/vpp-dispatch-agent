"""Compare wind feature and power-curve residual models on 2023-2024 only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from vpp.generation_forecast.model import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
    WIND_CHANGE_FEATURES,
    add_wind_change_features,
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
from vpp.generation_forecast.wind_hybrid import fit_wind_hybrid

CANDIDATES = (
    "current",
    "change_features",
    "power_curve_residual_change_features",
    "directional_power_curve_residual_change_features",
)
LEGACY_WIND_FEATURES = [
    feature for feature in FEATURES["wind"] if feature not in WIND_CHANGE_FEATURES
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--estimators", type=int, default=1800)
    parser.add_argument("--early-stopping", type=int, default=120)
    return parser.parse_args()


def evaluate(
    frame: pd.DataFrame,
    candidate: str,
    estimators: int,
    early_stopping_rounds: int,
    folds: list[TimeFold],
) -> list[dict[str, Any]]:
    import lightgbm as lgb

    expected_assets = int(frame["asset_id"].nunique())
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train, validation = split_fold(frame, fold)
        if candidate in {
            "power_curve_residual_change_features",
            "directional_power_curve_residual_change_features",
        }:
            model = fit_wind_hybrid(
                train,
                validation,
                estimators,
                early_stopping_rounds,
                directional_curve=candidate.startswith("directional_"),
            )
            prediction = model.predict(validation)
            best_iteration = int(model.residual_model.best_iteration_)
        else:
            if candidate == "change_features":
                train = add_wind_change_features(train)
                validation = add_wind_change_features(validation)
                features = FEATURES["wind"]
            else:
                features = LEGACY_WIND_FEATURES
            model = make_model(estimators)
            model.fit(
                train[features],
                train[TARGET],
                categorical_feature=CATEGORICAL_FEATURES,
                eval_X=validation[features],
                eval_y=validation[TARGET],
                eval_metric="l1",
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            prediction = enforce_physics(
                "wind",
                validation,
                model.predict(validation[features], num_iteration=model.best_iteration_),
            )
            best_iteration = int(model.best_iteration_)
        rows.append(
            {
                "energy_type": "wind",
                "candidate": candidate,
                "weight_mode": "none",
                "fold": fold.name,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "best_iteration": best_iteration,
                "portfolio_nmae_pct": portfolio_nmae(validation, prediction, expected_assets),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    frame, _ = development_frame(args.data / "wind_training_mart.parquet")
    rolling_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    full_2024 = [TimeFold("2024_full", "2024-01-01", "2025-01-01")]
    for candidate in CANDIDATES:
        print(f"evaluating wind: {candidate}", flush=True)
        rolling_rows.extend(evaluate(frame, candidate, args.estimators, args.early_stopping, FOLDS))
        holdout_rows.extend(
            evaluate(frame, candidate, args.estimators, args.early_stopping, full_2024)
        )
    rolling = pd.DataFrame(rolling_rows)
    holdout = pd.DataFrame(holdout_rows)
    ranking = aggregate_results(rolling)
    rolling.to_csv(args.reports / "wind_hybrid_tuning_folds.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(args.reports / "wind_hybrid_2024_holdout.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(
        args.reports / "wind_hybrid_tuning_ranking.csv", index=False, encoding="utf-8-sig"
    )
    print(ranking.to_string(index=False))
    print(holdout[["candidate", "portfolio_nmae_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
