"""Tune AWS-enhanced wind LightGBM parameters using 2023-2024 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from vpp.generation_forecast.aws_features import (
    AWS_WIND_FEATURES,
    PRIMARY_STATIONS,
    fit_aws_wind_calibrator,
    load_aws_observations,
)
from vpp.generation_forecast.improvement import (
    WIND_IMPROVEMENT_FEATURES,
    add_wind_improvement_features,
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
    select_jointly_stable_candidates,
    split_fold,
)

FULL_2024 = [TimeFold("2024_full", "2024-01-01", "2025-01-01")]

CANDIDATES: list[dict[str, Any]] = [
    {"name": "current", "weight_mode": "none", "params": {}},
    {
        "name": "regularized_15",
        "weight_mode": "none",
        "params": {"num_leaves": 15, "min_child_samples": 250, "reg_lambda": 2.0},
    },
    {
        "name": "regularized_31",
        "weight_mode": "none",
        "params": {"num_leaves": 31, "min_child_samples": 250, "reg_lambda": 2.0},
    },
    {
        "name": "balanced_63",
        "weight_mode": "none",
        "params": {"num_leaves": 63, "min_child_samples": 150, "reg_lambda": 1.0},
    },
    {
        "name": "slow_31",
        "weight_mode": "none",
        "params": {
            "learning_rate": 0.015,
            "num_leaves": 31,
            "min_child_samples": 150,
            "reg_lambda": 1.0,
        },
    },
    {
        "name": "conservative_31",
        "weight_mode": "none",
        "params": {
            "num_leaves": 31,
            "min_child_samples": 400,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 4.0,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--aws", type=Path, default=Path("data/raw/aws"))
    parser.add_argument("--estimators", type=int, default=2400)
    parser.add_argument("--early-stopping", type=int, default=150)
    return parser.parse_args()


def evaluate(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    folds: list[TimeFold],
    estimators: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    import lightgbm as lgb

    rows: list[dict[str, Any]] = []
    features = [*FEATURES["wind"], *WIND_IMPROVEMENT_FEATURES, *AWS_WIND_FEATURES]
    expected_assets = int(frame["asset_id"].nunique())
    for fold in folds:
        train, validation = split_fold(frame, fold)
        calibrator = fit_aws_wind_calibrator(train, observations, PRIMARY_STATIONS)
        train = calibrator.transform(train)
        validation = calibrator.transform(validation)
        for candidate in CANDIDATES:
            model = make_model(estimators, **candidate["params"])
            model.fit(
                train[features],
                train[TARGET],
                sample_weight=training_sample_weights(train, candidate["weight_mode"]),
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
            rows.append(
                {
                    "energy_type": "wind",
                    "candidate": candidate["name"],
                    "weight_mode": candidate["weight_mode"],
                    "fold": fold.name,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "best_iteration": int(model.best_iteration_),
                    "portfolio_nmae_pct": portfolio_nmae(
                        validation, prediction, expected_assets
                    ),
                    "aws_calibration_rows": sum(calibrator.fit_rows.values()),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    observations = load_aws_observations(args.aws)
    frame, _ = development_frame(args.data / "wind_training_mart.parquet")
    frame = add_wind_improvement_features(frame)
    folds = pd.DataFrame(
        evaluate(frame, observations, FOLDS, args.estimators, args.early_stopping)
    )
    holdout = pd.DataFrame(
        evaluate(frame, observations, FULL_2024, args.estimators, args.early_stopping)
    )
    ranking = aggregate_results(folds)
    selected = select_jointly_stable_candidates(ranking, holdout).iloc[0]
    candidate = next(item for item in CANDIDATES if item["name"] == selected["candidate"])
    summary = {
        "development_years_only": [2023, 2024],
        "locked_2025_loaded": False,
        "selection_rule": "improve rolling score and full-2024 nMAE versus current",
        "selected": {
            "candidate": candidate["name"],
            "params": candidate["params"],
            "median_best_iteration": int(round(selected["median_best_iteration"])),
            "selection_score": float(selected["selection_score"]),
            "holdout_2024_nmae_pct": float(selected["holdout_2024_nmae_pct"]),
        },
    }
    folds.to_csv(
        args.reports / "wind_aws_lgbm_tuning_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout.to_csv(
        args.reports / "wind_aws_lgbm_2024_holdout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking.to_csv(
        args.reports / "wind_aws_lgbm_tuning_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.reports / "wind_aws_lgbm_tuning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(ranking.to_string(index=False))
    print(holdout[["candidate", "portfolio_nmae_pct"]].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
