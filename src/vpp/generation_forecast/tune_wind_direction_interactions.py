"""Evaluate wind-direction sector interactions using 2023-2024 only."""

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
from vpp.generation_forecast.wind_direction_features import (
    WIND_DIRECTION_INTERACTION_FEATURES,
    add_wind_direction_interactions,
)

FULL_2024 = [TimeFold("2024_full", "2024-01-01", "2025-01-01")]
MODEL_PARAMS = {
    "learning_rate": 0.015,
    "num_leaves": 31,
    "min_child_samples": 150,
    "reg_lambda": 1.0,
}
CANDIDATES = {
    "current": [],
    "sector_8": ["aws_proxy_wind_sector_8"],
    "sector_interactions": WIND_DIRECTION_INTERACTION_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--aws", type=Path, default=Path("data/raw/aws"))
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--estimators", type=int, default=2400)
    parser.add_argument("--early-stopping", type=int, default=150)
    return parser.parse_args()


def evaluate_fold(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    fold: TimeFold,
    estimators: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    import lightgbm as lgb

    train, validation = split_fold(frame, fold)
    calibrator = fit_aws_wind_calibrator(train, observations, PRIMARY_STATIONS)
    train = add_wind_direction_interactions(calibrator.transform(train))
    validation = add_wind_direction_interactions(calibrator.transform(validation))
    base_features = [*FEATURES["wind"], *WIND_IMPROVEMENT_FEATURES, *AWS_WIND_FEATURES]
    expected_assets = int(frame["asset_id"].nunique())
    rows: list[dict[str, Any]] = []
    for candidate, extra_features in CANDIDATES.items():
        features = [*base_features, *extra_features]
        categorical = [*CATEGORICAL_FEATURES]
        if "aws_proxy_wind_sector_8" in extra_features:
            categorical.append("aws_proxy_wind_sector_8")
        model = make_model(estimators, **MODEL_PARAMS)
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
        prediction = enforce_physics(
            "wind",
            validation,
            model.predict(validation[features], num_iteration=model.best_iteration_),
        )
        rows.append(
            {
                "energy_type": "wind",
                "candidate": candidate,
                "weight_mode": "none",
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
    frame, _ = development_frame(args.data / "wind_training_mart.parquet")
    frame = add_wind_improvement_features(frame)
    observations = load_aws_observations(args.aws)
    folds = pd.DataFrame(
        [
            row
            for fold in FOLDS
            for row in evaluate_fold(
                frame, observations, fold, args.estimators, args.early_stopping
            )
        ]
    )
    holdout = pd.DataFrame(
        evaluate_fold(
            frame, observations, FULL_2024[0], args.estimators, args.early_stopping
        )
    )
    ranking = aggregate_results(folds)
    current = ranking[ranking["candidate"].eq("current")].iloc[0]
    holdout_current = holdout[holdout["candidate"].eq("current")].iloc[0]
    eligible = ranking[
        ranking["selection_score"].le(current["selection_score"] + 1e-12)
    ].merge(
        holdout[["candidate", "portfolio_nmae_pct"]],
        on="candidate",
        validate="one_to_one",
    )
    eligible = eligible[
        eligible["portfolio_nmae_pct"].le(
            holdout_current["portfolio_nmae_pct"] + 1e-12
        )
    ]
    selected = eligible.sort_values(
        ["selection_score", "portfolio_nmae_pct"]
    ).iloc[0]
    summary = {
        "development_years_only": [2023, 2024],
        "locked_2025_loaded": False,
        "selection_rule": "improve rolling score and full-2024 nMAE versus current",
        "selected_candidate": selected["candidate"],
        "selected_features": CANDIDATES[selected["candidate"]],
        "selected_selection_score": float(selected["selection_score"]),
        "selected_holdout_2024_nmae_pct": float(selected["portfolio_nmae_pct"]),
    }
    folds.to_csv(
        args.reports / "wind_direction_tuning_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout.to_csv(
        args.reports / "wind_direction_2024_holdout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking.to_csv(
        args.reports / "wind_direction_tuning_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.reports / "wind_direction_tuning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(ranking.to_string(index=False))
    print(holdout[["candidate", "portfolio_nmae_pct"]].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
