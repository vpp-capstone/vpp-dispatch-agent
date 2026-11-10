"""Compare leakage-safe AWS-calibrated wind features on 2023-2024 folds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from vpp.generation_forecast.aws_features import (
    AWS_WIND_FEATURES,
    MULTI_STATIONS,
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

CANDIDATES = {
    "trajectory_spatial": None,
    "aws_primary_proxy": PRIMARY_STATIONS,
    "aws_multi_proxy": MULTI_STATIONS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--aws", type=Path, default=Path("data/raw/aws"))
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--estimators", type=int, default=1800)
    parser.add_argument("--early-stopping", type=int, default=120)
    return parser.parse_args()


def evaluate(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    candidate: str,
    folds: list[TimeFold],
    estimators: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    import lightgbm as lgb

    station_map = CANDIDATES[candidate]
    expected_assets = int(frame["asset_id"].nunique())
    base_features = [*FEATURES["wind"], *WIND_IMPROVEMENT_FEATURES]
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train, validation = split_fold(frame, fold)
        features = base_features
        fit_rows: dict[str, int] = {}
        if station_map is not None:
            calibrator = fit_aws_wind_calibrator(train, observations, station_map)
            train = calibrator.transform(train)
            validation = calibrator.transform(validation)
            features = [*features, *AWS_WIND_FEATURES]
            fit_rows = calibrator.fit_rows
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
        rows.append(
            {
                "energy_type": "wind",
                "candidate": candidate,
                "weight_mode": "none",
                "fold": fold.name,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "best_iteration": int(model.best_iteration_),
                "portfolio_nmae_pct": portfolio_nmae(
                    validation, prediction, expected_assets
                ),
                "aws_calibration_rows": int(sum(fit_rows.values())),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    frame, _ = development_frame(args.data / "wind_training_mart.parquet")
    frame = add_wind_improvement_features(frame)
    observations = load_aws_observations(args.aws)
    full_2024 = [TimeFold("2024_full", "2024-01-01", "2025-01-01")]
    fold_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        print(f"evaluating wind AWS candidate: {candidate}", flush=True)
        fold_rows.extend(
            evaluate(
                frame, observations, candidate, FOLDS,
                args.estimators, args.early_stopping,
            )
        )
        holdout_rows.extend(
            evaluate(
                frame, observations, candidate, full_2024,
                args.estimators, args.early_stopping,
            )
        )
    folds = pd.DataFrame(fold_rows)
    holdout = pd.DataFrame(holdout_rows)
    ranking = aggregate_results(folds)
    folds.to_csv(
        args.reports / "wind_aws_tuning_folds.csv", index=False, encoding="utf-8-sig"
    )
    holdout.to_csv(
        args.reports / "wind_aws_2024_holdout.csv", index=False, encoding="utf-8-sig"
    )
    ranking.to_csv(
        args.reports / "wind_aws_tuning_ranking.csv", index=False, encoding="utf-8-sig"
    )
    print(ranking.to_string(index=False))
    print(holdout[["candidate", "portfolio_nmae_pct", "best_iteration"]].to_string(index=False))


if __name__ == "__main__":
    main()
