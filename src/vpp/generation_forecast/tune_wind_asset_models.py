"""Evaluate farm-specific wind LightGBMs and OOF blends on 2023-2024."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
from vpp.generation_forecast.tune_lightgbm import FOLDS, TimeFold, development_frame, split_fold
from vpp.generation_forecast.tune_wind_dynamic_regime_ensemble import portfolio_nmae
from vpp.generation_forecast.tune_wind_regime_models import MODEL_PARAMS
from vpp.generation_forecast.wind_direction_features import add_wind_direction_interactions

FULL_2024 = TimeFold("2024_full", "2024-01-01", "2025-01-01")
SECTOR_FEATURE = "aws_proxy_wind_sector_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--aws", type=Path, default=Path("data/raw/aws"))
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--estimators", type=int, default=2400)
    parser.add_argument("--early-stopping", type=int, default=150)
    return parser.parse_args()


def predict_asset_models(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    fold: TimeFold,
    estimators: int,
    early_stopping_rounds: int,
) -> pd.DataFrame:
    import lightgbm as lgb

    train, validation = split_fold(frame, fold)
    calibrator = fit_aws_wind_calibrator(train, observations, PRIMARY_STATIONS)
    train = add_wind_direction_interactions(calibrator.transform(train))
    validation = add_wind_direction_interactions(calibrator.transform(validation))
    features = [
        *FEATURES["wind"],
        *WIND_IMPROVEMENT_FEATURES,
        *AWS_WIND_FEATURES,
        SECTOR_FEATURE,
    ]
    prediction = np.full(len(validation), np.nan, dtype=float)
    iterations: dict[str, int] = {}
    for asset_id in sorted(validation["asset_id"].unique()):
        train_mask = train["asset_id"].eq(asset_id)
        validation_mask = validation["asset_id"].eq(asset_id)
        model = make_model(estimators, **MODEL_PARAMS)
        model.fit(
            train.loc[train_mask, features],
            train.loc[train_mask, TARGET],
            categorical_feature=[*CATEGORICAL_FEATURES, SECTOR_FEATURE],
            eval_X=validation.loc[validation_mask, features],
            eval_y=validation.loc[validation_mask, TARGET],
            eval_metric="l1",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        prediction[np.flatnonzero(validation_mask.to_numpy())] = model.predict(
            validation.loc[validation_mask, features], num_iteration=model.best_iteration_
        )
        iterations[str(asset_id)] = int(model.best_iteration_)
    if np.isnan(prediction).any():
        raise ValueError("asset-specific prediction contains missing rows")
    result = validation[
        [
            "issue_time_utc",
            "valid_time_utc",
            "asset_id",
            "generation_kwh",
            "capacity_kw",
        ]
    ].copy()
    result["asset_specific"] = enforce_physics("wind", validation, prediction)
    result["fold"] = fold.name
    result.attrs["best_iterations"] = iterations
    return result


def blend_prediction(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    row_weight = frame["asset_id"].map(weights).to_numpy(dtype=float)
    return frame["dynamic"].to_numpy() + row_weight * (
        frame["asset_specific"].to_numpy() - frame["dynamic"].to_numpy()
    )


def optimize_asset_weights(frame: pd.DataFrame) -> dict[str, float]:
    assets = sorted(frame["asset_id"].unique())

    def objective(values: np.ndarray) -> float:
        return portfolio_nmae(
            frame, blend_prediction(frame, dict(zip(assets, values, strict=True)))
        )

    result = minimize(
        objective,
        np.full(len(assets), 0.5),
        method="Powell",
        bounds=[(0.0, 1.0)] * len(assets),
        options={"maxiter": 300, "xtol": 1e-5, "ftol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"asset blend optimization failed: {result.message}")
    return dict(zip(assets, np.clip(result.x, 0.0, 1.0), strict=True))


def merge_base(base: pd.DataFrame, asset: pd.DataFrame) -> pd.DataFrame:
    keys = ["issue_time_utc", "valid_time_utc", "asset_id"]
    return base.merge(
        asset[keys + ["asset_specific"]], on=keys, validate="one_to_one"
    )


def score_folds(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, selected in frame.groupby("fold", observed=True):
        for candidate, prediction in {
            "dynamic": selected["dynamic"].to_numpy(),
            "asset_specific": selected["asset_specific"].to_numpy(),
            "asset_blend": blend_prediction(selected, weights),
        }.items():
            rows.append(
                {
                    "candidate": candidate,
                    "fold": fold,
                    "portfolio_nmae_pct": portfolio_nmae(selected, prediction),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    frame, _ = development_frame(args.data / "wind_training_mart.parquet")
    frame = add_wind_improvement_features(frame)
    observations = load_aws_observations(args.aws)
    base_folds = pd.read_parquet(
        args.reports / "wind_dynamic_regime_fold_predictions.parquet"
    )
    asset_parts: list[pd.DataFrame] = []
    iteration_records: list[dict[str, Any]] = []
    for fold in FOLDS:
        part = predict_asset_models(
            frame, observations, fold, args.estimators, args.early_stopping
        )
        asset_parts.append(part)
        iteration_records.append({"fold": fold.name, **part.attrs["best_iterations"]})
    folds = merge_base(base_folds, pd.concat(asset_parts, ignore_index=True))
    oof = folds[folds["fold"].isin(["2023_q3", "2023_q4"])]
    weights = optimize_asset_weights(oof)
    scores = score_folds(folds, weights)
    ranking = (
        scores.groupby("candidate", observed=True)
        .agg(
            mean_nmae_pct=("portfolio_nmae_pct", "mean"),
            std_nmae_pct=("portfolio_nmae_pct", "std"),
            worst_nmae_pct=("portfolio_nmae_pct", "max"),
        )
        .reset_index()
    )
    ranking["selection_score"] = ranking["mean_nmae_pct"] + 0.25 * ranking["std_nmae_pct"]

    base_holdout = pd.read_parquet(
        args.reports / "wind_dynamic_regime_2024_predictions.parquet"
    )
    asset_holdout = predict_asset_models(
        frame, observations, FULL_2024, args.estimators, args.early_stopping
    )
    holdout = merge_base(base_holdout, asset_holdout)
    holdout_scores = {
        "dynamic": portfolio_nmae(holdout, holdout["dynamic"].to_numpy()),
        "asset_specific": portfolio_nmae(
            holdout, holdout["asset_specific"].to_numpy()
        ),
        "asset_blend": portfolio_nmae(holdout, blend_prediction(holdout, weights)),
    }
    ranking["holdout_2024_nmae_pct"] = ranking["candidate"].map(holdout_scores)
    baseline = ranking[ranking["candidate"].eq("dynamic")].iloc[0]
    eligible = ranking[
        ranking["selection_score"].le(baseline["selection_score"] + 1e-12)
        & ranking["holdout_2024_nmae_pct"].le(
            baseline["holdout_2024_nmae_pct"] + 1e-12
        )
    ]
    selected = eligible.sort_values(
        ["selection_score", "holdout_2024_nmae_pct"]
    ).iloc[0]
    folds["asset_blend"] = blend_prediction(folds, weights)
    holdout["asset_blend"] = blend_prediction(holdout, weights)
    median_iterations = {
        asset: int(round(float(np.median([record[asset] for record in iteration_records]))))
        for asset in weights
    }
    summary = {
        "development_years_only": [2023, 2024],
        "locked_2025_loaded": False,
        "asset_blend_weights": weights,
        "selected_best_iterations": median_iterations,
        "selected_candidate": selected["candidate"],
        "selected_selection_score": float(selected["selection_score"]),
        "holdout_2024_nmae_pct": holdout_scores,
    }
    scores.to_csv(args.reports / "wind_asset_model_folds.csv", index=False)
    ranking.sort_values("selection_score").to_csv(
        args.reports / "wind_asset_model_ranking.csv", index=False
    )
    folds.to_parquet(args.reports / "wind_asset_model_fold_predictions.parquet")
    holdout.to_parquet(args.reports / "wind_asset_model_2024_predictions.parquet")
    (args.reports / "wind_asset_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(ranking.sort_values("selection_score").to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
