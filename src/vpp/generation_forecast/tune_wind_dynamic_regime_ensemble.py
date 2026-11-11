"""Tune weather-regime blend weights on 2023 OOF and evaluate on 2024."""

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
from vpp.generation_forecast.model import CATEGORICAL_FEATURES, FEATURES, enforce_physics
from vpp.generation_forecast.paths import PROCESSED_DIR, RESULTS_DIR
from vpp.generation_forecast.tune_lightgbm import FOLDS, TimeFold, development_frame, split_fold
from vpp.generation_forecast.tune_oof_ensemble import OOF_FOLDS
from vpp.generation_forecast.tune_wind_regime_models import fit_predict_regimes
from vpp.generation_forecast.wind_direction_features import add_wind_direction_interactions
from vpp.generation_forecast.wind_regime import wind_speed_regime

FULL_2024 = TimeFold("2024_full", "2024-01-01", "2025-01-01")
BOUNDARIES = (3.0, 7.0)
SECTOR_FEATURE = "aws_proxy_wind_sector_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--aws", type=Path, default=Path("data/raw/aws"))
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--estimators", type=int, default=2400)
    parser.add_argument("--early-stopping", type=int, default=150)
    return parser.parse_args()


def predict_fold(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    fold: TimeFold,
    estimators: int,
    early_stopping_rounds: int,
) -> pd.DataFrame:
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
    categorical = [*CATEGORICAL_FEATURES, SECTOR_FEATURE]
    joint, joint_iterations = fit_predict_regimes(
        train,
        validation,
        features,
        None,
        estimators,
        early_stopping_rounds,
        categorical=categorical,
    )
    expert, expert_iterations = fit_predict_regimes(
        train,
        validation,
        features,
        BOUNDARIES,
        estimators,
        early_stopping_rounds,
        categorical=categorical,
    )
    result = validation[
        [
            "issue_time_utc",
            "valid_time_utc",
            "asset_id",
            "valid_time_kst",
            "forecast_day_offset",
            "lead_hour",
            "generation_kwh",
            "capacity_kw",
        ]
    ].copy()
    result["joint"] = enforce_physics("wind", validation, joint)
    result["expert"] = enforce_physics("wind", validation, expert)
    result["regime"] = wind_speed_regime(
        validation["aws_proxy_wind_speed_ms"].to_numpy(), BOUNDARIES
    )
    result["aws_proxy_wind_speed_ms"] = validation[
        "aws_proxy_wind_speed_ms"
    ].to_numpy()
    result["fold"] = fold.name
    result.attrs["best_iterations"] = {
        "joint": joint_iterations,
        "expert": expert_iterations,
    }
    return result


def portfolio_nmae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    evaluated = frame[
        ["issue_time_utc", "valid_time_utc", "generation_kwh", "capacity_kw"]
    ].copy()
    evaluated["predicted_kwh"] = np.asarray(prediction) * evaluated["capacity_kw"]
    portfolio = evaluated.groupby(
        ["issue_time_utc", "valid_time_utc"], observed=True
    ).agg(
        actual_kwh=("generation_kwh", "sum"),
        predicted_kwh=("predicted_kwh", "sum"),
        capacity_kw=("capacity_kw", "sum"),
    )
    return float(
        100.0
        * np.abs(portfolio["predicted_kwh"] - portfolio["actual_kwh"]).sum()
        / portfolio["capacity_kw"].sum()
    )


def blend_prediction(frame: pd.DataFrame, expert_weights: np.ndarray) -> np.ndarray:
    row_weight = np.asarray(expert_weights)[frame["regime"].to_numpy(dtype=int)]
    return (1.0 - row_weight) * frame["joint"].to_numpy() + row_weight * frame[
        "expert"
    ].to_numpy()


def optimize_regime_weights(oof: pd.DataFrame) -> np.ndarray:
    result = minimize(
        lambda weights: portfolio_nmae(oof, blend_prediction(oof, weights)),
        np.full(len(BOUNDARIES) + 1, 0.5),
        method="Powell",
        bounds=[(0.0, 1.0)] * (len(BOUNDARIES) + 1),
        options={"maxiter": 300, "xtol": 1e-5, "ftol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"dynamic weight optimization failed: {result.message}")
    return np.clip(result.x, 0.0, 1.0)


def score_rows(frame: pd.DataFrame, weights: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = {
        "joint": frame["joint"].to_numpy(),
        "expert": frame["expert"].to_numpy(),
        "dynamic": blend_prediction(frame, weights),
    }
    for fold_name, selected in frame.groupby("fold", observed=True):
        positions = frame.index.get_indexer(selected.index)
        for candidate, prediction in candidates.items():
            rows.append(
                {
                    "energy_type": "wind",
                    "candidate": candidate,
                    "weight_mode": "2023_oof_regime" if candidate == "dynamic" else "none",
                    "fold": fold_name,
                    "train_rows": np.nan,
                    "validation_rows": len(selected),
                    "best_iteration": np.nan,
                    "portfolio_nmae_pct": portfolio_nmae(
                        selected, prediction[positions]
                    ),
                }
            )
    return rows


def main() -> None:
    from vpp.generation_forecast.tune_lightgbm import aggregate_results

    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    frame, _ = development_frame(args.data / "wind_training_mart.parquet")
    frame = add_wind_improvement_features(frame)
    observations = load_aws_observations(args.aws)
    fold_parts: list[pd.DataFrame] = []
    iteration_records: list[dict[str, Any]] = []
    for fold in FOLDS:
        part = predict_fold(
            frame, observations, fold, args.estimators, args.early_stopping
        )
        fold_parts.append(part)
        iteration_records.append(
            {"fold": fold.name, **part.attrs["best_iterations"]}
        )
    folds = pd.concat(fold_parts, ignore_index=True)
    oof = folds[folds["fold"].isin([fold.name for fold in OOF_FOLDS])].copy()
    weights = optimize_regime_weights(oof)
    fold_scores = pd.DataFrame(score_rows(folds, weights))
    ranking = aggregate_results(fold_scores)
    holdout = predict_fold(
        frame, observations, FULL_2024, args.estimators, args.early_stopping
    )
    holdout_scores = {
        "joint": portfolio_nmae(holdout, holdout["joint"].to_numpy()),
        "expert": portfolio_nmae(holdout, holdout["expert"].to_numpy()),
        "dynamic": portfolio_nmae(holdout, blend_prediction(holdout, weights)),
    }
    folds["dynamic"] = blend_prediction(folds, weights)
    holdout["dynamic"] = blend_prediction(holdout, weights)
    joint_iterations = [record["joint"][0] for record in iteration_records]
    expert_iterations = {
        str(regime): [record["expert"][regime] for record in iteration_records]
        for regime in range(len(BOUNDARIES) + 1)
    }
    current = ranking[ranking["candidate"].eq("joint")].iloc[0]
    eligible = ranking[
        ranking["selection_score"].le(current["selection_score"] + 1e-12)
        & ranking["candidate"].map(holdout_scores).le(
            holdout_scores["joint"] + 1e-12
        )
    ]
    selected = eligible.sort_values(
        ["selection_score", "worst_nmae_pct"]
    ).iloc[0]
    summary = {
        "development_years_only": [2023, 2024],
        "locked_2025_loaded": False,
        "weight_training_period": "2023 Q3-Q4 rolling OOF only",
        "regime_boundaries_ms": BOUNDARIES,
        "expert_weights_by_regime": weights.tolist(),
        "selected_best_iterations": {
            "joint": int(round(float(np.median(joint_iterations)))),
            "experts": {
                regime: int(round(float(np.median(values))))
                for regime, values in expert_iterations.items()
            },
        },
        "selected_candidate": selected["candidate"],
        "selected_selection_score": float(selected["selection_score"]),
        "holdout_2024_nmae_pct": holdout_scores,
    }
    fold_scores.to_csv(
        args.reports / "wind_dynamic_regime_tuning_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds.to_parquet(args.reports / "wind_dynamic_regime_fold_predictions.parquet")
    holdout.to_parquet(
        args.reports / "wind_dynamic_regime_2024_predictions.parquet"
    )
    ranking.to_csv(
        args.reports / "wind_dynamic_regime_tuning_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.reports / "wind_dynamic_regime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(ranking.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
