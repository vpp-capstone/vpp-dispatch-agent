"""Tune generation LightGBM models using 2023-2024 only; never load 2025 labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp.generation_forecast.model import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
    enforce_physics,
    make_model,
    prepare_frame,
    training_sample_weights,
)
from vpp.generation_forecast.paths import PROCESSED_DIR, RESULTS_DIR

DEFAULT_DATA = PROCESSED_DIR
DEFAULT_REPORTS = RESULTS_DIR


@dataclass(frozen=True)
class TimeFold:
    name: str
    validation_start: str
    validation_end: str


FOLDS = [
    TimeFold("2023_q3", "2023-07-01", "2023-10-01"),
    TimeFold("2023_q4", "2023-10-01", "2024-01-01"),
    TimeFold("2024_h1", "2024-01-01", "2024-07-01"),
    TimeFold("2024_h2", "2024-07-01", "2025-01-01"),
]

# A compact, pre-declared search avoids repeatedly adapting candidates to validation results.
CANDIDATES: list[dict[str, Any]] = [
    {"name": "current", "weight_mode": "none", "params": {}},
    {"name": "current_capacity", "weight_mode": "capacity", "params": {}},
    {"name": "current_sqrt_capacity", "weight_mode": "sqrt_capacity", "params": {}},
    {
        "name": "regularized_15_capacity",
        "weight_mode": "capacity",
        "params": {"num_leaves": 15, "min_child_samples": 250, "reg_lambda": 2.0},
    },
    {
        "name": "balanced_63_capacity",
        "weight_mode": "capacity",
        "params": {"num_leaves": 63, "min_child_samples": 150, "reg_lambda": 1.0},
    },
    {
        "name": "flexible_63_capacity",
        "weight_mode": "capacity",
        "params": {
            "num_leaves": 63,
            "min_child_samples": 50,
            "reg_alpha": 0.05,
            "reg_lambda": 0.5,
        },
    },
    {
        "name": "regularized_127_capacity",
        "weight_mode": "capacity",
        "params": {
            "num_leaves": 127,
            "min_child_samples": 300,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 4.0,
        },
    },
    {
        "name": "slow_63_capacity",
        "weight_mode": "capacity",
        "params": {
            "learning_rate": 0.015,
            "num_leaves": 63,
            "min_child_samples": 150,
            "reg_lambda": 1.0,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--energy", choices=["solar", "wind", "all"], default="all")
    parser.add_argument("--estimators", type=int, default=1800)
    parser.add_argument("--early-stopping", type=int, default=120)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="reuse existing fold and 2024 holdout CSV files without retraining",
    )
    return parser.parse_args()


def development_frame(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Load only train/validation row groups so locked 2025 labels never enter memory."""
    raw = pd.read_parquet(
        path,
        filters=[("dataset_split", "in", ["train", "validation"])],
    )
    if raw.empty or raw["dataset_split"].eq("test").any():
        raise ValueError("development data must contain 2023-2024 and no test rows")
    frame, category_maps = prepare_frame(raw)
    years = set(frame["valid_time_kst"].dt.year.unique())
    if not years.issubset({2023, 2024}):
        raise ValueError(f"unexpected development years: {sorted(years)}")
    return frame, category_maps


def split_fold(frame: pd.DataFrame, fold: TimeFold) -> tuple[pd.DataFrame, pd.DataFrame]:
    timezone = frame["valid_time_kst"].dt.tz
    start = pd.Timestamp(fold.validation_start, tz=timezone)
    end = pd.Timestamp(fold.validation_end, tz=timezone)
    train = frame[frame["valid_time_kst"] < start].copy()
    validation = frame[
        frame["valid_time_kst"].ge(start) & frame["valid_time_kst"].lt(end)
    ].copy()
    if train.empty or validation.empty:
        raise ValueError(f"empty train or validation rows for {fold.name}")
    overlap = set(train["valid_time_kst"]).intersection(validation["valid_time_kst"])
    if overlap:
        raise ValueError(f"valid_time leakage in {fold.name}")
    return train, validation


def portfolio_nmae(
    frame: pd.DataFrame, predicted_capacity_factor: np.ndarray, expected_assets: int
) -> float:
    evaluated = frame[["issue_time_utc", "valid_time_utc", "asset_id", "generation_kwh",
                       "capacity_kw"]].copy()  # fmt: skip
    evaluated["predicted_kwh"] = predicted_capacity_factor * evaluated["capacity_kw"]
    grouped = evaluated.groupby(
        ["issue_time_utc", "valid_time_utc"], observed=True
    ).agg(
        actual_kwh=("generation_kwh", "sum"),
        predicted_kwh=("predicted_kwh", "sum"),
        capacity_kw=("capacity_kw", "sum"),
        asset_count=("asset_id", "nunique"),
    )
    complete = grouped[grouped["asset_count"].eq(expected_assets)]
    if complete.empty:
        raise ValueError("no complete portfolio windows in validation fold")
    absolute_error = (complete["predicted_kwh"] - complete["actual_kwh"]).abs().sum()
    return float(100.0 * absolute_error / complete["capacity_kw"].sum())


def evaluate_candidate(
    energy_type: str,
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    estimators: int,
    early_stopping_rounds: int,
    folds: list[TimeFold] | None = None,
) -> list[dict[str, Any]]:
    import lightgbm as lgb

    features = FEATURES[energy_type]
    expected_assets = int(frame["asset_id"].nunique())
    rows: list[dict[str, Any]] = []
    for fold in FOLDS if folds is None else folds:
        train, validation = split_fold(frame, fold)
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
            energy_type,
            validation,
            model.predict(validation[features], num_iteration=model.best_iteration_),
        )
        rows.append(
            {
                "energy_type": energy_type,
                "candidate": candidate["name"],
                "weight_mode": candidate["weight_mode"],
                "fold": fold.name,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "best_iteration": int(model.best_iteration_),
                "portfolio_nmae_pct": portfolio_nmae(
                    validation, prediction, expected_assets
                ),
            }
        )
    return rows


def aggregate_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by mean error plus a fixed stability penalty."""
    aggregated = (
        fold_results.groupby(["energy_type", "candidate", "weight_mode"], observed=True)
        .agg(
            mean_nmae_pct=("portfolio_nmae_pct", "mean"),
            std_nmae_pct=("portfolio_nmae_pct", "std"),
            worst_nmae_pct=("portfolio_nmae_pct", "max"),
            median_best_iteration=("best_iteration", "median"),
            folds=("fold", "nunique"),
        )
        .reset_index()
    )
    aggregated["selection_score"] = (
        aggregated["mean_nmae_pct"] + 0.25 * aggregated["std_nmae_pct"]
    )
    return aggregated.sort_values(
        ["energy_type", "selection_score", "worst_nmae_pct"]
    ).reset_index(drop=True)


def select_jointly_stable_candidates(
    ranking: pd.DataFrame, holdout: pd.DataFrame
) -> pd.DataFrame:
    """Require improvement over current in rolling score and full-2024 nMAE."""
    full = holdout[["energy_type", "candidate", "portfolio_nmae_pct"]].rename(
        columns={"portfolio_nmae_pct": "holdout_2024_nmae_pct"}
    )
    combined = ranking.merge(full, on=["energy_type", "candidate"], validate="one_to_one")
    selected_rows: list[pd.Series] = []
    for _, group in combined.groupby("energy_type", observed=True):
        current = group[group["candidate"].eq("current")].iloc[0]
        eligible = group[
            group["selection_score"].le(current["selection_score"] + 1e-12)
            & group["holdout_2024_nmae_pct"].le(
                current["holdout_2024_nmae_pct"] + 1e-12
            )
        ]
        selected_rows.append(
            eligible.sort_values(
                ["selection_score", "holdout_2024_nmae_pct"]
            ).iloc[0]
        )
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    energy_types = ["solar", "wind"] if args.energy == "all" else [args.energy]
    all_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    mappings: dict[str, dict[str, dict[str, int]]] = {}
    fold_path = args.reports / "generation_lightgbm_tuning_folds.csv"
    holdout_path = args.reports / "generation_lightgbm_2024_holdout.csv"
    if args.summarize_existing:
        folds = pd.read_csv(fold_path)
        holdout = pd.read_csv(holdout_path)
    else:
        full_2024 = [TimeFold("2024_full", "2024-01-01", "2025-01-01")]
        for energy_type in energy_types:
            frame, category_maps = development_frame(
                args.data / f"{energy_type}_training_mart.parquet"
            )
            mappings[energy_type] = category_maps
            for candidate in CANDIDATES:
                print(f"tuning {energy_type}: {candidate['name']}", flush=True)
                all_rows.extend(
                    evaluate_candidate(
                        energy_type,
                        frame,
                        candidate,
                        args.estimators,
                        args.early_stopping,
                    )
                )
                holdout_rows.extend(
                    evaluate_candidate(
                        energy_type,
                        frame,
                        candidate,
                        args.estimators,
                        args.early_stopping,
                        folds=full_2024,
                    )
                )
        folds = pd.DataFrame(all_rows)
        holdout = pd.DataFrame(holdout_rows)
        folds.to_csv(fold_path, index=False, encoding="utf-8-sig")
        holdout.to_csv(holdout_path, index=False, encoding="utf-8-sig")
    ranking = aggregate_results(folds)
    ranking.to_csv(
        args.reports / "generation_lightgbm_tuning_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joint = select_jointly_stable_candidates(ranking, holdout)
    selected: dict[str, dict[str, Any]] = {}
    for energy_type in energy_types:
        best = joint[joint["energy_type"].eq(energy_type)].iloc[0]
        candidate = next(item for item in CANDIDATES if item["name"] == best["candidate"])
        selected[energy_type] = {
            "candidate": candidate["name"],
            "weight_mode": candidate["weight_mode"],
            "params": candidate["params"],
            "median_best_iteration": int(round(best["median_best_iteration"])),
            "mean_nmae_pct": float(best["mean_nmae_pct"]),
            "std_nmae_pct": float(best["std_nmae_pct"]),
            "worst_nmae_pct": float(best["worst_nmae_pct"]),
            "selection_score": float(best["selection_score"]),
            "holdout_2024_nmae_pct": float(best["holdout_2024_nmae_pct"]),
        }
    summary = {
        "development_years_only": [2023, 2024],
        "locked_2025_loaded": False,
        "selection_rule": (
            "improve over current in both rolling score "
            "and full-2024 portfolio nMAE"
        ),
        "folds": [fold.__dict__ for fold in FOLDS],
        "selected": selected,
        "category_maps": mappings,
    }
    (args.reports / "generation_lightgbm_tuning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
