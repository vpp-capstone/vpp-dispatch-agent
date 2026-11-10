"""Fit generation ensemble weights on 2023 OOF predictions and evaluate on 2024."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp.generation_forecast.ensemble import optimize_nonnegative_weights
from vpp.generation_forecast.improvement import (
    SOLAR_IMPROVEMENT_FEATURES,
    WIND_IMPROVEMENT_FEATURES,
    add_solar_improvement_features,
    add_wind_improvement_features,
    enforce_portfolio_physics,
    portfolio_frame,
)
from vpp.generation_forecast.model import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
    WIND_CHANGE_FEATURES,
    enforce_physics,
    make_model,
    training_sample_weights,
)
from vpp.generation_forecast.paths import PROCESSED_DIR, RESULTS_DIR
from vpp.generation_forecast.tune_lightgbm import TimeFold, development_frame, split_fold
from vpp.generation_forecast.wind_hybrid import fit_wind_hybrid

OOF_FOLDS = [
    TimeFold("2023_q3", "2023-07-01", "2023-10-01"),
    TimeFold("2023_q4", "2023-10-01", "2024-01-01"),
]
FULL_2024 = TimeFold("2024_full", "2024-01-01", "2025-01-01")
SOLAR_PORTFOLIO_WEATHER = [
    "shortwave_radiation_w_m2",
    "temperature_2m_c",
    "total_cloud_cover_pct",
    "solar_elevation_deg",
    "is_daylight",
    *SOLAR_IMPROVEMENT_FEATURES,
]
WIND_PORTFOLIO_WEATHER = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--estimators", type=int, default=1200)
    parser.add_argument("--early-stopping", type=int, default=100)
    return parser.parse_args()


def fit_lgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    estimators: int,
    early_stopping_rounds: int,
    *,
    categorical: list[str] | None = None,
    weight_mode: str = "none",
) -> Any:
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


def aggregate_asset_prediction(
    frame: pd.DataFrame, prediction: np.ndarray, name: str, expected_assets: int
) -> pd.DataFrame:
    evaluated = frame[
        [
            "issue_time_utc",
            "valid_time_utc",
            "valid_time_kst",
            "forecast_day_offset",
            "asset_id",
            "generation_kwh",
            "capacity_kw",
        ]
    ].copy()
    evaluated["predicted_kwh"] = prediction * evaluated["capacity_kw"]
    grouped = (
        evaluated.groupby(["issue_time_utc", "valid_time_utc"], observed=True)
        .agg(
            valid_time_kst=("valid_time_kst", "first"),
            forecast_day_offset=("forecast_day_offset", "first"),
            actual_kwh=("generation_kwh", "sum"),
            capacity_kw=("capacity_kw", "sum"),
            predicted_kwh=("predicted_kwh", "sum"),
            asset_count=("asset_id", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["asset_count"].eq(expected_assets)].copy()
    grouped[name] = grouped.pop("predicted_kwh") / grouped["capacity_kw"]
    grouped["actual_capacity_factor"] = grouped["actual_kwh"] / grouped["capacity_kw"]
    return grouped


def fit_predict_components(
    energy_type: str,
    asset_train: pd.DataFrame,
    asset_validation: pd.DataFrame,
    portfolio_train: pd.DataFrame,
    portfolio_validation: pd.DataFrame,
    portfolio_features: list[str],
    estimators: int,
    early_stopping_rounds: int,
    expected_assets: int,
) -> pd.DataFrame:
    asset_features = FEATURES[energy_type]
    if energy_type == "wind":
        asset_features = [*asset_features, *WIND_IMPROVEMENT_FEATURES]
    asset_model = fit_lgbm(
        asset_train,
        asset_validation,
        asset_features,
        estimators,
        early_stopping_rounds,
        categorical=CATEGORICAL_FEATURES,
        weight_mode="sqrt_capacity" if energy_type == "solar" else "none",
    )
    asset_prediction = enforce_physics(
        energy_type,
        asset_validation,
        asset_model.predict(
            asset_validation[asset_features],
            num_iteration=asset_model.best_iteration_,
        ),
    )
    combined = aggregate_asset_prediction(
        asset_validation, asset_prediction, "asset_lightgbm", expected_assets
    )

    portfolio_model = fit_lgbm(
        portfolio_train,
        portfolio_validation,
        portfolio_features,
        estimators,
        early_stopping_rounds,
    )
    direct = enforce_portfolio_physics(
        energy_type,
        portfolio_validation,
        portfolio_model.predict(
            portfolio_validation[portfolio_features],
            num_iteration=portfolio_model.best_iteration_,
        ),
    )
    direct_frame = portfolio_validation[["issue_time_utc", "valid_time_utc"]].copy()
    direct_frame["direct_portfolio"] = direct
    combined = combined.merge(
        direct_frame,
        on=["issue_time_utc", "valid_time_utc"],
        validate="one_to_one",
    )

    if energy_type == "wind":
        directional = fit_wind_hybrid(
            asset_train,
            asset_validation,
            estimators,
            early_stopping_rounds,
            directional_curve=True,
        )
        directional_prediction = directional.predict(asset_validation)
        directional_frame = aggregate_asset_prediction(
            asset_validation,
            directional_prediction,
            "directional_hybrid",
            expected_assets,
        )
        combined = combined.merge(
            directional_frame[["issue_time_utc", "valid_time_utc", "directional_hybrid"]],
            on=["issue_time_utc", "valid_time_utc"],
            validate="one_to_one",
        )
    best_iterations = {
        "asset_lightgbm": int(asset_model.best_iteration_),
        "direct_portfolio": int(portfolio_model.best_iteration_),
    }
    if energy_type == "wind":
        best_iterations["directional_hybrid"] = int(directional.residual_model.best_iteration_)
    combined.attrs["best_iterations"] = best_iterations
    return combined


def fit_bias_table(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    working = frame.copy()
    working["hour_kst"] = working["valid_time_kst"].dt.hour
    working["residual"] = working["actual_capacity_factor"] - working[prediction_column]
    table = (
        working.groupby(["forecast_day_offset", "hour_kst"], observed=True)
        .agg(bias=("residual", "mean"), samples=("residual", "size"))
        .reset_index()
    )
    table["bias"] *= table["samples"] / (table["samples"] + 100.0)
    return table


def apply_bias(frame: pd.DataFrame, prediction: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    keys = frame[["forecast_day_offset"]].copy()
    keys["hour_kst"] = frame["valid_time_kst"].dt.hour
    correction = keys.merge(
        table, on=["forecast_day_offset", "hour_kst"], how="left", validate="many_to_one"
    )["bias"].fillna(0.0)
    return np.clip(np.asarray(prediction) + correction.to_numpy(), 0.0, 1.0)


def nmae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    error_kwh = np.abs(
        np.asarray(prediction) * frame["capacity_kw"].to_numpy() - frame["actual_kwh"].to_numpy()
    )
    return float(100.0 * error_kwh.sum() / frame["capacity_kw"].sum())


def run_energy(energy_type: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    frame, _ = development_frame(args.data / f"{energy_type}_training_mart.parquet")
    if energy_type == "solar":
        frame = add_solar_improvement_features(frame)
        weather_features = SOLAR_PORTFOLIO_WEATHER
    else:
        frame = add_wind_improvement_features(frame)
        weather_features = WIND_PORTFOLIO_WEATHER
    portfolio, portfolio_features = portfolio_frame(frame, weather_features)
    expected_assets = int(frame["asset_id"].nunique())
    portfolio = portfolio[portfolio["asset_count"].eq(expected_assets)].copy()

    oof_parts = []
    for fold in OOF_FOLDS:
        asset_train, asset_validation = split_fold(frame, fold)
        portfolio_train, portfolio_validation = split_fold(portfolio, fold)
        oof_parts.append(
            fit_predict_components(
                energy_type,
                asset_train,
                asset_validation,
                portfolio_train,
                portfolio_validation,
                portfolio_features,
                args.estimators,
                args.early_stopping,
                expected_assets,
            )
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    model_names = ["asset_lightgbm", "direct_portfolio"]
    if energy_type == "wind":
        model_names.append("directional_hybrid")
    weights = optimize_nonnegative_weights(
        oof[model_names].to_numpy(),
        oof["actual_capacity_factor"].to_numpy(),
        oof["capacity_kw"].to_numpy(),
    )
    oof["oof_weighted"] = oof[model_names].to_numpy() @ weights
    bias_table = fit_bias_table(oof, "oof_weighted")

    asset_train, asset_validation = split_fold(frame, FULL_2024)
    portfolio_train, portfolio_validation = split_fold(portfolio, FULL_2024)
    holdout = fit_predict_components(
        energy_type,
        asset_train,
        asset_validation,
        portfolio_train,
        portfolio_validation,
        portfolio_features,
        args.estimators,
        args.early_stopping,
        expected_assets,
    )
    holdout["oof_weighted"] = holdout[model_names].to_numpy() @ weights
    holdout["oof_weighted_bias_corrected"] = apply_bias(
        holdout, holdout["oof_weighted"].to_numpy(), bias_table
    )
    timezone = holdout["valid_time_kst"].dt.tz
    period_masks = {
        "2024_full": pd.Series(True, index=holdout.index),
        "2024_h1": holdout["valid_time_kst"].lt(pd.Timestamp("2024-07-01", tz=timezone)),
        "2024_h2": holdout["valid_time_kst"].ge(pd.Timestamp("2024-07-01", tz=timezone)),
        "day0_24h": holdout["forecast_day_offset"].eq(0),
        "day1_24h": holdout["forecast_day_offset"].eq(1),
    }
    rows = []
    for name in [*model_names, "oof_weighted", "oof_weighted_bias_corrected"]:
        for period, mask in period_masks.items():
            selected = holdout.loc[mask]
            rows.append(
                {
                    "energy_type": energy_type,
                    "model": name,
                    "period": period,
                    "portfolio_nmae_pct": nmae(selected, selected[name].to_numpy()),
                }
            )
    full_scores = {
        row["model"]: row["portfolio_nmae_pct"]
        for row in rows
        if row["period"] == "2024_full" and row["model"] != "oof_weighted_bias_corrected"
    }
    selected_model = min(full_scores, key=full_scores.get)
    if selected_model == "oof_weighted":
        production_weights = dict(zip(model_names, weights.tolist(), strict=True))
    else:
        production_weights = {selected_model: 1.0}
    details = {
        "meta_training_period": "2023 Q3-Q4 rolling OOF only",
        "evaluation_period": 2024,
        "locked_2025_loaded": False,
        "model_names": model_names,
        "weights": dict(zip(model_names, weights.tolist(), strict=True)),
        "selected_production_model": selected_model,
        "production_weights": production_weights,
        "bias_groups": int(len(bias_table)),
        "selected_best_iterations": holdout.attrs["best_iterations"],
    }
    return pd.DataFrame(rows), details


def main() -> None:
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    reports = []
    summary = {}
    for energy_type in ("solar", "wind"):
        print(f"OOF ensemble: {energy_type}", flush=True)
        report, details = run_energy(energy_type, args)
        reports.append(report)
        summary[energy_type] = details
    combined = pd.concat(reports, ignore_index=True)
    combined.to_csv(
        args.reports / "generation_oof_ensemble_2024.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.reports / "generation_oof_ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(combined.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
