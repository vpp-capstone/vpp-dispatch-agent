"""Train and run the selected 2023-2024 generation portfolio ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp.generation_forecast.aws_features import (
    AWS_SOLAR_FEATURES,
    AWS_WIND_FEATURES,
    PRIMARY_STATIONS,
    SOLAR_PRIMARY_STATIONS,
    AwsSolarCalibrator,
    AwsWindCalibrator,
    fit_aws_solar_calibrator,
    fit_aws_wind_calibrator,
    load_aws_observations,
)
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
from vpp.generation_forecast.paths import MODEL_DIR, PROCESSED_DIR, RESULTS_DIR
from vpp.generation_forecast.tune_lightgbm import development_frame
from vpp.generation_forecast.wind_hybrid import fit_wind_hybrid_fixed

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
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument("--reports", type=Path, default=RESULTS_DIR)
    parser.add_argument("--aws", type=Path, default=Path("data/raw/aws"))
    return parser.parse_args()


def fit_fixed_lgbm(
    frame: pd.DataFrame,
    features: list[str],
    estimators: int,
    *,
    categorical: list[str] | None = None,
    weight_mode: str = "none",
) -> Any:
    import lightgbm as lgb

    model = make_model(estimators)
    model.fit(
        frame[features],
        frame[TARGET],
        sample_weight=training_sample_weights(frame, weight_mode),
        categorical_feature=categorical or [],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model


def train_selected(
    energy_type: str,
    frame: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
    selection: dict[str, Any],
    aws_observations: pd.DataFrame | None = None,
) -> dict[str, Any]:
    aws_calibrator: AwsWindCalibrator | None = None
    aws_solar_calibrator: AwsSolarCalibrator | None = None
    if energy_type == "solar":
        featured = add_solar_improvement_features(frame)
        weather_features = SOLAR_PORTFOLIO_WEATHER
        if aws_observations is not None:
            aws_solar_calibrator = fit_aws_solar_calibrator(
                featured, aws_observations, SOLAR_PRIMARY_STATIONS
            )
            featured = aws_solar_calibrator.transform(featured)
    else:
        featured = add_wind_improvement_features(frame)
        weather_features = WIND_PORTFOLIO_WEATHER
        if aws_observations is not None:
            aws_calibrator = fit_aws_wind_calibrator(
                featured, aws_observations, PRIMARY_STATIONS
            )
            featured = aws_calibrator.transform(featured)
    portfolio, portfolio_features = portfolio_frame(featured, weather_features)
    iterations = selection["selected_best_iterations"]
    selected_weights = selection.get("production_weights", selection["weights"])
    models: dict[str, Any] = {}
    if float(selected_weights.get("direct_portfolio", 0.0)) >= 1e-8:
        models["direct_portfolio"] = fit_fixed_lgbm(
            portfolio,
            portfolio_features,
            int(iterations["direct_portfolio"]),
        )
    if float(selected_weights.get("asset_lightgbm", 0.0)) >= 1e-8:
        asset_features = (
            [
                *FEATURES["solar"],
                *(AWS_SOLAR_FEATURES if aws_solar_calibrator is not None else []),
            ]
            if energy_type == "solar" else [
                *FEATURES["wind"],
                *WIND_IMPROVEMENT_FEATURES,
                *(AWS_WIND_FEATURES if aws_calibrator is not None else []),
            ]
        )
        models["asset_lightgbm"] = fit_fixed_lgbm(
            featured,
            asset_features,
            int(iterations["asset_lightgbm"]),
            categorical=CATEGORICAL_FEATURES,
            weight_mode="sqrt_capacity" if energy_type == "solar" else "none",
        )
    if float(selected_weights.get("directional_hybrid", 0.0)) >= 1e-8:
        models["directional_hybrid"] = fit_wind_hybrid_fixed(
            featured,
            int(iterations["directional_hybrid"]),
            directional_curve=True,
        )
    weights = {
        name: float(weight)
        for name, weight in selected_weights.items()
        if name in models and float(weight) >= 1e-8
    }
    total = sum(weights.values())
    weights = {name: weight / total for name, weight in weights.items()}
    return {
        "energy_type": energy_type,
        "models": models,
        "weights": weights,
        "asset_features": (
            [
                *FEATURES["solar"],
                *(AWS_SOLAR_FEATURES if aws_solar_calibrator is not None else []),
            ]
            if energy_type == "solar" else [
                *FEATURES["wind"],
                *WIND_IMPROVEMENT_FEATURES,
                *(AWS_WIND_FEATURES if aws_calibrator is not None else []),
            ]
        ),
        "portfolio_features": portfolio_features,
        "portfolio_weather_features": weather_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_maps": category_maps,
        "training_years": [2023, 2024],
        "selection_years": [2023, 2024],
        "locked_2025_loaded": False,
        "aws_wind_calibrator": aws_calibrator,
        "aws_solar_calibrator": aws_solar_calibrator,
        "aws_station_map": (
            aws_calibrator.station_map
            if aws_calibrator is not None
            else aws_solar_calibrator.station_map
            if aws_solar_calibrator is not None
            else None
        ),
        "aws_calibration_fit_rows": (
            aws_calibrator.fit_rows
            if aws_calibrator is not None
            else aws_solar_calibrator.fit_rows
            if aws_solar_calibrator is not None
            else None
        ),
        "bias_correction_applied": (
            aws_calibrator is not None or aws_solar_calibrator is not None
        ),
    }


def predict_rows(bundle: dict[str, Any], prepared_frame: pd.DataFrame) -> pd.DataFrame:
    """Predict per-asset rows while preserving the selected portfolio total."""
    energy_type = bundle["energy_type"]
    frame = (
        add_solar_improvement_features(prepared_frame)
        if energy_type == "solar"
        else add_wind_improvement_features(prepared_frame)
    )
    if energy_type == "wind" and bundle.get("aws_wind_calibrator") is not None:
        frame = bundle["aws_wind_calibrator"].transform(frame)
    if energy_type == "solar" and bundle.get("aws_solar_calibrator") is not None:
        frame = bundle["aws_solar_calibrator"].transform(frame)
    keys = ["issue_time_utc", "valid_time_utc"]
    component_rows: dict[str, np.ndarray] = {}
    if "asset_lightgbm" in bundle["models"]:
        asset_cf = enforce_physics(
            energy_type,
            frame,
            bundle["models"]["asset_lightgbm"].predict(frame[bundle["asset_features"]]),
        )
        component_rows["asset_lightgbm"] = asset_cf * frame["capacity_kw"].to_numpy()
    if "directional_hybrid" in bundle["models"]:
        directional_cf = bundle["models"]["directional_hybrid"].predict(frame)
        component_rows["directional_hybrid"] = directional_cf * frame["capacity_kw"].to_numpy()
    result = frame.copy()
    weights = bundle["weights"]
    prediction = np.zeros(len(result), dtype=float)
    for name, rows in component_rows.items():
        prediction = prediction + weights.get(name, 0.0) * rows
    if "direct_portfolio" in bundle["models"]:
        portfolio, _ = portfolio_frame(frame, bundle["portfolio_weather_features"])
        direct_cf = enforce_portfolio_physics(
            energy_type,
            portfolio,
            bundle["models"]["direct_portfolio"].predict(portfolio[bundle["portfolio_features"]]),
        )
        direct = portfolio[keys + ["capacity_kw"]].copy()
        direct["direct_total_kwh"] = direct_cf * direct["capacity_kw"]
        result = result.merge(direct[keys + ["direct_total_kwh"]], on=keys, validate="many_to_one")
        allocation_name = (
            "asset_lightgbm" if "asset_lightgbm" in component_rows else "directional_hybrid"
        )
        result["allocation_base_kwh"] = component_rows[allocation_name]
        base_total = result.groupby(keys, observed=True)["allocation_base_kwh"].transform("sum")
        capacity_total = result.groupby(keys, observed=True)["capacity_kw"].transform("sum")
        share = np.divide(
            result["allocation_base_kwh"],
            base_total,
            out=np.zeros(len(result), dtype=float),
            where=base_total.to_numpy() > 0,
        )
        zero_total = base_total.le(0).to_numpy()
        share[zero_total] = (result["capacity_kw"] / capacity_total).to_numpy()[zero_total]
        prediction = prediction + (
            weights["direct_portfolio"] * result["direct_total_kwh"].to_numpy() * share
        )
    result["improved_generation_kwh"] = prediction
    return result


def main() -> None:
    import joblib

    args = parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    selection = json.loads(
        (args.reports / "generation_oof_ensemble_summary.json").read_text(encoding="utf-8")
    )
    aws_observations = load_aws_observations(args.aws)
    aws_holdout = pd.read_csv(args.reports / "wind_aws_2024_holdout.csv")
    selected_aws = aws_holdout[aws_holdout["candidate"].eq("aws_primary_proxy")]
    if len(selected_aws) != 1:
        raise ValueError("wind AWS holdout must contain one aws_primary_proxy row")
    selection["wind"]["selected_best_iterations"]["asset_lightgbm"] = int(
        selected_aws.iloc[0]["best_iteration"]
    )
    solar_aws_holdout = pd.read_csv(args.reports / "solar_aws_2024_holdout.csv")
    selected_solar_aws = solar_aws_holdout[
        solar_aws_holdout["candidate"].eq("aws_asset_current_direct")
    ]
    if len(selected_solar_aws) != 1:
        raise ValueError("solar AWS holdout must contain one selected hybrid row")
    selection["solar"]["selected_best_iterations"]["asset_lightgbm"] = int(
        selected_solar_aws.iloc[0]["asset_best_iteration"]
    )
    summary: dict[str, Any] = {
        "training_years": [2023, 2024],
        "locked_2025_loaded": False,
        "models": {},
    }
    for energy_type in ("solar", "wind"):
        frame, category_maps = development_frame(args.data / f"{energy_type}_training_mart.parquet")
        bundle = train_selected(
            energy_type,
            frame,
            category_maps,
            selection[energy_type],
            aws_observations,
        )
        path = args.models / f"{energy_type}_improved_production.joblib"
        joblib.dump(bundle, path)
        summary["models"][energy_type] = {
            "path": path.name,
            "weights": bundle["weights"],
            "rows": len(frame),
            "components": list(bundle["models"]),
            "bias_correction_applied": bundle["bias_correction_applied"],
            "aws_station_map": bundle["aws_station_map"],
        }
    (args.reports / "generation_improved_production_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
