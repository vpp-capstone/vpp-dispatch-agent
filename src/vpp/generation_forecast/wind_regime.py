"""Wind-speed regime helpers shared by tuning and production inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vpp.generation_forecast.model import TARGET, make_model


def wind_speed_regime(speed: np.ndarray, boundaries: tuple[float, ...]) -> np.ndarray:
    """Assign deterministic zero-based regimes using forecast-derived wind speed."""
    values = np.asarray(speed, dtype=float)
    pairs = zip(boundaries, boundaries[1:], strict=False)
    if not boundaries or any(right <= left for left, right in pairs):
        raise ValueError("boundaries must be strictly increasing")
    return np.digitize(values, boundaries, right=False).astype("int8")


@dataclass
class DynamicWindRegimeModel:
    """Joint and speed-regime LightGBMs blended with OOF-derived weights."""

    joint_model: Any
    expert_models: dict[int, Any]
    features: list[str]
    boundaries: tuple[float, ...]
    expert_weights: np.ndarray

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        joint = np.asarray(self.joint_model.predict(frame[self.features]), dtype=float)
        regimes = wind_speed_regime(
            frame["aws_proxy_wind_speed_ms"].to_numpy(), self.boundaries
        )
        expert = np.full(len(frame), np.nan, dtype=float)
        for regime, model in self.expert_models.items():
            mask = regimes == regime
            if mask.any():
                expert[mask] = model.predict(frame.loc[mask, self.features])
        if np.isnan(expert).any():
            missing = sorted(np.unique(regimes[np.isnan(expert)]).tolist())
            raise ValueError(f"missing wind regime experts: {missing}")
        row_weights = self.expert_weights[regimes]
        return (1.0 - row_weights) * joint + row_weights * expert


@dataclass
class AssetBlendedWindModel:
    """Blend the shared regime model with farm-specific LightGBMs."""

    shared_model: DynamicWindRegimeModel
    asset_models: dict[str, Any]
    asset_weights: dict[str, float]
    features: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        shared = self.shared_model.predict(frame)
        result = shared.copy()
        for asset_id, model in self.asset_models.items():
            mask = frame["asset_id"].eq(asset_id)
            if not mask.any():
                continue
            asset_prediction = model.predict(frame.loc[mask, self.features])
            weight = self.asset_weights[asset_id]
            positions = np.flatnonzero(mask.to_numpy())
            result[positions] = shared[positions] + weight * (
                asset_prediction - shared[positions]
            )
        return result


def fit_dynamic_wind_regime_model(
    frame: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    boundaries: tuple[float, ...],
    expert_weights: np.ndarray,
    joint_iterations: int,
    expert_iterations: dict[int, int],
    model_params: dict[str, Any],
) -> DynamicWindRegimeModel:
    """Fit the selected regime ensemble on all available development rows."""
    import lightgbm as lgb

    joint = make_model(joint_iterations, **model_params)
    joint.fit(
        frame[features],
        frame[TARGET],
        categorical_feature=categorical,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    regimes = wind_speed_regime(
        frame["aws_proxy_wind_speed_ms"].to_numpy(), boundaries
    )
    experts: dict[int, Any] = {}
    for regime, iterations in expert_iterations.items():
        mask = regimes == regime
        if mask.sum() < 720:
            raise ValueError(f"regime {regime} has insufficient production rows: {mask.sum()}")
        model = make_model(iterations, **model_params)
        model.fit(
            frame.loc[mask, features],
            frame.loc[mask, TARGET],
            categorical_feature=categorical,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        experts[regime] = model
    return DynamicWindRegimeModel(
        joint_model=joint,
        expert_models=experts,
        features=features,
        boundaries=boundaries,
        expert_weights=np.asarray(expert_weights, dtype=float),
    )


def fit_asset_blended_wind_model(
    frame: pd.DataFrame,
    shared_model: DynamicWindRegimeModel,
    features: list[str],
    categorical: list[str],
    asset_weights: dict[str, float],
    asset_iterations: dict[str, int],
    model_params: dict[str, Any],
) -> AssetBlendedWindModel:
    """Fit farm-specific components using fixed OOF-selected iterations."""
    import lightgbm as lgb

    asset_models: dict[str, Any] = {}
    for asset_id, weight in asset_weights.items():
        if weight <= 1e-8:
            continue
        subset = frame[frame["asset_id"].eq(asset_id)]
        model = make_model(asset_iterations[asset_id], **model_params)
        model.fit(
            subset[features],
            subset[TARGET],
            categorical_feature=categorical,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        asset_models[asset_id] = model
    return AssetBlendedWindModel(
        shared_model=shared_model,
        asset_models=asset_models,
        asset_weights=asset_weights,
        features=features,
    )
