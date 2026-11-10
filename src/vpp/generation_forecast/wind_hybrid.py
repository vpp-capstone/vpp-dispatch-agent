"""Leakage-safe wind power-curve and residual LightGBM helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from vpp.generation_forecast.model import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
    add_wind_change_features,
    make_model,
    training_sample_weights,
)

WIND_HYBRID_FEATURES = [*FEATURES["wind"], "power_curve_capacity_factor"]


@dataclass
class WindPowerCurve:
    """Per-asset monotonic wind-speed-to-capacity-factor curves."""

    by_asset: dict[str, IsotonicRegression]
    fallback: IsotonicRegression
    bin_width_ms: float = 0.5

    @classmethod
    def fit(cls, frame: pd.DataFrame, bin_width_ms: float = 0.5) -> WindPowerCurve:
        def fit_curve(part: pd.DataFrame) -> IsotonicRegression:
            # GFS wind speeds are nearly continuous.  Binning prevents an isotonic
            # curve from memorizing small, label-specific steps in the training year.
            binned_speed = (
                part["wind_speed_80m_ms"].to_numpy(dtype=float) / bin_width_ms
            ).round() * bin_width_ms
            points = (
                pd.DataFrame({"speed": binned_speed, "target": part[TARGET].to_numpy()})
                .groupby("speed", observed=True)["target"]
                .agg(["mean", "size"])
                .reset_index()
            )
            curve = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            curve.fit(
                points["speed"].to_numpy(dtype=float),
                points["mean"].to_numpy(dtype=float),
                sample_weight=points["size"].to_numpy(dtype=float),
            )
            return curve

        return cls(
            by_asset={str(asset): fit_curve(part) for asset, part in frame.groupby("asset_id")},
            fallback=fit_curve(frame),
            bin_width_ms=bin_width_ms,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = np.empty(len(frame), dtype=float)
        asset_values = frame["asset_id"].astype(str).to_numpy()
        speed = frame["wind_speed_80m_ms"].to_numpy(dtype=float)
        for asset in np.unique(asset_values):
            mask = asset_values == asset
            curve = self.by_asset.get(asset, self.fallback)
            prediction[mask] = curve.predict(speed[mask])
        return prediction


@dataclass
class DirectionalWindPowerCurve:
    """Per-asset, wind-direction-sector monotonic power curves."""

    by_asset_sector: dict[tuple[str, int], IsotonicRegression]
    by_asset: WindPowerCurve
    sectors: int = 12

    @staticmethod
    def _sector(direction: pd.Series, sectors: int) -> np.ndarray:
        width = 360.0 / sectors
        return np.floor((direction.to_numpy(dtype=float) + width / 2) / width).astype(int) % sectors

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        sectors: int = 12,
        bin_width_ms: float = 0.5,
        minimum_rows: int = 200,
    ) -> DirectionalWindPowerCurve:
        base = WindPowerCurve.fit(frame, bin_width_ms=bin_width_ms)
        working = frame.copy()
        working["direction_sector"] = cls._sector(working["wind_direction_80m_deg"], sectors)
        curves: dict[tuple[str, int], IsotonicRegression] = {}
        for (asset, sector), part in working.groupby(
            ["asset_id", "direction_sector"], observed=True
        ):
            if len(part) >= minimum_rows and part["wind_speed_80m_ms"].nunique() >= 4:
                curves[(str(asset), int(sector))] = WindPowerCurve.fit(
                    part, bin_width_ms=bin_width_ms
                ).fallback
        return cls(curves, base, sectors)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = self.by_asset.predict(frame)
        assets = frame["asset_id"].astype(str).to_numpy()
        sectors = self._sector(frame["wind_direction_80m_deg"], self.sectors)
        speed = frame["wind_speed_80m_ms"].to_numpy(dtype=float)
        for key in set(zip(assets, sectors, strict=True)):
            curve = self.by_asset_sector.get(key)
            if curve is None:
                continue
            mask = (assets == key[0]) & (sectors == key[1])
            prediction[mask] = curve.predict(speed[mask])
        return prediction


@dataclass
class WindHybridModel:
    """Power-curve baseline plus a LightGBM residual correction."""

    power_curve: WindPowerCurve | DirectionalWindPowerCurve
    residual_model: Any
    features: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        featured = add_wind_change_features(frame)
        curve_prediction = self.power_curve.predict(featured)
        featured["power_curve_capacity_factor"] = curve_prediction
        iteration = getattr(self.residual_model, "best_iteration_", None)
        residual = self.residual_model.predict(featured[self.features], num_iteration=iteration)
        return np.clip(curve_prediction + residual, 0.0, 1.0)


def fit_wind_hybrid(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    estimators: int,
    early_stopping_rounds: int,
    *,
    weight_mode: str = "none",
    model_params: dict[str, Any] | None = None,
    directional_curve: bool = False,
) -> WindHybridModel:
    """Fit the curve on training labels and LightGBM on its remaining error."""
    import lightgbm as lgb

    train_featured = add_wind_change_features(train)
    validation_featured = add_wind_change_features(validation)
    curve = (
        DirectionalWindPowerCurve.fit(train_featured)
        if directional_curve
        else WindPowerCurve.fit(train_featured)
    )
    train_curve = curve.predict(train_featured)
    validation_curve = curve.predict(validation_featured)
    train_featured["power_curve_capacity_factor"] = train_curve
    validation_featured["power_curve_capacity_factor"] = validation_curve

    model = make_model(estimators, **(model_params or {}))
    model.fit(
        train_featured[WIND_HYBRID_FEATURES],
        train_featured[TARGET].to_numpy(dtype=float) - train_curve,
        sample_weight=training_sample_weights(train_featured, weight_mode),
        categorical_feature=CATEGORICAL_FEATURES,
        eval_X=validation_featured[WIND_HYBRID_FEATURES],
        eval_y=validation_featured[TARGET].to_numpy(dtype=float) - validation_curve,
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return WindHybridModel(curve, model, WIND_HYBRID_FEATURES)


def fit_wind_hybrid_fixed(
    train: pd.DataFrame,
    estimators: int,
    *,
    weight_mode: str = "none",
    model_params: dict[str, Any] | None = None,
    directional_curve: bool = False,
) -> WindHybridModel:
    """Refit a selected hybrid model without using another label split."""
    import lightgbm as lgb

    featured = add_wind_change_features(train)
    curve = (
        DirectionalWindPowerCurve.fit(featured)
        if directional_curve
        else WindPowerCurve.fit(featured)
    )
    curve_prediction = curve.predict(featured)
    featured["power_curve_capacity_factor"] = curve_prediction
    model = make_model(estimators, **(model_params or {}))
    model.fit(
        featured[WIND_HYBRID_FEATURES],
        featured[TARGET].to_numpy(dtype=float) - curve_prediction,
        sample_weight=training_sample_weights(featured, weight_mode),
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return WindHybridModel(curve, model, WIND_HYBRID_FEATURES)
