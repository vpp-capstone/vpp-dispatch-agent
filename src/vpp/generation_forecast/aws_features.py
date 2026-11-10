"""Leakage-safe AWS-derived wind calibration features for generation forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KST = "Asia/Seoul"

# Missing stations are skipped. This lets 금악(880) join automatically later.
PRIMARY_STATIONS = {
    "wind_hangyeong": (990,),
    "wind_sangmyeong": (880, 883),
    "wind_seongsan": (892,),
}
MULTI_STATIONS = {
    "wind_hangyeong": (990, 779, 793),
    "wind_sangmyeong": (880, 883, 779, 990),
    "wind_seongsan": (892, 781, 792, 862),
}

SOLAR_PRIMARY_STATIONS = {
    "solar_geumak": (883,),
    "solar_geumoreum": (883,),
    "solar_namwon": (792,),
    "solar_mirinae": (792,),
    "solar_bongseong": (883,),
    "solar_seongeup": (792,),
    "solar_sumang": (792,),
    "solar_aewol": (883,),
    "solar_oreumjeju": (792,),
    "solar_uribyeol": (792,),
    "solar_uri": (883,),
    "solar_wolpyeong": (862,),
    "solar_jejutop": (793,),
    "solar_kone": (792,),
    "solar_tamra": (792,),
    "solar_pyeongdae": (781,),
    "solar_pyoseon": (792,),
    "solar_handong": (781,),
}

CALIBRATOR_INPUTS = [
    "wind_u_80m_ms", "wind_v_80m_ms", "wind_speed_80m_ms",
    "wind_direction_sin", "wind_direction_cos", "temperature_2m_c",
    "total_cloud_cover_pct", "hour_sin", "hour_cos",
    "day_of_year_sin", "day_of_year_cos",
]

AWS_WIND_FEATURES = [
    "aws_proxy_wind_u_ms", "aws_proxy_wind_v_ms", "aws_proxy_wind_speed_ms",
    "aws_proxy_wind_direction_sin", "aws_proxy_wind_direction_cos",
    "aws_proxy_speed_minus_gfs_ms", "aws_proxy_u_minus_gfs_ms",
    "aws_proxy_v_minus_gfs_ms",
]

SOLAR_CALIBRATOR_INPUTS = [
    "shortwave_radiation_w_m2", "temperature_2m_c", "total_cloud_cover_pct",
    "solar_elevation_deg", "is_daylight", "hour_sin", "hour_cos",
    "day_of_year_sin", "day_of_year_cos", "lead_hour",
]

AWS_SOLAR_FEATURES = [
    "aws_proxy_temperature_c", "aws_proxy_humidity_pct",
    "aws_proxy_temperature_minus_gfs_c", "aws_proxy_dew_point_c",
    "aws_proxy_vapor_pressure_deficit_kpa", "aws_proxy_humidity_x_cloud",
    "aws_proxy_temperature_x_radiation",
]


def load_aws_observations(root: Path) -> pd.DataFrame:
    """Load KMA hourly AWS CSV files without assuming UTF-8 input."""
    files = sorted(root.glob("SURFACE_AWS_*_HR_*.csv"))
    if not files:
        raise FileNotFoundError(f"no KMA AWS hourly CSV files under {root}")
    parts: list[pd.DataFrame] = []
    for path in files:
        raw = pd.read_csv(path, encoding="cp949")
        if raw.shape[1] < 9:
            raise ValueError(f"{path}: expected at least nine AWS columns")
        part = raw.iloc[:, [0, 1, 2, 3, 4, 8]].copy()
        part.columns = [
            "station_id", "valid_time_kst", "temperature_c",
            "wind_direction_deg", "wind_speed_ms", "humidity_pct",
        ]
        part["station_id"] = pd.to_numeric(part["station_id"], errors="raise").astype("int16")
        part["valid_time_kst"] = pd.to_datetime(
            part["valid_time_kst"], errors="raise"
        ).dt.tz_localize(KST)
        for column in [
            "temperature_c", "wind_direction_deg", "wind_speed_ms", "humidity_pct"
        ]:
            part[column] = pd.to_numeric(part[column], errors="coerce")
        part["source_file"] = path.name
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    duplicate = result.duplicated(["station_id", "valid_time_kst"], keep=False)
    if duplicate.any():
        examples = result.loc[
            duplicate, ["station_id", "valid_time_kst", "source_file"]
        ].head()
        raise ValueError(f"duplicate AWS station timestamps:\n{examples.to_string(index=False)}")
    direction = np.deg2rad(result["wind_direction_deg"].to_numpy(dtype=float))
    speed = result["wind_speed_ms"].to_numpy(dtype=float)
    # Meteorological direction is where the wind comes from.
    result["aws_wind_u_ms"] = -speed * np.sin(direction)
    result["aws_wind_v_ms"] = -speed * np.cos(direction)
    return result.sort_values(["station_id", "valid_time_kst"]).reset_index(drop=True)


def aggregate_asset_observations(
    observations: pd.DataFrame,
    station_map: dict[str, tuple[int, ...]],
) -> pd.DataFrame:
    """Create an observed surface-wind target for each farm and hour."""
    available = set(observations["station_id"].unique())
    parts: list[pd.DataFrame] = []
    for asset_id, requested in station_map.items():
        selected = [station for station in requested if station in available]
        if not selected:
            raise ValueError(f"{asset_id}: none of AWS stations {requested} are available")
        subset = observations[observations["station_id"].isin(selected)].copy()
        grouped = (
            subset.groupby("valid_time_kst", observed=True)
            .agg(
                aws_observed_u_ms=("aws_wind_u_ms", "mean"),
                aws_observed_v_ms=("aws_wind_v_ms", "mean"),
            )
            .reset_index()
        )
        grouped["asset_id"] = asset_id
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True)


@dataclass
class AwsWindCalibrator:
    """Per-asset ridge models that turn operational GFS inputs into AWS proxies."""

    models: dict[str, object]
    station_map: dict[str, tuple[int, ...]]
    fit_rows: dict[str, int]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in AWS_WIND_FEATURES:
            result[column] = np.nan
        for asset_id, model in self.models.items():
            mask = result["asset_id"].eq(asset_id)
            if not mask.any():
                continue
            predicted = np.asarray(
                model.predict(result.loc[mask, CALIBRATOR_INPUTS]), dtype=float
            )
            u, v = predicted[:, 0], predicted[:, 1]
            speed = np.hypot(u, v)
            safe = np.maximum(speed, 1e-6)
            result.loc[mask, "aws_proxy_wind_u_ms"] = u
            result.loc[mask, "aws_proxy_wind_v_ms"] = v
            result.loc[mask, "aws_proxy_wind_speed_ms"] = speed
            result.loc[mask, "aws_proxy_wind_direction_sin"] = -u / safe
            result.loc[mask, "aws_proxy_wind_direction_cos"] = -v / safe
            result.loc[mask, "aws_proxy_speed_minus_gfs_ms"] = (
                speed - result.loc[mask, "wind_speed_80m_ms"].to_numpy(dtype=float)
            )
            result.loc[mask, "aws_proxy_u_minus_gfs_ms"] = (
                u - result.loc[mask, "wind_u_80m_ms"].to_numpy(dtype=float)
            )
            result.loc[mask, "aws_proxy_v_minus_gfs_ms"] = (
                v - result.loc[mask, "wind_v_80m_ms"].to_numpy(dtype=float)
            )
        if result[AWS_WIND_FEATURES].isna().any().any():
            missing = result[AWS_WIND_FEATURES].isna().any(axis=1)
            assets = sorted(result.loc[missing, "asset_id"].unique())
            raise ValueError(f"AWS calibration features missing for assets: {assets}")
        return result


def fit_aws_wind_calibrator(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    station_map: dict[str, tuple[int, ...]] = PRIMARY_STATIONS,
) -> AwsWindCalibrator:
    """Fit using AWS observations whose valid times occur in ``frame`` only."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    available = set(observations["station_id"].unique())
    effective_station_map = {
        asset_id: tuple(station for station in requested if station in available)
        for asset_id, requested in station_map.items()
    }
    targets = aggregate_asset_observations(observations, effective_station_map)
    training = frame.merge(
        targets, on=["asset_id", "valid_time_kst"], how="left", validate="many_to_one"
    )
    models: dict[str, object] = {}
    fit_rows: dict[str, int] = {}
    target_columns = ["aws_observed_u_ms", "aws_observed_v_ms"]
    for asset_id in sorted(frame["asset_id"].unique()):
        subset = training[training["asset_id"].eq(asset_id)].dropna(
            subset=[*CALIBRATOR_INPUTS, *target_columns]
        )
        if len(subset) < 24 * 30:
            raise ValueError(f"{asset_id}: only {len(subset)} complete AWS calibration rows")
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(subset[CALIBRATOR_INPUTS], subset[target_columns])
        models[asset_id] = model
        fit_rows[asset_id] = int(len(subset))
    return AwsWindCalibrator(
        models=models, station_map=effective_station_map, fit_rows=fit_rows
    )


def aggregate_solar_asset_observations(
    observations: pd.DataFrame,
    station_map: dict[str, tuple[int, ...]],
) -> pd.DataFrame:
    """Create hourly temperature and humidity targets for each solar asset."""
    available = set(observations["station_id"].unique())
    parts: list[pd.DataFrame] = []
    for asset_id, requested in station_map.items():
        selected = [station for station in requested if station in available]
        if not selected:
            raise ValueError(f"{asset_id}: none of AWS stations {requested} are available")
        subset = observations[observations["station_id"].isin(selected)]
        grouped = (
            subset.groupby("valid_time_kst", observed=True)
            .agg(
                aws_observed_temperature_c=("temperature_c", "mean"),
                aws_observed_humidity_pct=("humidity_pct", "mean"),
            )
            .reset_index()
        )
        grouped["asset_id"] = asset_id
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True)


@dataclass
class AwsSolarCalibrator:
    """Per-asset GFS-to-AWS temperature and humidity proxy models."""

    models: dict[str, object]
    station_map: dict[str, tuple[int, ...]]
    fit_rows: dict[str, int]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in AWS_SOLAR_FEATURES:
            result[column] = np.nan
        for asset_id, model in self.models.items():
            mask = result["asset_id"].eq(asset_id)
            if not mask.any():
                continue
            predicted = np.asarray(
                model.predict(result.loc[mask, SOLAR_CALIBRATOR_INPUTS]), dtype=float
            )
            temperature = predicted[:, 0]
            humidity = np.clip(predicted[:, 1], 1.0, 100.0)
            gamma = np.log(humidity / 100.0) + (
                17.625 * temperature / (243.04 + temperature)
            )
            dew_point = 243.04 * gamma / (17.625 - gamma)
            saturation = 0.6108 * np.exp(
                17.27 * temperature / (temperature + 237.3)
            )
            vapor_pressure_deficit = saturation * (1.0 - humidity / 100.0)
            result.loc[mask, "aws_proxy_temperature_c"] = temperature
            result.loc[mask, "aws_proxy_humidity_pct"] = humidity
            result.loc[mask, "aws_proxy_temperature_minus_gfs_c"] = (
                temperature - result.loc[mask, "temperature_2m_c"].to_numpy(dtype=float)
            )
            result.loc[mask, "aws_proxy_dew_point_c"] = dew_point
            result.loc[mask, "aws_proxy_vapor_pressure_deficit_kpa"] = vapor_pressure_deficit
            result.loc[mask, "aws_proxy_humidity_x_cloud"] = (
                humidity * result.loc[mask, "total_cloud_cover_pct"].to_numpy(dtype=float)
            )
            result.loc[mask, "aws_proxy_temperature_x_radiation"] = (
                temperature
                * result.loc[mask, "shortwave_radiation_w_m2"].to_numpy(dtype=float)
            )
        if result[AWS_SOLAR_FEATURES].isna().any().any():
            missing = result[AWS_SOLAR_FEATURES].isna().any(axis=1)
            assets = sorted(result.loc[missing, "asset_id"].unique())
            raise ValueError(f"AWS solar calibration features missing for assets: {assets}")
        return result


def fit_aws_solar_calibrator(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    station_map: dict[str, tuple[int, ...]] = SOLAR_PRIMARY_STATIONS,
) -> AwsSolarCalibrator:
    """Fit solar weather proxies using only observation times present in ``frame``."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    available = set(observations["station_id"].unique())
    effective_station_map = {
        asset_id: tuple(station for station in requested if station in available)
        for asset_id, requested in station_map.items()
    }
    targets = aggregate_solar_asset_observations(observations, effective_station_map)
    training = frame.merge(
        targets, on=["asset_id", "valid_time_kst"], how="left", validate="many_to_one"
    )
    target_columns = ["aws_observed_temperature_c", "aws_observed_humidity_pct"]
    models: dict[str, object] = {}
    fit_rows: dict[str, int] = {}
    for asset_id in sorted(frame["asset_id"].unique()):
        subset = training[training["asset_id"].eq(asset_id)].dropna(
            subset=[*SOLAR_CALIBRATOR_INPUTS, *target_columns]
        )
        if len(subset) < 24 * 30:
            raise ValueError(f"{asset_id}: only {len(subset)} complete AWS calibration rows")
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(subset[SOLAR_CALIBRATOR_INPUTS], subset[target_columns])
        models[asset_id] = model
        fit_rows[asset_id] = int(len(subset))
    return AwsSolarCalibrator(
        models=models, station_map=effective_station_map, fit_rows=fit_rows
    )
