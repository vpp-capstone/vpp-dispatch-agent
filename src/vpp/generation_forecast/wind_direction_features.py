"""Forecast-derived wind direction interaction features."""

from __future__ import annotations

import numpy as np
import pandas as pd

WIND_DIRECTION_INTERACTION_FEATURES = [
    "aws_proxy_wind_sector_8",
    "aws_proxy_speed_x_direction_sin",
    "aws_proxy_speed_x_direction_cos",
    "gfs_aws_direction_alignment",
    "gfs_aws_direction_cross",
]


def add_wind_direction_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add sector and smooth direction interactions without observed future weather."""
    result = frame.copy()
    aws_sin = result["aws_proxy_wind_direction_sin"].to_numpy(dtype=float)
    aws_cos = result["aws_proxy_wind_direction_cos"].to_numpy(dtype=float)
    angle = np.mod(np.arctan2(aws_sin, aws_cos), 2.0 * np.pi)
    result["aws_proxy_wind_sector_8"] = np.floor(angle / (2.0 * np.pi / 8.0)).astype(
        "int8"
    )
    speed = result["aws_proxy_wind_speed_ms"].to_numpy(dtype=float)
    result["aws_proxy_speed_x_direction_sin"] = speed * aws_sin
    result["aws_proxy_speed_x_direction_cos"] = speed * aws_cos
    gfs_sin = result["wind_direction_sin"].to_numpy(dtype=float)
    gfs_cos = result["wind_direction_cos"].to_numpy(dtype=float)
    result["gfs_aws_direction_alignment"] = gfs_sin * aws_sin + gfs_cos * aws_cos
    result["gfs_aws_direction_cross"] = gfs_sin * aws_cos - gfs_cos * aws_sin
    return result
