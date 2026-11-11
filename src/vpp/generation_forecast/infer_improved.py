"""Export improved per-asset predictions for MILP, using trained category maps.

Only dates after the production model's training years are accepted. Labels are
used for evaluation only, never to select or fill prediction rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vpp.generation_forecast.improved_model import predict_rows
from vpp.generation_forecast.model import add_wind_change_features

DELIVERY_COLUMNS = [
    "asset_id",
    "energy_type",
    "capacity_kw",
    "planning_day_kst",
    "issue_time_utc",
    "available_before_bid_deadline",
    "valid_time_utc",
    "valid_time_kst",
    "lead_hour",
    "forecast_day_offset",
    "improved_generation_kwh",
]


def prepare_inference(raw: pd.DataFrame, bundle: dict, start: str, end: str) -> pd.DataFrame:
    frame = raw.copy()
    ts = pd.to_datetime(frame.valid_time_kst, utc=True).dt.tz_convert("Asia/Seoul")
    if pd.Timestamp(start).year <= max(bundle["training_years"]):
        raise ValueError("학습에 사용한 연도로 미래 예측 성능을 평가할 수 없습니다.")
    if "planning_day_kst" in frame:
        planning_day = pd.to_datetime(frame["planning_day_kst"]).dt.date
    elif "forecast_day_offset" in frame:
        planning_day = (ts - pd.to_timedelta(frame["forecast_day_offset"], unit="D")).dt.date
    else:
        planning_day = ts.dt.date
    mask = (planning_day >= pd.Timestamp(start).date()) & (
        planning_day <= pd.Timestamp(end).date()
    )
    frame = frame.loc[mask].copy()
    if frame.empty:
        raise ValueError("개선 모델을 적용할 대상 기간의 기상 입력이 없습니다.")
    for source, target in (("asset_id", "asset_code"), ("gfs_grid_id", "grid_code")):
        values = frame[source].astype(str).map(bundle["category_maps"][source])
        if values.isna().any():
            raise ValueError(f"학습 때 없던 {source}가 있습니다.")
        frame[target] = values.astype("int16")
    if "wind_direction_80m_deg" in frame:
        rad = np.deg2rad(frame.wind_direction_80m_deg)
        frame["wind_direction_sin"], frame["wind_direction_cos"] = np.sin(rad), np.cos(rad)
        frame = add_wind_change_features(frame)
    if "is_daylight" in frame:
        frame["is_daylight"] = frame.is_daylight.astype("int8")
    if "label_is_valid" in frame and "generation_kwh" in frame:
        frame.loc[~frame.label_is_valid.fillna(False), "generation_kwh"] = np.nan
    return frame


def delivery_frame(result: pd.DataFrame) -> pd.DataFrame:
    """Return only operational inputs; never export generation labels or raw features."""
    optional = [
        column
        for column in (
            "improved_generation_p10_kwh",
            "improved_generation_p50_kwh",
            "improved_generation_p90_kwh",
            "model_version",
            "trained_until",
        )
        if column in result
    ]
    return result[DELIVERY_COLUMNS + optional].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for kind in ("solar", "wind"):
        bundle = joblib.load(args.models / f"{kind}_improved_production.joblib")
        raw = pd.read_parquet(args.data / f"{kind}_training_mart.parquet")
        frame = prepare_inference(raw, bundle, args.start, args.end)
        result = predict_rows(bundle, frame)
        result["model_version"] = "improved_production"
        result["trained_until"] = f"{max(bundle['training_years'])}-12-31"
        delivery_frame(result).to_parquet(
            args.output / f"{kind}_improved_predictions.parquet", index=False
        )


if __name__ == "__main__":
    main()
