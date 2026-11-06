"""담당: 최적화·대시보드 연결 — 발전량·SMP 예측 성능을 한 가지 정의로 계산하는 공통 모듈.

    python -m vpp.forecast_metrics --start 2024-06-01 --end 2024-12-31 \
        [--solar ..] [--wind ..] [--smp ..] [--plants ..] [--out data/processed/forecast_metrics]

모델 내부는 건드리지 않는다. 저장된 예측 결과 파일(발전량 parquet, SMP csv)에서만 계산한다.

지표 정의 (팀 공통)
- 발전량: 오차 = 예측 − 실측 (kWh). MAE·RMSE·nMAE(= 100 × Σ|오차| / Σ설비용량kW, 리나의
  generation_forecast.model.error_metrics 와 같은 식) + 편향(bias = 평균 오차, 양수면 과대예측).
  기준 예측은 파일의 baseline_generation_kwh(기후값). 포트폴리오/발전원 합산은 그 시각의 발전소가
  모두 있을 때만(complete) 평가하고, 빠진 시각은 제외 건수로 센다. 1일차(offset 0)·2일차(offset 1)는
  항상 따로 집계한다 — 같은 대상 시각을 두 번 세지 않는다.
- SMP: 단위 원/kWh(smp_hat, smp_b1, smp_actual). 오차 = 예측 − 실측. MAE·RMSE(smp_forecast.metrics
  와 같은 식)·편향, horizon(1=내일, 2=모레)별. 고가 4시간 적중률 = 대상일마다 예측 상위 4시간과
  실측 상위 4시간의 교집합 / 4 를 24시간이 모두 있는 날만 평균. 동률은 이른 시각 우선으로 고정.
- 평가 범위는 백테스트에서 살아남은 날이 아니라 예측·실측이 모두 있는 모든 행. 건수·결측·제외를
  함께 돌려준다.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp import forecast_inputs as fi
from vpp.generation_forecast.model import error_metrics

KST = fi.KST
GENERATION_MODELS = {"lightgbm": "lightgbm_generation_kwh", "baseline": "baseline_generation_kwh"}
GENERATION_EVAL_COLUMNS = fi.GENERATION_KEY + [
    "energy_type", "plant_name", "capacity_kw", "valid_time_kst", "forecast_day_offset",
    "lead_hour", "lightgbm_generation_kwh",
]  # forecast_inputs.GENERATION_COLUMNS 의 상위집합 → aggregate_portfolio 에도 그대로 넣을 수 있다
GENERATION_OPTIONAL = ("generation_kwh", "baseline_generation_kwh")
GENERATION_LEVELS = ("portfolio", "energy_type", "asset")
SMP_MODELS = {"model": "smp_hat", "b1": "smp_b1"}
TOP_K = 4
# 빈 기간에도 같은 열을 돌려준다 (화면이 열 이름으로 접근하므로)
GENERATION_METRIC_COLUMNS = [
    "level", "group", "forecast_day_offset", "model", "n_target_hours", "n_evaluated",
    "n_excluded_incomplete_assets", "n_missing_actual", "n_missing_forecast",
    "mae_kwh", "rmse_kwh", "nmae_pct", "mae_capacity_factor", "bias_kwh", "bias_pct",
    "actual_total_mwh", "forecast_total_mwh",
]
SMP_METRIC_COLUMNS = [
    "horizon", "model", "n_target_hours", "n_evaluated", "n_missing_actual",
    "n_missing_forecast", "mae", "rmse", "bias", f"top{TOP_K}_hit_rate", "n_days_scored",
    "n_days_excluded_incomplete", "unit",
]


# ---------------------------------------------------------------- 발전량


def load_generation_eval_frame(
    solar_path: Path = fi.DEFAULT_SOLAR_PREDICTIONS,
    wind_path: Path = fi.DEFAULT_WIND_PREDICTIONS,
    plants_path: Path = fi.PLANT_MASTER_PATH,
) -> pd.DataFrame:
    """평가용 발전소별 행. 실측·기준 예측 열이 없으면 NaN 으로 둔다 (평가 불가 건수로 집계)."""
    pred = pd.concat([pd.read_parquet(solar_path), pd.read_parquet(wind_path)], ignore_index=True)
    missing = [c for c in GENERATION_EVAL_COLUMNS if c not in pred.columns]
    if missing:
        raise ValueError(f"발전량 예측 파일에 평가에 필요한 열이 없습니다: {missing}")
    for col in GENERATION_OPTIONAL:
        if col not in pred.columns:
            pred[col] = np.nan
    cols = GENERATION_EVAL_COLUMNS + list(GENERATION_OPTIONAL)
    return fi.validate_generation_predictions(pred[cols], fi.expected_assets(plants_path))


def _in_range(days: pd.Series, start: date | None, end: date | None) -> pd.Series:
    mask = pd.Series(True, index=days.index)
    if start is not None:
        mask &= days >= start
    if end is not None:
        mask &= days <= end
    return mask


def _group_columns(level: str) -> list[str]:
    if level not in GENERATION_LEVELS:
        raise ValueError(f"level 은 {GENERATION_LEVELS} 중 하나: {level!r}")
    return {"portfolio": [], "energy_type": ["energy_type"], "asset": ["asset_id"]}[level]


def aggregate_generation(
    pred: pd.DataFrame, level: str, expected: dict[str, set[str]]
) -> pd.DataFrame:
    """(그룹, 발표분, 대상시각, 일차) 단위 합산. 그룹의 발전소가 모두 있을 때만 complete_assets."""
    group_cols = _group_columns(level)
    keys = group_cols + ["issue_time_utc", "valid_time_kst", "forecast_day_offset"]
    n_expected = {kind: len(ids) for kind, ids in expected.items()}
    total_expected = sum(n_expected.values())

    # 합계는 벡터 연산으로 내고, 그룹 안에 NaN 이 하나라도 있으면(count < size) NaN 으로 둔다.
    # (파이썬 람다 집계는 발전소 단위에서 수십만 그룹이라 너무 느리다.)
    grouped = pred.groupby(keys, observed=True, sort=True)
    agg = grouped.agg(
        n_assets=("asset_id", "nunique"),
        n_rows=("asset_id", "size"),
        capacity_kw=("capacity_kw", "sum"),
        actual_kwh=("generation_kwh", "sum"),
        n_actual=("generation_kwh", "count"),
        lightgbm_kwh=("lightgbm_generation_kwh", "sum"),
        n_lightgbm=("lightgbm_generation_kwh", "count"),
        baseline_kwh=("baseline_generation_kwh", "sum"),
        n_baseline=("baseline_generation_kwh", "count"),
    ).reset_index()
    for col, n_col in (("actual_kwh", "n_actual"), ("lightgbm_kwh", "n_lightgbm"),
                       ("baseline_kwh", "n_baseline")):
        agg.loc[agg[n_col] < agg["n_rows"], col] = np.nan
    agg = agg.drop(columns=["n_rows", "n_lightgbm", "n_baseline"])
    if level == "portfolio":
        agg["expected_assets"] = total_expected
    elif level == "energy_type":
        agg["expected_assets"] = agg["energy_type"].map(n_expected).astype(int)
    else:
        agg["expected_assets"] = 1
    agg["complete_assets"] = agg["n_assets"].eq(agg["expected_assets"])
    agg["group"] = "portfolio" if level == "portfolio" else agg[group_cols[0]].astype(str)
    return agg


def generation_metrics(
    pred: pd.DataFrame,
    expected: dict[str, set[str]],
    *,
    level: str = "portfolio",
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """그룹 × 일차 × 모델별 지표 한 표. 대상시각(valid_time_kst) 날짜가 [start, end] 인 행만."""
    frame = pred[_in_range(pred["valid_time_kst"].dt.date, start, end)]
    agg = aggregate_generation(frame, level, expected)
    rows: list[dict[str, Any]] = []
    for (group, offset), part in agg.groupby(["group", "forecast_day_offset"], sort=True):
        has_actual = part["actual_kwh"].notna()
        for model, col in (("lightgbm", "lightgbm_kwh"), ("baseline", "baseline_kwh")):
            has_pred = part[col].notna()
            ok = part["complete_assets"] & has_actual & has_pred
            scored = part[ok]
            row: dict[str, Any] = {
                "level": level, "group": group, "forecast_day_offset": int(offset),
                "model": model,
                "n_target_hours": int(len(part)),
                "n_evaluated": int(len(scored)),
                "n_excluded_incomplete_assets": int((~part["complete_assets"]).sum()),
                "n_missing_actual": int((part["complete_assets"] & ~has_actual).sum()),
                "n_missing_forecast": int((part["complete_assets"] & has_actual & ~has_pred).sum()),
            }
            if len(scored):
                m = error_metrics(scored["actual_kwh"].to_numpy(), scored[col].to_numpy(),
                                  scored["capacity_kw"].to_numpy())
                err = scored[col].to_numpy() - scored["actual_kwh"].to_numpy()
                row.update({
                    "mae_kwh": m["mae_kwh"], "rmse_kwh": m["rmse_kwh"],
                    "nmae_pct": m["nmae_pct"], "mae_capacity_factor": m["mae_capacity_factor"],
                    "bias_kwh": float(err.mean()),
                    "bias_pct": float(100.0 * err.sum() / scored["capacity_kw"].sum()),
                    "actual_total_mwh": float(scored["actual_kwh"].sum() / 1000.0),
                    "forecast_total_mwh": float(scored[col].sum() / 1000.0),
                })
            else:
                row.update({k: np.nan for k in GENERATION_METRIC_COLUMNS[9:]})
            rows.append(row)
    return pd.DataFrame(rows, columns=GENERATION_METRIC_COLUMNS)


def generation_timeseries(
    pred: pd.DataFrame,
    expected: dict[str, set[str]],
    *,
    level: str = "portfolio",
    group: str | None = None,
    forecast_day_offset: int = 0,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """그래프용 시계열 (ts, actual_kwh, lightgbm_kwh, baseline_kwh, complete_assets). 한 일차만."""
    frame = pred[_in_range(pred["valid_time_kst"].dt.date, start, end)]
    agg = aggregate_generation(frame, level, expected)
    agg = agg[agg["forecast_day_offset"].eq(forecast_day_offset)]
    if level != "portfolio":
        if group is None:
            raise ValueError("level 이 portfolio 가 아니면 group 을 지정하세요.")
        agg = agg[agg["group"].eq(group)]
    out = agg.rename(columns={"valid_time_kst": "ts"})[
        ["ts", "actual_kwh", "lightgbm_kwh", "baseline_kwh", "capacity_kw", "complete_assets"]
    ].copy()
    # 발전소가 빠진 시각의 부분 합계는 포트폴리오 값이 아니므로 그리지 않는다 (0 으로도 채우지 않음)
    out.loc[~out["complete_assets"], ["actual_kwh", "lightgbm_kwh", "baseline_kwh"]] = np.nan
    return out.sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------- SMP


def load_smp_eval_frame(path: Path = fi.DEFAULT_SMP_FORECAST) -> pd.DataFrame:
    """평가용 SMP 행 (원/kWh). target_time 은 KST, target_day·hour 추가. smp_b1 없으면 NaN."""
    smp = fi.load_smp_forecast(path)
    if "smp_hat" not in smp.columns:
        raise ValueError("SMP 예측 파일에 smp_hat(원/kWh) 열이 없습니다.")
    if "smp_b1" not in smp.columns:
        smp["smp_b1"] = np.nan
    out = pd.DataFrame({
        "issue_date": pd.to_datetime(smp["issue_date"]).dt.date,
        "horizon": smp["horizon"].astype(int),
        "target_time": pd.to_datetime(smp["target_time"], utc=True).dt.tz_convert(KST),
        "smp_hat": smp["smp_hat"].astype(float),
        "smp_b1": smp["smp_b1"].astype(float),
        "smp_actual": smp["smp_actual"].astype(float),
    })
    out["target_day"] = out["target_time"].dt.date
    out["hour"] = out["target_time"].dt.hour
    if "smp_hat_krw_per_mwh" in smp.columns:
        out["unit_abs_diff_krw_per_mwh"] = (
            smp["smp_hat_krw_per_mwh"].astype(float) - out["smp_hat"] * 1000.0
        ).abs()
    return out


def top_hours(frame: pd.DataFrame, col: str, k: int = TOP_K) -> set[int]:
    """값 내림차순 상위 k 시간. 동률은 이른 시각 우선 (고정 규칙)."""
    ordered = frame.sort_values([col, "hour"], ascending=[False, True], kind="mergesort")
    return set(ordered["hour"].head(k).astype(int))


def smp_metrics(
    smp: pd.DataFrame, *, start: date | None = None, end: date | None = None, k: int = TOP_K
) -> pd.DataFrame:
    """horizon × 모델(model=smp_hat, b1=smp_b1) 별 MAE·RMSE·편향·고가 k시간 적중률. 단위 원/kWh."""
    frame = smp[_in_range(smp["target_day"], start, end)]
    rows: list[dict[str, Any]] = []
    for horizon, part in frame.groupby("horizon", sort=True):
        has_actual = part["smp_actual"].notna()
        for model, col in SMP_MODELS.items():
            has_pred = part[col].notna()
            scored = part[has_actual & has_pred]
            err = (scored[col] - scored["smp_actual"]).to_numpy()
            # 적중률: 24시간이 모두 평가 가능한 날만
            per_day = scored.groupby("target_day")
            full_days = [d for d, g in per_day if len(g) == 24 and g["hour"].nunique() == 24]
            hits = [
                len(top_hours(per_day.get_group(d), col, k)
                    & top_hours(per_day.get_group(d), "smp_actual", k)) / k
                for d in full_days
            ]
            rows.append({
                "horizon": int(horizon), "model": model,
                "n_target_hours": int(len(part)),
                "n_evaluated": int(len(scored)),
                "n_missing_actual": int((~has_actual).sum()),
                "n_missing_forecast": int((has_actual & ~has_pred).sum()),
                "mae": float(np.abs(err).mean()) if len(err) else np.nan,
                "rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
                "bias": float(err.mean()) if len(err) else np.nan,
                f"top{k}_hit_rate": float(np.mean(hits)) if hits else np.nan,
                "n_days_scored": len(full_days),
                "n_days_excluded_incomplete": int(part["target_day"].nunique() - len(full_days)),
                "unit": "원/kWh",
            })
    out = pd.DataFrame(rows, columns=SMP_METRIC_COLUMNS if k == TOP_K else None)
    if "unit_abs_diff_krw_per_mwh" in frame.columns and len(frame):
        out["unit_check_max_abs_diff_krw_per_mwh"] = float(
            frame["unit_abs_diff_krw_per_mwh"].max()
        )
    return out


def smp_timeseries(
    smp: pd.DataFrame, *, horizon: int = 1, start: date | None = None, end: date | None = None
) -> pd.DataFrame:
    frame = smp[_in_range(smp["target_day"], start, end) & smp["horizon"].eq(horizon)]
    return frame[["target_time", "smp_hat", "smp_b1", "smp_actual"]].rename(
        columns={"target_time": "ts"}
    ).sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--solar", type=Path, default=fi.DEFAULT_SOLAR_PREDICTIONS)
    parser.add_argument("--wind", type=Path, default=fi.DEFAULT_WIND_PREDICTIONS)
    parser.add_argument("--smp", type=Path, default=fi.DEFAULT_SMP_FORECAST)
    parser.add_argument("--plants", type=Path, default=fi.PLANT_MASTER_PATH)
    parser.add_argument("--out", type=Path, default=Path("data/processed/forecast_metrics"))
    args = parser.parse_args(argv)

    expected = fi.expected_assets(args.plants)
    pred = load_generation_eval_frame(args.solar, args.wind, args.plants)
    gen = pd.concat([
        generation_metrics(pred, expected, level=level, start=args.start, end=args.end)
        for level in GENERATION_LEVELS
    ], ignore_index=True)
    smp = smp_metrics(load_smp_eval_frame(args.smp), start=args.start, end=args.end)
    args.out.mkdir(parents=True, exist_ok=True)
    gen.to_csv(args.out / "generation_metrics.csv", index=False, encoding="utf-8-sig")
    smp.to_csv(args.out / "smp_metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    show = gen[gen["level"] != "asset"][
        ["group", "forecast_day_offset", "model", "n_evaluated", "n_excluded_incomplete_assets",
         "mae_kwh", "nmae_pct", "bias_pct"]
    ]
    print(show.round(2).to_string(index=False))
    print(smp.round(3).to_string(index=False))
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
