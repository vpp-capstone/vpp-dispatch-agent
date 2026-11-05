"""담당: 최적화 — 발전량·SMP 예측을 계획일 D의 48시간 ESS 입력 창으로 결합.

계획일 D(KST)에 대해
  - 발전량: generation_forecast 가 D-2 18 UTC GFS 로 낸 예측. 대상 D 00:00 ~ D+1 23:00
            (forecast_day_offset 0·1 이 같은 발표분)
  - 가격:   smp_forecast 가 발행일 D-1 11:00 에 낸 예측. 같은 대상 구간 (horizon 1·2)
을 대상시각(ts)으로 결합해 solve_window 에 넣을 48행을 만든다.

예측 창(forecast)과 비교용 실측 창(actual)은 별도 DataFrame 으로 돌려주며, 예측 창에는 실측 열이
없다. 불완전한 창(발전소 누락, 48행 미만, 시간 끊김, 결측)은 만들지 않고 사유를 남긴다.
0 으로 채우지 않는다.

가용시각: SMP 는 발행시각이 D-1 11:00 으로 고정돼 있다. 발전량 예측 파일에는 공개시각 플래그가 없어
D-2 18 UTC 발표분이 D-1 11:00 KST 전에 공개됐는지 행 단위로 확인하지 못한다 (한계로 기록).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from vpp.generation_forecast.paths import PLANT_MASTER_PATH, PREDICTION_DIR
from vpp.smp_forecast import window_for_issue

KST = "Asia/Seoul"
WINDOW_HOURS = 48
GENERATION_KEY = ["asset_id", "issue_time_utc", "valid_time_utc"]
GENERATION_COLUMNS = GENERATION_KEY + [
    "energy_type", "valid_time_kst", "forecast_day_offset", "lead_hour",
    "lightgbm_generation_kwh", "generation_kwh",
]
SMP_COLUMNS = [
    "issue_date", "horizon", "lead_hour", "target_time", "smp_hat_krw_per_mwh", "smp_actual",
]

DEFAULT_SOLAR_PREDICTIONS = PREDICTION_DIR / "solar_validation_predictions.parquet"
DEFAULT_WIND_PREDICTIONS = PREDICTION_DIR / "wind_validation_predictions.parquet"
DEFAULT_SMP_FORECAST = Path("data/processed/forecast/smp_forecast.csv")


@dataclass(frozen=True)
class PlanWindow:
    """계획일 하나의 48시간 입력. forecast 로 계획을 세우고 actual 은 비교에만 쓴다."""

    planning_day: date
    # forecast: ts, generation_mwh, price_krw_per_mwh, dt_hours, forecast_day_offset, horizon
    forecast: pd.DataFrame
    # actual:   ts, generation_mwh, price_krw_per_mwh, dt_hours
    actual: pd.DataFrame
    sources: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- 입력 로딩


def expected_assets(plants_path: Path = PLANT_MASTER_PATH) -> dict[str, set[str]]:
    """기준정보의 발전원별 asset_id 집합 (태양광 18, 풍력 3)."""
    plants = pd.read_csv(plants_path, encoding="utf-8-sig")
    return {kind: set(group["asset_id"]) for kind, group in plants.groupby("energy_type")}


def load_generation_predictions(
    solar_path: Path = DEFAULT_SOLAR_PREDICTIONS,
    wind_path: Path = DEFAULT_WIND_PREDICTIONS,
    plants_path: Path = PLANT_MASTER_PATH,
) -> pd.DataFrame:
    """발전소별 예측 파일 두 개를 합쳐 검사한다. 중복 키·기준정보 밖 발전소는 예외."""
    frames = [pd.read_parquet(solar_path), pd.read_parquet(wind_path)]
    pred = pd.concat(frames, ignore_index=True)
    missing = [c for c in GENERATION_COLUMNS if c not in pred.columns]
    if missing:
        raise ValueError(f"발전량 예측 파일에 열이 없습니다: {missing}")
    return validate_generation_predictions(pred[GENERATION_COLUMNS], expected_assets(plants_path))


def validate_generation_predictions(
    pred: pd.DataFrame, expected: dict[str, set[str]]
) -> pd.DataFrame:
    dup = pred.duplicated(GENERATION_KEY)
    if dup.any():
        raise ValueError(f"발전량 예측에 중복 키 {int(dup.sum())}건: {GENERATION_KEY}")
    known = set().union(*expected.values())
    unknown = sorted(set(pred["asset_id"]) - known)
    if unknown:
        raise ValueError(f"기준정보에 없는 발전소: {unknown}")
    out = pred.copy()
    out["valid_time_kst"] = pd.to_datetime(out["valid_time_kst"], utc=True).dt.tz_convert(KST)
    out["issue_time_utc"] = pd.to_datetime(out["issue_time_utc"], utc=True)
    return out


def load_smp_forecast(path: Path = DEFAULT_SMP_FORECAST) -> pd.DataFrame:
    smp = pd.read_csv(path)
    missing = [c for c in SMP_COLUMNS if c not in smp.columns]
    if missing:
        raise ValueError(f"SMP 예측 파일에 열이 없습니다: {missing}")
    return smp


# ---------------------------------------------------------------- 포트폴리오 합산


def aggregate_portfolio(pred: pd.DataFrame, expected: dict[str, set[str]]) -> pd.DataFrame:
    """발표시각·대상시각별 21개소 합산. 발전원별 발전소 수가 기준과 같을 때만 complete.

    planning_day 는 대상시각에서 forecast_day_offset 일을 뺀 날짜 (offset 0 → D, 1 → D+1 이 D 로).
    """
    n_expected = {kind: len(ids) for kind, ids in expected.items()}
    keys = ["issue_time_utc", "valid_time_kst"]
    counts = (
        pred.groupby(keys + ["energy_type"], observed=True)["asset_id"].nunique()
        .unstack("energy_type", fill_value=0)
    )
    for kind in n_expected:
        if kind not in counts.columns:
            counts[kind] = 0
    # 발전소 하나라도 결측이면 합계도 결측 (0 으로 채워 더하지 않는다)
    def strict_sum_mwh(values: pd.Series) -> float:
        return values.sum(min_count=len(values)) / 1000.0

    agg = pred.groupby(keys, observed=True).agg(
        generation_forecast_mwh=("lightgbm_generation_kwh", strict_sum_mwh),
        generation_actual_mwh=("generation_kwh", strict_sum_mwh),
        forecast_day_offset=("forecast_day_offset", "first"),
        lead_hour=("lead_hour", "first"),
        n_offsets=("forecast_day_offset", "nunique"),
    )
    out = agg.join(counts.add_suffix("_assets")).reset_index()
    out["complete"] = True
    for kind, n in n_expected.items():
        out["complete"] &= out[f"{kind}_assets"].eq(n)
    out["complete"] &= out["n_offsets"].eq(1)
    out["planning_day"] = [
        (ts - timedelta(days=int(off))).date()
        for ts, off in zip(out["valid_time_kst"], out["forecast_day_offset"], strict=True)
    ]
    return out.drop(columns="n_offsets").sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------- 창 구성


def _expected_ts(planning_day: date) -> pd.DatetimeIndex:
    start = pd.Timestamp(planning_day, tz=KST)
    return pd.date_range(start, periods=WINDOW_HOURS, freq="h", name="ts")


def _check_48h(ts: pd.Series, planning_day: date, label: str) -> str | None:
    """D 00:00 KST 부터 1시간 간격 48개와 같은 시각인지. 시간대 표기와 무관하게 절대시각 비교."""
    if len(ts) != WINDOW_HOURS:
        return f"{label}_not_48h:{len(ts)}"
    actual = pd.DatetimeIndex(ts).as_unit("ns").tz_convert("UTC").asi8
    expected = _expected_ts(planning_day).as_unit("ns").tz_convert("UTC").asi8
    if not (actual == expected).all():
        return f"{label}_gap"
    return None


def generation_window(
    portfolio: pd.DataFrame, planning_day: date
) -> tuple[pd.DataFrame | None, str | None]:
    """계획일 D 의 발표분 하나로 만든 48행. 실패하면 (None, 사유)."""
    sel = portfolio[portfolio["planning_day"] == planning_day]
    if sel.empty:
        return None, "generation_missing"
    if sel["issue_time_utc"].nunique() != 1:
        return None, f"generation_multiple_issues:{sel['issue_time_utc'].nunique()}"
    if not sel["complete"].all():
        bad = sel.loc[~sel["complete"], ["solar_assets", "wind_assets"]].drop_duplicates()
        return None, f"generation_incomplete_assets:{bad.to_dict('records')[0]}"
    sel = sel.sort_values("valid_time_kst").rename(columns={"valid_time_kst": "ts"})
    reason = _check_48h(sel["ts"], planning_day, "generation")
    if reason:
        return None, reason
    if sel[["generation_forecast_mwh", "generation_actual_mwh"]].isna().any().any():
        return None, "generation_nan"
    return sel.reset_index(drop=True), None


def price_window(smp: pd.DataFrame, planning_day: date) -> tuple[pd.DataFrame | None, str | None]:
    """발행일 D-1 의 SMP 48행. window_for_issue 로 예측 열을, 같은 행에서 smp_actual 을 가져온다."""
    issue_date = planning_day - timedelta(days=1)
    fc = window_for_issue(smp, issue_date)
    if fc.empty:
        return None, "price_missing"
    sel = smp[pd.to_datetime(smp["issue_date"]).dt.date == issue_date].sort_values("target_time")
    out = fc.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.tz_convert(KST)
    out["price_actual_krw_per_mwh"] = sel["smp_actual"].to_numpy(dtype=float) * 1000.0
    reason = _check_48h(out["ts"], planning_day, "price")
    if reason:
        return None, reason
    if out[["price_krw_per_mwh", "price_actual_krw_per_mwh"]].isna().any().any():
        return None, "price_nan"
    return out.reset_index(drop=True), None


def build_plan_window(
    portfolio: pd.DataFrame, smp: pd.DataFrame, planning_day: date
) -> tuple[PlanWindow | None, str | None]:
    gen, reason = generation_window(portfolio, planning_day)
    if reason:
        return None, reason
    price, reason = price_window(smp, planning_day)
    if reason:
        return None, reason
    if not gen["ts"].equals(price["ts"]):
        return None, "ts_mismatch"
    forecast = pd.DataFrame({
        "ts": gen["ts"],
        "generation_mwh": gen["generation_forecast_mwh"].to_numpy(),
        "price_krw_per_mwh": price["price_krw_per_mwh"].to_numpy(),
        "dt_hours": 1.0,
        "forecast_day_offset": gen["forecast_day_offset"].to_numpy(),
        "horizon": price["horizon"].to_numpy(),
    })
    actual = pd.DataFrame({
        "ts": gen["ts"],
        "generation_mwh": gen["generation_actual_mwh"].to_numpy(),
        "price_krw_per_mwh": price["price_actual_krw_per_mwh"].to_numpy(),
        "dt_hours": 1.0,
    })
    sources = {
        "generation_issue_time_utc": gen["issue_time_utc"].iloc[0].isoformat(),
        "generation_issue_time_kst": gen["issue_time_utc"].iloc[0].tz_convert(KST).isoformat(),
        "generation_availability_verified": False,
        "smp_issue_date": (planning_day - timedelta(days=1)).isoformat(),
        "smp_issue_time_kst": f"{(planning_day - timedelta(days=1)).isoformat()}T11:00:00+09:00",
        "solar_assets": int(gen["solar_assets"].iloc[0]),
        "wind_assets": int(gen["wind_assets"].iloc[0]),
    }
    return PlanWindow(planning_day, forecast, actual, sources), None


def build_plan_windows(
    portfolio: pd.DataFrame, smp: pd.DataFrame, start: date, end: date
) -> tuple[list[PlanWindow], pd.DataFrame]:
    """[start, end] 의 계획일마다 창을 만들고, 못 만든 날은 (planning_day, reason) 표로 돌려준다."""
    windows: list[PlanWindow] = []
    excluded: list[dict[str, Any]] = []
    day = start
    while day <= end:
        window, reason = build_plan_window(portfolio, smp, day)
        if window is None:
            excluded.append({"planning_day": day, "reason": reason})
        else:
            windows.append(window)
        day += timedelta(days=1)
    return windows, pd.DataFrame(excluded, columns=["planning_day", "reason"])


def contiguous_segments(windows: list[PlanWindow]) -> list[list[PlanWindow]]:
    """계획일이 하루씩 이어지는 구간으로 나눈다. 끊긴 날은 연속 운영처럼 잇지 않는다."""
    segments: list[list[PlanWindow]] = []
    for window in sorted(windows, key=lambda w: w.planning_day):
        if segments and window.planning_day == segments[-1][-1].planning_day + timedelta(days=1):
            segments[-1].append(window)
        else:
            segments.append([window])
    return segments
