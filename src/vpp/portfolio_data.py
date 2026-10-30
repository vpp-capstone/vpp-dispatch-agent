"""담당: 최적화·대시보드팀 — 중개사업자 가상 포트폴리오(태양광 18곳 + 풍력 3곳) 시간별 합산.

원본 4개 CSV를 `raw_dir`에서 읽어 시간별 발전량(MWh)과 하루전 SMP(원/MWh)를 만든다.
원본은 레포에 복사하지 않으며 수정하지 않는다. 결측·중복·이상치는 보간하지 않고,
결측이 있는 날은 제외 목록에 기록한다. 규칙은 docs/formulation.md §5.

시각: 원본 hour h = (h−1)시~h시 구간. `ts`는 구간 시작 시각 KST (원본 1시 → 00:00).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KST = "Asia/Seoul"
HOURS = list(range(1, 25))

# 원본 파일명 (공공데이터포털 다운로드 그대로)
PV_FILE = "제주#1_기상정보 융합 태양광 발전 시계열 데이터.csv"
SEONGSAN_FILE = "한국남부발전(주)_성산풍력발전실적_20260331.csv"
HANGYEONG_FILE = "한국남부발전(주)_한경풍력발전실적_20260331.csv"
SANGMYEONG_FILE = "한국중부발전(주)_(AI 친화 데이터)풍력 발전 실적(상명풍력)_20260330.csv"

# 설비용량 kW — 이상치 플래그 기준 (설비 현황 기준, 자동 수정에는 쓰지 않음)
CAPACITY_KW = {
    "seongsan": {1: 12_000.0, 2: 8_000.0},
    "hangyeong": {1: 6_000.0, 2: 15_000.0},
    "sangmyeong": 21_000.0,
}
PV_PLANT_COUNT = 18
# 상명: 이 날짜까지 원본이 Wh (÷1000 → kWh), 다음 날부터 kWh 그대로. 두 구간 모두 최대 ≈ 20.1 MWh.
SANGMYEONG_WH_UNTIL = date(2025, 1, 31)

DEFAULT_MARKET_PATH = Path("data/processed/market/market_hourly.csv")


@dataclass(frozen=True)
class SourceSpec:
    key: str
    file: str
    encoding: str


SOURCES = (
    SourceSpec("pv", PV_FILE, "utf-8-sig"),
    SourceSpec("seongsan", SEONGSAN_FILE, "cp949"),
    SourceSpec("hangyeong", HANGYEONG_FILE, "cp949"),
    SourceSpec("sangmyeong", SANGMYEONG_FILE, "cp949"),
)


def _to_numeric(frame: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """'-', ' - ' 같은 비숫자 셀을 NaN으로. 바뀐 셀 수를 함께 반환."""
    raw = frame[cols]
    out = raw.apply(lambda s: pd.to_numeric(s.astype(str).str.strip(), errors="coerce"))
    non_numeric = int((out.isna() & raw.notna()).sum().sum())
    frame = frame.copy()
    frame[cols] = out
    return frame, non_numeric


def _load_pv(path: Path, expected_plants: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """long → (date, hour) 별 18곳 합산 kWh + 시간별 단가 1개."""
    raw = pd.read_csv(
        path, encoding="utf-8-sig",
        usecols=["거래일", "시간", "발전기코드", "설비용량", "계량전력량", "단가"],
    )
    raw["date"] = pd.to_datetime(raw["거래일"]).dt.date
    raw["hour"] = raw["시간"].astype(int)
    capacity_kw = float(raw.drop_duplicates("발전기코드")["설비용량"].sum())
    grouped = raw.groupby(["date", "hour"])
    pv = grouped.agg(
        pv_kwh=("계량전력량", lambda s: s.sum(skipna=False)),
        pv_plant_count=("발전기코드", "nunique"),
        price_nunique=("단가", "nunique"),
        price_krw_per_kwh=("단가", "first"),
    ).reset_index()
    pv["pv_price_inconsistent"] = pv["price_nunique"] > 1
    pv["pv_plant_count_short"] = pv["pv_plant_count"] < expected_plants
    info = {
        "file": path.name, "encoding": "utf-8-sig", "rows": int(len(raw)),
        "plants": int(raw["발전기코드"].nunique()), "capacity_kw": capacity_kw,
        "date_min": str(raw["date"].min()), "date_max": str(raw["date"].max()),
        "hours_price_inconsistent": int(pv["pv_price_inconsistent"].sum()),
        "hours_plant_count_short": int(pv["pv_plant_count_short"].sum()),
        "missing_cells": int(raw["계량전력량"].isna().sum() + raw["단가"].isna().sum()),
    }
    return pv.drop(columns=["price_nunique"]), info


def _load_knp_wind(path: Path, key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """남부발전 wide(년월일, 단계, 1~24) → (date, hour) 별 단계 1+2 합산 kWh.

    중복 (년월일, 단계): 값이 같으면 하나만 남기고, 다르면 그 날을 중복 충돌로 표시한다
    (자동 선택하지 않음 → 백테스트에서 제외).
    """
    raw = pd.read_csv(path, encoding="cp949", dtype=str)
    hour_cols = [str(h) for h in HOURS]
    raw, non_numeric = _to_numeric(raw, hour_cols)
    raw["date"] = pd.to_datetime(raw["년월일"]).dt.date
    raw["stage"] = raw["단계"].astype(int)
    exact_dup = raw.duplicated(["date", "stage", *hour_cols], keep="first")
    raw = raw[~exact_dup]
    conflict_mask = raw.duplicated(["date", "stage"], keep=False)
    conflict_dates = sorted({str(d) for d in raw.loc[conflict_mask, "date"]})
    raw = raw[~conflict_mask]

    long = raw[["date", "stage", *hour_cols]].melt(
        id_vars=["date", "stage"], value_vars=hour_cols, var_name="hour", value_name="kwh"
    )
    long["hour"] = long["hour"].astype(int)
    caps = CAPACITY_KW[key]
    long["cap_kwh"] = long["stage"].map(caps)
    long["over"] = long["kwh"] > long["cap_kwh"]
    wide = long.pivot_table(index=["date", "hour"], columns="stage", values="kwh", aggfunc="first")
    stages = sorted(caps)
    for s in stages:
        if s not in wide.columns:
            wide[s] = np.nan
    over = long.groupby(["date", "hour"])["over"].any()
    out = pd.DataFrame({
        f"{key}_kwh": sum(wide[s] for s in stages),  # NaN이 있으면 NaN 유지 (보간 안 함)
        f"{key}_over_capacity": over.reindex(wide.index).fillna(False).astype(bool),
    }).reset_index()
    clipped = sum(wide[s].clip(upper=caps[s]) for s in stages)
    out[f"{key}_kwh_clipped"] = clipped.to_numpy()
    info = {
        "file": path.name, "encoding": "cp949", "rows": int(len(raw) + exact_dup.sum()),
        "date_min": str(raw["date"].min()), "date_max": str(raw["date"].max()),
        "non_numeric_cells_to_nan": non_numeric,
        "blank_cells": int(raw[hour_cols].isna().sum().sum()) - non_numeric,
        "exact_duplicate_rows_dropped": int(exact_dup.sum()),
        "conflicting_duplicate_dates_excluded": conflict_dates,
        "over_capacity_cells": int(long["over"].sum()),
        "capacity_kw_by_stage": {str(k): v for k, v in caps.items()},
        "stage_sum": "stage 1 + stage 2",
    }
    return out, info


def _load_sangmyeong(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """중부발전 상명 wide(기준일, 호기, 1시~24시). 2025-01-31까지 Wh→÷1000, 이후 kWh."""
    raw = pd.read_csv(path, encoding="cp949", dtype=str)
    hour_cols = [f"{h}시" for h in HOURS]
    raw = raw[raw["기준일"].notna()]
    raw, non_numeric = _to_numeric(raw, hour_cols)
    raw["date"] = pd.to_datetime(raw["기준일"]).dt.date
    dup_dates = sorted({str(d) for d in raw.loc[raw.duplicated("date", keep=False), "date"]})
    # 호기가 여러 개면 합산 (현재 원본은 호기 1 하나). NaN은 전파.
    daily = raw.groupby("date")[hour_cols].agg(lambda s: s.sum(skipna=False))
    scaled = pd.Series(
        [d <= SANGMYEONG_WH_UNTIL for d in daily.index], index=daily.index, dtype=bool
    )
    values = daily.where(~scaled, daily / 1000.0)
    long = values.reset_index().melt(
        id_vars=["date"], value_vars=hour_cols, var_name="hour", value_name="sangmyeong_kwh"
    )
    long["hour"] = long["hour"].str.replace("시", "", regex=False).astype(int)
    long["sangmyeong_scaled_from_wh"] = long["date"].map(scaled).astype(bool)
    cap = CAPACITY_KW["sangmyeong"]
    long["sangmyeong_over_capacity"] = long["sangmyeong_kwh"] > cap
    long["sangmyeong_kwh_clipped"] = long["sangmyeong_kwh"].clip(upper=cap)
    info = {
        "file": path.name, "encoding": "cp949", "rows": int(len(raw)),
        "date_min": str(raw["date"].min()), "date_max": str(raw["date"].max()),
        "non_numeric_cells_to_nan": non_numeric,
        "blank_cells": int(raw[hour_cols].isna().sum().sum()) - non_numeric,
        "unit_rule": f"<= {SANGMYEONG_WH_UNTIL}: Wh ÷ 1000 → kWh; after: kWh as is",
        "days_scaled_from_wh": int(scaled.sum()),
        "max_kwh_before_switch": float(values[scaled].max().max()) if scaled.any() else None,
        "max_kwh_after_switch": float(values[~scaled].max().max()) if (~scaled).any() else None,
        "duplicate_dates_summed_by_unit": dup_dates,
        "over_capacity_cells": int(long["sangmyeong_over_capacity"].sum()),
        "capacity_kw": cap,
    }
    return long, info


def _price_cross_check(frame: pd.DataFrame, market_path: Path | None) -> dict[str, Any]:
    """있으면 market_hourly.csv의 smp_da(원/kWh)와 PV 단가를 시간별 대조."""
    if market_path is None or not Path(market_path).exists():
        return {"status": "skipped", "reason": f"파일 없음: {market_path}"}
    market = pd.read_csv(market_path, usecols=["ts", "smp_da"])
    ts = pd.to_datetime(market["ts"])
    ts = ts.dt.tz_localize(KST) if ts.dt.tz is None else ts.dt.tz_convert(KST)
    market = pd.DataFrame({"ts": ts, "smp_da": market["smp_da"]}).dropna()
    merged = frame[["ts", "price_krw_per_kwh"]].merge(market, on="ts", how="inner")
    diff = (merged["price_krw_per_kwh"] - merged["smp_da"]).abs()
    return {
        "status": "done", "file": str(market_path), "compared_hours": int(len(merged)),
        "mismatch_hours": int((diff > 1e-6).sum()),
        "max_abs_diff_krw_per_kwh": float(diff.max()) if len(diff) else None,
    }


def build_portfolio_hourly(
    raw_dir: str | Path,
    start: str | date,
    end: str | date,
    *,
    clip_to_capacity: bool = False,
    market_path: str | Path | None = DEFAULT_MARKET_PATH,
    expected_pv_plants: int = PV_PLANT_COUNT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """[start, end] 기간의 시간별 포트폴리오 DataFrame과 데이터 검사 결과(dict)를 반환.

    결측·중복 충돌·PV 발전소 수 부족·단가 불일치가 있는 날은 제외하고 `checks["excluded_days"]`에
    기록한다. 용량 초과는 플래그 컬럼으로만 남기고 `clip_to_capacity=True`일 때만 클리핑한다.
    """
    raw_dir = Path(raw_dir)
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    if end < start:
        raise ValueError("end는 start 이후여야 합니다.")
    missing = [s.file for s in SOURCES if not (raw_dir / s.file).exists()]
    if missing:
        raise FileNotFoundError(f"원본 파일이 없습니다 ({raw_dir}): {missing}")

    pv, pv_info = _load_pv(raw_dir / PV_FILE, expected_pv_plants)
    seongsan, ss_info = _load_knp_wind(raw_dir / SEONGSAN_FILE, "seongsan")
    hangyeong, hg_info = _load_knp_wind(raw_dir / HANGYEONG_FILE, "hangyeong")
    sangmyeong, sm_info = _load_sangmyeong(raw_dir / SANGMYEONG_FILE)

    days = pd.date_range(start, end, freq="D").date
    grid = pd.MultiIndex.from_product([days, HOURS], names=["date", "hour"]).to_frame(index=False)
    frame = grid
    for part in (pv, seongsan, hangyeong, sangmyeong):
        frame = frame.merge(part, on=["date", "hour"], how="left")

    # 날 단위 제외 사유 수집 (보간·자동 선택 없음)
    reasons: dict[str, set[str]] = {}

    def exclude(mask: pd.Series, reason: str) -> None:
        for d in frame.loc[mask, "date"].unique():
            reasons.setdefault(str(d), set()).add(reason)

    conflict_days = set(ss_info["conflicting_duplicate_dates_excluded"]) | set(
        hg_info["conflicting_duplicate_dates_excluded"]
    )
    exclude(frame["date"].astype(str).isin(conflict_days), "duplicate_conflict_wind")
    for col, reason in (
        ("pv_kwh", "missing_pv"), ("price_krw_per_kwh", "missing_price"),
        ("seongsan_kwh", "missing_seongsan"), ("hangyeong_kwh", "missing_hangyeong"),
        ("sangmyeong_kwh", "missing_sangmyeong"),
    ):
        exclude(frame[col].isna(), reason)
    exclude(frame["pv_plant_count_short"].fillna(True).astype(bool), "pv_plant_count_short")
    exclude(frame["pv_price_inconsistent"].fillna(False).astype(bool), "pv_price_inconsistent")

    excluded_days = [{"date": d, "reasons": sorted(r)} for d, r in sorted(reasons.items())]
    keep = ~frame["date"].astype(str).isin(set(reasons))
    frame = frame[keep].reset_index(drop=True)

    wind_cols = ["seongsan", "hangyeong", "sangmyeong"]
    if clip_to_capacity:
        for key in wind_cols:
            frame[f"{key}_kwh"] = frame[f"{key}_kwh_clipped"]
    frame = frame.drop(columns=[f"{k}_kwh_clipped" for k in wind_cols])
    frame["pv_over_capacity"] = frame["pv_kwh"] > pv_info["capacity_kw"]
    frame["wind_kwh"] = frame[[f"{k}_kwh" for k in wind_cols]].sum(axis=1, skipna=False)
    frame["generation_mwh"] = (frame["pv_kwh"] + frame["wind_kwh"]) / 1000.0
    frame["price_krw_per_mwh"] = frame["price_krw_per_kwh"] * 1000.0
    frame["dt_hours"] = 1.0
    frame["ts"] = (
        pd.to_datetime(frame["date"]) + pd.to_timedelta(frame["hour"] - 1, unit="h")
    ).dt.tz_localize(KST)
    frame["any_flag"] = frame[
        [f"{k}_over_capacity" for k in wind_cols] + ["pv_over_capacity"]
    ].any(axis=1)

    ordered = [
        "ts", "date", "hour", "dt_hours",
        "pv_kwh", "seongsan_kwh", "hangyeong_kwh", "sangmyeong_kwh", "wind_kwh", "generation_mwh",
        "price_krw_per_kwh", "price_krw_per_mwh",
        "pv_plant_count", "pv_over_capacity", "seongsan_over_capacity", "hangyeong_over_capacity",
        "sangmyeong_over_capacity", "sangmyeong_scaled_from_wh", "any_flag",
    ]
    frame = frame[ordered].sort_values("ts").reset_index(drop=True)
    for col in ("pv_over_capacity", "seongsan_over_capacity", "hangyeong_over_capacity",
                "sangmyeong_over_capacity", "sangmyeong_scaled_from_wh", "any_flag"):
        frame[col] = frame[col].astype(bool)

    checks: dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "requested": {"start": str(start), "end": str(end), "days": int(len(days))},
        "included_days": int(frame["date"].nunique()),
        "included_hours": int(len(frame)),
        "excluded_days": excluded_days,
        "clip_to_capacity": clip_to_capacity,
        "time_rule": "원본 hour h = (h-1)시~h시; ts = 구간 시작 KST (1시 → 00:00)",
        "sources": {
            "pv": pv_info, "seongsan": ss_info, "hangyeong": hg_info, "sangmyeong": sm_info,
        },
        "over_capacity_hours_in_range": {
            k: int(frame[f"{k}_over_capacity"].sum()) for k in [*wind_cols, "pv"]
        },
        "sangmyeong_scaled_hours_in_range": int(frame["sangmyeong_scaled_from_wh"].sum()),
        "negative_price_hours": int((frame["price_krw_per_mwh"] < 0).sum()),
        "price_cross_check": _price_cross_check(
            frame, None if market_path is None else Path(market_path)
        ),
        "hourly_continuity_ok": bool(
            frame["ts"].is_monotonic_increasing and frame["ts"].is_unique
        ),
    }
    return frame, checks
