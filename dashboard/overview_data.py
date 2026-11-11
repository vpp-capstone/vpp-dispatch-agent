"""Read saved experiments without rerunning a model or changing an operating policy."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from vpp.baselines import no_ess, plan_revenue_breakdown
from vpp.optimization import EssParams
from vpp.rec import REC_COLUMN

MODE = "forecast_vs_perfect_foresight_plan"
REQUIRED = (
    "generation_mwh",
    "price_krw_per_mwh",
    "dt_hours",
    "sell_mwh",
    "charge_mwh",
    "discharge_mwh",
    "curtailment_mwh",
    "energy_end_mwh",
    "soc_end",
)


def experiments(base: Path) -> list[tuple[Path, dict]]:
    """Only paired forecast/perfect-information experiments, including legacy results."""
    found = []
    for path in base.glob("*/summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("mode") == MODE:
                found.append((path.parent, summary))
        except (OSError, ValueError):
            continue
    return sorted(
        found,
        key=lambda x: (
            x[1].get("days", 0),
            bool(x[1].get("rec", {}).get("enabled")),
            (x[0] / "summary.json").stat().st_mtime,
        ),
        reverse=True,
    )


def read_plan(path: Path, *, require_rec: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"ts", *REQUIRED} - set(frame)
    if missing:
        raise ValueError(f"{path.name}: 필요한 열이 없습니다: {sorted(missing)}")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert("Asia/Seoul")
    if frame["ts"].isna().any() or frame["ts"].duplicated().any():
        raise ValueError(f"{path.name}: 시각 누락·중복이 있어 비교를 중단합니다.")
    if require_rec and REC_COLUMN not in frame:
        raise ValueError(f"{path.name}: REC 적용 결과인데 REC 단가 열이 없습니다.")
    numeric = [*REQUIRED, *([REC_COLUMN] if REC_COLUMN in frame else [])]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: 숫자 결측이 있어 비교를 중단합니다.")
    if not frame["dt_hours"].eq(1).all():
        raise ValueError("이 화면은 시간별 확정 계획(1시간 간격)만 지원합니다.")
    return frame.sort_values("ts").reset_index(drop=True)


def load_experiment(folder: Path) -> dict:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary.get("mode") != MODE:
        raise ValueError("예측 계획과 실측을 미리 알고 만든 계획의 비교 결과를 선택하세요.")
    require_rec = bool((summary.get("rec") or {}).get("enabled"))
    forecast = read_plan(folder / "hourly_forecast.csv", require_rec=require_rec)
    if forecast.empty:
        raise ValueError("예측 계획이 비어 있습니다.")
    # 실측이 없는 기간은 예측 계획만 있다(actual=None). 화면은 비교 부분만 비우고 계획은 보여준다.
    actual_csv = folder / "hourly_actual.csv"
    has_actual = bool(summary.get("actual_available", actual_csv.exists()))
    actual = read_plan(actual_csv, require_rec=require_rec) if has_actual else None
    if actual is not None:
        if not forecast["ts"].equals(actual["ts"]):
            raise ValueError("두 계획의 비교 시각이 다릅니다. 일부만 골라 비교하지 않습니다.")
        rec_f = forecast.get(REC_COLUMN, pd.Series(0.0, index=forecast.index))
        rec_a = actual.get(REC_COLUMN, pd.Series(0.0, index=actual.index))
        if not np.allclose(rec_f, rec_a, rtol=0, atol=1e-8):
            raise ValueError("두 계획의 REC 단가가 달라 공정하게 비교할 수 없습니다.")
    known = {f.name for f in fields(EssParams)}
    ess = EssParams(**{k: v for k, v in summary["ess"].items() if k in known})
    revenues = {}
    pairs = [("forecast", forecast)] + ([("actual", actual)] if actual is not None else [])
    for name, plan in pairs:
        revenue = plan_revenue_breakdown(plan, ess)
        base = plan_revenue_breakdown(no_ess(plan), ess)
        revenue["no_ess_net_krw"] = base["net_revenue_krw"]
        revenue["ess_gain_krw"] = revenue["net_revenue_krw"] - base["net_revenue_krw"]
        stored = summary.get("scenarios", {}).get(name, {}).get("planned_objective_krw")
        if stored is not None and not np.isclose(
            revenue["net_revenue_krw"],
            stored,
            rtol=1e-9,
            atol=0.1,
        ):
            raise ValueError(
                "저장된 요약과 시간별 수익이 다릅니다. 같은 실행의 파일인지 확인하세요."
            )
        revenues[name] = revenue
    gaps = forecast["ts"].diff().gt(pd.Timedelta(hours=1))
    metrics: dict = {
        "hours": len(forecast),
        "days": forecast["ts"].dt.date.nunique(),
        "segments": int(gaps.sum()) + 1,
        "actual_available": actual is not None,
    }
    if actual is not None:
        error_gen = forecast["generation_mwh"] - actual["generation_mwh"]
        error_smp = (forecast["price_krw_per_mwh"] - actual["price_krw_per_mwh"]) / 1000
        actual_low = actual["price_krw_per_mwh"] <= 0
        forecast_low = forecast["price_krw_per_mwh"] <= 0
        metrics.update({
            "generation_mae_mwh": float(error_gen.abs().mean()),
            "generation_bias_mwh": float(error_gen.mean()),
            "smp_mae_krw_per_kwh": float(error_smp.abs().mean()),
            "smp_bias_krw_per_kwh": float(error_smp.mean()),
            "actual_low_hours": int(actual_low.sum()),
            "forecast_low_hits": int((actual_low & forecast_low).sum()),
            "forecast_false_alarms": int((~actual_low & forecast_low).sum()),
        })
    excluded_path = folder / "excluded_days.csv"
    excluded = pd.read_csv(excluded_path) if excluded_path.exists() else pd.DataFrame()
    return {
        "summary": summary,
        "forecast": forecast,
        "actual": actual,
        "ess": ess,
        "revenues": revenues,
        "realized_profit_krw": None,
        "metrics": metrics,
        "excluded": excluded,
    }
