"""예측 평가: 예측·대시보드팀 / 운영비용 평가: 최적화·대시보드팀. 모든 계산은 담당팀이 구현한다."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vpp.baselines import plan_revenue_krw
from vpp.optimization import EssParams, RollingResult
from vpp.schemas import CostEvaluation, ForecastEvaluation, ForecastResponse, OptimizationResult


def evaluate_forecast(forecast: ForecastResponse, actuals: pd.DataFrame) -> ForecastEvaluation:
    """TODO: 지표·산식·결측 처리 합의 후 실측 비교와 5% 목표 판정을 구현한다."""
    raise NotImplementedError("미구현: 예측·대시보드팀이 예측 평가를 구현해 주세요.")


def evaluate_cost(plan: OptimizationResult, actuals: pd.DataFrame) -> CostEvaluation:
    """TODO: 고정된 계획을 실측에 적용한 비용과 동일 조건 기준 방식의 비용을 비교한다.

    스키마 합의 후 `summarize_backtest` 결과를 `CostEvaluation` 필드에 연결할 예정.
    """
    raise NotImplementedError("미구현: 최적화·대시보드팀이 운영비용 평가를 구현해 주세요.")


def _pct(delta: float, base: float) -> float | None:
    return None if abs(base) < 1e-9 else delta / base * 100.0


def summarize_backtest(
    plan: RollingResult, baselines: dict[str, pd.DataFrame], ess: EssParams
) -> dict[str, Any]:
    """MILP 계획(run_rolling 결과)과 기준 계획(baselines.*)들의 수익을 한 표로 요약.

    baselines는 {"no_ess": DataFrame, "rule_based": DataFrame, ...} 형태. 각 DataFrame은
    price_krw_per_mwh·sell_mwh·charge_mwh·discharge_mwh 컬럼을 가진 시간별 계획.
    완전예측 백테스트 결과이며 실제 입찰 성과가 아니다.
    """
    hourly = plan.hourly
    days = int(pd.to_datetime(hourly["ts"]).dt.date.nunique()) if len(hourly) else 0
    discharge_total = float(hourly["discharge_mwh"].sum())
    usable = ess.usable_capacity_mwh
    net = plan.objective_krw
    summary: dict[str, Any] = {
        "mode": "perfect_foresight_backtest_not_live_bidding",
        "policy": plan.policy,
        "days": days,
        "hours": int(len(hourly)),
        "sales_revenue_krw": plan.sales_revenue_krw,
        "throughput_cost_krw": plan.throughput_cost_krw,
        "net_revenue_krw": net,
        "generation_mwh": float(hourly["generation_mwh"].sum()),
        "sold_mwh": float(hourly["sell_mwh"].sum()),
        "charge_mwh": float(hourly["charge_mwh"].sum()),
        "discharge_mwh": discharge_total,
        "curtailment_mwh": float(hourly["curtailment_mwh"].sum()),
        "cycles_per_day": (discharge_total / usable / days) if days and usable > 0 else None,
        "period_initial_energy_mwh": plan.period_initial_energy_mwh,
        "period_final_energy_mwh": plan.period_final_energy_mwh,
        "stored_energy_change_mwh": plan.period_final_energy_mwh - plan.period_initial_energy_mwh,
        "baselines": {},
    }
    for name, frame in baselines.items():
        base_net = plan_revenue_krw(frame, ess)
        delta = net - base_net
        summary["baselines"][name] = {
            "net_revenue_krw": base_net,
            "sold_mwh": float(frame["sell_mwh"].sum()),
            "curtailment_mwh": float(frame["curtailment_mwh"].sum()),
            "discharge_mwh": float(frame["discharge_mwh"].sum()),
            "milp_gain_krw": delta,
            "milp_gain_pct": _pct(delta, base_net),
        }
    return summary
