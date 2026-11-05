"""담당: 최적화 — 계획일별 48시간 예측 창으로 ESS 계획을 롤링 실행한다.

run_rolling 은 기간 전체의 시간별 값 한 줄을 받아 창을 자르지만, 예측은 발표분마다 같은 시각의 값이
다르다. 그래서 계획일 D 마다 그 날의 PlanWindow(forecast_inputs)를 solve_window 에 넣고
앞 24시간(D)만 확정한 뒤, 확정 구간 끝 에너지를 다음 계획일의 시작 에너지로 넘긴다.

- 이월하는 것은 **계획상** SOC 다. 실측에 따른 실행 조정·정산은 하지 않는다 (다음 단계).
- scenario="forecast" 는 window.forecast(예측 발전량·예측 가격), "actual" 은 window.actual
  (실측 발전량·실측 SMP)을 넣는다. 같은 창 목록·ESS 설정·최초 SOC·종료 정책으로 돌려 비교한다.
- 계획일이 끊기면 연속 구간(segment)으로 나누고, 구간마다 설정값 SOC 에서 다시 시작한다.
- 종료 정책은 run_rolling 의 rolling48 과 같다: 구간 마지막 계획일은 D 24시간만 풀고
  E[끝] ≥ 구간 시작 에너지 제약을 건다 (초기 저장분을 팔아 만든 수익 분리).
- 합계(수익·충방전·판매량)는 확정한 24시간만 집계한다. D+1 미확정 계획은 넣지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from vpp.forecast_inputs import PlanWindow, contiguous_segments
from vpp.optimization import EssParams, RollingResult, solve_window

SCENARIOS = ("forecast", "actual")
COMMIT_HOURS = 24


@dataclass(frozen=True)
class SegmentResult:
    """연속 계획일 구간 하나의 결과. result.hourly 는 확정 구간만, daily 는 계획일별 요약."""

    scenario: str
    segment_id: int
    start_day: date
    end_day: date
    result: RollingResult
    sources: list[dict[str, Any]] = field(default_factory=list)


def _inputs(window: PlanWindow, scenario: str) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario 는 {SCENARIOS} 중 하나여야 합니다: {scenario!r}")
    frame = window.forecast if scenario == "forecast" else window.actual
    required = {"ts", "generation_mwh", "price_krw_per_mwh", "dt_hours"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{scenario} 창에 열이 없습니다: {sorted(missing)}")
    return frame


def run_segment(
    segment: list[PlanWindow],
    ess: EssParams,
    scenario: str,
    *,
    segment_id: int = 0,
    tol: float = 1e-6,
) -> SegmentResult:
    """연속 계획일 창 목록을 순서대로 풀어 확정 계획을 잇는다. 끊긴 구간이 섞이면 예외."""
    if not segment:
        raise ValueError("빈 구간입니다.")
    days = [w.planning_day for w in segment]
    if any((b - a).days != 1 for a, b in zip(days[:-1], days[1:], strict=True)):
        raise ValueError(f"계획일이 연속이 아닙니다: {days}")

    period_start_energy = ess.initial_energy_mwh
    start = period_start_energy
    hourly_parts: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for window_id, window in enumerate(segment):
        frame = _inputs(window, scenario)
        is_last = window_id == len(segment) - 1
        # 마지막 계획일은 D 24시간만 풀고 마감 (rolling48 과 동일). 그 외는 48h 풀고 24h 확정.
        block = frame.iloc[:COMMIT_HOURS] if is_last else frame
        n_commit = min(COMMIT_HOURS, len(block))
        final_lo = period_start_energy if is_last else None
        solved = solve_window(
            block["generation_mwh"].to_numpy(),
            block["price_krw_per_mwh"].to_numpy(),
            block["dt_hours"].to_numpy(),
            ess,
            start,
            final_energy_min_mwh=final_lo,
            tol=tol,
        )
        sl = slice(0, n_commit)
        committed = block.iloc[sl].copy()
        committed["sell_mwh"] = solved.sell_mwh[sl]
        committed["charge_mwh"] = solved.charge_mwh[sl]
        committed["discharge_mwh"] = solved.discharge_mwh[sl]
        committed["curtailment_mwh"] = solved.curtailment_mwh[sl]
        committed["energy_end_mwh"] = solved.energy_end_mwh[sl]
        committed["soc_end"] = solved.soc_end[sl]
        committed["window_id"] = window_id
        committed["planning_day"] = window.planning_day
        committed["scenario"] = scenario
        hourly_parts.append(committed)

        price_c = committed["price_krw_per_mwh"].to_numpy()
        gen_c = committed["generation_mwh"].to_numpy()
        sales = float(price_c @ solved.sell_mwh[sl])
        throughput = float(solved.charge_mwh[sl].sum() + solved.discharge_mwh[sl].sum())
        wear_cost = ess.throughput_cost_krw_per_mwh * throughput
        final_energy = ess.clip_energy(solved.energy_end_mwh[n_commit - 1])
        daily_rows.append({
            "window_id": window_id,
            "planning_day": window.planning_day,
            "start_ts": committed["ts"].iloc[0],
            "end_ts": committed["ts"].iloc[-1],
            "solved_intervals": int(len(block)),
            "committed_intervals": n_commit,
            "committed_hours": float(committed["dt_hours"].sum()),
            "initial_energy_mwh": start,
            "final_energy_mwh": final_energy,
            "sales_revenue_krw": sales,
            "throughput_cost_krw": wear_cost,
            "objective_krw": sales - wear_cost,
            "no_ess_same_curtailment_policy_krw": float(np.maximum(price_c, 0) @ gen_c),
            "generation_mwh": float(gen_c.sum()),
            "sell_mwh": float(solved.sell_mwh[sl].sum()),
            "charge_mwh": float(solved.charge_mwh[sl].sum()),
            "discharge_mwh": float(solved.discharge_mwh[sl].sum()),
            "curtailment_mwh": float(solved.curtailment_mwh[sl].sum()),
            "mip_gap": solved.mip_gap,
            "terminal_constraint": "period_end_ge_initial" if is_last else "none",
        })
        sources.append({"planning_day": window.planning_day.isoformat(), **window.sources})
        start = final_energy

    hourly = pd.concat(hourly_parts, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    initial, final = daily["initial_energy_mwh"].to_numpy(), daily["final_energy_mwh"].to_numpy()
    continuity = float(np.max(np.abs(initial[1:] - final[:-1]))) if len(daily) > 1 else 0.0
    gap = period_start_energy - start
    checks = [
        {"item": "rolling_energy_continuity_max_abs_error_mwh", "passed": continuity <= tol,
         "value": continuity},
        {"item": "hourly_rows_equal_committed_hours",
         "passed": len(hourly) == COMMIT_HOURS * len(segment), "value": float(len(hourly))},
        {"item": "committed_days_unique",
         "passed": hourly["planning_day"].nunique() == len(segment),
         "value": float(hourly["planning_day"].nunique())},
        {"item": "period_final_energy_ge_initial_shortfall_mwh", "passed": gap <= tol,
         "value": float(gap)},
    ]
    failed = [chk for chk in checks if not chk["passed"]]
    if failed:
        raise RuntimeError(f"예측 롤링 검산 실패 ({scenario}, segment {segment_id}): {failed}")
    result = RollingResult(
        hourly=hourly, daily=daily, policy="rolling48", horizon_hours=48.0,
        commit_hours=float(COMMIT_HOURS), period_initial_energy_mwh=period_start_energy,
        period_final_energy_mwh=start, checks=checks,
    )
    return SegmentResult(scenario, segment_id, days[0], days[-1], result, sources)


def run_scenario(
    windows: list[PlanWindow], ess: EssParams, scenario: str, *, tol: float = 1e-6
) -> list[SegmentResult]:
    """창 목록을 연속 구간으로 나눠 각각 설정값 SOC 에서 시작해 푼다."""
    return [
        run_segment(segment, ess, scenario, segment_id=i, tol=tol)
        for i, segment in enumerate(contiguous_segments(windows))
    ]


def compare_daily(forecast: list[SegmentResult], actual: list[SegmentResult]) -> pd.DataFrame:
    """두 시나리오의 계획일별 요약을 나란히 놓는다.

    수익 차이는 '예측이 맞았다면'의 계획 수익 차이이지 실제 손익이 아니다.
    """
    def stack(results: list[SegmentResult], suffix: str) -> pd.DataFrame:
        parts = [r.result.daily.assign(segment_id=r.segment_id) for r in results]
        cols = ["planning_day", "segment_id", "initial_energy_mwh", "final_energy_mwh",
                "objective_krw", "generation_mwh", "sell_mwh", "charge_mwh", "discharge_mwh",
                "curtailment_mwh"]
        out = pd.concat(parts, ignore_index=True)[cols]
        return out.rename(columns={c: f"{c}_{suffix}" for c in cols if c != "planning_day"})

    merged = stack(forecast, "forecast").merge(
        stack(actual, "actual"), on="planning_day", how="inner"
    )
    merged["planned_objective_diff_krw"] = (
        merged["objective_krw_forecast"] - merged["objective_krw_actual"]
    )
    return merged.sort_values("planning_day").reset_index(drop=True)
