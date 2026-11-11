"""최적화 — 계획일별 48시간 예측 창으로 ESS 계획을 롤링 실행한다.

run_rolling 은 기간 전체의 시간별 값 한 줄을 받아 창을 자르지만, 예측은 발표분마다 같은 시각의 값이
다르다. 그래서 계획일 D 마다 그 날의 PlanWindow(forecast_inputs)를 solve_window 에 넣고
앞 24시간(D)만 확정한 뒤, 확정 구간 끝 에너지를 다음 계획일의 시작 에너지로 넘긴다.

- 이월하는 것은 **계획상** SOC 다. 실측에 따른 실행 조정·정산은 하지 않는다 (다음 단계).
- scenario="forecast" 는 window.forecast(예측 발전량·예측 가격), "actual" 은 window.actual
  (실측 발전량·실측 SMP)을 넣는다. 같은 창 목록·ESS 설정·최초 SOC·종료 정책으로 돌려 비교한다.
- 계획일이 끊기면 연속 구간(segment)으로 나누고, 구간마다 설정값 SOC 에서 다시 시작한다.
- 종료 정책은 run_rolling 의 rolling48 과 같다: 구간 마지막 계획일은 D 24시간만 풀고
  E[끝] ≥ 구간 시작 에너지 제약을 건다 (초기 저장분을 팔아 만든 수익 분리). 그 제약을 풀었을 때
  늘어나는 계산상 수익은 terminal_constraint_cost_krw 로 따로 적는다 (못 판 잔량의 기회비용 포함.
  실제 충전 비용이 아니므로 참고 지표).
- 합계(수익·충방전·판매량)는 확정한 24시간만 집계한다. D+1 미확정 계획은 넣지 않는다.
- rec(vpp.rec.RecParams)를 주면 두 시나리오 모두 같은 REC 가산 열을 붙여 푼다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from vpp.forecast_inputs import PlanWindow, contiguous_segments
from vpp.optimization import (
    EssParams,
    RollingResult,
    Terminal,
    plan_window,
    terminal_constraint_cost_krw,
)
from vpp.rec import RecParams

SCENARIOS = ("forecast", "actual")
COMMIT_HOURS = 24
TERMINAL_MODES = ("last_day_only", "horizon_aware")


@dataclass(frozen=True)
class TerminalValue:
    """종료 조건이 없는 창 끝의 잔량 1 MWh 평가액 (소프트 종료). 계획 시점 정보만 쓴다.

    window_mean_sell_value: 그 창의 예측 판매 가치(SMP+REC, 음수는 0) 평균을 나중에 받을
    단가로 보고, 방전 효율과 방전 열화비를 뺀 값. 단위 원/MWh.
    = fraction × η_d × max(0, 평균 − 방전 열화비).
    평가액은 계획 목적함수에만 들어가고 수익 집계에는 합산하지 않는다.
    """

    method: str = "window_mean_sell_value"
    fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.method != "window_mean_sell_value":
            raise ValueError("terminal value method 는 window_mean_sell_value 만 지원합니다.")
        if not (0 <= self.fraction <= 1):
            raise ValueError("fraction 은 0~1 이어야 합니다.")

    def value_krw_per_mwh(self, frame: pd.DataFrame, ess: EssParams) -> float:
        from vpp.rec import effective_price_krw_per_mwh

        mean_value = float(np.maximum(effective_price_krw_per_mwh(frame), 0.0).mean())
        _, discharge_wear = ess.degradation().milp_coefficients(1)
        return max(0.0, self.fraction * ess.discharge_efficiency
                   * (mean_value - float(discharge_wear[0])))

    def metadata(self) -> dict[str, Any]:
        return {"method": self.method, "fraction": self.fraction, "unit": "krw_per_mwh",
                "basis": "창의 예측 판매가치(SMP+REC) 평균 × 방전효율 − 방전 열화비. "
                         "계획 시점 정보만",
                "note": "평가액은 목적함수에만 반영. 수익에 합산하지 않음"}


def window_terminal(
    frame: pd.DataFrame, window_id: int, n_windows: int, period_start_energy: float, ess: EssParams,
    *, terminal_mode: str = "last_day_only", terminal_value: TerminalValue | None = None,
) -> tuple[pd.DataFrame, Terminal]:
    """계획일 창의 (풀 구간, 종료 조건).

    - last_day_only (기존): 구간 마지막 날만 24시간을 풀고 E[끝] ≥ 구간 시작 에너지. 그 전날
      48시간 창은 구간 끝을 포함하는데도 종료 조건이 없어 D+1 잠정 계획이 실험 종료를 모른다.
    - horizon_aware: 48시간 창이 구간 끝을 포함하는 날(마지막 전날)부터 창 끝에 같은 조건을
      미리 건다. 마지막 날은 24시간 + 같은 조건. 비교의 시작·종료 잔량 기준은 그대로다.
    종료 조건이 없는 창에는 terminal_value 가 있으면 창 끝 잔량 평가(소프트)를 둔다.
    """
    if terminal_mode not in TERMINAL_MODES:
        raise ValueError(f"terminal_mode 는 {TERMINAL_MODES} 중 하나여야 합니다: {terminal_mode!r}")
    is_last = window_id == n_windows - 1
    if is_last:
        return frame.iloc[:COMMIT_HOURS], Terminal.period_end_ge_initial(period_start_energy)
    if terminal_mode == "horizon_aware" and window_id == n_windows - 2:
        return frame, Terminal.period_end_ge_initial(period_start_energy)
    if terminal_value is not None:
        return frame, Terminal.valued(terminal_value.value_krw_per_mwh(frame, ess))
    return frame, Terminal()


@dataclass(frozen=True)
class SegmentResult:
    """연속 계획일 구간 하나의 결과. result.hourly 는 확정 구간만, daily 는 계획일별 요약."""

    scenario: str
    segment_id: int
    start_day: date
    end_day: date
    result: RollingResult
    sources: list[dict[str, Any]] = field(default_factory=list)


def _inputs(window: PlanWindow, scenario: str, rec: RecParams | None = None) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario 는 {SCENARIOS} 중 하나여야 합니다: {scenario!r}")
    frame = window.forecast if scenario == "forecast" else window.actual
    if frame is None:
        raise ValueError(
            f"{window.planning_day} 는 실측이 없어 actual 시나리오를 풀 수 없습니다 "
            f"({window.comparison_reason}). comparable_windows 로 거르세요."
        )
    required = {"ts", "generation_mwh", "price_krw_per_mwh", "dt_hours"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{scenario} 창에 열이 없습니다: {sorted(missing)}")
    return frame if rec is None else rec.apply(frame)


def run_segment(
    segment: list[PlanWindow],
    ess: EssParams,
    scenario: str,
    *,
    segment_id: int = 0,
    rec: RecParams | None = None,
    terminal_mode: str = "last_day_only",
    terminal_value: TerminalValue | None = None,
    tol: float = 1e-6,
) -> SegmentResult:
    """연속 계획일 창 목록을 순서대로 풀어 확정 계획을 잇는다. 끊긴 구간이 섞이면 예외.

    terminal_mode / terminal_value 는 window_terminal 참고. 기본은 기존 방식
    (last_day_only, 평가 없음).
    """
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
    terminal_cost = 0.0
    for window_id, window in enumerate(segment):
        frame = _inputs(window, scenario, rec)
        is_last = window_id == len(segment) - 1
        # 구간 종료 규칙은 window_terminal. 운영계획은 이 규칙을 쓰지 않고 항상 48h 를 푼다.
        block, terminal = window_terminal(
            frame, window_id, len(segment), period_start_energy, ess,
            terminal_mode=terminal_mode, terminal_value=terminal_value,
        )
        plan = plan_window(block, ess, start, commit_hours=COMMIT_HOURS, terminal=terminal, tol=tol)
        if is_last:
            terminal_cost = terminal_constraint_cost_krw(
                block, ess, start, plan, commit_hours=COMMIT_HOURS, tol=tol,
            )
        committed = plan.committed.drop(columns="committed")
        committed["window_id"] = window_id
        committed["planning_day"] = window.planning_day
        committed["scenario"] = scenario
        hourly_parts.append(committed)
        daily_rows.append({
            "window_id": window_id, "planning_day": window.planning_day, **plan.summary_row(),
        })
        sources.append({"planning_day": window.planning_day.isoformat(), **window.sources})
        start = plan.final_committed_energy_mwh

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
        terminal_constraint_cost_krw=terminal_cost,
    )
    return SegmentResult(scenario, segment_id, days[0], days[-1], result, sources)


def run_scenario(
    windows: list[PlanWindow], ess: EssParams, scenario: str, *,
    rec: RecParams | None = None, terminal_mode: str = "last_day_only",
    terminal_value: TerminalValue | None = None, tol: float = 1e-6,
) -> list[SegmentResult]:
    """창 목록을 연속 구간으로 나눠 각각 설정값 SOC 에서 시작해 푼다."""
    return [
        run_segment(segment, ess, scenario, segment_id=i, rec=rec, terminal_mode=terminal_mode,
                    terminal_value=terminal_value, tol=tol)
        for i, segment in enumerate(contiguous_segments(windows))
    ]


def soc_trace(hourly: pd.DataFrame, segment_initial_mwh: dict[int, float]) -> pd.DataFrame:
    """SOC 그래프용 점 목록. 구간 첫 시각에 시작 잔량, 각 시간의 끝(ts+dt)에 종료 잔량을 놓는다.

    hourly 는 ts, dt_hours, energy_end_mwh, segment_id 를 가진 확정 계획. 구간 사이에는 선이
    이어지지 않도록 빈 점(NaN)을 넣는다.
    """
    parts: list[pd.DataFrame] = []
    for seg, group in hourly.groupby("segment_id", sort=True):
        group = group.sort_values("ts")
        ts_end = group["ts"] + pd.to_timedelta(group["dt_hours"], unit="h")
        parts.append(pd.DataFrame({
            "ts": [group["ts"].iloc[0], *ts_end.tolist(), ts_end.iloc[-1]],
            "energy_mwh": [segment_initial_mwh[int(seg)], *group["energy_end_mwh"].tolist(),
                           float("nan")],
            "segment_id": int(seg),
            "point": ["start", *(["end"] * len(group)), "gap"],
        }))
    return pd.concat(parts, ignore_index=True)


def compare_daily(forecast: list[SegmentResult], actual: list[SegmentResult]) -> pd.DataFrame:
    """두 시나리오의 계획일별 요약을 나란히 놓는다.

    수익 차이는 '예측이 맞았다면'의 계획 수익 차이이지 실제 손익이 아니다.
    """
    def stack(results: list[SegmentResult], suffix: str) -> pd.DataFrame:
        parts = [r.result.daily.assign(segment_id=r.segment_id) for r in results]
        daily = pd.concat(parts, ignore_index=True).sort_values("planning_day")
        # 이월 확인: 오늘 시작 − 전날 종료 (같은 구간 안에서만). 구간 첫날은 설정값으로 초기화.
        prev_final = daily.groupby("segment_id")["final_energy_mwh"].shift(1)
        daily["segment_start"] = prev_final.isna()
        daily["carry_gap_mwh"] = daily["initial_energy_mwh"] - prev_final
        cols = ["planning_day", "segment_id", "segment_start", "initial_energy_mwh",
                "final_energy_mwh", "carry_gap_mwh", "sales_revenue_krw", "rec_revenue_krw",
                "throughput_cost_krw", "objective_krw", "no_ess_same_curtailment_policy_krw",
                "generation_mwh", "sell_mwh", "charge_mwh", "discharge_mwh", "curtailment_mwh"]
        out = daily[cols]
        return out.rename(columns={c: f"{c}_{suffix}" for c in cols if c != "planning_day"})

    merged = stack(forecast, "forecast").merge(
        stack(actual, "actual"), on="planning_day", how="inner"
    )
    merged["planned_objective_diff_krw"] = (
        merged["objective_krw_forecast"] - merged["objective_krw_actual"]
    )
    return merged.sort_values("planning_day").reset_index(drop=True)
