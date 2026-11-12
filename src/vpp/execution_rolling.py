"""최적화 — 실행 SOC 이월 일별 사슬: 계획 → 판매계획 확정 → 실측 적용 운전 → 정산 → SOC 전달.

연속 구간(segment) 안에서 계획일 D 마다
 1. 시작 잔량 = 전날 **실행** 종료 잔량 (plan_soc_source="executed").
    "planned" 는 기존 방식(계획상 SOC 이월) 비교용이며, 실행은 언제나 실제 잔량에서 시작한다.
 2. 그 시점에 쓸 수 있는 48시간 예측으로 계획 (종료 조건은 forecast_rolling.window_terminal).
 3. 앞 24시간을 확정한다 (판매계획 고정).
 4. 실측 발전량을 시간순으로 적용해 운전하고 (vpp.execution), 실측 SMP 로 정산한다 (vpp.settlement).
 5. 실행 종료 잔량을 다음 계획일로 넘긴다.
계획 종료 SOC(planned_final_energy_mwh)와 실행 종료 SOC(executed_final_energy_mwh)는 따로 저장한다.

실측이 없는 날은 사슬에 넣지 않는다. 그런 날이 끼면 구간이 끊기고(contiguous_segments) 다음 구간은
설정값 잔량에서 다시 시작하며, 그 사유를 segment_start_reason 에 적는다. 빠진 시간을 건너뛰어 SOC 를
이어 붙이거나 보간하지 않는다.

ESS 미사용 기준(run_no_ess_execution)은 같은 예측으로 ESS 없이 판매계획을 세우고
(baselines.no_ess) 같은 실측·정산으로 운전한 결과다. 실행 운전에서 배터리를 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from vpp.baselines import no_ess
from vpp.execution import ExecutionPolicy, RuleBasedPolicy, execute_plan, execution_totals
from vpp.forecast_inputs import PlanWindow, contiguous_segments
from vpp.forecast_rolling import COMMIT_HOURS, TerminalValue, window_terminal
from vpp.optimization import EssParams, plan_window
from vpp.rec import RecParams
from vpp.settlement import REVENUE_COLUMNS, SettlementParams, settle, settlement_totals

PLAN_SOC_SOURCES = ("executed", "planned")
NO_ESS_POLICY = RuleBasedPolicy(extra_discharge=False, surplus_order=("sell", "curtail"),
                                name="no_ess_passthrough")


@dataclass(frozen=True)
class ExecutionSegmentResult:
    scenario: str
    segment_id: int
    start_day: date
    end_day: date
    plan_hourly: pd.DataFrame      # 확정 24시간 계획 (계획일별)
    exec_hourly: pd.DataFrame      # 실행 + 정산 (계획일별)
    daily: pd.DataFrame
    period_initial_energy_mwh: float
    planned_final_energy_mwh: float
    executed_final_energy_mwh: float
    segment_start_reason: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


def _check(name: str, value: float, passed: bool) -> dict[str, Any]:
    return {"item": name, "passed": bool(passed), "value": float(value)}


def _actual_day(window: PlanWindow) -> pd.DataFrame:
    if window.actual is None:
        raise ValueError(
            f"{window.planning_day} 는 실측이 없어 실행 운전을 계산하지 않습니다 "
            f"({window.comparison_reason})."
        )
    return window.actual.iloc[:COMMIT_HOURS].reset_index(drop=True)


def _forecast_inputs(window: PlanWindow, rec: RecParams | None) -> pd.DataFrame:
    return window.forecast if rec is None else rec.apply(window.forecast)


def run_execution_segment(
    segment: list[PlanWindow],
    ess: EssParams,
    *,
    rec: RecParams | None = None,
    policy: ExecutionPolicy | None = None,
    settlement: SettlementParams | None = None,
    plan_soc_source: str = "executed",
    terminal_mode: str = "horizon_aware",
    terminal_value: TerminalValue | None = None,
    segment_id: int = 0,
    scenario: str = "executed",
    segment_start_reason: str = "period_start",
    tol: float = 1e-6,
) -> ExecutionSegmentResult:
    if plan_soc_source not in PLAN_SOC_SOURCES:
        raise ValueError(f"plan_soc_source 는 {PLAN_SOC_SOURCES} 중 하나: {plan_soc_source!r}")
    if not segment:
        raise ValueError("빈 구간입니다.")
    days = [w.planning_day for w in segment]
    if any((b - a).days != 1 for a, b in zip(days[:-1], days[1:], strict=True)):
        raise ValueError(f"계획일이 연속이 아닙니다: {days}")
    policy = policy or RuleBasedPolicy()
    settlement = settlement or SettlementParams()

    period_start = ess.initial_energy_mwh
    plan_start = exec_start = period_start
    plan_parts: list[pd.DataFrame] = []
    exec_parts: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for window_id, window in enumerate(segment):
        frame = _forecast_inputs(window, rec)
        block, terminal = window_terminal(
            frame, window_id, len(segment), period_start, ess,
            terminal_mode=terminal_mode, terminal_value=terminal_value,
        )
        start = exec_start if plan_soc_source == "executed" else plan_start
        plan = plan_window(block, ess, start, commit_hours=COMMIT_HOURS, terminal=terminal, tol=tol)
        committed = plan.committed.drop(columns="committed").reset_index(drop=True)
        actual = _actual_day(window)
        if not committed["ts"].reset_index(drop=True).equals(actual["ts"]):
            raise ValueError(f"{window.planning_day}: 확정 계획과 실측의 시각이 다릅니다.")
        executed = execute_plan(committed, actual["generation_mwh"], exec_start, ess, policy,
                                tol=tol)
        settled = settle(executed.hourly, actual["price_krw_per_mwh"], ess, settlement)
        for part in (committed, settled):
            part["window_id"] = window_id
            part["planning_day"] = window.planning_day
            part["scenario"] = scenario
        plan_parts.append(committed)
        exec_parts.append(settled)
        totals = plan.totals(committed_only=True)
        rows.append({
            "window_id": window_id, "planning_day": window.planning_day,
            "plan_start_energy_mwh": plan.initial_energy_mwh,
            "exec_start_energy_mwh": executed.energy_start_mwh,
            "planned_final_energy_mwh": plan.final_committed_energy_mwh,
            "executed_final_energy_mwh": executed.energy_end_mwh,
            "terminal_constraint": terminal.name,
            "terminal_value_krw": plan.terminal_value_krw,
            "solved_intervals": int(len(plan.hourly)),
            **{f"planned_{k}": v for k, v in totals.items()},
            **{f"exec_{k}": v for k, v in execution_totals(settled, ess).items()},
            **settlement_totals(settled),
            "mip_gap": plan.mip_gap,
        })
        plan_start = plan.final_committed_energy_mwh
        exec_start = executed.energy_end_mwh

    daily = pd.DataFrame(rows)
    exec_hourly = pd.concat(exec_parts, ignore_index=True)
    plan_hourly = pd.concat(plan_parts, ignore_index=True)
    e_start = daily["exec_start_energy_mwh"].to_numpy()
    e_end = daily["executed_final_energy_mwh"].to_numpy()
    carry = float(np.max(np.abs(e_start[1:] - e_end[:-1]))) if len(daily) > 1 else 0.0
    checks = [
        _check("executed_soc_carry_max_abs_error_mwh", carry, carry <= tol),
        _check("plan_rows_equal_committed_hours", float(len(plan_hourly)),
               len(plan_hourly) == COMMIT_HOURS * len(segment)),
        _check("exec_rows_equal_plan_rows", float(len(exec_hourly)),
               len(exec_hourly) == len(plan_hourly)),
        _check("committed_sales_unchanged_by_execution",
               float(np.max(np.abs(exec_hourly["committed_sell_mwh"].to_numpy()
                                   - plan_hourly["sell_mwh"].to_numpy()))),
               np.allclose(exec_hourly["committed_sell_mwh"], plan_hourly["sell_mwh"], atol=tol)),
    ]
    if plan_soc_source == "executed":
        gap = float(np.max(np.abs(daily["plan_start_energy_mwh"] - daily["exec_start_energy_mwh"])))
        checks.append(_check("plan_starts_from_executed_soc_max_abs_error_mwh", gap, gap <= tol))
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise RuntimeError(f"실행 사슬 검산 실패 ({scenario}, segment {segment_id}): {failed}")
    return ExecutionSegmentResult(
        scenario, segment_id, days[0], days[-1], plan_hourly, exec_hourly, daily,
        period_start, plan_start, exec_start, segment_start_reason, checks,
        settings={"plan_soc_source": plan_soc_source, "terminal_mode": terminal_mode,
                  "terminal_value": None if terminal_value is None else terminal_value.metadata(),
                  "policy": policy.metadata(), "settlement": settlement.metadata()},
    )


def run_no_ess_execution(
    segment: list[PlanWindow], ess: EssParams, *, rec: RecParams | None = None,
    settlement: SettlementParams | None = None, segment_id: int = 0,
    segment_start_reason: str = "period_start", tol: float = 1e-6,
) -> ExecutionSegmentResult:
    """ESS 미사용 기준: 같은 예측으로 ESS 없이 판매계획을 세우고 같은 실측·정산으로 운전한다."""
    settlement = settlement or SettlementParams()
    plan_parts, exec_parts, rows = [], [], []
    start = ess.initial_energy_mwh
    for window_id, window in enumerate(segment):
        forecast = _forecast_inputs(window, rec).iloc[:COMMIT_HOURS].reset_index(drop=True)
        committed = no_ess(forecast)
        committed["energy_end_mwh"] = start
        committed["soc_end"] = start / ess.capacity_mwh
        actual = _actual_day(window)
        executed = execute_plan(committed, actual["generation_mwh"], start, ess, NO_ESS_POLICY,
                                tol=tol)
        settled = settle(executed.hourly, actual["price_krw_per_mwh"], ess, settlement)
        if settled[["charge_mwh", "discharge_mwh"]].abs().to_numpy().max() > tol:
            raise RuntimeError("ESS 미사용 기준에서 배터리가 움직였습니다.")
        for part in (committed, settled):
            part["window_id"] = window_id
            part["planning_day"] = window.planning_day
            part["scenario"] = "no_ess_executed"
        plan_parts.append(committed)
        exec_parts.append(settled)
        rows.append({
            "window_id": window_id, "planning_day": window.planning_day,
            "plan_start_energy_mwh": start, "exec_start_energy_mwh": start,
            "planned_final_energy_mwh": start, "executed_final_energy_mwh": start,
            "terminal_constraint": "none", "terminal_value_krw": 0.0,
            "solved_intervals": COMMIT_HOURS,
            "planned_sales_revenue_krw": float(committed["price_krw_per_mwh"]
                                               @ committed["sell_mwh"]),
            "planned_objective_krw": float(
                (committed["price_krw_per_mwh"] + committed.get("rec_krw_per_mwh", 0.0))
                @ committed["sell_mwh"]
            ),
            **{f"exec_{k}": v for k, v in execution_totals(settled, ess).items()},
            **settlement_totals(settled),
            "mip_gap": 0.0,
        })
    daily = pd.DataFrame(rows)
    days = [w.planning_day for w in segment]
    return ExecutionSegmentResult(
        "no_ess_executed", segment_id, days[0], days[-1],
        pd.concat(plan_parts, ignore_index=True), pd.concat(exec_parts, ignore_index=True), daily,
        start, start, start, segment_start_reason,
        checks=[_check("no_ess_battery_untouched", 0.0, True)],
        settings={"policy": NO_ESS_POLICY.metadata(), "settlement": settlement.metadata()},
    )


def segment_reasons(windows: list[PlanWindow], gap_reasons: dict[date, str] | None = None) -> list:
    """구간마다 시작 사유. 첫 구간은 period_start, 이후는 바로 전날이 빠진 이유.

    사유를 모르면 calendar_gap.
    """
    segments = contiguous_segments(windows)
    reasons = []
    for i, seg in enumerate(segments):
        if i == 0:
            reasons.append("period_start")
            continue
        previous = seg[0].planning_day - pd.Timedelta(days=1)
        reason = (gap_reasons or {}).get(previous.date() if hasattr(previous, "date") else previous)
        reasons.append(f"restart_after_gap:{reason or 'calendar_gap'}")
    return reasons


def run_execution_scenario(
    windows: list[PlanWindow], ess: EssParams, *, scenario: str = "executed",
    gap_reasons: dict[date, str] | None = None, **kwargs,
) -> list[ExecutionSegmentResult]:
    """실측이 있는 창만 받아 연속 구간별로 실행 사슬을 돈다. scenario='no_ess_executed' 는 기준."""
    segments = contiguous_segments(windows)
    reasons = segment_reasons(windows, gap_reasons)
    out = []
    for i, (seg, reason) in enumerate(zip(segments, reasons, strict=True)):
        if scenario == "no_ess_executed":
            out.append(run_no_ess_execution(
                seg, ess, rec=kwargs.get("rec"), settlement=kwargs.get("settlement"),
                segment_id=i, segment_start_reason=reason,
            ))
        else:
            out.append(run_execution_segment(
                seg, ess, segment_id=i, scenario=scenario, segment_start_reason=reason, **kwargs,
            ))
    return out


def stack_exec_hourly(results: list[ExecutionSegmentResult]) -> pd.DataFrame:
    return pd.concat([r.exec_hourly.assign(segment_id=r.segment_id) for r in results],
                     ignore_index=True)


def stack_plan_hourly(results: list[ExecutionSegmentResult]) -> pd.DataFrame:
    return pd.concat([r.plan_hourly.assign(segment_id=r.segment_id) for r in results],
                     ignore_index=True)


def stack_exec_daily(results: list[ExecutionSegmentResult]) -> pd.DataFrame:
    parts = []
    for r in results:
        daily = r.daily.assign(segment_id=r.segment_id, scenario=r.scenario,
                               segment_start=False, segment_start_reason="")
        daily.loc[daily.index[0], ["segment_start", "segment_start_reason"]] = [
            True, r.segment_start_reason,
        ]
        parts.append(daily)
    return pd.concat(parts, ignore_index=True)


TERMINAL_POLICIES = (
    ("last_day_only", None),
    ("horizon_aware", None),
    ("horizon_aware", 0.5),
    ("horizon_aware", 1.0),
)


def compare_terminal_policies(
    windows: list[PlanWindow], ess: EssParams, *, rec: RecParams | None = None,
    policy: ExecutionPolicy | None = None, settlement: SettlementParams | None = None,
    gap_reasons: dict[date, str] | None = None, policies=TERMINAL_POLICIES,
) -> pd.DataFrame:
    """같은 창·실측·정산으로 종료 정책만 바꿔 실행 사슬을 돌린 비교표.

    realized_net_krw 가 크다고 좋은 정책이 아니다. executed_end_shortfall_mwh(시작보다 비운
    잔량)가 크면 배터리를 비워 번 돈이 섞인 것이므로 함께 본다. 평가액은 계획 목적함수에만
    들어간 값이다.
    """
    rows = []
    for mode, fraction in policies:
        value = None if fraction is None else TerminalValue(fraction=fraction)
        results = run_execution_scenario(
            windows, ess, rec=rec, policy=policy, settlement=settlement, gap_reasons=gap_reasons,
            plan_soc_source="executed", terminal_mode=mode, terminal_value=value,
        )
        t = execution_scenario_totals(results)
        rows.append({
            "terminal_mode": mode,
            "terminal_value_fraction": fraction,
            "realized_net_krw": t["realized_net_krw"],
            "planned_objective_krw": t["planned_objective_krw"],
            "shortfall_mwh": t["shortfall_mwh"],
            "surplus_mwh": t["surplus_mwh"],
            "charge_mwh": t["charge_mwh"],
            "discharge_mwh": t["discharge_mwh"],
            "wear_krw": t["wear_krw"],
            "executed_end_shortfall_mwh": t["executed_end_shortfall_mwh"],
            "terminal_value_krw_plan_only": t["terminal_value_krw_plan_only"],
        })
    return pd.DataFrame(rows)


def execution_scenario_totals(results: list[ExecutionSegmentResult]) -> dict[str, Any]:
    """시나리오 합계. 정산 불가 날이 하나라도 있으면 금액은 None."""
    daily = stack_exec_daily(results)
    computable = bool(daily["computable"].all())
    out: dict[str, Any] = {
        "days": int(len(daily)), "computable": computable,
        "not_computable_days": int((~daily["computable"]).sum()),
        "segments": [
            {"segment_id": r.segment_id, "start": r.start_day.isoformat(),
             "end": r.end_day.isoformat(), "days": int(len(r.daily)),
             "start_reason": r.segment_start_reason,
             "initial_energy_mwh": r.period_initial_energy_mwh,
             "planned_final_energy_mwh": r.planned_final_energy_mwh,
             "executed_final_energy_mwh": r.executed_final_energy_mwh,
             # 공정 비교용: 실행 종료 잔량이 시작보다 모자란 양 (양수면 배터리를 비워 번 돈이 섞임)
             "executed_end_shortfall_mwh": r.period_initial_energy_mwh
             - r.executed_final_energy_mwh}
            for r in results
        ],
        "planned_objective_krw": float(daily["planned_objective_krw"].sum()),
        "terminal_value_krw_plan_only": float(daily["terminal_value_krw"].sum()),
        "supply_mwh": float(daily["exec_supply_mwh"].sum()),
        "committed_sell_mwh": float(daily["exec_committed_sell_mwh"].sum()),
        "shortfall_mwh": float(daily["exec_shortfall_mwh"].sum()),
        "surplus_mwh": float(daily["exec_surplus_mwh"].sum()),
        "shortfall_hours": int(daily["exec_shortfall_hours"].sum()),
        "charge_mwh": float(daily["exec_charge_mwh"].sum()),
        "discharge_mwh": float(daily["exec_discharge_mwh"].sum()),
        "curtailment_mwh": float(daily["exec_curtailment_mwh"].sum()),
        "executed_end_shortfall_mwh": float(sum(
            r.period_initial_energy_mwh - r.executed_final_energy_mwh for r in results
        )),
        "settings": results[0].settings if results else {},
    }
    for col in REVENUE_COLUMNS:
        out[col] = float(daily[col].sum()) if computable else None
    return out
