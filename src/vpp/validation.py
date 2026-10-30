"""담당: 최적화·대시보드팀 — ESS 계획 검산과 속성 검사.

위반을 예외로 던지지 않고 {item, passed, value} 목록으로 반환한다. optimization은 이 결과에
실패가 있으면 예외를 던지고, 대시보드는 표(✅/❌)로 보여 준다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # 순환 import 방지 (optimization → validation)
    from vpp.optimization import EssParams

PLAN_COLUMNS = (
    "generation_mwh", "price_krw_per_mwh", "dt_hours",
    "sell_mwh", "charge_mwh", "discharge_mwh", "curtailment_mwh", "energy_end_mwh",
)


def _item(name: str, value: float, passed: bool) -> dict[str, Any]:
    return {"item": name, "passed": bool(passed), "value": float(value)}


def check_plan(
    plan: pd.DataFrame,
    ess: EssParams,
    tol: float = 1e-6,
    *,
    initial_energy_mwh: float | None = None,
    objective_krw: float | None = None,
    solver_status: str = "optimal",
    export_cap_mwh: np.ndarray | None = None,
    final_energy_min_mwh: float | None = None,
    final_energy_max_mwh: float | None = None,
) -> list[dict[str, Any]]:
    """시간별 계획(DataFrame)의 제약 검산.

    항목: 풀이 상태, 전력균형, SOC 재계산 일치, SOC 범위, 충전·방전 상한, 판매·감축 범위,
    (판매 상한), 동시 충방전 없음, 종료 에너지, 목적함수 재계산 일치, 롤링 에너지 연속.
    initial_energy_mwh가 None이면 ess.initial_energy_mwh(기간 첫 시작)를 쓴다.
    objective_krw가 None이면 계획에서 재계산한 값과 비교할 대상이 없으므로 그 항목은 생략.
    """
    missing = [c for c in PLAN_COLUMNS if c not in plan.columns]
    if missing:
        raise ValueError(f"계획 DataFrame에 컬럼이 없습니다: {missing}")
    g = plan["generation_mwh"].to_numpy(dtype=float)
    price = plan["price_krw_per_mwh"].to_numpy(dtype=float)
    dt = plan["dt_hours"].to_numpy(dtype=float)
    sell = plan["sell_mwh"].to_numpy(dtype=float)
    charge = plan["charge_mwh"].to_numpy(dtype=float)
    discharge = plan["discharge_mwh"].to_numpy(dtype=float)
    curtailed = plan["curtailment_mwh"].to_numpy(dtype=float)
    energy = plan["energy_end_mwh"].to_numpy(dtype=float)
    start = ess.initial_energy_mwh if initial_energy_mwh is None else float(initial_energy_mwh)
    ec, ed = ess.charge_efficiency, ess.discharge_efficiency
    previous = np.r_[start, energy[:-1]]
    cap_c, cap_d = ess.charge_power_mw * dt, ess.discharge_power_mw * dt

    checks: list[dict[str, Any]] = [
        _item("solver_status_optimal", 1.0 if solver_status == "optimal" else 0.0,
              solver_status == "optimal"),
    ]
    v = float(np.max(np.abs(g + discharge - sell - charge - curtailed)))
    checks.append(_item("power_balance_max_abs_error_mwh", v, v < tol))
    recomputed = previous + ec * charge - discharge / ed
    v = float(np.max(np.abs(energy - recomputed)))
    checks.append(_item("energy_balance_max_abs_error_mwh", v, v < tol))
    v = float(max(ess.energy_min_mwh - energy.min(), energy.max() - ess.energy_max_mwh))
    checks.append(_item("soc_bounds_max_violation_mwh", v, v <= tol))
    v = float(max(-charge.min(), (charge - np.minimum(g, cap_c)).max()))
    checks.append(_item("charge_bounds_max_violation_mwh", v, v <= tol))
    v = float(max(-discharge.min(), (discharge - cap_d).max()))
    checks.append(_item("discharge_bounds_max_violation_mwh", v, v <= tol))
    v = float(max(-sell.min(), -curtailed.min(), (curtailed - g).max()))
    checks.append(_item("export_curtailment_bounds_max_violation_mwh", v, v <= tol))
    if export_cap_mwh is not None:
        v = float((sell - np.asarray(export_cap_mwh, dtype=float)).max())
        checks.append(_item("export_cap_max_violation_mwh", v, v <= tol))
    v = float(np.minimum(charge, discharge).max())
    checks.append(_item("simultaneous_charge_discharge_max_mwh", v, v <= tol))
    if final_energy_min_mwh is not None or final_energy_max_mwh is not None:
        lo = -np.inf if final_energy_min_mwh is None else final_energy_min_mwh
        hi = np.inf if final_energy_max_mwh is None else final_energy_max_mwh
        v = float(max(lo - energy[-1], energy[-1] - hi))
        checks.append(_item("terminal_energy_max_violation_mwh", v, v <= tol))
    if objective_krw is not None:
        recomputed_obj = float(price @ sell) - ess.throughput_cost_krw_per_mwh * float(
            charge.sum() + discharge.sum()
        )
        v = abs(recomputed_obj - objective_krw)
        checks.append(_item("objective_recomputed_abs_error_krw", v,
                            v <= 1e-3 + 1e-6 * abs(objective_krw)))
    if "window_id" in plan.columns:
        window = plan["window_id"].to_numpy()
        boundary = np.flatnonzero(window[1:] != window[:-1]) + 1  # 각 창의 첫 행
        v = float(np.max(np.abs(energy - recomputed)[boundary])) if len(boundary) else 0.0
        checks.append(_item("rolling_energy_continuity_max_abs_error_mwh", v, v < tol))
    return checks


def check_properties(summary: dict[str, Any], tol_krw: float = 1e-3) -> list[dict[str, Any]]:
    """백테스트 요약(evaluation.summarize_backtest)의 속성 검사.

    - MILP 순수익 ≥ ESS 없음, ≥ 규칙 계획 (baselines에 있는 기준마다)
    - summary["size_sweep"] = [{"capacity_mwh", "net_revenue_krw"}, ...]가 있으면 용량 증가 시
      수익 비감소
    """
    checks: list[dict[str, Any]] = []
    for name, base in summary.get("baselines", {}).items():
        gain = float(base["milp_gain_krw"])
        checks.append(_item(f"milp_ge_{name}_gain_krw", gain, gain >= -tol_krw))
    sweep = summary.get("size_sweep")
    if sweep:
        ordered = sorted(sweep, key=lambda r: float(r["capacity_mwh"]))
        revenue = np.array([float(r["net_revenue_krw"]) for r in ordered])
        worst = float(np.min(np.diff(revenue))) if len(revenue) > 1 else 0.0
        checks.append(_item("revenue_nondecreasing_in_capacity_min_step_krw", worst,
                            worst >= -tol_krw))
    return checks
