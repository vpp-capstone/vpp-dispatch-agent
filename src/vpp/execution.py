"""확정 판매계획을 고정한 채 실측 발전량에 맞춰 ESS 충·방전을 조정하는 실행 운전.

확정 24시간 계획(forecast_rolling / operating_plan)은 예측 발전량으로 세운 것이라 실측과 어긋난다.
이 모듈은 **판매계획(sell_mwh)을 다시 쓰지 않고** 시간 순서대로 운전한다. 첫 버전은 설명 가능한 규칙
(RuleBasedPolicy)이며 수익 최적 운전을 보장하지 않는다. 재최적화 정책은 ExecutionPolicy 인터페이스로
교체할 수 있다.

시간 t 의 입력: 확정 판매량 q*, 계획 충전 c*, 계획 방전 d*, 실측 발전량 g, 직전 시간의 실제 잔량 e0.
1. 계획을 실제 상태에서 가능한 범위로 자른다. c ≤ min(c*, g, 충전 출력, 남은 용량),
   d ≤ min(d*, 방전 출력, 가용 잔량). (계통 충전 금지: 충전은 발전량 이내)
2. 부족 (g − c + d < q*): 계획 충전을 먼저 줄이고, 그래도 모자라면 허용 범위에서 추가 방전한다.
   남는 부족분은 shortfall_mwh 로 기록한다 (숨기지 않는다).
3. 잉여 (g − c + d > q*): surplus_order 순서로 처리한다. 기본은
   계획 방전 축소 → 초과 판매 → 추가 충전 → 감축. 초과 판매는 계획 시점 판매 가치(예측 SMP + REC)가
   0 이상일 때만 한다 (그 시각의 실측 가격은 운전 시점에 모른다). 따라서 계획에 없던 충전은 팔 수
   없는(가치 < 0) 전력에만 한다. 2024년 하반기 실측으로 '추가 충전 → 초과 판매' 순서를 돌려 보면
   계획에 없던 충전 1,712 MWh 의 열화비가 실행 수익을 약 2.2억 원 깎았다
   (docs/execution_settlement.md).
4. e1 = e0 + η_c·c − d/η_d. 매 시간 g + d = 공급 + c + 감축, 동시 충방전 없음, SOC·출력 범위를
   검산한다.

미래 시각의 실측(발전량·가격)은 보지 않는다. 실행 중 판매계획은 바뀌지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from vpp.optimization import EssParams
from vpp.rec import REC_COLUMN, rec_adder

SURPLUS_STEPS = ("reduce_discharge", "sell", "charge", "curtail")  # 허용 단계 (기본 순서이기도 함)
PLAN_COLUMNS = ("ts", "sell_mwh", "charge_mwh", "discharge_mwh", "price_krw_per_mwh")
EXECUTION_COLUMNS = (
    "charge_mwh", "discharge_mwh", "supply_mwh", "curtailment_mwh", "shortfall_mwh",
    "surplus_mwh", "energy_start_mwh", "energy_end_mwh",
)


@dataclass(frozen=True)
class HourInput:
    committed_sell_mwh: float
    planned_charge_mwh: float
    planned_discharge_mwh: float
    actual_generation_mwh: float
    energy_start_mwh: float
    dt_hours: float
    plan_sell_value_krw_per_mwh: float  # 계획 시점 판매 가치 (예측 SMP + REC)


@dataclass(frozen=True)
class HourResult:
    charge_mwh: float
    discharge_mwh: float
    supply_mwh: float
    curtailment_mwh: float
    shortfall_mwh: float
    surplus_mwh: float
    energy_end_mwh: float
    actions: tuple[str, ...]


class ExecutionPolicy(Protocol):
    """시간 하나를 운전하는 정책. 미래 시각 정보는 받지 않는다."""

    name: str

    def execute_hour(self, step: HourInput, ess: EssParams) -> HourResult: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RuleBasedPolicy:
    """판매 약속 우선 규칙. 수익 최적을 보장하지 않는다 (status 참고)."""

    extra_discharge: bool = True  # 부족 시 계획에 없던 추가 방전 허용
    surplus_order: tuple[str, ...] = SURPLUS_STEPS
    sell_surplus_only_if_value_nonnegative: bool = True
    name: str = "committed_sales_first_v0"
    status: str = "rule_based_not_profit_optimal"

    def __post_init__(self) -> None:
        order = tuple(self.surplus_order)
        unknown = sorted(set(order) - set(SURPLUS_STEPS))
        if unknown or len(set(order)) != len(order):
            raise ValueError(f"surplus_order 는 {SURPLUS_STEPS} 의 부분 순열이어야 합니다: {order}")
        object.__setattr__(self, "surplus_order", order)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RuleBasedPolicy:
        raw = dict(raw or {})
        if "surplus_order" in raw:
            raw["surplus_order"] = tuple(raw["surplus_order"])
        return cls(**raw)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name, "status": self.status, "extra_discharge": self.extra_discharge,
            "surplus_order": list(self.surplus_order),
            "sell_surplus_only_if_value_nonnegative": self.sell_surplus_only_if_value_nonnegative,
            "note": "판매계획 고정. 부족: 충전 축소 → 추가 방전 → 부족분 기록. "
                    "잉여: surplus_order 순. 미래 실측은 보지 않음.",
        }

    def execute_hour(self, step: HourInput, ess: EssParams, tol: float = 1e-9) -> HourResult:
        g = max(0.0, step.actual_generation_mwh)
        q = max(0.0, step.committed_sell_mwh)
        e0 = step.energy_start_mwh
        dt = step.dt_hours
        cap_c, cap_d = ess.charge_power_mw * dt, ess.discharge_power_mw * dt
        room = max(0.0, (ess.energy_max_mwh - e0) / ess.charge_efficiency)
        available = max(0.0, (e0 - ess.energy_min_mwh) * ess.discharge_efficiency)
        actions: list[str] = []

        # 1. 계획을 실제 상태에서 가능한 범위로 자른다 (동시 충방전은 계획에 없다)
        c = min(max(0.0, step.planned_charge_mwh), g, cap_c, room)
        d = min(max(0.0, step.planned_discharge_mwh), cap_d, available)
        if c > tol and d > tol:  # 계획이 동시 충방전이면 더 큰 쪽만 남긴다 (검산 통과 계획엔 없음)
            if c >= d:
                d = 0.0
            else:
                c = 0.0
            actions.append("drop_simultaneous")
        if c < step.planned_charge_mwh - tol:
            actions.append("charge_limited")
        if d < step.planned_discharge_mwh - tol:
            actions.append("discharge_limited")

        net = g - c + d
        shortfall = surplus_sold = curtail = 0.0
        if net < q - tol:
            deficit = q - net
            cut = min(c, deficit)
            if cut > tol:
                c -= cut
                deficit -= cut
                actions.append("charge_cut")
            if deficit > tol and self.extra_discharge and c <= tol:
                extra = max(0.0, min(deficit, cap_d - d, available - d))
                if extra > tol:
                    d += extra
                    deficit -= extra
                    actions.append("extra_discharge")
            shortfall = max(0.0, deficit)
            if shortfall > tol:
                actions.append("shortfall")
        elif net > q + tol:
            extra = net - q
            for rule in self.surplus_order:
                if extra <= tol:
                    break
                if rule == "reduce_discharge" and d > tol:
                    cut = min(d, extra)
                    d -= cut
                    extra -= cut
                    actions.append("reduce_discharge")
                elif rule == "charge" and d <= tol:
                    add = max(0.0, min(extra, cap_c - c, room - c))
                    if add > tol:
                        c += add
                        extra -= add
                        actions.append("extra_charge")
                elif rule == "sell":
                    allowed = (not self.sell_surplus_only_if_value_nonnegative
                               or step.plan_sell_value_krw_per_mwh >= 0)
                    if allowed:
                        surplus_sold = extra
                        extra = 0.0
                        actions.append("sell_surplus")
                elif rule == "curtail":
                    curtail = extra
                    extra = 0.0
                    actions.append("curtail")
            if extra > tol:  # 정책이 감축을 빼놓았어도 남는 전력은 버린다 (물리 균형)
                curtail += extra
                actions.append("curtail")
        shortfall = 0.0 if shortfall <= tol else shortfall
        surplus_sold = 0.0 if surplus_sold <= tol else surplus_sold
        supply = q - shortfall + surplus_sold
        e1 = ess.clip_energy(e0 + ess.charge_efficiency * c - d / ess.discharge_efficiency)
        return HourResult(c, d, supply, curtail, shortfall, surplus_sold, e1, tuple(actions))


def load_policy(path: str | Path | None) -> RuleBasedPolicy:
    """config/execution_v0.yaml 의 policy 절. 없으면 기본 규칙."""
    if path is None:
        return RuleBasedPolicy()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RuleBasedPolicy.from_dict(raw.get("policy"))


@dataclass(frozen=True)
class ExecutionResult:
    hourly: pd.DataFrame
    energy_start_mwh: float
    energy_end_mwh: float
    policy: dict[str, Any]
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def execute_plan(
    committed: pd.DataFrame,
    actual_generation_mwh,
    energy_start_mwh: float,
    ess: EssParams,
    policy: ExecutionPolicy | None = None,
    *,
    tol: float = 1e-6,
) -> ExecutionResult:
    """확정 계획(시간순)을 실측 발전량에 적용한다. 실측이 하나라도 없으면 계산하지 않는다.

    committed: ts, sell_mwh, charge_mwh, discharge_mwh, price_krw_per_mwh[, rec_krw_per_mwh,
               dt_hours, generation_mwh(예측), energy_end_mwh(계획)].
    actual_generation_mwh: committed 와 같은 순서·길이의 실측 발전량(MWh).
    """
    policy = policy or RuleBasedPolicy()
    missing = [c for c in PLAN_COLUMNS if c not in committed.columns]
    if missing:
        raise ValueError(f"확정 계획에 열이 없습니다: {missing}")
    plan = committed.reset_index(drop=True)
    if plan.empty:
        raise ValueError("확정 계획이 비어 있습니다.")
    ts = pd.to_datetime(plan["ts"])
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("확정 계획의 ts 는 중복 없이 오름차순이어야 합니다.")
    actual = np.asarray(actual_generation_mwh, dtype=float)
    if actual.shape != (len(plan),):
        raise ValueError("실측 발전량 길이가 확정 계획과 다릅니다.")
    if not np.isfinite(actual).all():
        raise ValueError("실측 발전량에 결측이 있어 그 날은 실행 운전을 계산하지 않습니다.")
    if (actual < -tol).any():
        raise ValueError("실측 발전량이 음수입니다.")
    dt = plan["dt_hours"].to_numpy(dtype=float) if "dt_hours" in plan else np.ones(len(plan))
    value = plan["price_krw_per_mwh"].to_numpy(dtype=float) + rec_adder(plan)
    start = float(energy_start_mwh)
    if not ess.energy_min_mwh - tol <= start <= ess.energy_max_mwh + tol:
        raise ValueError("시작 잔량이 SOC 범위를 벗어났습니다.")
    e = ess.clip_energy(start)
    rows: list[dict[str, Any]] = []
    for i in range(len(plan)):
        step = HourInput(
            committed_sell_mwh=float(plan.at[i, "sell_mwh"]),
            planned_charge_mwh=float(plan.at[i, "charge_mwh"]),
            planned_discharge_mwh=float(plan.at[i, "discharge_mwh"]),
            actual_generation_mwh=float(actual[i]),
            energy_start_mwh=e,
            dt_hours=float(dt[i]),
            plan_sell_value_krw_per_mwh=float(value[i]),
        )
        out = policy.execute_hour(step, ess)
        rows.append({
            "ts": plan.at[i, "ts"], "dt_hours": float(dt[i]),
            "committed_sell_mwh": step.committed_sell_mwh,
            "planned_charge_mwh": step.planned_charge_mwh,
            "planned_discharge_mwh": step.planned_discharge_mwh,
            "planned_energy_end_mwh": (float(plan.at[i, "energy_end_mwh"])
                                       if "energy_end_mwh" in plan else np.nan),
            "forecast_generation_mwh": (float(plan.at[i, "generation_mwh"])
                                        if "generation_mwh" in plan else np.nan),
            "actual_generation_mwh": step.actual_generation_mwh,
            "plan_price_krw_per_mwh": float(plan.at[i, "price_krw_per_mwh"]),
            REC_COLUMN: float(rec_adder(plan)[i]),
            "charge_mwh": out.charge_mwh, "discharge_mwh": out.discharge_mwh,
            "supply_mwh": out.supply_mwh, "curtailment_mwh": out.curtailment_mwh,
            "shortfall_mwh": out.shortfall_mwh, "surplus_mwh": out.surplus_mwh,
            "energy_start_mwh": e, "energy_end_mwh": out.energy_end_mwh,
            "soc_end": out.energy_end_mwh / ess.capacity_mwh,
            "actions": "|".join(out.actions),
        })
        e = out.energy_end_mwh
    hourly = pd.DataFrame(rows)
    checks = check_execution(hourly, ess, tol=tol)
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise RuntimeError(f"실행 운전 검산 실패: {failed}")
    return ExecutionResult(hourly, ess.clip_energy(start), e, policy.metadata(), checks)


def _item(name: str, value: float, passed: bool) -> dict[str, Any]:
    return {"item": name, "passed": bool(passed), "value": float(value)}


def check_execution(hourly: pd.DataFrame, ess: EssParams, *, tol: float = 1e-6) -> list[dict]:
    """실행 결과의 물리·약속 검산. 위반을 예외로 던지지 않고 목록으로 돌려준다."""
    g = hourly["actual_generation_mwh"].to_numpy(dtype=float)
    q = hourly["committed_sell_mwh"].to_numpy(dtype=float)
    c = hourly["charge_mwh"].to_numpy(dtype=float)
    d = hourly["discharge_mwh"].to_numpy(dtype=float)
    s = hourly["supply_mwh"].to_numpy(dtype=float)
    r = hourly["curtailment_mwh"].to_numpy(dtype=float)
    short = hourly["shortfall_mwh"].to_numpy(dtype=float)
    plus = hourly["surplus_mwh"].to_numpy(dtype=float)
    e0 = hourly["energy_start_mwh"].to_numpy(dtype=float)
    e1 = hourly["energy_end_mwh"].to_numpy(dtype=float)
    dt = hourly["dt_hours"].to_numpy(dtype=float)
    cap_c, cap_d = ess.charge_power_mw * dt, ess.discharge_power_mw * dt
    checks = []
    v = float(np.max(np.abs(g + d - s - c - r)))
    checks.append(_item("power_balance_max_abs_error_mwh", v, v < tol))
    v = float(np.max(np.abs(e1 - (e0 + ess.charge_efficiency * c - d / ess.discharge_efficiency))))
    checks.append(_item("energy_balance_max_abs_error_mwh", v, v < tol))
    v = float(np.max(np.abs(e0[1:] - e1[:-1]))) if len(e0) > 1 else 0.0
    checks.append(_item("hourly_energy_continuity_max_abs_error_mwh", v, v < tol))
    v = float(max(ess.energy_min_mwh - e1.min(), e1.max() - ess.energy_max_mwh))
    checks.append(_item("soc_bounds_max_violation_mwh", v, v <= tol))
    v = float(max(-c.min(), (c - np.minimum(g, cap_c)).max()))
    checks.append(_item("charge_bounds_max_violation_mwh", v, v <= tol))
    v = float(max(-d.min(), (d - cap_d).max()))
    checks.append(_item("discharge_bounds_max_violation_mwh", v, v <= tol))
    v = float(np.minimum(c, d).max())
    checks.append(_item("simultaneous_charge_discharge_max_mwh", v, v <= tol))
    v = float(max(-short.min(), -plus.min(), -s.min(), -r.min()))
    checks.append(_item("nonnegative_flows_max_violation_mwh", v, v <= tol))
    v = float(np.max(np.abs(s - (q - short + plus))))
    checks.append(_item("supply_equals_committed_minus_shortfall_plus_surplus_mwh", v, v < tol))
    v = float(np.minimum(short, plus).max())
    checks.append(_item("shortfall_and_surplus_not_both_mwh", v, v <= tol))
    return checks


def execution_totals(hourly: pd.DataFrame, ess: EssParams) -> dict[str, float]:
    return {
        "committed_sell_mwh": float(hourly["committed_sell_mwh"].sum()),
        "supply_mwh": float(hourly["supply_mwh"].sum()),
        "shortfall_mwh": float(hourly["shortfall_mwh"].sum()),
        "surplus_mwh": float(hourly["surplus_mwh"].sum()),
        "shortfall_hours": int((hourly["shortfall_mwh"] > 1e-9).sum()),
        "actual_generation_mwh": float(hourly["actual_generation_mwh"].sum()),
        "charge_mwh": float(hourly["charge_mwh"].sum()),
        "discharge_mwh": float(hourly["discharge_mwh"].sum()),
        "planned_charge_mwh": float(hourly["planned_charge_mwh"].sum()),
        "planned_discharge_mwh": float(hourly["planned_discharge_mwh"].sum()),
        "curtailment_mwh": float(hourly["curtailment_mwh"].sum()),
        "wear_krw": ess.degradation_cost_krw(hourly["charge_mwh"], hourly["discharge_mwh"]),
    }
