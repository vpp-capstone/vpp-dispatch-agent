"""담당: 최적화·대시보드팀 — ESS 운영계획 v0 (MILP, scipy.optimize.milp).

단위: 에너지 MWh, 출력 MW, 가격 원/MWh, 구간 길이 h. 에너지 변수에 dt를 다시 곱하지 않는다.
수식·가정은 docs/formulation.md 참조. 가격에 실측 하루전 SMP를 넣는 것은 완전예측 백테스트이며
실제 입찰 성과가 아니다. 가정값(config/ess_v0.yaml)은 설비 사양이 아니다.

`schemas.py`는 전원 합의 전까지 수정하지 않으므로 필요한 타입은 이 파일의 dataclass로 둔다.
제약 검산은 vpp.validation.check_plan 에 있다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp

from vpp.schemas import OptimizationRequest, OptimizationResult
from vpp.validation import check_plan

TERMINAL_POLICIES = ("rolling48", "free", "equal_initial")
_MIP_OPTIONS = {"mip_rel_gap": 1e-8, "time_limit": 60}


@dataclass(frozen=True)
class EssParams:
    """ESS 사양·운영 범위. 값은 생성 시 검증한다"""

    capacity_mwh: float
    charge_power_mw: float
    discharge_power_mw: float
    soc_min: float
    soc_max: float
    soc_initial: float
    charge_efficiency: float
    discharge_efficiency: float
    throughput_cost_krw_per_mwh: float = 0.0
    grid_charging_allowed: bool = False
    status: str = "assumption_not_equipment_spec"

    def __post_init__(self) -> None:
        numeric = (
            "capacity_mwh", "charge_power_mw", "discharge_power_mw", "soc_min", "soc_max",
            "soc_initial", "charge_efficiency", "discharge_efficiency",
            "throughput_cost_krw_per_mwh",
        )
        for name in numeric:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"ESS 수치 {name}는 숫자여야 합니다: {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"ESS 수치 {name}는 유한해야 합니다: {value!r}")
            object.__setattr__(self, name, float(value))
        ok = (
            self.capacity_mwh > 0
            and self.charge_power_mw >= 0
            and self.discharge_power_mw >= 0
            and 0 <= self.soc_min <= self.soc_max <= 1
            and self.soc_min <= self.soc_initial <= self.soc_max
            and 0 < self.charge_efficiency <= 1
            and 0 < self.discharge_efficiency <= 1
            and self.throughput_cost_krw_per_mwh >= 0
        )
        if not ok:
            raise ValueError("ESS 용량/출력/SOC/효율/비용 범위를 확인하세요.")
        if self.grid_charging_allowed:
            raise ValueError("v0는 계통 충전 금지만 지원합니다 (grid_charging_allowed=false).")

    @property
    def energy_min_mwh(self) -> float:
        return self.soc_min * self.capacity_mwh

    @property
    def energy_max_mwh(self) -> float:
        return self.soc_max * self.capacity_mwh

    @property
    def usable_capacity_mwh(self) -> float:
        return self.energy_max_mwh - self.energy_min_mwh

    @property
    def initial_energy_mwh(self) -> float:
        return self.soc_initial * self.capacity_mwh

    def clip_energy(self, energy_mwh: float) -> float:
        """부동소수점 오차가 다음 창의 시작 에너지 검사를 깨지 않도록 SOC 범위로 보정."""
        return min(self.energy_max_mwh, max(self.energy_min_mwh, float(energy_mwh)))

    @classmethod
    def from_yaml(cls, path: str | Path) -> EssParams:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"ESS 설정은 매핑이어야 합니다: {path}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"알 수 없는 ESS 설정 키: {unknown}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowResult:
    """한 최적화 창(horizon)의 전 구간 결과. 길이 n 배열은 창의 각 구간에 대응."""

    sell_mwh: np.ndarray
    charge_mwh: np.ndarray
    discharge_mwh: np.ndarray
    curtailment_mwh: np.ndarray
    energy_end_mwh: np.ndarray
    soc_end: np.ndarray
    initial_energy_mwh: float
    sales_revenue_krw: float
    throughput_cost_krw: float
    objective_krw: float
    no_ess_same_curtailment_policy_krw: float
    solver_status: str
    mip_gap: float
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.sell_mwh)

    def to_frame(self, generation_mwh, price_krw_per_mwh, dt_hours) -> pd.DataFrame:
        """입력과 결과를 validation.check_plan이 받는 시간별 DataFrame으로 합친다."""
        return pd.DataFrame({
            "generation_mwh": np.asarray(generation_mwh, dtype=float),
            "price_krw_per_mwh": np.asarray(price_krw_per_mwh, dtype=float),
            "dt_hours": np.asarray(dt_hours, dtype=float),
            "sell_mwh": self.sell_mwh, "charge_mwh": self.charge_mwh,
            "discharge_mwh": self.discharge_mwh, "curtailment_mwh": self.curtailment_mwh,
            "energy_end_mwh": self.energy_end_mwh, "soc_end": self.soc_end,
        })


@dataclass(frozen=True)
class RollingResult:
    """롤링 백테스트 결과. hourly는 확정된 구간만, daily는 창(window)별 요약."""

    hourly: pd.DataFrame
    daily: pd.DataFrame
    policy: str
    horizon_hours: float
    commit_hours: float
    period_initial_energy_mwh: float
    period_final_energy_mwh: float
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sales_revenue_krw(self) -> float:
        return float(self.daily["sales_revenue_krw"].sum())

    @property
    def throughput_cost_krw(self) -> float:
        return float(self.daily["throughput_cost_krw"].sum())

    @property
    def objective_krw(self) -> float:
        return float(self.daily["objective_krw"].sum())

    @property
    def no_ess_same_curtailment_policy_krw(self) -> float:
        return float(self.daily["no_ess_same_curtailment_policy_krw"].sum())


def _as_window_array(value: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name}는 길이 {n}의 1차원 배열이거나 스칼라여야 합니다.")
    return arr


def solve_window(
    generation_mwh,
    price_krw_per_mwh,
    dt_hours,
    ess: EssParams,
    initial_energy_mwh: float,
    final_energy_min_mwh: float | None = None,
    export_cap_mwh=None,
    *,
    final_energy_max_mwh: float | None = None,
    tol: float = 1e-6,
) -> WindowResult:
    """한 창을 MILP로 푼다 (docs/formulation.md §2).

    final_energy_min_mwh: 창 마지막 구간 끝 에너지 하한 (기간 끝 잔량 제약 등).
    final_energy_max_mwh: 상한. min과 같게 주면 equal_initial 등식이 된다.
    export_cap_mwh: 구간별 판매 상한(MWh). None이면 제약 없음.
    최적해가 아니거나 결과 검산에 실패하면 예외.
    """
    g, price, dt = (
        np.asarray(x, dtype=float) for x in (generation_mwh, price_krw_per_mwh, dt_hours)
    )
    if g.ndim != 1 or not len(g) or price.shape != g.shape or dt.shape != g.shape:
        raise ValueError("입력은 길이가 같은 비어 있지 않은 1차원 배열이어야 합니다.")
    if not all(np.isfinite(x).all() for x in (g, price, dt)):
        raise ValueError("결측/비유한 입력은 사용할 수 없습니다.")
    if (g < 0).any() or (dt <= 0).any():
        raise ValueError("발전량은 0 이상, 시간 간격은 양수여야 합니다.")
    n = len(g)
    start = float(initial_energy_mwh)
    in_range = ess.energy_min_mwh - tol <= start <= ess.energy_max_mwh + tol
    if not math.isfinite(start) or not in_range:
        raise ValueError("시작 에너지가 SOC 범위를 벗어났습니다.")
    start = ess.clip_energy(start)
    export_cap = None
    if export_cap_mwh is not None:
        export_cap = _as_window_array(export_cap_mwh, n, "export_cap_mwh")
    if export_cap is not None and ((export_cap < 0).any() or not np.isfinite(export_cap).all()):
        raise ValueError("판매 상한은 0 이상 유한해야 합니다.")
    e_lo, e_hi = ess.energy_min_mwh, ess.energy_max_mwh
    final_lo = e_lo if final_energy_min_mwh is None else max(e_lo, float(final_energy_min_mwh))
    final_hi = e_hi if final_energy_max_mwh is None else min(e_hi, float(final_energy_max_mwh))
    if final_lo > final_hi + tol:
        raise ValueError("종료 에너지 하한이 상한보다 큽니다.")

    cap_c, cap_d = ess.charge_power_mw * dt, ess.discharge_power_mw * dt
    ec, ed, wear = ess.charge_efficiency, ess.discharge_efficiency, ess.throughput_cost_krw_per_mwh

    # 변수 블록: 판매 q, 충전 c, 방전 d, 감축 r, 구간 끝 에너지 e, 충전 모드 z∈{0,1}.
    q, c, d, r, e, z = (np.arange(n) + k * n for k in range(6))
    lower = np.zeros(6 * n)
    upper = np.full(6 * n, np.inf)
    lower[e], upper[e] = e_lo, e_hi
    lower[e[-1]], upper[e[-1]] = final_lo, final_hi
    upper[c] = np.minimum(g, cap_c)  # 계통 충전 금지: 발전량 이내로만 충전
    upper[d], upper[r], upper[z] = cap_d, g, 1
    if export_cap is not None:
        upper[q] = export_cap
    objective = np.zeros(6 * n)
    objective[q], objective[c], objective[d] = -price, wear, wear  # milp는 최소화
    integrality = np.zeros(6 * n, dtype=int)
    integrality[z] = 1

    rows = np.zeros((4 * n, 6 * n))
    lbs, ubs = np.zeros(4 * n), np.zeros(4 * n)
    for t in range(n):
        k = 4 * t
        # 전력균형 g + d = q + c + r
        rows[k, [q[t], c[t], r[t], d[t]]] = 1, 1, 1, -1
        lbs[k] = ubs[k] = g[t]
        # SOC 전이 e[t] - e[t-1] - ec*c + d/ed = 0 (t=0은 e[-1] := start)
        rows[k + 1, [e[t], c[t], d[t]]] = 1, -ec, 1 / ed
        if t:
            rows[k + 1, e[t - 1]] = -1
        lbs[k + 1] = ubs[k + 1] = start if t == 0 else 0.0
        # 동시 충방전 금지: c ≤ P_ch·dt·z, d ≤ P_dis·dt·(1-z)
        rows[k + 2, [c[t], z[t]]] = 1, -cap_c[t]
        lbs[k + 2], ubs[k + 2] = -np.inf, 0.0
        rows[k + 3, [d[t], z[t]]] = 1, cap_d[t]
        lbs[k + 3], ubs[k + 3] = -np.inf, cap_d[t]

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(rows, lbs, ubs),
        options=_MIP_OPTIONS,
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"최적해 확인 실패 (status={result.status}): {result.message}")
    x = result.x
    sell, charge, discharge, curtailed, energy = (x[idx] for idx in (q, c, d, r, e))
    sales = float(price @ sell)
    wear_cost = float(wear * (charge.sum() + discharge.sum()))
    window = WindowResult(
        sell_mwh=sell,
        charge_mwh=charge,
        discharge_mwh=discharge,
        curtailment_mwh=curtailed,
        energy_end_mwh=energy,
        soc_end=energy / ess.capacity_mwh,
        initial_energy_mwh=start,
        sales_revenue_krw=sales,
        throughput_cost_krw=wear_cost,
        objective_krw=float(-result.fun),
        no_ess_same_curtailment_policy_krw=float(np.maximum(price, 0) @ g),
        solver_status="optimal",
        mip_gap=float(result.mip_gap),
    )
    plan = window.to_frame(g, price, dt)
    checks = check_plan(
        plan, ess, tol,
        initial_energy_mwh=start, objective_krw=window.objective_krw,
        solver_status=window.solver_status, export_cap_mwh=export_cap,
        final_energy_min_mwh=final_lo, final_energy_max_mwh=final_hi,
    )
    failed = [chk for chk in checks if not chk["passed"]]
    if failed:
        raise RuntimeError(f"계산 결과 제약 위반: {failed}")
    return WindowResult(**{**asdict(window), "checks": checks})


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"generation_mwh", "price_krw_per_mwh"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"입력 DataFrame에 컬럼이 없습니다: {missing}")
    if frame.empty:
        raise ValueError("입력 DataFrame이 비어 있습니다.")
    out = frame.copy()
    if "dt_hours" not in out.columns:
        out["dt_hours"] = 1.0
    if "ts" in out.columns:
        ts = pd.to_datetime(out["ts"])
        if not ts.is_monotonic_increasing or ts.duplicated().any():
            raise ValueError("ts는 중복 없이 오름차순이어야 합니다.")
        out["ts"] = ts
    else:
        out.insert(0, "ts", pd.RangeIndex(len(out)))
    out = out.reset_index(drop=True)
    for col in ("generation_mwh", "price_krw_per_mwh", "dt_hours"):
        out[col] = out[col].astype(float)
        if not np.isfinite(out[col]).all():
            raise ValueError(f"{col}에 결측/비유한 값이 있습니다 (보간 안 함, 그 날은 제외).")
    return out


def run_rolling(
    frame: pd.DataFrame,
    ess: EssParams,
    horizon_hours: float = 48,
    commit_hours: float = 24,
    policy: str = "rolling48",
    *,
    export_cap_col: str | None = None,
    tol: float = 1e-6,
) -> RollingResult:
    """시간별 DataFrame(ts, generation_mwh, price_krw_per_mwh[, dt_hours])을 롤링으로 푼다.

    - rolling48: 매 commit_hours마다 horizon_hours 창을 풀고 앞 commit_hours만 확정.
      확정 구간 끝 에너지를 다음 창 시작 에너지로 넘긴다. 데이터 끝에서는 남은 시간만 푼다.
      마지막 창에는 E[마지막] ≥ 기간 시작 에너지 제약을 건다 (초기 저장분 판매 수익 분리).
    - free: 창 = commit_hours, 종료 제약 없음 .
    - equal_initial: 창 = commit_hours, 창 끝 에너지 = 창 시작 에너지 (equal_initial).
    hourly에는 확정 구간만 들어가며, daily의 수익도 확정 구간만으로 계산한다.
    """
    if policy not in TERMINAL_POLICIES:
        raise ValueError(f"policy는 {TERMINAL_POLICIES} 중 하나여야 합니다: {policy!r}")
    if commit_hours <= 0 or horizon_hours < commit_hours:
        raise ValueError("commit_hours > 0, horizon_hours ≥ commit_hours 이어야 합니다.")
    if policy != "rolling48":
        horizon_hours = commit_hours
    data = _prepare_frame(frame)
    if export_cap_col is not None and export_cap_col not in data.columns:
        raise ValueError(f"판매 상한 컬럼이 없습니다: {export_cap_col}")
    dt = data["dt_hours"].to_numpy()
    elapsed = np.cumsum(dt) - dt  # 각 구간의 시작 시각(기간 시작 기준, h)
    total_hours = float(dt.sum())

    period_start_energy = ess.initial_energy_mwh
    start = period_start_energy
    hourly_parts: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    offset = 0.0
    window_id = 0
    eps = 1e-9
    while offset < total_hours - eps:
        in_window = (elapsed >= offset - eps) & (elapsed < offset + horizon_hours - eps)
        in_commit = (elapsed >= offset - eps) & (elapsed < offset + commit_hours - eps)
        idx = np.flatnonzero(in_window)
        n_commit = int(in_commit.sum())
        is_last = offset + commit_hours >= total_hours - eps
        block = data.iloc[idx]
        final_lo = final_hi = None
        if policy == "rolling48" and is_last:
            final_lo = period_start_energy
        elif policy == "equal_initial":
            final_lo = final_hi = start
        export_cap = None if export_cap_col is None else block[export_cap_col].to_numpy()
        window = solve_window(
            block["generation_mwh"].to_numpy(),
            block["price_krw_per_mwh"].to_numpy(),
            block["dt_hours"].to_numpy(),
            ess,
            start,
            final_energy_min_mwh=final_lo,
            export_cap_mwh=export_cap,
            final_energy_max_mwh=final_hi,
            tol=tol,
        )
        sl = slice(0, n_commit)
        committed = block.iloc[sl].copy()
        committed["sell_mwh"] = window.sell_mwh[sl]
        committed["charge_mwh"] = window.charge_mwh[sl]
        committed["discharge_mwh"] = window.discharge_mwh[sl]
        committed["curtailment_mwh"] = window.curtailment_mwh[sl]
        committed["energy_end_mwh"] = window.energy_end_mwh[sl]
        committed["soc_end"] = window.soc_end[sl]
        committed["window_id"] = window_id
        hourly_parts.append(committed)

        price_c = committed["price_krw_per_mwh"].to_numpy()
        gen_c = committed["generation_mwh"].to_numpy()
        sales = float(price_c @ window.sell_mwh[sl])
        throughput = window.charge_mwh[sl].sum() + window.discharge_mwh[sl].sum()
        wear_cost = float(ess.throughput_cost_krw_per_mwh * throughput)
        final_energy = ess.clip_energy(window.energy_end_mwh[n_commit - 1])
        daily_rows.append(
            {
                "window_id": window_id,
                "start_ts": committed["ts"].iloc[0],
                "end_ts": committed["ts"].iloc[-1],
                "solved_intervals": int(len(idx)),
                "committed_intervals": n_commit,
                "committed_hours": float(committed["dt_hours"].sum()),
                "initial_energy_mwh": start,
                "final_energy_mwh": final_energy,
                "sales_revenue_krw": sales,
                "throughput_cost_krw": wear_cost,
                "objective_krw": sales - wear_cost,
                "no_ess_same_curtailment_policy_krw": float(np.maximum(price_c, 0) @ gen_c),
                "charge_mwh": float(window.charge_mwh[sl].sum()),
                "discharge_mwh": float(window.discharge_mwh[sl].sum()),
                "curtailment_mwh": float(window.curtailment_mwh[sl].sum()),
                "mip_gap": window.mip_gap,
                "terminal_constraint": (
                    "period_end_ge_initial" if policy == "rolling48" and is_last
                    else "equal_window_initial" if policy == "equal_initial"
                    else "none"
                ),
            }
        )
        start = final_energy
        offset += commit_hours
        window_id += 1

    hourly = pd.concat(hourly_parts, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    initial, final = daily["initial_energy_mwh"].to_numpy(), daily["final_energy_mwh"].to_numpy()
    continuity = float(np.max(np.abs(initial[1:] - final[:-1]))) if len(daily) > 1 else 0.0
    checks = [
        {"item": "rolling_energy_continuity_max_abs_error_mwh", "passed": continuity <= tol,
         "value": continuity},
        {"item": "hourly_rows_equal_input_rows", "passed": len(hourly) == len(data),
         "value": float(len(hourly))},
    ]
    if policy == "rolling48":
        gap = period_start_energy - start
        checks.append({"item": "period_final_energy_ge_initial_shortfall_mwh",
                       "passed": gap <= tol, "value": float(gap)})
    failed = [chk for chk in checks if not chk["passed"]]
    if failed:
        raise RuntimeError(f"롤링 결과 검산 실패: {failed}")
    return RollingResult(
        hourly=hourly,
        daily=daily,
        policy=policy,
        horizon_hours=float(horizon_hours),
        commit_hours=float(commit_hours),
        period_initial_energy_mwh=period_start_energy,
        period_final_energy_mwh=start,
        checks=checks,
    )


def plan_ess(request: OptimizationRequest) -> OptimizationResult:
    """TODO: 입력·ESS 사양·제약·비교 기준을 합의하고 운영계획과 솔버 상태를 반환한다.

    출력 필드(시간별 충·방전, SOC 등)도 담당팀이 정의한다. 현재 계산/결과 생성 없음.
    스키마 합의 후 `run_rolling`(발전량·가격·ESS 사양 입력)을 연결할 예정.
    """
    raise NotImplementedError("미구현: 스키마 합의 후 run_rolling을 연결합니다.")
