"""담당: 최적화·대시보드팀 — MILP 계획과 비교할 기준 운영 방식.

입력은 run_rolling과 같은 시간별 DataFrame(ts, generation_mwh, price_krw_per_mwh[, dt_hours]
[, rec_krw_per_mwh]).
출력은 run_rolling.hourly와 같은 계획 컬럼(sell/charge/discharge/curtailment/energy_end/soc_end).

판매/감축 판단은 MILP 와 같은 '판매 가치' = SMP + REC 가산(vpp.rec, 열이 없으면 0)으로 한다.
REC 를 MILP 에만 더하면 ESS 추가수익이 부풀려지므로 기준 방식에도 같은 가산을 적용한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vpp.optimization import EssParams
from vpp.rec import effective_price_krw_per_mwh, rec_revenue_krw

PLAN_COLUMNS = [
    "sell_mwh", "charge_mwh", "discharge_mwh", "curtailment_mwh", "energy_end_mwh", "soc_end",
]


def _base(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"generation_mwh", "price_krw_per_mwh"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"입력 DataFrame에 컬럼이 없습니다: {missing}")
    out = frame.copy().reset_index(drop=True)
    if "dt_hours" not in out.columns:
        out["dt_hours"] = 1.0
    if "ts" not in out.columns:
        out.insert(0, "ts", pd.RangeIndex(len(out)))
    return out


def plan_revenue_breakdown(plan: pd.DataFrame, ess: EssParams | None = None) -> dict[str, Any]:
    """판매수익(SMP)·REC 수익·열화비·순수익. 열화비는 ESS 의 열화모델 비용 함수로만 계산."""
    sales = float(plan["price_krw_per_mwh"] @ plan["sell_mwh"])
    rec = rec_revenue_krw(plan)
    wear = 0.0 if ess is None else ess.degradation_cost_krw(plan["charge_mwh"],
                                                             plan["discharge_mwh"])
    return {
        "sales_revenue_krw": sales,
        "rec_revenue_krw": rec,
        "throughput_cost_krw": wear,
        "net_revenue_krw": sales + rec - wear,
    }


def plan_revenue_krw(plan: pd.DataFrame, ess: EssParams | None = None) -> float:
    """판매수익 + REC 수익 − 열화비 (ess가 None이면 열화비 0)."""
    return plan_revenue_breakdown(plan, ess)["net_revenue_krw"]


def no_ess(frame: pd.DataFrame) -> pd.DataFrame:
    """ESS 없음: 판매 가치(SMP+REC) ≥ 0이면 전량 판매, 음수면 전량 감축 (Codex 기준과 동일)."""
    out = _base(frame)
    positive = effective_price_krw_per_mwh(out) >= 0
    out["sell_mwh"] = np.where(positive, out["generation_mwh"], 0.0)
    out["curtailment_mwh"] = np.where(positive, 0.0, out["generation_mwh"])
    out["charge_mwh"] = 0.0
    out["discharge_mwh"] = 0.0
    out["energy_end_mwh"] = np.nan
    out["soc_end"] = np.nan
    return out


def rule_based(frame: pd.DataFrame, ess: EssParams, n_hours: int = 4) -> pd.DataFrame:
    """규칙 계획: 매일 판매 가치가 가장 낮은 n시간에 충전(발전 범위 내), 가장 높은 n시간에 방전.

    같은 효율·SOC 범위·출력 한도·계통 충전 금지를 적용하고 날 사이 에너지를 이월한다.
    판매 가치(SMP+REC)가 음수인 시간은 판매하지 않고 남는 발전을 감축한다. 방전은 가치 > 0일 때만.
    """
    out = _base(frame)
    ts = pd.to_datetime(out["ts"]) if not isinstance(out["ts"].iloc[0], int | np.integer) else None
    day_key = ts.dt.date if ts is not None else out.index // 24
    value = effective_price_krw_per_mwh(out)
    n = len(out)
    sell = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    curtail = np.zeros(n)
    energy = np.zeros(n)
    e = ess.initial_energy_mwh
    for _, idx in out.groupby(day_key, sort=False).indices.items():
        idx = np.sort(idx)
        order = np.argsort(value[idx], kind="stable")
        charge_slots = set(idx[order[:n_hours]])
        discharge_slots = set(idx[order[::-1][:n_hours]]) - charge_slots
        for i in idx:
            g = out.at[i, "generation_mwh"]
            p = value[i]
            dt = out.at[i, "dt_hours"]
            c = d = 0.0
            if i in charge_slots:
                room = (ess.energy_max_mwh - e) / ess.charge_efficiency
                c = max(0.0, min(g, ess.charge_power_mw * dt, room))
            elif i in discharge_slots and p > 0:
                available = (e - ess.energy_min_mwh) * ess.discharge_efficiency
                d = max(0.0, min(ess.discharge_power_mw * dt, available))
            e = e + ess.charge_efficiency * c - d / ess.discharge_efficiency
            e = ess.clip_energy(e)
            remaining = g - c + d
            if p >= 0:
                sell[i] = remaining
            else:
                curtail[i] = remaining
            charge[i], discharge[i], energy[i] = c, d, e
    out["sell_mwh"] = sell
    out["charge_mwh"] = charge
    out["discharge_mwh"] = discharge
    out["curtailment_mwh"] = curtail
    out["energy_end_mwh"] = energy
    out["soc_end"] = energy / ess.capacity_mwh
    return out
