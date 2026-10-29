"""담당: 최적화·대시보드팀 — MILP 계획과 비교할 기준 운영 방식.

입력은 run_rolling과 같은 시간별 DataFrame(ts, generation_mwh, price_krw_per_mwh[, dt_hours]).
출력은 run_rolling.hourly와 같은 계획 컬럼(sell/charge/discharge/curtailment/energy_end/soc_end).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vpp.optimization import EssParams

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


def plan_revenue_krw(plan: pd.DataFrame, ess: EssParams | None = None) -> float:
    """판매수익 − 열화비 (ess가 None이면 열화비 0)."""
    sales = float(plan["price_krw_per_mwh"] @ plan["sell_mwh"])
    if ess is None:
        return sales
    throughput = float(plan["charge_mwh"].sum() + plan["discharge_mwh"].sum())
    return sales - ess.throughput_cost_krw_per_mwh * throughput


def no_ess(frame: pd.DataFrame) -> pd.DataFrame:
    """ESS 없음: 가격 ≥ 0이면 전량 판매, 음수면 전량 감축 (Codex 기준과 동일)."""
    out = _base(frame)
    positive = out["price_krw_per_mwh"] >= 0
    out["sell_mwh"] = np.where(positive, out["generation_mwh"], 0.0)
    out["curtailment_mwh"] = np.where(positive, 0.0, out["generation_mwh"])
    out["charge_mwh"] = 0.0
    out["discharge_mwh"] = 0.0
    out["energy_end_mwh"] = np.nan
    out["soc_end"] = np.nan
    return out


def rule_based(frame: pd.DataFrame, ess: EssParams, n_hours: int = 4) -> pd.DataFrame:
    """규칙 계획: 매일 가장 싼 n시간에 충전(발전 범위 내), 가장 비싼 n시간에 방전.

    같은 효율·SOC 범위·출력 한도·계통 충전 금지를 적용하고 날 사이 에너지를 이월한다.
    가격이 음수인 시간은 판매하지 않고 남는 발전을 감축한다. 방전은 가격 > 0일 때만.
    """
    out = _base(frame)
    ts = pd.to_datetime(out["ts"]) if not isinstance(out["ts"].iloc[0], int | np.integer) else None
    day_key = ts.dt.date if ts is not None else out.index // 24
    n = len(out)
    sell = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    curtail = np.zeros(n)
    energy = np.zeros(n)
    e = ess.initial_energy_mwh
    for _, idx in out.groupby(day_key, sort=False).indices.items():
        idx = np.sort(idx)
        price = out.loc[idx, "price_krw_per_mwh"].to_numpy()
        order = np.argsort(price, kind="stable")
        charge_slots = set(idx[order[:n_hours]])
        discharge_slots = set(idx[order[::-1][:n_hours]]) - charge_slots
        for i in idx:
            g = out.at[i, "generation_mwh"]
            p = out.at[i, "price_krw_per_mwh"]
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
