"""별표 33 (2026-04 시행) 임밸런스 페널티의 1시간 근사. 반환 금액은 양의 비용."""

from __future__ import annotations

import math

import numpy as np

from vpp.settlement_inputs import RULE_VERSION


def hourly_imbalance(
    *, supply_mwh: float, resource_capacity_mw: float = np.nan,
    set_point_mw: float = np.nan, available_capacity_mw: float = np.nan,
    realtime_price_krw_per_mwh: float = np.nan, fcr_price_krw_per_mwh: float = np.nan,
    min_offer_price_krw_per_mwh: float = np.nan, offer_floor_krw_per_mwh: float = np.nan,
) -> dict:
    """단일 자원·1시간. 공급량은 MWh, 출력은 MW, 가격은 원/MWh.

    부족분의 면제 또는 허용범위 이내임이 확인되면 단가 입력 없이도 비용 0을 확정한다.
    그 외 필수 입력 결측은 NaN이며, 누락 단가를 0으로 대체하지 않는다.
    """
    values = dict(locals())
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float | np.number) \
                or math.isinf(value):
            raise ValueError(f"{name}: 유한한 숫자 또는 결측(NaN)이 필요합니다.")
        if name in {"supply_mwh", "resource_capacity_mw", "set_point_mw",
                    "available_capacity_mw", "fcr_price_krw_per_mwh"} and value < 0:
            raise ValueError(f"{name}: 음수는 허용하지 않습니다.")
    if resource_capacity_mw == 0:
        raise ValueError("resource_capacity_mw: 양의 발전설비 용량이 필요합니다.")
    out = {
        "imbalance_rule_version": RULE_VERSION,
        "imbalance_branch": "unknown", "imbalance_direction": "unknown",
        "tolerance_ratio": np.nan, "allowed_deviation_mwh": np.nan,
        "dispatch_deviation_mwh": np.nan, "penalized_mwh": np.nan,
        "penalty_price_krw_per_mwh": np.nan, "imbalance_penalty_krw": np.nan,
        "imbalance_status": "missing_inputs", "imbalance_missing_inputs": "",
    }

    def missing(*names):
        absent = [name for name in names if math.isnan(values[name])]
        out["imbalance_missing_inputs"] = ",".join(absent)
        return bool(absent)

    if missing("supply_mwh", "resource_capacity_mw", "set_point_mw", "realtime_price_krw_per_mwh"):
        return out
    if realtime_price_krw_per_mwh > 0 and missing("available_capacity_mw"):
        return out
    if realtime_price_krw_per_mwh <= 0:
        branch = "nonpositive_rt"
    elif set_point_mw < available_capacity_mw:
        branch = "dispatch_reduction"
    else:
        branch = "normal"
    tolerance = 0.06 if branch == "normal" else 0.015
    deviation = supply_mwh - set_point_mw  # set_point MW × 1 h
    allowed = resource_capacity_mw * tolerance  # ICDM MW × tolerance × 1 h
    excess = max(abs(deviation) - allowed, 0.0)
    # 경계의 부동소수점 오차에 따른 미소 과금을 방지한다 (1e-9 MWh = 0.001 Wh).
    if excess <= 1e-9:
        excess = 0.0
    out.update(
        imbalance_branch=branch,
        imbalance_direction=(
            "surplus" if deviation > 0 else "shortfall" if deviation < 0 else "none"
        ),
        tolerance_ratio=tolerance, allowed_deviation_mwh=allowed,
        dispatch_deviation_mwh=deviation, penalized_mwh=excess,
    )
    if excess == 0 or (deviation < 0 and branch != "normal"):
        price = 0.0
    else:
        required = ["fcr_price_krw_per_mwh"]
        if branch == "nonpositive_rt":
            required.append("offer_floor_krw_per_mwh")
        elif branch == "dispatch_reduction":
            required.append("min_offer_price_krw_per_mwh")
        if missing(*required):
            return out
        price = fcr_price_krw_per_mwh
        if branch == "nonpositive_rt":
            price += max(-offer_floor_krw_per_mwh, 0) * 2.5
        elif branch == "dispatch_reduction":
            price += max(realtime_price_krw_per_mwh - min_offer_price_krw_per_mwh, 0)
    out.update(
        penalty_price_krw_per_mwh=price, imbalance_penalty_krw=excess * price,
        imbalance_status="computed", imbalance_missing_inputs="",
    )
    return out
