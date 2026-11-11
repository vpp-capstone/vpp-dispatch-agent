"""ESS 중앙계약시장식 계약가격 정산 시나리오.

이 모듈은 재생에너지의 SMP·REC·용량정산금과 분리된 ESS 계약가격 정산을
계산한다. 2025년 제2차 제주 ESS 중앙계약시장 공고의 공개된 구조를
반영하되, 실제 낙찰자료와 시간별 이행자료가 없을 때 사용할 수 있는
시나리오 계산기이다.

공고에 따른 핵심 구조는 다음과 같다.

    계약가격 = 입찰가격 × 60%
    공급가능용량 = 최대전력저장량 ÷ 6시간
    계약가격 정산 = 계약가격 × 공급가능용량 × 이행률

공고의 일간 식을 시간별 자료로 계산하기 위해 각 시간의 정산액을
``계약가격 × 공급가능용량(kW) × 시간 × 이행률``로 표현한다. 실제 계약의
월별 정산, 계약가격 조정, 지연·고장·정비 페널티는 별도 자료가 필요하다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


def supply_capacity_mw_from_storage(
    storage_energy_mwh: float,
    *,
    supply_duration_hours: float = 6.0,
) -> float:
    """중앙계약시장식 공급가능용량(MW)을 저장량에서 환산한다."""
    if (
        isinstance(storage_energy_mwh, bool)
        or not isinstance(storage_energy_mwh, int | float)
        or not math.isfinite(float(storage_energy_mwh))
        or storage_energy_mwh <= 0
    ):
        raise ValueError("저장량은 0보다 큰 숫자여야 합니다.")
    if (
        isinstance(supply_duration_hours, bool)
        or not isinstance(supply_duration_hours, int | float)
        or not math.isfinite(float(supply_duration_hours))
        or supply_duration_hours <= 0
    ):
        raise ValueError("공급 지속시간은 0보다 큰 숫자여야 합니다.")
    return float(storage_energy_mwh) / float(supply_duration_hours)


def calculate_ess_contract_performance(
    charge_dispatch_mwh: pd.Series,
    discharge_dispatch_mwh: pd.Series,
    charge_metered_mwh: pd.Series,
    discharge_metered_mwh: pd.Series,
) -> pd.Series:
    """공고의 충·방전 급전지시량과 계량량으로 이행률을 계산한다.

    급전지시가 없는 시간에는 미이행이 발생하지 않은 것으로 보아 이행률을
    1.0으로 둔다. 실제 월 정산에서는 거래소의 최종 정산자료가 우선하며,
    이 함수는 시간별 원자료가 있을 때 사용할 수 있는 공식식의 구현이다.
    """
    series = {
        "charge_dispatch_mwh": charge_dispatch_mwh,
        "discharge_dispatch_mwh": discharge_dispatch_mwh,
        "charge_metered_mwh": charge_metered_mwh,
        "discharge_metered_mwh": discharge_metered_mwh,
    }
    if len({len(value) for value in series.values()}) != 1:
        raise ValueError("급전지시량과 계량량의 길이가 같아야 합니다.")
    converted: dict[str, pd.Series] = {}
    for name, value in series.items():
        numeric = pd.to_numeric(value, errors="coerce")
        if (
            numeric.isna().any()
            or (~numeric.map(math.isfinite)).any()
            or (numeric < 0).any()
        ):
            raise ValueError(f"{name}에는 음수·결측·비수치 값이 없어야 합니다.")
        converted[name] = numeric.astype(float)

    charge_dispatch = converted["charge_dispatch_mwh"]
    discharge_dispatch = converted["discharge_dispatch_mwh"]
    charge_metered = converted["charge_metered_mwh"]
    discharge_metered = converted["discharge_metered_mwh"]
    total_dispatch = charge_dispatch + discharge_dispatch
    alpha = charge_dispatch.div(total_dispatch.where(total_dispatch > 0))
    alpha = alpha.fillna(0.0)
    charge_ratio = (
        (charge_metered - charge_dispatch).abs()
        .div(charge_dispatch.where(charge_dispatch > 0))
        .fillna(0.0)
        .clip(upper=1.0)
    )
    discharge_ratio = (
        (discharge_metered - discharge_dispatch).abs()
        .div(discharge_dispatch.where(discharge_dispatch > 0))
        .fillna(0.0)
        .clip(upper=1.0)
    )
    performance = 1.0 - (charge_ratio * alpha + discharge_ratio * (1.0 - alpha))
    return performance.clip(lower=0.0, upper=1.0)


@dataclass(frozen=True)
class EssCentralContractConfig:
    """ESS 중앙계약시장식 계약가격 정산 입력.

    ``contracted_power_mw``가 없으면 공고의 6시간 기준을 적용해
    ``storage_energy_mwh / supply_duration_hours``로 공급가능용량을 계산한다.
    실제 낙찰 계약물량을 확인한 뒤에는 ``contracted_power_mw``를 직접 넣어
    그 값을 우선 사용할 수 있다.

    단가 단위는 원/kW·h, 저장량은 MWh, 출력·공급가능용량은 MW이다.
    """

    bid_price_krw_per_kw_h: float
    storage_energy_mwh: float = 16.0
    supply_duration_hours: float = 6.0
    contracted_power_mw: float | None = None
    contract_price_ratio: float = 0.60
    availability_rate: float = 1.0
    performance_rate: float = 1.0

    def __post_init__(self) -> None:
        values = {
            "bid_price_krw_per_kw_h": self.bid_price_krw_per_kw_h,
            "storage_energy_mwh": self.storage_energy_mwh,
            "supply_duration_hours": self.supply_duration_hours,
            "contract_price_ratio": self.contract_price_ratio,
            "availability_rate": self.availability_rate,
            "performance_rate": self.performance_rate,
        }
        if self.contracted_power_mw is not None:
            values["contracted_power_mw"] = self.contracted_power_mw
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in values.values()
        ):
            raise ValueError("계약수익 입력은 숫자여야 합니다.")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("계약수익 입력은 유한한 숫자여야 합니다.")
        if self.bid_price_krw_per_kw_h < 0:
            raise ValueError("입찰단가는 0 이상이어야 합니다.")
        if self.storage_energy_mwh <= 0:
            raise ValueError("저장량은 0보다 커야 합니다.")
        if self.supply_duration_hours <= 0:
            raise ValueError("공급 지속시간은 0보다 커야 합니다.")
        if self.contracted_power_mw is not None and self.contracted_power_mw <= 0:
            raise ValueError("계약용량은 0보다 커야 합니다.")
        if not 0 <= self.contract_price_ratio <= 1:
            raise ValueError("계약가격 환산비율은 0~1이어야 합니다.")
        if not 0 <= self.availability_rate <= 1:
            raise ValueError("가용률은 0~1이어야 합니다.")
        if not 0 <= self.performance_rate <= 1:
            raise ValueError("이행률은 0~1이어야 합니다.")

    @property
    def contract_price_krw_per_kw_h(self) -> float:
        """입찰가격의 60% 구조를 반영한 계약가격."""
        return self.bid_price_krw_per_kw_h * self.contract_price_ratio

    @property
    def effective_contracted_power_mw(self) -> float:
        """계약수익에 사용할 공급가능용량."""
        if self.contracted_power_mw is not None:
            return float(self.contracted_power_mw)
        return supply_capacity_mw_from_storage(
            self.storage_energy_mwh,
            supply_duration_hours=self.supply_duration_hours,
        )


def _rate_series(
    frame: pd.DataFrame,
    column: str,
    default: float,
) -> pd.Series:
    values = (
        pd.to_numeric(frame[column], errors="coerce")
        if column in frame
        else pd.Series(default, index=frame.index, dtype=float)
    )
    if (
        values.isna().any()
        or (~values.map(math.isfinite)).any()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError(f"{column}은 0~1 범위의 숫자여야 합니다.")
    return values.astype(float)


def calculate_ess_contract_revenue(
    hourly: pd.DataFrame,
    config: EssCentralContractConfig,
) -> pd.DataFrame:
    """시간별 계약가격 정산을 계산한다.

    ``available_capacity_mw`` 컬럼이 있으면 시간별 공급가능용량으로 직접
    사용한다. 없으면 계약용량에 ``availability_rate``를 곱한다.
    충·방전 급전지시량과 계량량 4개 컬럼이 모두 있으면 공식 이행률을
    계산하고, 없으면 ``performance_rate``를 가정값으로 사용한다.
    """
    out = hourly.copy()
    if "dt_hours" not in out:
        out["dt_hours"] = 1.0
    dt = pd.to_numeric(out["dt_hours"], errors="coerce")
    if dt.isna().any() or (~dt.map(math.isfinite)).any() or (dt <= 0).any():
        raise ValueError("dt_hours는 양수인 숫자여야 합니다.")
    power_mw = config.effective_contracted_power_mw
    if "available_capacity_mw" in out:
        available_power = pd.to_numeric(out["available_capacity_mw"], errors="coerce")
        if (
            available_power.isna().any()
            or (~available_power.map(math.isfinite)).any()
            or (available_power < 0).any()
            or (available_power > power_mw).any()
        ):
            raise ValueError(
                "available_capacity_mw는 0 이상 계약용량 이하의 숫자여야 합니다."
            )
        available_power = available_power.astype(float)
        availability = available_power / power_mw
    else:
        availability = _rate_series(
            out, "availability_rate", config.availability_rate
        )
        available_power = power_mw * availability

    dispatch_columns = {
        "charge_dispatch_mwh",
        "discharge_dispatch_mwh",
        "charge_metered_mwh",
        "discharge_metered_mwh",
    }
    present_dispatch_columns = dispatch_columns.intersection(out.columns)
    if present_dispatch_columns and present_dispatch_columns != dispatch_columns:
        missing = sorted(dispatch_columns - present_dispatch_columns)
        raise ValueError(
            "공식 이행률 계산에는 다음 컬럼이 모두 필요합니다: " + ", ".join(missing)
        )
    if present_dispatch_columns == dispatch_columns:
        performance = calculate_ess_contract_performance(
            out["charge_dispatch_mwh"],
            out["discharge_dispatch_mwh"],
            out["charge_metered_mwh"],
            out["discharge_metered_mwh"],
        )
    else:
        performance = _rate_series(
            out, "performance_rate", config.performance_rate
        )
    out["dt_hours"] = dt
    out["ess_contract_price_krw_per_kw_h"] = config.contract_price_krw_per_kw_h
    out["ess_contract_power_mw"] = power_mw
    out["ess_contract_available_power_mw"] = available_power
    out["ess_contract_availability_rate"] = availability
    out["ess_contract_performance_rate"] = performance
    out["ess_contract_revenue_krw"] = (
        config.contract_price_krw_per_kw_h
        * available_power
        * 1000.0
        * dt
        * performance
    )
    return out


def summarize_ess_contract_revenue(hourly: pd.DataFrame) -> dict[str, float | int]:
    """시간별 계약수익 계산 결과를 합산한다."""
    required = {"dt_hours", "ess_contract_revenue_krw"}
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise KeyError(f"계약수익 계산 결과에 컬럼이 없습니다: {missing}")
    return {
        "hours": int(len(hourly)),
        "contract_hours": float(hourly["dt_hours"].sum()),
        "ess_contract_revenue_krw": float(hourly["ess_contract_revenue_krw"].sum()),
    }


def annualized_ess_contract_revenue(
    config: EssCentralContractConfig,
    *,
    hours_per_year: float = 8_760.0,
) -> float:
    """365일 계약이 유지된다는 가정의 연환산 계약가격 정산액."""
    if hours_per_year <= 0:
        raise ValueError("연간 시간은 0보다 커야 합니다.")
    return (
        config.contract_price_krw_per_kw_h
        * config.effective_contracted_power_mw
        * 1000.0
        * hours_per_year
        * config.availability_rate
        * config.performance_rate
    )
