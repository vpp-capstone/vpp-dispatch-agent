"""ESS 열화비 모델의 작은 확장점.

현재 v0.1은 충전량과 방전량에 비례하는 고정 처리량 비용만 구현한다.
모델을 별도 객체로 분리해 향후 DoD/온도/경로 의존 모델을 추가할 때
최적화 코드의 목적함수와 검산 코드를 동시에 다시 쓰지 않도록 한다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class DegradationModel(Protocol):
    """MILP 선형 목적함수에 연결할 수 있는 열화비 인터페이스."""

    name: str

    def milp_coefficients(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """충전·방전 변수에 적용할 원/MWh 비용 계수."""

    def cost_for_arrays(self, charge_mwh: Any, discharge_mwh: Any) -> float:
        """계획 배열의 열화비를 계산한다."""

    def metadata(self) -> dict[str, Any]:
        """결과·로그에 남길 모델 메타데이터."""


@dataclass(frozen=True)
class NoDegradationModel:
    name: str = "none"

    def milp_coefficients(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)

    def cost_for_arrays(self, charge_mwh: Any, discharge_mwh: Any) -> float:
        return 0.0

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "throughput_cost_krw_per_mwh": 0.0}


@dataclass(frozen=True)
class FixedThroughputModel:
    """충전+방전 처리량에 일정한 원/MWh 비용을 부과하는 1단계 모델."""

    cost_krw_per_mwh: float
    throughput_basis: str = "charge_plus_discharge"
    name: str = "fixed_throughput"

    def __post_init__(self) -> None:
        if not np.isfinite(self.cost_krw_per_mwh) or self.cost_krw_per_mwh < 0:
            raise ValueError("고정 처리량 열화비는 0 이상 유한해야 합니다.")
        if self.throughput_basis not in {"charge_plus_discharge", "discharge_only"}:
            raise ValueError(
                "throughput_basis는 charge_plus_discharge 또는 discharge_only여야 합니다."
            )

    def milp_coefficients(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        charge_cost = (
            self.cost_krw_per_mwh if self.throughput_basis == "charge_plus_discharge" else 0.0
        )
        charge = np.full(n, charge_cost)
        discharge = np.full(n, self.cost_krw_per_mwh)
        return charge, discharge

    def cost_for_arrays(self, charge_mwh: Any, discharge_mwh: Any) -> float:
        charge = np.asarray(charge_mwh, dtype=float)
        discharge = np.asarray(discharge_mwh, dtype=float)
        throughput = discharge.sum()
        if self.throughput_basis == "charge_plus_discharge":
            throughput += charge.sum()
        return float(self.cost_krw_per_mwh * throughput)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "throughput_basis": self.throughput_basis,
            "throughput_cost_krw_per_mwh": float(self.cost_krw_per_mwh),
        }


ModelFactory = Callable[[float, str], DegradationModel]
_MODEL_FACTORIES: dict[str, ModelFactory] = {}


def register_degradation_model(
    name: str, factory: ModelFactory, *, overwrite: bool = False
) -> None:
    """사용자 정의 모델을 등록한다."""
    if not name or name in {"none", "fixed_throughput"} and not overwrite:
        raise ValueError("사용자 모델 이름은 비어 있지 않고 기본 모델과 달라야 합니다.")
    if name in _MODEL_FACTORIES and not overwrite:
        raise ValueError(f"이미 등록된 degradation_model입니다: {name!r}")
    _MODEL_FACTORIES[name] = factory


def build_degradation_model(
    name: str, cost_krw_per_mwh: float, throughput_basis: str = "charge_plus_discharge"
) -> DegradationModel:
    """설정명으로 현재 지원 모델을 생성한다."""
    if name == "none":
        if cost_krw_per_mwh != 0:
            raise ValueError(
                "degradation_model=none이면 throughput_cost_krw_per_mwh는 0이어야 합니다."
            )
        return NoDegradationModel()
    if name == "fixed_throughput":
        return FixedThroughputModel(cost_krw_per_mwh, throughput_basis)
    factory = _MODEL_FACTORIES.get(name)
    if factory is not None:
        return factory(cost_krw_per_mwh, throughput_basis)
    raise ValueError(f"지원하지 않는 degradation_model입니다: {name!r}")
