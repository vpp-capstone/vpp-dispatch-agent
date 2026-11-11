"""최적화 — 실행 운전 결과의 실현 수익 (실험용 기본 정산 방식).

    확정 판매 정산액 = q* × P_da
    편차 정산액     = surplus × P_dev × m_surplus − shortfall × (P_dev × m_short + penalty)
    REC 수익        = ρ × 실제 공급량 (판매 시점 1회, 계획과 같은 가정)
    실행 열화비     = 열화모델(실제 충전, 실제 방전)
    실현 순수익     = 확정 판매 정산액 + 편차 정산액 + REC − 열화비

- P_da(하루전 정산 가격)·P_dev(편차 정산 가격) 자료가 없어 둘 다 **그 시각 실측 SMP** 로 둔다
  (config/execution_v0.yaml, 실험 가정). 설정으로 분리돼 있어 자료가 오면 바꾼다.
- 기본 m_surplus = m_short = 1, penalty = 0. 이러면 SMP 부분 합계가 정확히 `실제 공급량 × P` 가 되어
  확정 판매 수익과 공급량 수익을 중복으로 더하는 일이 없다. 음수 가격에서는 초과 공급이 비용,
  부족이 이득으로 부호가 맞게 나온다 (부족을 무조건 벌금으로 보지 않는다).
- 정산 가격이 없는 시간이 하나라도 있으면 그 날은 '계산 불가'(NaN + 사유)로 둔다. 예측값을 대신 넣지
  않는다. 가정 가격(실측 SMP 대체)으로 계산한 결과는 basis="assumed_actual_smp" 로 구분한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from vpp.optimization import EssParams
from vpp.rec import REC_COLUMN

PRICE_SOURCES = ("actual_smp",)
STATUS = "experiment_assumption_not_market_rule"
REVENUE_COLUMNS = ("committed_sales_krw", "deviation_krw", "rec_krw", "wear_krw",
                   "realized_net_krw")


@dataclass(frozen=True)
class SettlementParams:
    day_ahead_price: str = "actual_smp"
    deviation_price: str = "actual_smp"
    shortfall_price_multiplier: float = 1.0
    surplus_price_multiplier: float = 1.0
    shortfall_penalty_krw_per_mwh: float = 0.0
    rec_basis: str = "supplied_mwh_once"
    status: str = STATUS

    def __post_init__(self) -> None:
        for name in ("day_ahead_price", "deviation_price"):
            if getattr(self, name) not in PRICE_SOURCES:
                raise ValueError(f"{name} 은 {PRICE_SOURCES} 중 하나여야 합니다 (자료가 그것뿐).")
        for name in ("shortfall_price_multiplier", "surplus_price_multiplier",
                     "shortfall_penalty_krw_per_mwh"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 은 0 이상 유한한 숫자여야 합니다: {value!r}")
            object.__setattr__(self, name, float(value))
        if self.rec_basis != "supplied_mwh_once":
            raise ValueError("rec_basis 는 supplied_mwh_once 만 지원합니다.")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SettlementParams:
        return cls(**dict(raw or {}))

    @property
    def basis(self) -> str:
        """정산 가격의 출처 표시. 실측 SMP 를 하루전·편차 가격으로 쓰면 '가정 기반'."""
        return "assumed_actual_smp"

    def metadata(self) -> dict[str, Any]:
        return {
            "status": self.status, "basis": self.basis,
            "day_ahead_price": self.day_ahead_price, "deviation_price": self.deviation_price,
            "shortfall_price_multiplier": self.shortfall_price_multiplier,
            "surplus_price_multiplier": self.surplus_price_multiplier,
            "shortfall_penalty_krw_per_mwh": self.shortfall_penalty_krw_per_mwh,
            "rec_basis": self.rec_basis,
            "note": "하루전·실시간 가격 자료가 없어 둘 다 그 시각 실측 SMP. 배수 1·벌칙 0 은 "
                    "대칭 기본값(근거 없음). 실제 시장 규칙 아님.",
        }


def load_settlement(path: str | Path | None) -> SettlementParams:
    if path is None:
        return SettlementParams()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SettlementParams.from_dict(raw.get("settlement"))


def settle(
    execution: pd.DataFrame,
    actual_price_krw_per_mwh,
    ess: EssParams,
    params: SettlementParams | None = None,
) -> pd.DataFrame:
    """실행 결과(execute_plan.hourly)에 정산 열을 붙인 복사본. 가격이 없는 시간은 NaN 으로 둔다.

    actual_price_krw_per_mwh: execution 과 같은 길이의 실측 SMP(원/MWh). NaN 은 그 시간 계산 불가.
    """
    params = params or SettlementParams()
    required = {"committed_sell_mwh", "supply_mwh", "shortfall_mwh", "surplus_mwh",
                "charge_mwh", "discharge_mwh"}
    missing = sorted(required - set(execution.columns))
    if missing:
        raise ValueError(f"실행 결과에 열이 없습니다: {missing}")
    out = execution.reset_index(drop=True).copy()
    price = np.asarray(actual_price_krw_per_mwh, dtype=float)
    if price.shape != (len(out),):
        raise ValueError("실측 가격 길이가 실행 결과와 다릅니다.")
    out["settlement_price_krw_per_mwh"] = price
    q = out["committed_sell_mwh"].to_numpy(dtype=float)
    short = out["shortfall_mwh"].to_numpy(dtype=float)
    plus = out["surplus_mwh"].to_numpy(dtype=float)
    supply = out["supply_mwh"].to_numpy(dtype=float)
    rec = out[REC_COLUMN].to_numpy(dtype=float) if REC_COLUMN in out else np.zeros(len(out))
    wear = np.array([
        ess.degradation_cost_krw(np.array([c]), np.array([d]))
        for c, d in zip(out["charge_mwh"], out["discharge_mwh"], strict=True)
    ])
    out["committed_sales_krw"] = q * price
    out["deviation_krw"] = (
        plus * price * params.surplus_price_multiplier
        - short * price * params.shortfall_price_multiplier
        - short * params.shortfall_penalty_krw_per_mwh
    )
    out["rec_krw"] = rec * supply
    out["wear_krw"] = wear
    out["realized_net_krw"] = (out["committed_sales_krw"] + out["deviation_krw"]
                               + out["rec_krw"] - out["wear_krw"])
    out["settlement_basis"] = np.where(np.isfinite(price), params.basis, "price_missing")
    return out


def settlement_totals(settled: pd.DataFrame) -> dict[str, Any]:
    """합계. 가격이 없는 시간이 하나라도 있으면 금액은 None(계산 불가)이고 사유를 적는다."""
    missing = int((settled["settlement_basis"] == "price_missing").sum())
    out: dict[str, Any] = {
        "hours": int(len(settled)), "price_missing_hours": missing,
        "computable": missing == 0,
        "basis": None if missing else str(settled["settlement_basis"].iloc[0]),
    }
    for col in REVENUE_COLUMNS:
        out[col] = None if missing else float(settled[col].sum())
    if missing:
        out["note"] = f"정산 가격이 없는 시간 {missing}개 → 계산 불가 (예측값으로 대체하지 않음)"
    return out
