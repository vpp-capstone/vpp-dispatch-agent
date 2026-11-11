"""담당: 최적화 — REC(신재생에너지 공급인증서) 수익 가정 v0.

    REC 수익(원) = 계통 판매 전력량(MWh) × 가중치 × REC 단가(원/REC, 1 REC = 1 MWh)

- 판매 시점에 **한 번만** 붙인다 (eligible_basis=sold_mwh_once). 충전한 전력에는 붙지 않고 방전해서
  팔 때 붙으므로, 저장을 거친 전력은 왕복 손실만큼 REC 도 줄어든다. 같은 전력에 두 번 붙지 않는다.
- 목적함수에는 판매 단가 가산으로 들어간다: 판매 1 MWh 의 가치 = SMP + rec_krw_per_mwh
  (optimization.solve_window). ESS 없음·규칙 기준도 같은 가산으로 판매/감축을 정한다 (baselines).
  그래서 음수 SMP 라도 SMP + REC 가 양수면 판매한다.
- 단가는 월별 표(price_table CSV) 또는 단일 시나리오 값(price_krw_per_rec) 중 하나. 표에 없는 월은
  0 원으로 채우지 않고 오류를 낸다.

가정이며 실제 계약·발급 기준이 아니다 (status: experiment_assumption_not_contract):
포트폴리오 단일 가중치, 태양광·풍력 가중치 분리·전력 출처 추적·ESS 연계 가중치 없음.
**시작 SOC 에 들어 있던 전력도 팔리면 REC 가 붙는다** (모델은 전력 출처를 추적하지 않는다 — 제외는
미구현). 대신 그 금액의 보수적 상한을 요약에 따로 적는다 (initial_energy_rec_upper_bound_krw: 실제
시작 잔량 × 방전 효율 × 기간 최고 가산). 실제 발급 기준(발전기 계량 vs 송전량)과 계약 형태(고정가격
vs 현물)가 확인되면 이 모듈의 기준을 교체한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REC_COLUMN = "rec_krw_per_mwh"  # 시간별 입력 프레임에 붙는 판매 단가 가산 (가중치 × 단가)
ELIGIBLE_BASES = ("sold_mwh_once",)
STATUS = "experiment_assumption_not_contract"
PRICE_TABLE_COLUMNS = ("month", "price_krw_per_rec")
_YAML_KEYS = {"status", "eligible_basis", "weight", "price_krw_per_rec", "price_table"}


def _months(ts: pd.Series) -> pd.PeriodIndex:
    values = pd.to_datetime(ts)
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return pd.PeriodIndex(values, freq="M")


@dataclass(frozen=True)
class RecParams:
    """REC 가정값. 단가는 price_krw_per_rec(단일) 또는 price_table(월별) 중 하나만."""

    weight: float
    price_krw_per_rec: float | None = None
    price_table: pd.DataFrame | None = None  # 열: month(YYYY-MM), price_krw_per_rec
    eligible_basis: str = "sold_mwh_once"
    status: str = STATUS
    source: str = ""
    table_source: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.weight, bool) or not isinstance(self.weight, int | float):
            raise ValueError(f"REC 가중치는 숫자여야 합니다: {self.weight!r}")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("REC 가중치는 0 이상 유한해야 합니다.")
        object.__setattr__(self, "weight", float(self.weight))
        if self.eligible_basis not in ELIGIBLE_BASES:
            raise ValueError(f"eligible_basis 는 {ELIGIBLE_BASES} 중 하나여야 합니다.")
        has_flat, has_table = self.price_krw_per_rec is not None, self.price_table is not None
        if has_flat == has_table:
            raise ValueError("REC 단가는 price_krw_per_rec 또는 price_table 중 하나만 주세요.")
        if has_flat:
            price = self.price_krw_per_rec
            if isinstance(price, bool) or not isinstance(price, int | float):
                raise ValueError(f"REC 단가는 숫자여야 합니다: {price!r}")
            if not math.isfinite(price) or price < 0:
                raise ValueError("REC 단가는 0 이상 유한해야 합니다.")
            object.__setattr__(self, "price_krw_per_rec", float(price))
        else:
            object.__setattr__(self, "price_table", _validate_table(self.price_table))

    # ------------------------------------------------------------ 읽기
    @classmethod
    def from_yaml(cls, path: str | Path) -> RecParams:
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"REC 설정은 매핑이어야 합니다: {path}")
        unknown = sorted(set(raw) - _YAML_KEYS)
        if unknown:
            raise ValueError(f"알 수 없는 REC 설정 키: {unknown}")
        table = None
        table_source = raw.get("price_table")
        if table_source is not None:
            table_path = Path(table_source)
            if not table_path.is_absolute() and not table_path.exists():
                table_path = path.parent / table_path  # 설정 파일 옆에 둔 경우
            table = pd.read_csv(table_path)
            table_source = str(table_path)
        return cls(
            weight=raw.get("weight", 1.0),
            price_krw_per_rec=raw.get("price_krw_per_rec"),
            price_table=table,
            eligible_basis=raw.get("eligible_basis", "sold_mwh_once"),
            status=raw.get("status", STATUS),
            source=str(path),
            table_source=table_source,
        )

    # ------------------------------------------------------------ 적용
    def price_for(self, ts: pd.Series) -> np.ndarray:
        """각 시각의 REC 단가(원/REC). 월별 표에 없는 월이 있으면 예외 (0 으로 채우지 않음)."""
        n = len(ts)
        if self.price_krw_per_rec is not None:
            return np.full(n, self.price_krw_per_rec)
        if n == 0:
            return np.zeros(0)
        if not pd.api.types.is_datetime64_any_dtype(pd.Series(ts)):
            raise ValueError("월별 REC 단가표를 쓰려면 ts 가 시각이어야 합니다.")
        months = _months(pd.Series(ts).reset_index(drop=True))
        lookup = dict(zip(self.price_table["month"], self.price_table["price_krw_per_rec"],
                          strict=True))
        missing = sorted({str(m) for m in months if m not in lookup})
        if missing:
            raise ValueError(
                f"REC 단가가 없는 월: {missing} ({self.table_source}). 0 원으로 채우지 않습니다."
            )
        return np.array([lookup[m] for m in months], dtype=float)

    def adder_krw_per_mwh(self, ts: pd.Series) -> np.ndarray:
        """판매 단가 가산 = 가중치 × 단가 (원/MWh)."""
        return self.weight * self.price_for(ts)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """시간별 입력 프레임 복사본에 rec_krw_per_mwh 열을 붙인다."""
        if "ts" not in frame.columns:
            raise ValueError("REC 를 붙이려면 ts 열이 필요합니다.")
        out = frame.copy()
        out[REC_COLUMN] = self.adder_krw_per_mwh(out["ts"])
        return out

    def initial_energy_rec_upper_bound_krw(
        self, ess, ts, *, initial_energy_mwh: float | None = None
    ) -> float:
        """시작 저장분(하한 초과분)을 전부 팔았을 때 붙을 수 있는 REC 의 보수적 상한 (보정값).

        ts: 그 계획이 다루는 시각들(Series) 또는 첫 시각 하나. 월별 단가표면 그 기간의 **최고**
        단가를 쓴다 (뒤 달에 팔면 더 받을 수 있으므로 첫 달 단가는 상한이 아니다).
        initial_energy_mwh: 실제 시작 잔량 (--start-soc, --from-plan 등). None 이면 설정값.
        """
        start = ess.initial_energy_mwh if initial_energy_mwh is None else float(initial_energy_mwh)
        sellable = max(0.0, start - ess.energy_min_mwh) * ess.discharge_efficiency
        series = pd.Series(ts) if isinstance(ts, pd.Series | list | tuple | np.ndarray) \
            else pd.Series([pd.Timestamp(ts)])
        adders = self.adder_krw_per_mwh(series.reset_index(drop=True))
        return sellable * (float(np.max(adders)) if len(adders) else 0.0)

    def metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "enabled": True,
            "status": self.status,
            "eligible_basis": self.eligible_basis,
            "weight": self.weight,
            "source": self.source,
        }
        if self.price_krw_per_rec is not None:
            out["pricing"] = {"type": "flat_scenario", "price_krw_per_rec": self.price_krw_per_rec}
        else:
            months = self.price_table["month"]
            out["pricing"] = {
                "type": "monthly_table", "table": self.table_source,
                "months": int(len(months)), "first_month": str(months.min()),
                "last_month": str(months.max()),
                "price_min": float(self.price_table["price_krw_per_rec"].min()),
                "price_max": float(self.price_table["price_krw_per_rec"].max()),
            }
        return out


def _validate_table(table: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PRICE_TABLE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"REC 단가표에 열이 없습니다: {missing}")
    out = table.loc[:, list(PRICE_TABLE_COLUMNS)].copy()
    prices = pd.to_numeric(out["price_krw_per_rec"], errors="coerce")
    if prices.isna().any() or not np.isfinite(prices).all() or (prices < 0).any():
        bad = out.loc[prices.isna() | (prices < 0), "month"].astype(str).tolist()
        raise ValueError(f"REC 단가가 비었거나 음수인 월: {bad}. 값을 채우거나 행을 지우세요.")
    out["price_krw_per_rec"] = prices.astype(float)
    out["month"] = pd.PeriodIndex(out["month"].astype(str), freq="M")
    if out["month"].duplicated().any():
        raise ValueError("REC 단가표에 같은 월이 두 번 있습니다.")
    return out.reset_index(drop=True)


NO_REC_METADATA = {"enabled": False, "note": "REC 미적용 (판매수익 = SMP × 판매량)"}


def load_rec(path: str | Path | None) -> RecParams | None:
    """경로가 None 이면 REC 미적용."""
    return None if path is None else RecParams.from_yaml(path)


def rec_metadata(rec: RecParams | None) -> dict[str, Any]:
    return NO_REC_METADATA if rec is None else rec.metadata()


def rec_adder(frame: pd.DataFrame) -> np.ndarray:
    """프레임의 REC 가산 열 (없으면 0)."""
    if REC_COLUMN in frame.columns:
        return frame[REC_COLUMN].to_numpy(dtype=float)
    return np.zeros(len(frame))


def effective_price_krw_per_mwh(frame: pd.DataFrame) -> np.ndarray:
    """판매 1 MWh 의 가치 = SMP + REC 가산."""
    return frame["price_krw_per_mwh"].to_numpy(dtype=float) + rec_adder(frame)


def rec_revenue_krw(plan: pd.DataFrame) -> float:
    """계획의 REC 수익 = Σ 가산 × 판매량. REC 열이 없으면 0."""
    return float(rec_adder(plan) @ plan["sell_mwh"].to_numpy(dtype=float))
