"""시간 정산 입력: KST 시각 정렬, 단위 검증, 명시적인 시나리오/손실계수 가정."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vpp.rec import REC_COLUMN

RULE_VERSION = "jeju_2026_04"
RULE_EFFECTIVE_DATE = "2026-04-01"
DISPATCH_COLUMNS = (
    "resource_capacity_mw", "set_point_mw", "available_capacity_mw",
    "fcr_price_krw_per_kwh", "min_offer_price_krw_per_kwh", "offer_floor_krw_per_kwh",
)


@dataclass(frozen=True)
class HourlySettlementParams:
    resource_id: str
    rule_version: str = RULE_VERSION
    day_ahead_loss_factor: float = 1.0
    realtime_loss_factor: float = 1.0
    loss_factor_source: str = "assumption"
    loss_factor_note: str = "자원별 손실계수 미연결: 하루전·실시간 모두 1로 가정"
    rec_mode: str = "supplied_mwh_once"

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise ValueError("resource_id: 단일 정산 자원 식별자가 필요합니다.")
        if self.rule_version != RULE_VERSION:
            raise ValueError(f"rule_version: {RULE_VERSION}만 지원합니다.")
        for name in ("day_ahead_loss_factor", "realtime_loss_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) \
                    or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: 양의 유한한 값이어야 합니다.")
        if self.loss_factor_source not in {"observed", "assumption"}:
            raise ValueError("loss_factor_source: observed 또는 assumption이어야 합니다.")
        if not isinstance(self.loss_factor_note, str) or not self.loss_factor_note.strip():
            raise ValueError("loss_factor_note: 출처 또는 가정을 적어야 합니다.")
        if self.rec_mode not in {"excluded", "supplied_mwh_once"}:
            raise ValueError("rec_mode: excluded 또는 supplied_mwh_once이어야 합니다.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> HourlySettlementParams:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("시간 정산 설정은 매핑이어야 합니다.")
        return cls(**raw)

    def metadata(self) -> dict:
        return {
            **asdict(self),
            "rule_effective_date": RULE_EFFECTIVE_DATE,
            "interval_hours": 1.0,
            "status": "hourly_approximation_not_official_settlement",
            "resource_scope": "single_modeled_dispatchable_renewable_resource",
        }


def _hourly_frame(frame: pd.DataFrame, name: str, resource_id: str) -> pd.DataFrame:
    if "ts" not in frame:
        raise ValueError(f"{name}: ts 열이 필요합니다.")
    out = frame.copy()
    times = []
    for value in out["ts"]:
        # 문자열/타임스탬프만 허용: 숫자를 ns epoch로 오해하지 않는다.
        if isinstance(value, int | float | np.number):
            raise ValueError(f"{name}: ts는 날짜·시간이어야 합니다.")
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError(f"{name}: ts 결측은 허용하지 않습니다.")
        ts = ts.tz_localize("Asia/Seoul") if ts.tzinfo is None else ts.tz_convert("Asia/Seoul")
        if ts != ts.floor("h"):
            raise ValueError(f"{name}: 1시간 시작 시각만 허용합니다: {ts}")
        times.append(ts)
    out["ts"] = pd.DatetimeIndex(times, tz="Asia/Seoul")
    if out["ts"].duplicated().any():
        raise ValueError(f"{name}: 중복 ts가 있습니다. 자원/시나리오 하나를 선택하세요.")
    if "resource_id" in out and not out["resource_id"].eq(resource_id).all():
        raise ValueError(f"{name}: resource_id가 설정과 다릅니다. 자원별로 계산하세요.")
    if "dt_hours" in out:
        _numeric(out, "dt_hours", name, required=True)
        if not out["dt_hours"].eq(1.0).all():
            raise ValueError(f"{name}: dt_hours는 반드시 1이어야 합니다.")
    return out.sort_values("ts").reset_index(drop=True)


def _numeric(frame, col, name, *, required=False, nonnegative=False):
    if col not in frame:
        if required:
            raise ValueError(f"{name}: {col} 열이 필요합니다.")
        frame[col] = np.nan
    if frame[col].map(lambda x: isinstance(x, bool | np.bool_)).any():
        raise ValueError(f"{name}: {col}에 bool을 사용할 수 없습니다.")
    try:
        values = pd.to_numeric(frame[col], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {col}은 숫자여야 합니다.") from exc
    if np.isinf(values).any() or (required and values.isna().any()):
        raise ValueError(f"{name}: {col}에 유한한 값이 필요합니다.")
    if nonnegative and (values < 0).any():
        raise ValueError(f"{name}: {col}은 0 이상이어야 합니다.")
    frame[col] = values


def prepare_hourly_inputs(
    execution: pd.DataFrame,
    market: pd.DataFrame,
    dispatch: pd.DataFrame | None,
    params: HourlySettlementParams,
    *,
    fcr: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """행 순서가 아닌 ts로 결합한다. SMP 입력 단위는 market_hourly.csv의 원/kWh."""
    out = _hourly_frame(execution, "execution", params.resource_id)
    if out.empty:
        raise ValueError("execution: 정산할 행이 없습니다.")
    if "scenario" in out and out["scenario"].nunique(dropna=False) > 1:
        raise ValueError("execution: 시나리오 하나를 선택하세요.")
    for col in ("dt_hours", "committed_sell_mwh", "supply_mwh", "charge_mwh", "discharge_mwh"):
        _numeric(out, col, "execution", required=True, nonnegative=True)
    if params.rec_mode == "supplied_mwh_once":
        _numeric(out, REC_COLUMN, "execution", required=True, nonnegative=True)
    out["resource_id"] = params.resource_id

    prices = _hourly_frame(market, "market", params.resource_id)
    for col in ("smp_da", "smp_rt_mean"):
        _numeric(prices, col, "market")
    prices = prices.set_index("ts").reindex(out["ts"])
    for col in ("smp_da", "smp_rt_mean"):
        out[col] = prices[col].to_numpy()
    out["day_ahead_price_krw_per_mwh"] = out["smp_da"] * 1000 * params.day_ahead_loss_factor
    out["realtime_price_krw_per_mwh"] = out["smp_rt_mean"] * 1000 * params.realtime_loss_factor

    instructions = pd.DataFrame({"ts": []}) if dispatch is None else dispatch
    instructions = _hourly_frame(instructions, "dispatch", params.resource_id)
    for col in DISPATCH_COLUMNS:
        nonnegative = col in {
            "resource_capacity_mw", "set_point_mw", "available_capacity_mw",
            "fcr_price_krw_per_kwh",
        }
        _numeric(instructions, col, "dispatch", nonnegative=nonnegative)
    if instructions["resource_capacity_mw"].eq(0).any():
        raise ValueError("dispatch: resource_capacity_mw는 0보다 커야 합니다 (ESS 용량 아님).")
    if "dispatch_source" not in instructions:
        instructions["dispatch_source"] = "unknown"
    instructions["dispatch_source"] = instructions["dispatch_source"].fillna("unknown")
    if not instructions["dispatch_source"].isin(["observed", "scenario", "unknown"]).all():
        raise ValueError("dispatch_source: observed/scenario/unknown 중 하나여야 합니다.")
    if "dispatch_note" not in instructions:
        instructions["dispatch_note"] = ""
    instructions["dispatch_note"] = instructions["dispatch_note"].fillna("").astype(str)
    noted = instructions["dispatch_source"].isin(["observed", "scenario"])
    if (noted & instructions["dispatch_note"].str.strip().eq("")).any():
        raise ValueError("dispatch_note: 관측 출처 또는 시나리오 가정을 적어야 합니다.")
    instructions = instructions.set_index("ts").reindex(out["ts"])
    for col in (*DISPATCH_COLUMNS, "dispatch_source", "dispatch_note"):
        out[col] = instructions[col].to_numpy()
    out["dispatch_source"] = out["dispatch_source"].fillna("missing")
    out["dispatch_note"] = out["dispatch_note"].fillna("급전 자료 없음")
    # 시장 공통 FCR 가격은 급전 자료 없이도 연결한다. 급전 목표/출처를 생성하지 않는다.
    price_col = "fcr_price_krw_per_kwh"
    out["fcr_price_source"] = np.where(out[price_col].notna(), "dispatch_input", "missing")
    if fcr is not None:
        reserves = _hourly_frame(fcr, "fcr", params.resource_id)
        if price_col not in reserves:
            raise ValueError(f"fcr: {price_col} 열이 필요합니다.")
        _numeric(reserves, price_col, "fcr", nonnegative=True)
        reserves = reserves.set_index("ts").reindex(out["ts"])
        linked = reserves[price_col].reset_index(drop=True)
        # 두 경로에 다른 단가가 있으면 어느 쪽을 쓸지 조용히 결정하지 않는다.
        conflict = out[price_col].notna() & linked.notna() & ~np.isclose(
            out[price_col], linked, rtol=1e-10, atol=1e-10,
        )
        if conflict.any():
            raise ValueError("fcr: 별도 FCR 가격과 dispatch의 FCR 가격이 다릅니다.")
        out[price_col] = linked  # 명시한 FCR 파일이 없던 시간은 다른 값으로 채우지 않는다.
        source = reserves.get("fcr_price_source", pd.Series("fcr_input", index=reserves.index))
        out["fcr_price_source"] = source.fillna("fcr_input").to_numpy()
        out.loc[linked.isna(), "fcr_price_source"] = "missing"
    for col in ("fcr_price", "min_offer_price", "offer_floor"):
        out[f"{col}_krw_per_mwh"] = out[f"{col}_krw_per_kwh"] * 1000
    return out
