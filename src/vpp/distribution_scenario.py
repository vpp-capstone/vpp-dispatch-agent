"""배전망 ESS v0 시나리오용 입력 변환과 운영 결과 요약.

기존 발전량·가격 예측과 ESS 최적화 코어를 유지하면서, 배전망 ESS 공모
시나리오의 입력 조건만 덧붙인다. 이 모듈은 20년 투자경제성 계산을 하지
않으며, 연간 운영 결과를 경제성 계산 모듈에 넘길 수 있는 형태로 만든다.

핵심 가정
----------
* 태양광 발전량은 기존 포트폴리오의 태양광 설비용량 비율로 5.7 MW에 맞춘다.
* ESS 사양은 별도 설정에서 4 MW / 20 MWh로 받는다.
* 선로 상한은 시간별 MW 컬럼 또는 하나의 상수 MW로 입력받아 MWh 상한으로
  변환한다. 공식 선로 상한이 정해지기 전까지 상수는 임시 시나리오로만 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from vpp.optimization import EssParams, RollingResult, run_rolling


@dataclass(frozen=True)
class DistributionESSConfig:
    """배전망 ESS v0 입력 조건.

    ``line_export_cap_mw``와 ``line_export_cap_column`` 중 하나만 사용한다.
    시간별 상한이 아직 없으면 둘 다 비워 둔 뒤 실행 시 입력으로 전달한다.
    """

    source_solar_capacity_mw: float
    target_solar_capacity_mw: float = 5.7
    line_export_cap_mw: float | None = None
    line_export_cap_column: str | None = None
    ess: EssParams | None = None

    def __post_init__(self) -> None:
        if self.source_solar_capacity_mw <= 0:
            raise ValueError("source_solar_capacity_mw는 0보다 커야 합니다.")
        if self.target_solar_capacity_mw <= 0:
            raise ValueError("target_solar_capacity_mw는 0보다 커야 합니다.")
        if self.line_export_cap_mw is not None and self.line_export_cap_mw < 0:
            raise ValueError("line_export_cap_mw는 0 이상이어야 합니다.")
        if self.line_export_cap_mw is not None and self.line_export_cap_column:
            raise ValueError("상수 상한과 시간별 상한 컬럼을 동시에 지정할 수 없습니다.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> DistributionESSConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"배전망 ESS 설정은 매핑이어야 합니다: {path}")
        ess_raw = raw.pop("ess", None)
        if not isinstance(ess_raw, dict):
            raise ValueError("설정에 ess 매핑이 필요합니다.")
        known = {f.name for f in fields(cls)} - {"ess"}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"알 수 없는 배전망 ESS 설정 키: {unknown}")
        return cls(ess=EssParams(**ess_raw), **raw)


def scale_solar_to_capacity(
    frame: pd.DataFrame,
    *,
    source_solar_capacity_mw: float,
    target_solar_capacity_mw: float = 5.7,
    solar_kwh_col: str = "pv_kwh",
    non_solar_kwh_col: str = "wind_kwh",
    total_generation_mwh_col: str = "generation_mwh",
) -> pd.DataFrame:
    """태양광 부분만 설비용량 비율로 환산한다.

    ``portfolio_data.build_portfolio_hourly``가 반환하는 ``pv_kwh``와
    ``wind_kwh`` 컬럼을 우선 지원한다. 풍력 컬럼이 없더라도 전체 발전량에서
    기존 태양광 부분을 빼고 다시 더해 비태양광 발전량을 보존한다.
    """

    if source_solar_capacity_mw <= 0 or target_solar_capacity_mw <= 0:
        raise ValueError("태양광 원본·목표 설비용량은 0보다 커야 합니다.")
    if solar_kwh_col not in frame.columns and "solar_generation_mwh" not in frame.columns:
        raise ValueError("태양광 발전량 컬럼(pv_kwh 또는 solar_generation_mwh)이 없습니다.")

    out = frame.copy()
    factor = float(target_solar_capacity_mw) / float(source_solar_capacity_mw)

    if solar_kwh_col in out.columns:
        solar_kwh = pd.to_numeric(out[solar_kwh_col], errors="raise").astype(float)
        scaled_solar_kwh = solar_kwh * factor
        out[solar_kwh_col] = scaled_solar_kwh
        if non_solar_kwh_col in out.columns:
            non_solar_kwh = pd.to_numeric(out[non_solar_kwh_col], errors="raise").astype(float)
            out[total_generation_mwh_col] = (scaled_solar_kwh + non_solar_kwh) / 1000.0
        elif total_generation_mwh_col in out.columns:
            total = pd.to_numeric(out[total_generation_mwh_col], errors="raise").astype(float)
            out[total_generation_mwh_col] = total - solar_kwh / 1000.0 + scaled_solar_kwh / 1000.0
    else:
        solar_mwh = pd.to_numeric(out["solar_generation_mwh"], errors="raise").astype(float)
        scaled_solar_mwh = solar_mwh * factor
        out["solar_generation_mwh"] = scaled_solar_mwh
        if total_generation_mwh_col in out.columns:
            total = pd.to_numeric(out[total_generation_mwh_col], errors="raise").astype(float)
            out[total_generation_mwh_col] = total - solar_mwh + scaled_solar_mwh
        else:
            out[total_generation_mwh_col] = scaled_solar_mwh

    out["solar_scale_factor"] = factor
    out["solar_source_capacity_mw"] = float(source_solar_capacity_mw)
    out["solar_target_capacity_mw"] = float(target_solar_capacity_mw)
    return out


def attach_line_export_cap(
    frame: pd.DataFrame,
    *,
    cap_mw: float | None = None,
    cap_mw_col: str | None = None,
    dt_hours_col: str = "dt_hours",
    output_col: str = "export_cap_mwh",
) -> pd.DataFrame:
    """시간별 선로 송전 상한을 최적화 입력용 MWh 컬럼으로 추가한다."""

    if cap_mw is not None and cap_mw_col is not None:
        raise ValueError("상수 상한과 시간별 상한 컬럼을 동시에 지정할 수 없습니다.")
    if cap_mw is None and cap_mw_col is None:
        raise ValueError("cap_mw 또는 cap_mw_col 중 하나가 필요합니다.")
    if dt_hours_col not in frame.columns:
        raise ValueError(f"시간 간격 컬럼이 없습니다: {dt_hours_col}")

    out = frame.copy()
    dt = pd.to_numeric(out[dt_hours_col], errors="raise").astype(float)
    if (dt <= 0).any() or not np.isfinite(dt).all():
        raise ValueError("시간 간격은 양의 유한값이어야 합니다.")
    if cap_mw_col is not None:
        if cap_mw_col not in out.columns:
            raise ValueError(f"선로 상한 컬럼이 없습니다: {cap_mw_col}")
        cap = pd.to_numeric(out[cap_mw_col], errors="raise").astype(float)
    else:
        cap = pd.Series(float(cap_mw), index=out.index, dtype=float)
    if (cap < 0).any() or not np.isfinite(cap).all():
        raise ValueError("선로 송전 상한은 0 이상인 유한값이어야 합니다.")

    out["line_export_cap_mw"] = cap
    out[output_col] = cap * dt
    return out


def prepare_distribution_v0_frame(
    frame: pd.DataFrame,
    config: DistributionESSConfig,
    *,
    line_cap_mw: float | None = None,
    line_cap_mw_col: str | None = None,
) -> pd.DataFrame:
    """5.7MW 환산과 선로 상한 부여를 한 번에 수행한다."""

    out = scale_solar_to_capacity(
        frame,
        source_solar_capacity_mw=config.source_solar_capacity_mw,
        target_solar_capacity_mw=config.target_solar_capacity_mw,
    )
    cap = config.line_export_cap_mw if line_cap_mw is None else line_cap_mw
    cap_col = config.line_export_cap_column if line_cap_mw_col is None else line_cap_mw_col
    return attach_line_export_cap(out, cap_mw=cap, cap_mw_col=cap_col)


def run_distribution_v0(
    frame: pd.DataFrame,
    config: DistributionESSConfig,
    *,
    horizon_hours: float = 48,
    commit_hours: float = 24,
    policy: str = "rolling48",
) -> RollingResult:
    """배전망 v0 계획을 실행한다. 선로 상한은 최적화 제약으로 적용된다."""

    if config.ess is None:
        raise ValueError("DistributionESSConfig.ess가 필요합니다.")
    if "export_cap_mwh" not in frame.columns:
        raise ValueError("frame에 export_cap_mwh가 필요합니다.")
    return run_rolling(
        frame,
        config.ess,
        horizon_hours=horizon_hours,
        commit_hours=commit_hours,
        policy=policy,
        export_cap_col="export_cap_mwh",
    )


def summarize_distribution_outputs(
    plan: RollingResult,
    ess: EssParams,
    *,
    annual_days: int = 365,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """경제성 계산 모듈에 넘길 운영 결과를 기간·연환산으로 요약한다."""

    if annual_days <= 0:
        raise ValueError("annual_days는 0보다 커야 합니다.")
    hourly = plan.hourly
    operating_days = int(pd.to_datetime(hourly["ts"]).dt.date.nunique()) if len(hourly) else 0
    if operating_days <= 0:
        raise ValueError("운영 결과에 날짜가 없습니다.")
    factor = annual_days / operating_days
    period = {
        "sales_revenue_krw": plan.sales_revenue_krw,
        "throughput_cost_krw": plan.throughput_cost_krw,
        "net_revenue_krw": plan.objective_krw,
        "charge_mwh": float(hourly["charge_mwh"].sum()),
        "discharge_mwh": float(hourly["discharge_mwh"].sum()),
        "curtailment_mwh": float(hourly["curtailment_mwh"].sum()),
        "sold_mwh": float(hourly["sell_mwh"].sum()),
    }
    annual = {f"annualized_{key}": float(value) * factor for key, value in period.items()}
    cap_binding = 0
    cap_violation = 0.0
    if "export_cap_mwh" in hourly.columns:
        cap = pd.to_numeric(hourly["export_cap_mwh"], errors="raise")
        sell = pd.to_numeric(hourly["sell_mwh"], errors="raise")
        cap_binding = int((sell >= cap - tol).sum())
        cap_violation = float(np.maximum(sell - cap, 0.0).sum())
    return {
        "mode": "distribution_ess_v0_perfect_foresight_backtest",
        "operating_days": operating_days,
        "annualization_factor": factor,
        "ess_capacity_mwh": ess.capacity_mwh,
        "ess_power_mw": ess.charge_power_mw,
        "period": period,
        "annualized": annual,
        "line_export_cap_binding_hours": cap_binding,
        "line_export_cap_violation_mwh": cap_violation,
    }
