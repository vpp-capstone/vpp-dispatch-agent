"""제주 급전가능 재생에너지 용량정산금 계산.

2024-09-12 KPX Q&A의 제주 재생e 설명을 기준으로 한다.

    시간별 용량정산금
      = min(발전량, 입찰량, 실효용량) × 기준용량가격(RCP)

이 모듈은 2026년 4월에 도입된 RPCF를 과거(2024-06~2025-12) 자료에
소급 적용하지 않는다. 2024~2025년의 자원별 실효용량비율과 거래기간별 RCP를
CSV에서 읽어 시간별로 적용한다.

주의:
* 프로젝트의 원본 발전량은 kWh, 계산 결과의 시간별 정산량은 MWh이다.
* RCP는 원/kWh이므로 정산량(MWh)에 1,000을 곱해 원으로 변환한다.
* acceptance_rate는 실제 낙찰자료가 없을 때 이름 그대로 설비용량 대비 입찰량
  또는 계획 발전량 대비 입찰량 비율로 사용하는 시나리오 값이다. 실제 입찰량이
  있으면 이를 직접 넣어야 한다.
* 2024-07-01 이전의 실효용량비율은 현재 CSV에 없으므로 strict=True일 때 오류를
  발생시킨다. 누락 구간을 0원으로 조용히 처리하지 않는 것이 목적이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_RCP_PATH = Path("data/processed/capacity_settlement/rcp_jeju_by_period.csv")
DEFAULT_RATIO_PATH = Path(
    "data/processed/capacity_settlement/effective_capacity_ratio_jeju.csv"
)


@dataclass(frozen=True)
class RenewableCapacityConfig:
    """프로젝트의 발전자원별 용량정산 입력.

    현재 프로젝트의 포트폴리오 용량은 ``formulation.md``의 가정 및
    ``portfolio_data.py``의 용량 기준을 따른다. 실제 등록용량·입찰량이 확인되면
    이 값을 교체해야 한다.
    """

    pv_capacity_mw: float = 14.081
    wind_capacity_mw: float = 62.0
    acceptance_rate: float = 1.0
    bid_basis: str = "generation"

    def __post_init__(self) -> None:
        if self.pv_capacity_mw < 0 or self.wind_capacity_mw < 0:
            raise ValueError("발전원 용량은 음수가 될 수 없습니다.")
        if not 0 <= self.acceptance_rate:
            raise ValueError("acceptance_rate는 0 이상이어야 합니다.")
        if self.bid_basis not in {"generation", "nameplate"}:
            raise ValueError("bid_basis는 generation 또는 nameplate이어야 합니다.")


def _date_series(frame: pd.DataFrame) -> pd.Series:
    if "ts" not in frame:
        raise KeyError("시간별 입력에는 ts 컬럼이 필요합니다.")
    ts = pd.to_datetime(frame["ts"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return ts.dt.normalize()


def _lookup_period_values(
    dates: pd.Series,
    table: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    """날짜별 적용기간·월 자료를 반환한다. 매칭되지 않은 날짜는 NaN이다."""
    label_column = (
        "application_label" if "application_label" in table else "period_label"
    )
    required = {"period_start", "period_end", label_column, *value_columns}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"참조표에 필요한 컬럼이 없습니다: {sorted(missing)}")

    starts = pd.to_datetime(table["period_start"]).dt.normalize()
    ends = pd.to_datetime(table["period_end"]).dt.normalize()
    out = pd.DataFrame(index=dates.index)
    out["application_label"] = pd.NA
    for column in value_columns:
        out[column] = pd.NA

    for row_idx, row in table.iterrows():
        mask = (
            dates.ge(starts.loc[row_idx])
            & dates.le(ends.loc[row_idx])
        )
        if "month" in table:
            mask &= dates.dt.month.eq(int(row["month"]))
        out.loc[mask, "application_label"] = row[label_column]
        for column in value_columns:
            out.loc[mask, column] = float(row[column])
    return out


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise KeyError(f"시간별 입력에 {column} 컬럼이 없습니다.")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column}에 결측 또는 숫자가 아닌 값이 있습니다.")
    if (values < 0).any():
        raise ValueError(f"{column}에 음수 발전량이 있습니다.")
    return values.astype(float)


def kpx_capacity_quantity_mwh(
    available_mwh: pd.Series,
    bid_mwh: pd.Series,
    metered_mwh: pd.Series,
    effective_mwh: pd.Series,
    *,
    set_point_mwh: pd.Series | None = None,
    tolerance_mwh: float = 1e-9,
) -> pd.Series:
    """KPX 제주 재생e 정산교육자료의 공급가능용량 분기를 계산한다.

    ``SET_POINT = RA``이면 ``MGO``(계량발전량)를 최소값에 포함하고,
    ``SET_POINT < RA``이면 ``MGO``를 제외하는 규칙이다. 실제 프로젝트에
    SET_POINT 자료가 없을 때는 ``None``을 넣어 보수적인 Q&A 산식
    ``min(MGO, RA, EA)``을 사용한다.

    입력 단위는 모두 해당 거래구간의 MWh이다. 음수나 결측은 허용하지 않는다.
    """
    values = pd.concat(
        [available_mwh, bid_mwh, metered_mwh, effective_mwh], axis=1
    ).apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or (values < 0).any().any():
        raise ValueError("용량정산 입력에는 음수·결측이 없어야 합니다.")
    available, bid, metered, effective = [values.iloc[:, i] for i in range(4)]
    full_dispatch = pd.concat([available, bid, metered, effective], axis=1).min(axis=1)
    partial_dispatch = pd.concat([available, bid, effective], axis=1).min(axis=1)
    if set_point_mwh is None:
        return full_dispatch
    set_point = pd.to_numeric(set_point_mwh, errors="coerce")
    if set_point.isna().any() or (set_point < 0).any():
        raise ValueError("SET_POINT에는 음수·결측이 없어야 합니다.")
    is_full_dispatch = (set_point - bid).abs() <= tolerance_mwh
    return full_dispatch.where(is_full_dispatch, partial_dispatch)


def calculate_renewable_capacity_settlement(
    hourly: pd.DataFrame,
    *,
    config: RenewableCapacityConfig | None = None,
    rcp_path: str | Path = DEFAULT_RCP_PATH,
    ratio_path: str | Path = DEFAULT_RATIO_PATH,
    strict: bool = True,
) -> pd.DataFrame:
    """시간별 태양광·풍력 용량정산금을 계산한다.

    Parameters
    ----------
    hourly:
        ``build_portfolio_hourly`` 결과처럼 ``ts``, ``pv_kwh``, ``wind_kwh``를
        포함하는 시간별 자료. ``dt_hours``가 없으면 1시간으로 본다.
    config:
        발전원별 등록용량 및 낙찰 시나리오. ``bid_basis="generation"``이면
        정답 발전량(팀 문서의 판매계획 대용)의 90% 등을 입찰량으로 사용하고,
        ``bid_basis="nameplate"``이면 설비용량의 90%를 입찰량으로 사용한다.
    strict:
        RCP 또는 실효용량비율이 없는 기간을 오류로 처리할지 여부. False이면
        해당 행의 정산 관련 값은 NaN으로 남아 누락 구간을 확인할 수 있다.

    Returns
    -------
    pandas.DataFrame
        원본 컬럼에 적용기간, 실효용량, min 정산량, 자원별·합계 정산금을 추가한
        복사본. 금액 단위는 원, 정산량 단위는 MWh이다.
    """
    if hourly.empty:
        return hourly.copy()
    config = config or RenewableCapacityConfig()
    out = hourly.copy()
    dates = _date_series(out)
    dt = (
        pd.to_numeric(out["dt_hours"], errors="coerce")
        if "dt_hours" in out
        else pd.Series(1.0, index=out.index)
    )
    if dt.isna().any() or (dt <= 0).any():
        raise ValueError("dt_hours는 양수여야 합니다.")

    rcp = pd.read_csv(rcp_path)
    ratio = pd.read_csv(ratio_path)
    rcp_values = _lookup_period_values(dates, rcp, ["rcp_krw_per_kwh"])
    ratio_values = _lookup_period_values(dates, ratio, ["solar_pct", "wind_pct"])
    out["capacity_rcp_period"] = rcp_values["application_label"].to_numpy()
    out["capacity_ratio_period"] = ratio_values["application_label"].to_numpy()
    out["capacity_rcp_krw_per_kwh"] = pd.to_numeric(
        rcp_values["rcp_krw_per_kwh"], errors="coerce"
    ).to_numpy()
    out["capacity_missing_reference"] = (
        out["capacity_rcp_krw_per_kwh"].isna()
        | ratio_values[["solar_pct", "wind_pct"]].isna().any(axis=1).to_numpy()
    )
    if strict and out["capacity_missing_reference"].any():
        missing_dates = dates.loc[out["capacity_missing_reference"]].dt.strftime("%Y-%m-%d")
        raise ValueError(
            "RCP 또는 실효용량비율이 없는 날짜가 있습니다: "
            + ", ".join(missing_dates.drop_duplicates().tolist())
        )

    out["capacity_acceptance_rate"] = config.acceptance_rate
    out["capacity_bid_basis"] = config.bid_basis
    total_revenue = pd.Series(0.0, index=out.index)
    total_quantity = pd.Series(0.0, index=out.index)
    for resource, generation_column, ratio_column, capacity_mw in (
        ("pv", "pv_kwh", "solar_pct", config.pv_capacity_mw),
        ("wind", "wind_kwh", "wind_pct", config.wind_capacity_mw),
    ):
        generation_mwh = _numeric_series(out, generation_column) / 1000.0
        ratio_pct = pd.to_numeric(ratio_values[ratio_column], errors="coerce")
        effective_mw = capacity_mw * ratio_pct / 100.0
        generation_quantity = generation_mwh
        if config.bid_basis == "generation":
            bid_quantity = generation_quantity * config.acceptance_rate
            bid_mw = bid_quantity / dt
        else:
            bid_mw = capacity_mw * config.acceptance_rate
            bid_quantity = bid_mw * dt
        effective_quantity = effective_mw * dt
        valid_reference = (~out["capacity_missing_reference"]) & effective_quantity.notna()
        settlement_quantity = pd.Series(float("nan"), index=out.index)
        if valid_reference.any():
            settlement_quantity.loc[valid_reference] = kpx_capacity_quantity_mwh(
                bid_quantity.loc[valid_reference],
                bid_quantity.loc[valid_reference],
                generation_quantity.loc[valid_reference],
                effective_quantity.loc[valid_reference],
            )
        revenue = settlement_quantity * 1000.0 * out["capacity_rcp_krw_per_kwh"]

        out[f"{resource}_capacity_bid_mw"] = bid_mw
        out[f"{resource}_effective_capacity_ratio_pct"] = ratio_pct.to_numpy()
        out[f"{resource}_effective_capacity_mw"] = effective_mw.to_numpy()
        out[f"{resource}_capacity_settlement_quantity_mwh"] = settlement_quantity
        out[f"{resource}_capacity_settlement_krw"] = revenue
        total_quantity = total_quantity.add(settlement_quantity, fill_value=0.0)
        total_revenue = total_revenue.add(revenue, fill_value=0.0)

    out["capacity_settlement_quantity_mwh"] = total_quantity
    out["capacity_settlement_krw"] = total_revenue
    out.loc[out["capacity_missing_reference"], "capacity_settlement_quantity_mwh"] = pd.NA
    out.loc[out["capacity_missing_reference"], "capacity_settlement_krw"] = pd.NA
    out["capacity_settlement_formula"] = (
        "min(generation_mwh,bid_mwh,effective_capacity_mw*dt)*1000*rcp_krw_per_kwh"
    )
    return out


def summarize_capacity_settlement(hourly: pd.DataFrame) -> dict[str, float | int]:
    """계산 결과를 합계로 요약한다. 누락 참조가 있으면 합계를 내지 않는다."""
    if "capacity_settlement_krw" not in hourly:
        raise KeyError("먼저 calculate_renewable_capacity_settlement를 실행해야 합니다.")
    if hourly["capacity_settlement_krw"].isna().any():
        raise ValueError("누락된 기간이 있어 용량정산금 합계를 확정할 수 없습니다.")
    return {
        "hours": int(len(hourly)),
        "capacity_settlement_quantity_mwh": float(
            hourly["capacity_settlement_quantity_mwh"].sum()
        ),
        "capacity_settlement_krw": float(hourly["capacity_settlement_krw"].sum()),
    }
