"""배전망 ESS 1년 운영 백테스트 → 사업성 입력 (운영 결과 CSV).

실행 쪽(``distribution_scenario``)과 사업성 쪽(``project_economics``)을 잇는다.

    python -m vpp.distribution_backtest \
        --generation <발전량 예측 평가 parquet> --fcr <예비력 가격 CSV> \
        --forecast <SMP v6 예측 CSV> \
        --start 2025-01-01 --end 2025-12-31 \
        --out docs/results/distribution_ess_operating_<날짜>.csv

한 조합(선로 조건 x 배터리 x 운전 방식)마다 하루씩 48시간 MILP를 풀고 앞 24시간을 확정한다.

- 계획 가격: ``forecast`` = 전날 11시에 낸 SMP v6 예측(48시간), ``perfect`` = 정답 SMP.
  어느 쪽이든 확정한 판매량은 실제 SMP로 정산한다.
- 판매 1MWh 가치 = SMP + REC 가산(``config/rec_v0_monthly.yaml``).
  표 밖의 달은 표의 처음·마지막 달 값을 쓴다.
- ``forecast``·``perfect`` 는 발전량을 실측 그대로 안다고 본다 (예측 오차·페널티 없음).
- ``forecast_exec`` (``--generation`` 필요): 발전량 예측 + SMP v6 예측으로 하루 전 계획 →
  실측 발전량으로 실행 운전(``vpp.execution``, 확정 판매량 우선, 초과 판매 없음) →
  하루전·실시간 이중가격 정산과 임밸런스 페널티(``vpp.hourly_settlement``).
  급전지시량 = 확정 판매량, 공급가능용량 = 확정 판매량(감발 지시 없음)으로 가정한다.
  FCR 가격이 없는 시간은 있는 자료의 시간대별 평균으로 채운다.
- 선로 상한(새 태양광+ESS가 선로로 내보낼 수 있는 MW):
  - ``fixed``: 낮(08~18시)에만 c MW, 밤은 제한 없음
  - ``dynamic``: c + 기존 태양광 MW x (1 - 일사 비율). 해가 가장 강할 때 c MW.
    한전이 D-1에 주는 시간별 허용범위를 흉내 낸 가정이다.
- 배터리별 용량(정격 x SoH)·효율·열화비는 ``config/project_economics_v0.yaml`` 값을 쓴다.
  열화비는 운전 판단에만 쓰고, 운영 결과의 수익에서는 빼지 않는다.
- ``battery=none`` 은 ESS 없이 상한 안에서 태양광만 판 기준선
  (SMP+REC 계획값이 음수인 시간은 판매 안 함).
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vpp import project_economics as pe
from vpp.distribution_scenario import attach_line_export_cap
from vpp.execution import RuleBasedPolicy, execute_plan
from vpp.hourly_settlement import settle_hourly
from vpp.optimization import EssParams, solve_window
from vpp.rec import RecParams
from vpp.settlement_inputs import HourlySettlementParams

DAY_HOURS = (8, 18)  # fixed 상한이 걸리는 시간 [시작, 끝)
SOLAR_EFCR = 0.0384  # 태양광 용량정산 실효용량 비율
ESS_EFCR = 0.6563  # 4시간 ESS 실효용량 비율
JEJU_RCP_KRW_PER_KWH_H = 22.05
BASELINE = pe.BASELINE_BATTERY
EXEC = "forecast_exec"
OPERATIONS = ("forecast", "perfect", EXEC)
# 선로 상한을 넘지 않도록 실행 중 초과 판매는 하지 않는다 (남는 전력은 충전 또는 감축)
EXEC_POLICY = RuleBasedPolicy(surplus_order=("reduce_discharge", "charge", "curtail"))
DISPATCH_NOTE = "급전지시량 = 하루전 확정 판매량, 공급가능용량 = 확정 판매량 (감발 지시 없음 가정)"


@dataclass(frozen=True)
class LineCase:
    """선로 조건 하나. kind: fixed / dynamic."""

    kind: str
    cap_mw: float
    existing_pv_mw: float = 5.0
    ess_power_mw: float | None = None  # None 이면 사업성 설정값
    ess_energy_mwh: float | None = None

    @property
    def name(self) -> str:
        head = "cap" if self.kind == "fixed" else "dyn"
        out = f"{head}_{self.cap_mw:.2f}MW"
        if self.ess_power_mw is not None:
            out += f"_ess{self.ess_power_mw:g}MW{self.ess_energy_mwh:g}MWh"
        return out

    def __post_init__(self) -> None:
        if self.kind not in ("fixed", "dynamic"):
            raise ValueError("kind 는 fixed 또는 dynamic 이어야 합니다.")
        if self.cap_mw < 0 or self.existing_pv_mw < 0:
            raise ValueError("상한·기존 태양광은 0 이상이어야 합니다.")
        if (self.ess_power_mw is None) != (self.ess_energy_mwh is None):
            raise ValueError("ess_power_mw 와 ess_energy_mwh 는 함께 지정합니다.")


def line_cap_mw(
    case: LineCase, ts: pd.DatetimeIndex, solar_mw: np.ndarray, solar_capacity_mw: float
):
    """시간별 선로 상한 (MW)."""
    if case.kind == "fixed":
        day = (ts.hour >= DAY_HOURS[0]) & (ts.hour < DAY_HOURS[1])
        return np.where(day, case.cap_mw, 1e6)
    ratio = np.clip(np.asarray(solar_mw, float) / solar_capacity_mw, 0.0, 1.0)
    return case.cap_mw + case.existing_pv_mw * (1.0 - ratio)


def ess_for(base: EssParams, cfg: pe.EconomicsConfig, battery: pe.Battery, case: LineCase):
    power = cfg.ess_power_mw if case.ess_power_mw is None else case.ess_power_mw
    energy = cfg.ess_energy_mwh if case.ess_energy_mwh is None else case.ess_energy_mwh
    return dataclasses.replace(
        base,
        capacity_mwh=energy * battery.soh,
        charge_power_mw=power,
        discharge_power_mw=power,
        charge_efficiency=battery.efficiency,
        discharge_efficiency=battery.efficiency,
        throughput_cost_krw_per_mwh=battery.throughput_cost_krw_per_mwh(energy, cfg.fx),
    )


@dataclass
class Inputs:
    """시간별 입력. 가격은 모두 원/MWh."""

    hourly: pd.DataFrame  # index ts: solar_mwh, smp_krw_per_mwh, rec_krw_per_mwh[, smp_rt, fcr]
    forecast: dict[str, np.ndarray]  # 발행일(YYYY-MM-DD) -> 다음 날 0시부터 48시간 예측
    solar_capacity_mw: float
    # 계획일(YYYY-MM-DD) -> 그날 0시부터 48시간 발전량 예측 (MWh, 5.7MW 환산)
    generation_forecast: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    notes: dict = dataclasses.field(default_factory=dict)


def load_generation(path: str | Path, target_mw: float) -> tuple[pd.Series, dict, float]:
    """발전량 예측 평가 파일 → (실측, 계획일별 예측, 원본 MW).

    파일: 발전량 예측 ``artifacts/2025/evaluation/solar_improved_2025_with_actual.parquet``.

    자원별 행을 합쳐 포트폴리오 합계로 만들고 target_mw / 원본 설비용량 비율로 환산한다.
    실측은 계획 당일(forecast_day_offset=0) 행에서 시간마다 한 번만 쓴다.
    """
    d = pd.read_parquet(path)
    need = {"asset_id", "capacity_kw", "planning_day_kst", "valid_time_kst",
            "forecast_day_offset", "improved_generation_kwh", "generation_kwh"}  # fmt: skip
    missing = sorted(need - set(d.columns))
    if missing:
        raise ValueError(f"발전량 예측 파일에 열이 없습니다: {missing}")
    if "available_before_bid_deadline" in d and not d["available_before_bid_deadline"].all():
        d = d[d["available_before_bid_deadline"]]
    source_mw = float(d.groupby("asset_id")["capacity_kw"].first().sum()) / 1000.0
    k = target_mw / source_mw
    d = d.assign(t=pd.to_datetime(d["valid_time_kst"]).dt.tz_localize(None))
    same_day = d[d["forecast_day_offset"] == 0]
    actual = same_day.groupby("t")["generation_kwh"].sum(min_count=1) / 1000.0 * k
    fc = {}
    for day, g in d.groupby("planning_day_kst"):
        s = g.groupby("t")["improved_generation_kwh"].sum().sort_index()
        if len(s) == 48:
            fc[str(day)] = s.to_numpy(float) / 1000.0 * k
    return actual.clip(lower=0.0), fc, source_mw


def load_fcr_csv(paths: Iterable[str | Path]) -> pd.Series:
    """KPX 예비력 가격 → 주파수제어예비력 시간 평균 (원/kWh). 두 형식을 받는다.

    - 공공데이터포털 15145101 원본 CSV (거래일자, 예비력 구분, 01시_1 ~ 24시_4):
      01시_1~4 구간 → 00:00. 네 구간 중 하나라도 없으면 그 시간은 결측.
    - ``python -m vpp.market.reserve_prices`` 가 만든 시간 CSV (ts, fcr_price_krw_per_kwh).
      원본 엑셀(실시간예비력가격_*.xlsx)은 그 명령으로 먼저 변환한다.
    겹치는 시각은 뒤에 준 파일 값을 쓴다.
    """
    parts = []
    for path in paths:
        raw = pd.read_csv(path, encoding="utf-8-sig")
        if {"ts", "fcr_price_krw_per_kwh"} <= set(raw.columns):
            ts = _naive(raw["ts"])
            vals = raw["fcr_price_krw_per_kwh"].astype(float).to_numpy()
            parts.append(pd.Series(vals, index=pd.DatetimeIndex(ts)))
            continue
        raw = raw[raw.iloc[:, 1].astype(str).str.strip() == "주파수제어예비력"]
        cols = [c for c in raw.columns if "시_" in str(c)]
        if len(cols) != 96:
            raise ValueError(f"예비력 CSV는 96구간 열이 필요합니다: {path}")
        v = raw[cols].to_numpy(float).reshape(len(raw), 24, 4)
        hourly = v.mean(axis=2)  # 하나라도 NaN 이면 NaN
        days = pd.to_datetime(raw.iloc[:, 0].astype(str))
        ts = [day + pd.Timedelta(hours=h) for day in days for h in range(24)]
        parts.append(pd.Series(hourly.ravel(), index=pd.DatetimeIndex(ts)))
    if not parts:
        raise ValueError("FCR 파일이 없습니다.")
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="last")].rename("fcr_price_krw_per_kwh")


def _naive(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts.astype(str).str[:19])
    return t


def load_inputs(
    solar_path: str | Path,
    market_path: str | Path,
    forecast_path: str | Path | None,
    rec_config: str | Path,
    *,
    solar_col: str = "solar_traded_mwh",
    solar_capacity_mw: float = 5.7,
    scale_year: str = "2025",
    generation_path: str | Path | None = None,
    fcr_paths: Iterable[str | Path] = (),
) -> Inputs:
    """태양광 실측은 두 가지 중 하나.

    - generation_path 가 있으면 포트폴리오 실측(발전량 예측 평가 파일)을 5.7MW로 비례 환산
    - 없으면 solar_path 의 scale_year 최대값이 solar_capacity_mw 가 되도록 비례 환산
    """
    notes: dict = {}
    gen_fc: dict[str, np.ndarray] = {}
    if generation_path is not None:
        s, gen_fc, src_mw = load_generation(generation_path, solar_capacity_mw)
        notes["solar_source"] = f"portfolio {src_mw:.3f}MW -> {solar_capacity_mw}MW"
    else:
        sol = pd.read_csv(solar_path)
        sol["ts"] = _naive(sol["ts"])
        s = sol.set_index("ts")[solar_col].astype(float)
        peak = s[scale_year].max()
        if not peak > 0:
            raise ValueError(f"{scale_year}년 태양광 최대값이 0입니다.")
        s = (s / peak * solar_capacity_mw).clip(lower=0.0)
        notes["solar_source"] = f"{Path(solar_path).name} {scale_year} max -> {solar_capacity_mw}MW"
    mk = pd.read_csv(market_path)
    mk["ts"] = _naive(mk["ts"])
    mk = mk.set_index("ts")
    hourly = pd.DataFrame({"solar_mwh": s}).join(
        pd.DataFrame(
            {
                "smp_krw_per_mwh": mk["smp_da"].astype(float) * 1000.0,
                "smp_rt_krw_per_mwh": mk["smp_rt_mean"].astype(float) * 1000.0
                if "smp_rt_mean" in mk
                else np.nan,
            }
        ),
        how="outer",
    )
    hourly = hourly[~hourly.index.duplicated()].sort_index()
    rec = RecParams.from_yaml(rec_config)
    months = (
        pd.PeriodIndex(rec.price_table["month"], freq="M") if rec.price_table is not None else None
    )
    ts_for_rec = pd.Series(hourly.index)
    if months is not None:
        last = months.max().to_timestamp(how="end").floor("h")
        first = months.min().to_timestamp()
        ts_for_rec = ts_for_rec.clip(lower=first, upper=last)
    hourly["rec_krw_per_mwh"] = rec.adder_krw_per_mwh(ts_for_rec)
    fc: dict[str, np.ndarray] = {}
    if forecast_path is not None:
        f = pd.read_csv(forecast_path)
        col = "smp_hat_krw_per_mwh" if "smp_hat_krw_per_mwh" in f.columns else "smp_hat"
        scale = 1.0 if col.endswith("_krw_per_mwh") else 1000.0
        f["t"] = _naive(f["target_time"])
        for d, g in f.groupby("issue_date"):
            g = g.sort_values("t")
            if len(g) == 48:
                fc[str(d)] = g[col].to_numpy(float) * scale
    fcr_paths = list(fcr_paths)
    if fcr_paths:
        fcr = load_fcr_csv(fcr_paths).reindex(hourly.index)
        have = fcr.notna()
        by_hour = fcr.groupby(fcr.index.hour).mean()
        filled = fcr.fillna(pd.Series(fcr.index.hour.map(by_hour), index=fcr.index))
        hourly["fcr_price_krw_per_kwh"] = filled
        hourly["fcr_observed"] = have
        notes["fcr"] = "observed where available, else hour-of-day mean of observed"
    return Inputs(hourly, fc, solar_capacity_mw, gen_fc, notes)


def run_case(
    inputs: Inputs,
    cfg: pe.EconomicsConfig,
    base_ess: EssParams,
    case: LineCase,
    battery_key: str,
    operation: str,
    days: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """확정 24시간씩 이어 붙인 시간별 결과.

    forecast/perfect: operating_row_from_hourly 입력 형식.
    forecast_exec: 실행 운전 결과 (committed_sell_mwh, supply_mwh, charge/discharge 등).
    """
    if operation not in OPERATIONS:
        raise ValueError(f"operation 은 {OPERATIONS} 중 하나입니다.")
    if operation == EXEC:
        return _run_exec(inputs, cfg, base_ess, case, battery_key, days)
    h = inputs.hourly
    ess = None
    energy = 0.0
    if battery_key != BASELINE:
        ess = ess_for(base_ess, cfg, cfg.batteries[battery_key], case)
        energy = ess.initial_energy_mwh
    parts = []
    for d in days:
        idx = pd.date_range(d, periods=48, freq="h")
        w = h.reindex(idx)
        act = w["smp_krw_per_mwh"].to_numpy(float)
        if np.isnan(act[:24]).any():
            continue
        if operation == "perfect":
            plan = act.copy()
        else:
            key = (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if key not in inputs.forecast:
                continue
            plan = inputs.forecast[key].copy()
        plan = np.where(np.isnan(plan), np.nanmean(plan), plan)
        gen = w["solar_mwh"].fillna(0.0).to_numpy(float)
        rec = w["rec_krw_per_mwh"].ffill().bfill().to_numpy(float)
        cap_frame = attach_line_export_cap(
            pd.DataFrame(
                {
                    "cap": line_cap_mw(case, idx, gen, inputs.solar_capacity_mw),
                    "dt_hours": 1.0,
                }
            ),
            cap_mw_col="cap",
        )
        cap = cap_frame["export_cap_mwh"].to_numpy(float)
        if ess is None:
            sell = np.where(plan[:24] + rec[:24] > 0, np.minimum(gen[:24], cap[:24]), 0.0)
            ch = dis = np.zeros(24)
        else:
            res = solve_window(
                gen, plan, np.ones(48), ess, energy, export_cap_mwh=cap, rec_krw_per_mwh=rec
            )
            sell, ch, dis = res.sell_mwh[:24], res.charge_mwh[:24], res.discharge_mwh[:24]
            energy = float(res.energy_end_mwh[23])
        parts.append(
            pd.DataFrame(
                {
                    "ts": idx[:24],
                    "price_krw_per_mwh": act[:24],
                    "rec_krw_per_mwh": rec[:24],
                    "sell_mwh": sell,
                    "charge_mwh": ch,
                    "discharge_mwh": dis,
                }
            )
        )
    if not parts:
        raise ValueError(f"계산할 날이 없습니다: {case.name} {battery_key} {operation}")
    return pd.concat(parts, ignore_index=True)


def _run_exec(
    inputs: Inputs,
    cfg: pe.EconomicsConfig,
    base_ess: EssParams,
    case: LineCase,
    battery_key: str,
    days: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """발전량·SMP 예측으로 계획 → 실측 발전량으로 실행 운전."""
    if not inputs.generation_forecast:
        raise ValueError("forecast_exec 에는 발전량 예측(--generation)이 필요합니다.")
    h = inputs.hourly
    ess = None
    energy = 0.0
    if battery_key != BASELINE:
        ess = ess_for(base_ess, cfg, cfg.batteries[battery_key], case)
        energy = ess.initial_energy_mwh
    parts = []
    for d in days:
        idx = pd.date_range(d, periods=48, freq="h")
        w = h.reindex(idx)
        act = w["smp_krw_per_mwh"].to_numpy(float)
        actual_gen = w["solar_mwh"].to_numpy(float)[:24]
        smp_key = (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        gen_key = d.strftime("%Y-%m-%d")
        if (
            np.isnan(act[:24]).any()
            or np.isnan(actual_gen).any()
            or smp_key not in inputs.forecast
            or gen_key not in inputs.generation_forecast
        ):
            continue
        plan = inputs.forecast[smp_key].copy()
        plan = np.where(np.isnan(plan), np.nanmean(plan), plan)
        gen_fc = inputs.generation_forecast[gen_key]
        rec = w["rec_krw_per_mwh"].ffill().bfill().to_numpy(float)
        # 한전도 D-1 예측으로 허용범위를 낸다고 보고, 동적 상한은 발전량 예측으로 계산
        cap = line_cap_mw(case, idx, gen_fc, inputs.solar_capacity_mw).astype(float)
        if ess is None:
            sell = np.where(plan[:24] + rec[:24] > 0, np.minimum(gen_fc[:24], cap[:24]), 0.0)
            supply = np.minimum(actual_gen, sell)  # 약속보다 많이는 팔지 않는다
            ex = pd.DataFrame(
                {
                    "committed_sell_mwh": sell,
                    "supply_mwh": supply,
                    "charge_mwh": 0.0,
                    "discharge_mwh": 0.0,
                    "actual_generation_mwh": actual_gen,
                    "forecast_generation_mwh": gen_fc[:24],
                    "shortfall_mwh": np.maximum(sell - actual_gen, 0.0),
                    "curtailment_mwh": np.maximum(actual_gen - sell, 0.0),
                }
            )
        else:
            res = solve_window(
                gen_fc, plan, np.ones(48), ess, energy, export_cap_mwh=cap, rec_krw_per_mwh=rec
            )
            committed = pd.DataFrame(
                {
                    "ts": idx[:24],
                    "sell_mwh": res.sell_mwh[:24],
                    "charge_mwh": res.charge_mwh[:24],
                    "discharge_mwh": res.discharge_mwh[:24],
                    "price_krw_per_mwh": plan[:24],
                    "rec_krw_per_mwh": rec[:24],
                    "generation_mwh": gen_fc[:24],
                    "energy_end_mwh": res.energy_end_mwh[:24],
                }
            )
            out = execute_plan(committed, actual_gen, energy, ess, EXEC_POLICY)
            energy = out.energy_end_mwh
            ex = out.hourly.drop(columns=["ts", "rec_krw_per_mwh"])
        ex.insert(0, "ts", idx[:24])
        ex["price_krw_per_mwh"] = act[:24]
        ex["realtime_price_krw_per_mwh"] = w["smp_rt_krw_per_mwh"].to_numpy(float)[:24]
        ex["rec_krw_per_mwh"] = rec[:24]
        if "fcr_price_krw_per_kwh" in w:
            ex["fcr_price_krw_per_kwh"] = w["fcr_price_krw_per_kwh"].to_numpy(float)[:24]
        parts.append(ex)
    if not parts:
        raise ValueError(f"계산할 날이 없습니다: {case.name} {battery_key} {EXEC}")
    return pd.concat(parts, ignore_index=True)


def settle_exec(hourly: pd.DataFrame, ess: EssParams, resource_capacity_mw: float) -> pd.DataFrame:
    """실행 운전 결과를 하루전·실시간 이중가격 + 임밸런스 페널티로 정산한다.

    정산 규칙은 vpp.hourly_settlement (2026-04 시행 별표33, 허용오차 6% / 1.5%).
    """
    ts = pd.DatetimeIndex(hourly["ts"]).tz_localize("Asia/Seoul")
    execution = pd.DataFrame(
        {
            "ts": ts,
            "dt_hours": 1.0,
            "committed_sell_mwh": np.maximum(hourly["committed_sell_mwh"].to_numpy(float), 0.0),
            "supply_mwh": np.maximum(hourly["supply_mwh"].to_numpy(float), 0.0),
            "charge_mwh": np.maximum(hourly["charge_mwh"].to_numpy(float), 0.0),
            "discharge_mwh": np.maximum(hourly["discharge_mwh"].to_numpy(float), 0.0),
            "rec_krw_per_mwh": hourly["rec_krw_per_mwh"].to_numpy(float),
        }
    )
    market = pd.DataFrame(
        {
            "ts": ts,
            "smp_da": hourly["price_krw_per_mwh"].to_numpy(float) / 1000.0,
            "smp_rt_mean": hourly["realtime_price_krw_per_mwh"].to_numpy(float) / 1000.0,
        }
    )
    # MILP 해의 -1e-12 같은 수치 오차를 0으로 (정산 입력은 음수를 받지 않는다)
    committed_mw = np.maximum(hourly["committed_sell_mwh"].to_numpy(float), 0.0)  # dt = 1h
    fcr = (
        hourly["fcr_price_krw_per_kwh"].to_numpy(float)
        if "fcr_price_krw_per_kwh" in hourly
        else np.full(len(hourly), np.nan)
    )
    dispatch = pd.DataFrame(
        {
            "ts": ts,
            "resource_capacity_mw": resource_capacity_mw,
            "set_point_mw": committed_mw,
            "available_capacity_mw": committed_mw,
            "fcr_price_krw_per_kwh": fcr,
            "min_offer_price_krw_per_kwh": np.nan,
            "offer_floor_krw_per_kwh": np.nan,
            "dispatch_source": "scenario",
            "dispatch_note": DISPATCH_NOTE,
        }
    )
    params = HourlySettlementParams(resource_id="distribution_ess_v0")
    return settle_hourly(execution, market, ess, params=params, dispatch=dispatch)


def operating_row_from_settlement(
    settled: pd.DataFrame,
    *,
    case: str,
    battery: str,
    operation: str,
    solar_capacity_credit_mw: float,
    ess_capacity_credit_mw: float,
    rcp_krw_per_kwh_h: float,
) -> dict:
    """정산 결과 → 운영 결과 행. 열화비(wear)는 빼지 않는다.

    시간마다 (하루전 확정분 + 실시간 편차분) 에너지 수익과 REC를 실제 공급량 중 방전 비율로
    태양광 몫과 ESS 몫으로 나눈다. 페널티는 따로 둔다 (표준 배분에서 VPP 100%).
    """
    supply = settled["supply_mwh"].to_numpy(float)
    dis = np.minimum(settled["discharge_mwh"].to_numpy(float), supply)
    share = np.divide(dis, supply, out=np.zeros_like(supply), where=supply > 1e-12)
    value = settled["energy_revenue_krw"].to_numpy(float) + settled["rec_krw"].to_numpy(float)
    direct = supply - dis
    to_krw = 1000.0 * rcp_krw_per_kwh_h
    penalty = settled["imbalance_penalty_krw"].to_numpy(float)
    solar_rev = float(np.nansum(value * (1 - share)))
    ess_rev = float(np.nansum(value * share))
    solar_cp = float(np.minimum(direct, solar_capacity_credit_mw).sum() * to_krw)
    ess_cp = float(np.minimum(dis, ess_capacity_credit_mw).sum() * to_krw)
    pen = float(np.nansum(penalty))
    ch = settled["charge_mwh"].to_numpy(float)
    return {
        "case": case,
        "battery": battery,
        "operation": operation,
        "market_revenue_krw": solar_rev + ess_rev + solar_cp - pen,
        "solar_to_vpp_mwh": float(direct.sum() + ch.sum()),
        "ess_charge_mwh": float(ch.sum()),
        "ess_discharge_mwh": float(settled["discharge_mwh"].sum()),
        "days": int(pd.to_datetime(settled["ts"]).dt.date.nunique()),
        "solar_direct_revenue_krw": solar_rev,
        "ess_discharge_revenue_krw": ess_rev,
        "solar_capacity_payment_krw": solar_cp,
        "ess_capacity_payment_krw": ess_cp,
        "imbalance_penalty_krw": pen,
        "deviation_krw": float(np.nansum(settled["deviation_krw"])),
        "shortfall_mwh": float(np.maximum(-settled["sales_deviation_mwh"], 0).sum()),
        "penalized_mwh": float(np.nansum(settled["penalized_mwh"])),
        "penalty_missing_hours": int(np.isnan(penalty).sum()),
    }


def operating_row(
    inputs: Inputs,
    cfg: pe.EconomicsConfig,
    base_ess: EssParams,
    case: LineCase,
    battery_key: str,
    operation: str,
    days: list[pd.Timestamp],
    rcp_krw_per_kwh_h: float = JEJU_RCP_KRW_PER_KWH_H,
) -> dict:
    hourly = run_case(inputs, cfg, base_ess, case, battery_key, operation, days)
    power = cfg.ess_power_mw if case.ess_power_mw is None else case.ess_power_mw
    energy = cfg.ess_energy_mwh if case.ess_energy_mwh is None else case.ess_energy_mwh
    kw = {
        "case": case.name,
        "battery": battery_key,
        "operation": operation,
        "solar_capacity_credit_mw": inputs.solar_capacity_mw * SOLAR_EFCR,
        "ess_capacity_credit_mw": 0.0 if battery_key == BASELINE else power * ESS_EFCR,
        "rcp_krw_per_kwh_h": rcp_krw_per_kwh_h,
    }
    if operation == EXEC:
        ess = (
            base_ess
            if battery_key == BASELINE
            else ess_for(base_ess, cfg, cfg.batteries[battery_key], case)
        )
        settled = settle_exec(hourly, ess, inputs.solar_capacity_mw)
        row = operating_row_from_settlement(settled, **kw)
    else:
        row = pe.operating_row_from_hourly(hourly, **kw)
    row["line_case"] = dataclasses.replace(case, ess_power_mw=None, ess_energy_mwh=None).name
    if case.ess_power_mw is not None:
        row["ess_power_mw"] = power
        row["ess_energy_mwh"] = energy
    return row


_WORKER: dict = {}


def _init_worker(inputs, cfg, base_ess, days, rcp):
    _WORKER.update(inputs=inputs, cfg=cfg, base_ess=base_ess, days=days, rcp=rcp)


def _job(args):
    case, battery_key, operation = args
    w = _WORKER
    return operating_row(
        w["inputs"], w["cfg"], w["base_ess"], case, battery_key, operation, w["days"], w["rcp"]
    )


def run_all(
    inputs: Inputs,
    cfg: pe.EconomicsConfig,
    base_ess: EssParams,
    jobs: list[tuple[LineCase, str, str]],
    days: list[pd.Timestamp],
    *,
    workers: int = 1,
    rcp_krw_per_kwh_h: float = JEJU_RCP_KRW_PER_KWH_H,
) -> pd.DataFrame:
    init = (inputs, cfg, base_ess, days, rcp_krw_per_kwh_h)
    if workers <= 1:
        _init_worker(*init)
        rows = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=init) as ex:
            rows = list(ex.map(_job, jobs))
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def build_jobs(
    batteries: list[str],
    *,
    fixed_caps: list[float],
    dynamic_caps: list[float],
    existing_pv_mw: float,
    operations: list[str],
    sizes: list[tuple[float, float]] | None = None,
) -> list[tuple[LineCase, str, str]]:
    cases = [LineCase("fixed", c) for c in fixed_caps]
    cases += [LineCase("dynamic", c, existing_pv_mw) for c in dynamic_caps]
    if sizes:
        cases = [
            dataclasses.replace(c, ess_power_mw=p, ess_energy_mwh=e)
            for c in cases
            for p, e in sizes
        ]
    jobs = []
    for c in cases:
        for op in operations:
            jobs += [(c, b, op) for b in batteries]
            if not sizes or (c.ess_power_mw, c.ess_energy_mwh) == sizes[0]:
                jobs.append(
                    (dataclasses.replace(c, ess_power_mw=None, ess_energy_mwh=None), BASELINE, op)
                )
    # 같은 기준선이 여러 번 생기지 않게
    seen, out = set(), []
    for c, b, op in jobs:
        key = (c.name, b, op)
        if key not in seen:
            seen.add(key)
            out.append((c, b, op))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="배전망 ESS 1년 운영 백테스트 → 운영 결과 CSV")
    p.add_argument("--solar", help="시간별 태양광 CSV (ts, solar_traded_mwh). --generation 없을 때")
    p.add_argument(
        "--generation",
        help="발전량 예측 평가 parquet (solar_improved_2025_with_actual). 주면 이 실측을 쓰고 "
        "forecast_exec 운전을 할 수 있다",
    )
    p.add_argument("--fcr", nargs="*", default=[], help="KPX 예비력 가격 CSV (여러 개 가능)")
    p.add_argument("--solar-col", default="solar_traded_mwh")
    p.add_argument("--market", default="data/processed/market/market_hourly.csv")
    p.add_argument("--forecast", help="SMP v6 예측 CSV (smp_forecast 출력). 없으면 perfect 만")
    p.add_argument("--rec", default="config/rec_v0_monthly.yaml")
    p.add_argument("--economics-config", default="config/project_economics_v0.yaml")
    p.add_argument("--ess-config", default="config/ess_v0_1.yaml", help="SoC 범위 등 기본 ESS 설정")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--batteries", help="쉼표 구분 (기본: 설정의 모든 배터리)")
    p.add_argument("--operations", help="쉼표 구분. 기본: forecast,perfect (+forecast_exec)")
    p.add_argument("--fixed-caps", default="1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0")
    p.add_argument("--dynamic-caps", default="1.0,1.5,2.0,2.5")
    p.add_argument("--existing-pv-mw", type=float, default=5.0)
    p.add_argument("--sizes", help="ESS 용량 후보 'MW:MWh' 쉼표 구분 (예: 2:10,3:15,4:20)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True, help="운영 결과 CSV 경로")
    a = p.parse_args(argv)

    cfg = pe.EconomicsConfig.from_yaml(a.economics_config)
    base_ess = EssParams.from_yaml(a.ess_config)
    if a.solar is None and a.generation is None:
        raise SystemExit("--solar 또는 --generation 중 하나가 필요합니다.")
    inputs = load_inputs(
        a.solar,
        a.market,
        a.forecast,
        a.rec,
        solar_col=a.solar_col,
        solar_capacity_mw=cfg.solar_mw,
        generation_path=a.generation,
        fcr_paths=a.fcr,
    )
    default_ops = "forecast,perfect" + (f",{EXEC}" if a.generation else "")
    ops = [o for o in (a.operations or default_ops).split(",") if o]
    if EXEC in ops and not inputs.generation_forecast:
        raise SystemExit("forecast_exec 에는 --generation 이 필요합니다.")
    if "forecast" in ops and not inputs.forecast:
        raise SystemExit("--forecast 없이 forecast 운전은 할 수 없습니다.")
    sizes = None
    if a.sizes:
        sizes = [tuple(float(x) for x in s.split(":")) for s in a.sizes.split(",")]
    batteries = a.batteries.split(",") if a.batteries else list(cfg.batteries)
    jobs = build_jobs(
        batteries,
        fixed_caps=_floats(a.fixed_caps),
        dynamic_caps=_floats(a.dynamic_caps),
        existing_pv_mw=a.existing_pv_mw,
        operations=ops,
        sizes=sizes,
    )
    days = list(pd.date_range(a.start, a.end, freq="D"))
    out = run_all(inputs, cfg, base_ess, jobs, days, workers=a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"{len(out)}행 저장: {a.out}")


if __name__ == "__main__":
    main()
