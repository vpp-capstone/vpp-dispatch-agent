"""배전망 ESS 1년 운영 백테스트 → 사업성 입력 (운영 결과 CSV).

실행 쪽(``distribution_scenario``)과 사업성 쪽(``project_economics``)을 잇는다.

    python -m vpp.distribution_backtest \
        --solar <시간별 태양광 CSV> --forecast <SMP v6 예측 CSV> \
        --start 2025-01-01 --end 2025-12-31 \
        --out docs/results/distribution_ess_operating_<날짜>.csv

한 조합(선로 조건 x 배터리 x 운전 방식)마다 하루씩 48시간 MILP를 풀고 앞 24시간을 확정한다.

- 계획 가격: ``forecast`` = 전날 11시에 낸 SMP v6 예측(48시간), ``perfect`` = 정답 SMP.
  어느 쪽이든 확정한 판매량은 실제 SMP로 정산한다.
- 판매 1MWh 가치 = SMP + REC 가산(``config/rec_v0_monthly.yaml``).
  표 밖의 달은 표의 처음·마지막 달 값을 쓴다.
- 발전량은 실측을 그대로 쓴다 (발전량 예측 오차·임밸런스 페널티는 아직 넣지 않는다).
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
from vpp.optimization import EssParams, solve_window
from vpp.rec import RecParams

DAY_HOURS = (8, 18)  # fixed 상한이 걸리는 시간 [시작, 끝)
SOLAR_EFCR = 0.0384  # 태양광 용량정산 실효용량 비율
ESS_EFCR = 0.6563  # 4시간 ESS 실효용량 비율
JEJU_RCP_KRW_PER_KWH_H = 22.05
BASELINE = pe.BASELINE_BATTERY


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

    hourly: pd.DataFrame  # index ts: solar_mwh, smp_krw_per_mwh, rec_krw_per_mwh
    forecast: dict[str, np.ndarray]  # 발행일(YYYY-MM-DD) -> 다음 날 0시부터 48시간 예측
    solar_capacity_mw: float


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
) -> Inputs:
    """태양광은 scale_year 최대값이 solar_capacity_mw 가 되도록 비례 환산한다."""
    sol = pd.read_csv(solar_path)
    sol["ts"] = _naive(sol["ts"])
    s = sol.set_index("ts")[solar_col].astype(float)
    peak = s[scale_year].max()
    if not peak > 0:
        raise ValueError(f"{scale_year}년 태양광 최대값이 0입니다.")
    s = (s / peak * solar_capacity_mw).clip(lower=0.0)
    mk = pd.read_csv(market_path)
    mk["ts"] = _naive(mk["ts"])
    smp = mk.set_index("ts")["smp_da"].astype(float) * 1000.0
    hourly = pd.DataFrame({"solar_mwh": s}).join(smp.rename("smp_krw_per_mwh"), how="outer")
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
    return Inputs(hourly, fc, solar_capacity_mw)


def run_case(
    inputs: Inputs,
    cfg: pe.EconomicsConfig,
    base_ess: EssParams,
    case: LineCase,
    battery_key: str,
    operation: str,
    days: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """확정 24시간씩 이어 붙인 시간별 결과 (operating_row_from_hourly 입력 형식)."""
    if operation not in ("forecast", "perfect"):
        raise ValueError("operation 은 forecast 또는 perfect 입니다.")
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
    row = pe.operating_row_from_hourly(
        hourly,
        case=case.name,
        battery=battery_key,
        operation=operation,
        solar_capacity_credit_mw=inputs.solar_capacity_mw * SOLAR_EFCR,
        ess_capacity_credit_mw=0.0 if battery_key == BASELINE else power * ESS_EFCR,
        rcp_krw_per_kwh_h=rcp_krw_per_kwh_h,
    )
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
    p.add_argument("--solar", required=True, help="시간별 태양광 CSV (ts, solar_traded_mwh)")
    p.add_argument("--solar-col", default="solar_traded_mwh")
    p.add_argument("--market", default="data/processed/market/market_hourly.csv")
    p.add_argument("--forecast", help="SMP v6 예측 CSV (smp_forecast 출력). 없으면 perfect 만")
    p.add_argument("--rec", default="config/rec_v0_monthly.yaml")
    p.add_argument("--economics-config", default="config/project_economics_v0.yaml")
    p.add_argument("--ess-config", default="config/ess_v0_1.yaml", help="SoC 범위 등 기본 ESS 설정")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--batteries", help="쉼표 구분 (기본: 설정의 모든 배터리)")
    p.add_argument("--operations", default="forecast,perfect")
    p.add_argument("--fixed-caps", default="1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0")
    p.add_argument("--dynamic-caps", default="1.0,1.5,2.0,2.5")
    p.add_argument("--existing-pv-mw", type=float, default=5.0)
    p.add_argument("--sizes", help="ESS 용량 후보 'MW:MWh' 쉼표 구분 (예: 2:10,3:15,4:20)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True, help="운영 결과 CSV 경로")
    a = p.parse_args(argv)

    cfg = pe.EconomicsConfig.from_yaml(a.economics_config)
    base_ess = EssParams.from_yaml(a.ess_config)
    inputs = load_inputs(
        a.solar, a.market, a.forecast, a.rec, solar_col=a.solar_col, solar_capacity_mw=cfg.solar_mw
    )
    ops = [o for o in a.operations.split(",") if o]
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
