"""Save hourly execution from the existing engine, reconciled to the published year.

No dispatch, settlement or economic formula is implemented here. A candidate is run
chronologically for the whole year so that SOC carries across midnight. Date changes
only read that result. Cache identity includes inputs, configurations and engine code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from vpp import distribution_backtest as db
from vpp import project_economics as pe
from vpp.optimization import EssParams

YEAR = pd.date_range("2025-01-01", periods=8760, freq="h")
TOTALS = (
    "ess_charge_mwh",
    "ess_discharge_mwh",
    "market_revenue_krw",
    "solar_to_vpp_mwh",
    "solar_direct_revenue_krw",
    "ess_discharge_revenue_krw",
    "solar_capacity_payment_krw",
    "ess_capacity_payment_krw",
    "imbalance_penalty_krw",
    "deviation_krw",
    "shortfall_mwh",
    "penalized_mwh",
    "penalty_missing_hours",
)
DISPLAY = (
    "actual_generation_mwh",
    "charge_mwh",
    "discharge_mwh",
    "supply_mwh",
    "energy_start_mwh",
    "energy_end_mwh",
    "soc_end",
    "vpp_income_krw",
    "vpp_ess_krw",
    "vpp_solar_cp_krw",
    "vpp_penalty_krw",
    "expected_plan_vpp_income_krw",
    "expected_plan_vpp_ess_krw",
    "expected_plan_vpp_solar_cp_krw",
    "expected_plan_vpp_penalty_krw",
    "fixed_plan_vpp_income_krw",
    "fixed_plan_vpp_ess_krw",
    "fixed_plan_vpp_solar_cp_krw",
    "fixed_plan_vpp_penalty_krw",
    "fixed_plan_supply_mwh",
    "fixed_plan_charge_mwh",
    "fixed_plan_discharge_mwh",
    "fixed_plan_energy_start_mwh",
    "fixed_plan_energy_end_mwh",
    "fixed_plan_soc_end",
    "fixed_plan_shortfall_mwh",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(root: Path) -> dict:
    config = root / "config/distribution_replay.yaml"
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    paths = {
        k: root / raw[k] for k in ("generation", "forecast", "market", "rec", "ess", "economics")
    }
    paths["fcr"] = [root / p for p in raw["fcr"]]
    all_paths = [*paths["fcr"], *(v for k, v in paths.items() if k != "fcr"), config]
    missing = [str(p) for p in all_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("필요한 입력 파일: " + ", ".join(missing))
    rec = yaml.safe_load(paths["rec"].read_text(encoding="utf-8"))
    if rec.get("price_table"):
        # Same resolution used by RecParams.from_yaml.
        table = Path(rec["price_table"])
        if not table.is_absolute():
            table = paths["rec"].resolve().parents[1] / table
        all_paths.append(table)
    paths["dependencies"] = all_paths
    return paths


def context(root: Path, row: pd.Series) -> dict:
    if row.operation != db.EXEC or row.days != 365:
        raise ValueError("forecast_exec 기준 365일 결과만 연결할 수 있습니다.")
    paths = sources(root)["dependencies"]
    paths += list((root / "src/vpp").glob("*.py"))
    paths += [Path(__file__)]
    return {
        "schema": 2,
        "candidate": str(row.id),
        "operation": db.EXEC,
        "basis": "standard",
        "ess_capacity_payment": False,
        "expected": {k: float(row[k]) for k in TOTALS},
        "sources": {
            str(p.relative_to(root) if p.is_relative_to(root) else p): _sha(p) for p in paths
        },
    }


def cache_path(root: Path, stamp: dict) -> Path:
    key = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:24]
    return root / "data/processed/distribution_replay" / key


def reconcile(annual: dict, row: pd.Series) -> None:
    differences = []
    for key in TOTALS:
        actual, expected = float(annual[key]), float(row[key])
        # Published CSVs round to two decimal places; no material divergence is accepted.
        if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > 0.011:
            differences.append(f"{key}: 재실행 {actual:,.2f} / 보고 {expected:,.2f}")
    if differences:
        raise ValueError(
            "연간 원본 결과와 일치하지 않아 연결을 중단했습니다. " + "; ".join(differences)
        )


def validate_hours(hourly: pd.DataFrame, capacity: float, expected=YEAR) -> None:
    ts = pd.DatetimeIndex(pd.to_datetime(hourly.ts))
    if not ts.equals(pd.DatetimeIndex(expected)):
        raise ValueError("시간별 결과에 누락·중복·순서 오류가 있습니다.")
    if not np.isfinite(hourly[list(DISPLAY)].to_numpy(float)).all():
        raise ValueError("시간별 실행·정산 값에 결측이 있습니다.")
    start, end = hourly.energy_start_mwh.to_numpy(), hourly.energy_end_mwh.to_numpy()
    if (
        capacity <= 0
        or np.any(start < -1e-6)
        or np.any(end < -1e-6)
        or np.any(start > capacity + 1e-6)
        or np.any(end > capacity + 1e-6)
        or not np.allclose(end[:-1], start[1:], atol=1e-6, rtol=0)
        or not np.allclose(hourly.soc_end, end / capacity, atol=1e-6, rtol=0)
    ):
        raise ValueError("배터리 잔량 또는 시간 간 SOC 연결을 확인하세요.")
    fixed_start = hourly.fixed_plan_energy_start_mwh.to_numpy(float)
    fixed_end = hourly.fixed_plan_energy_end_mwh.to_numpy(float)
    if (
        np.any(fixed_start < -1e-6)
        or np.any(fixed_end < -1e-6)
        or np.any(fixed_start > capacity + 1e-6)
        or np.any(fixed_end > capacity + 1e-6)
        or not np.allclose(fixed_end[:-1], fixed_start[1:], atol=1e-6, rtol=0)
        or not np.allclose(hourly.fixed_plan_soc_end, fixed_end / capacity, atol=1e-6, rtol=0)
    ):
        raise ValueError("고정계획 사후 실행의 SOC 연결을 확인하세요.")


def standard_income(cfg, settled: pd.DataFrame, kw: dict) -> pd.DataFrame:
    """Apply the same saved-report allocation to one settlement scenario, hourly."""
    components = pd.DataFrame(
        [
            db.operating_row_from_settlement(settled.iloc[i : i + 1], **kw)
            for i in range(len(settled))
        ]
    )
    sh = cfg.standard_vpp_shares
    if any(
        sh.get(k, 0) != v
        for k, v in {
            "ess_energy": 1,
            "solar_energy": 0,
            "solar_capacity_payment": 0.5,
            "imbalance_penalty": 1,
        }.items()
    ):
        raise ValueError("수익 배분 설정이 보고서의 standard 기준과 다릅니다.")
    return pd.DataFrame(
        {
            "income": [
                pe._standard_view(
                    cfg, r, r.ess_charge_mwh + r.ess_discharge_mwh
                ).op.market_revenue_krw
                for _, r in components.iterrows()
            ],
            "ess": components.ess_discharge_revenue_krw * sh["ess_energy"],
            "solar_cp": (
                components.solar_capacity_payment_krw * sh["solar_capacity_payment"]
            ),
            "penalty": -components.imbalance_penalty_krw * sh["imbalance_penalty"],
        }
    )


def execute(inputs, cfg, base, case, battery, days):
    """Call existing calculation functions unchanged, including standard allocation."""
    ess = db.ess_for(base, cfg, cfg.batteries[battery], case)
    hourly = db.run_case(inputs, cfg, base, case, battery, db.EXEC, days)
    scenarios = {
        name: db.settle_exec(
            db.income_scenario_hourly(hourly, name), ess, inputs.solar_capacity_mw
        )
        for name in ("expected_plan", "fixed_plan", "executed")
    }
    if any(s.imbalance_penalty_krw.isna().any() for s in scenarios.values()):
        raise ValueError("예비력 가격이 없어 페널티 정산을 완료할 수 없습니다.")
    kw = dict(
        case=case.name,
        battery=battery,
        operation=db.EXEC,
        solar_capacity_credit_mw=inputs.solar_capacity_mw * db.SOLAR_EFCR,
        ess_capacity_credit_mw=case.ess_power_mw * db.ESS_EFCR,
        rcp_krw_per_kwh_h=db.JEJU_RCP_KRW_PER_KWH_H,
    )
    settled = scenarios["executed"]
    annual = db.operating_row_from_settlement(settled, **kw)
    allocated = {name: standard_income(cfg, value, kw) for name, value in scenarios.items()}
    for scenario, prefix in (
        ("expected_plan", "expected_plan_vpp"),
        ("fixed_plan", "fixed_plan_vpp"),
        ("executed", "vpp"),
    ):
        values = allocated[scenario]
        hourly[f"{prefix}_income_krw"] = values.income.to_numpy()
        hourly[f"{prefix}_ess_krw"] = values.ess.to_numpy()
        hourly[f"{prefix}_solar_cp_krw"] = values.solar_cp.to_numpy()
        hourly[f"{prefix}_penalty_krw"] = values.penalty.to_numpy()
    for col in ("energy_revenue_krw", "rec_krw", "deviation_krw", "imbalance_penalty_krw"):
        hourly[col] = settled[col].to_numpy()
    expected = pd.date_range(days[0], periods=len(days) * 24, freq="h")
    validate_hours(hourly, ess.capacity_mwh, expected)
    return hourly, annual, ess.capacity_mwh


def build(root: Path, row: pd.Series) -> tuple[pd.DataFrame, dict]:
    stamp = context(root, row)
    p = sources(root)
    cfg, base = pe.EconomicsConfig.from_yaml(p["economics"]), EssParams.from_yaml(p["ess"])
    inputs = db.load_inputs(
        None,
        p["market"],
        p["forecast"],
        p["rec"],
        generation_path=p["generation"],
        fcr_paths=p["fcr"],
        solar_capacity_mw=cfg.solar_mw,
    )
    if not inputs.hourly.reindex(YEAR).fcr_observed.fillna(False).all():
        raise ValueError("정식 결과와 같이 2025년 전체의 관측 예비력 가격이 필요합니다.")
    days = list(pd.date_range("2025-01-01", "2025-12-31"))
    for day in days:
        gen = inputs.generation_forecast.get(str(day.date()))
        smp = inputs.forecast.get(str((day - pd.Timedelta(days=1)).date()))
        if gen is None or smp is None or not np.isfinite(gen).all() or not np.isfinite(smp).all():
            raise ValueError(f"{day.date()}의 48시간 발전량·SMP 예측이 불완전합니다.")
    kind, cap = row.line_case.split("_")
    case = db.LineCase(
        "fixed" if kind == "cap" else "dynamic",
        float(cap[:-2]),
        ess_power_mw=float(row.ess_power_mw),
        ess_energy_mwh=float(row.ess_energy_mwh),
    )
    hourly, annual, capacity = execute(inputs, cfg, base, case, row.battery, days)
    reconcile(annual, row)
    validate_hours(hourly, capacity)
    if context(root, row) != stamp:
        raise ValueError("계산 중 입력 파일이 변경됐습니다. 다시 연결해 주세요.")
    folder = cache_path(root, stamp)
    folder.mkdir(parents=True, exist_ok=True)
    # Unique immutable data file + atomic manifest prevents readers seeing partial output.
    target = folder / f"hourly-{uuid4().hex}.parquet"
    hourly.to_parquet(target, index=False)
    manifest = {
        **stamp,
        "capacity_mwh": capacity,
        "annual": annual,
        "file": target.name,
        "sha256": _sha(target),
        "hours": len(hourly),
    }
    temp = folder / f"manifest-{uuid4().hex}.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(folder / "manifest.json")
    return hourly, manifest


def load(root: Path, row: pd.Series) -> tuple[pd.DataFrame, dict] | None:
    stamp = context(root, row)
    folder = cache_path(root, stamp)
    path = folder / "manifest.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    if any(meta.get(k) != v for k, v in stamp.items()):
        raise ValueError("선택한 설비와 시간별 결과의 기준이 다릅니다.")
    file = folder / Path(meta["file"]).name
    if _sha(file) != meta["sha256"]:
        raise ValueError("저장된 시간별 결과가 변경됐습니다. 다시 연결해 주세요.")
    hourly = pd.read_parquet(file)
    validate_hours(hourly, meta["capacity_mwh"])
    reconcile(meta["annual"], row)
    return hourly, meta


def day_payload(hourly: pd.DataFrame, meta: dict, date: str) -> dict:
    day = hourly[pd.to_datetime(hourly.ts).dt.strftime("%Y-%m-%d").eq(date)].copy()
    if len(day) != 24:
        raise ValueError("선택한 날짜의 24시간 결과가 없습니다.")
    for column in (
        "vpp_income_krw",
        "charge_mwh",
        "discharge_mwh",
        "vpp_ess_krw",
        "vpp_solar_cp_krw",
        "vpp_penalty_krw",
    ):
        day["cumulative_" + column] = day[column].cumsum()
    day["hour"] = pd.to_datetime(day.ts).dt.hour
    columns = [*DISPLAY, *(c for c in day if c.startswith("cumulative_")), "hour"]
    return {
        "candidate": meta["candidate"],
        "date": date,
        "operation": db.EXEC,
        "basis": "standard",
        "capacity": meta["capacity_mwh"],
        "initial_energy": float(day.energy_start_mwh.iloc[0]),
        "hours": day[columns].to_dict("records"),
    }


def main():
    parser = argparse.ArgumentParser(description="배전망 ESS 시간별 실행·정산 결과 연결")
    parser.add_argument("--candidate", default="cap_2.50MW|kr_notice_example|2:8")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from dashboard.distribution_display import load_display

    rows, _ = load_display(root)
    selected = rows[rows.id.eq(args.candidate)]
    if selected.empty:
        parser.error("저장된 결과에 없는 설비입니다.")
    hourly, meta = build(root, selected.iloc[0])
    print(f"{meta['candidate']}: {len(hourly)}시간, forecast_exec / standard, 연간 원본 대조 통과")
    print(f"VPP 귀속 수입: {hourly.vpp_income_krw.sum():,.2f}원")


if __name__ == "__main__":
    main()
