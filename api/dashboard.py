"""Read-only React adapter over verified, saved operator results.

No optimizer, model training, device command, or settlement formula runs here.
Archived distribution results are validated against their own manifest and published
annual totals, and explicitly labelled historical (not a current-input cache hit).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dashboard.distribution_display import load_display
from dashboard.distribution_operation import revenue_breakdown
from dashboard.operator_simulation import replay_payload
from dashboard.overview_data import load_experiment
from fastapi import APIRouter, HTTPException, Query, Response

from api.economics_pdf import build_economics_pdf
from dashboard import distribution_replay as replay
from vpp.baselines import plan_revenue_breakdown
from vpp.project_paths import resolve_project_path

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


def project_root() -> Path:
    return Path(os.environ.get("VPP_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    return float(value) if value is not None and pd.notna(value) and np.isfinite(value) else None


def economics_result(
    root: Path, selection: str
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series | None]:
    """Load one candidate's saved summary and annual cashflow without recalculation."""
    rows, _ = load_display(root)
    matched = rows[rows.id.eq(selection)]
    if matched.empty:
        raise LookupError("경제성 후보를 찾을 수 없습니다.")
    candidate = matched.iloc[0]
    folder = "economics_operating" if "operating" in candidate.source_file else "economics_sizing"
    summary_path = within(root, root / "docs/results" / folder / "economics_summary.csv")
    cashflow_path = within(root, root / "docs/results" / folder / "economics_cashflow.csv")
    if not summary_path.exists() or not cashflow_path.exists():
        raise FileNotFoundError("상세 경제성 결과 파일을 먼저 생성해야 합니다.")
    summary = pd.read_csv(summary_path)
    cashflows = pd.read_csv(cashflow_path)
    common = (
        summary["case"].eq(candidate["case"])
        & summary["battery"].eq(candidate.battery)
        & summary["operation"].eq("forecast_exec")
        & summary["basis"].eq("standard")
        & summary["revenue_scenario"].eq("base")
        & summary["ess_power_mw"].eq(candidate.ess_power_mw)
        & summary["ess_energy_mwh"].eq(candidate.ess_energy_mwh)
    )
    summary_cp = summary["ess_capacity_payment"].astype(str).str.lower().eq("true")
    summary_match = summary[common & ~summary_cp]
    capacity_match = summary[common & summary_cp]
    flow_cp = cashflows["ess_capacity_payment"].astype(str).str.lower().eq("true")
    flow_match = cashflows[
        cashflows["case"].eq(candidate["case"])
        & cashflows["battery"].eq(candidate.battery)
        & cashflows["ess_power_mw"].eq(candidate.ess_power_mw)
        & cashflows["ess_energy_mwh"].eq(candidate.ess_energy_mwh)
        & ~flow_cp
    ].copy()
    if len(summary_match) != 1 or len(capacity_match) > 1 or flow_match.empty:
        raise ValueError("선택 후보와 일치하는 상세 경제성 결과가 없습니다.")
    capacity = capacity_match.iloc[0] if len(capacity_match) == 1 else None
    return candidate, summary_match.iloc[0], flow_match.sort_values("year"), capacity


def within(root: Path, path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("결과 파일이 프로젝트 범위를 벗어났습니다.")
    return path


def releases(root: Path) -> tuple[dict, list[dict]]:
    status_file = root / "data/processed/operator_pipeline/status.json"
    status = read_json(status_file) if status_file.exists() else {}
    values = list((status.get("releases") or {}).values())
    if status.get("active") and not any(v.get("id") == status["active"]["id"] for v in values):
        values.insert(0, status["active"])
    return status, [r for r in values if r.get("backtest")]


def archives(root: Path, candidate: str) -> list[tuple[Path, dict]]:
    found = []
    for path in (root / "data/processed/distribution_replay").glob("*/manifest.json"):
        try:
            meta = read_json(path)
        except (OSError, ValueError):
            continue
        if meta.get("candidate") == candidate:
            found.append((path, meta))
    return sorted(found, key=lambda item: item[0].stat().st_mtime_ns, reverse=True)


def load_archive(root: Path, row: pd.Series) -> tuple[pd.DataFrame, dict, str]:
    found = archives(root, str(row.id))
    if not found:
        raise FileNotFoundError(
            "이 설비의 시간별 저장 결과가 없습니다. 기존 CLI에서 결과를 연결해 주세요."
        )
    # Do not silently fall back to an older result if the selected newest archive is corrupt.
    path, meta = found[0]
    if (
        meta.get("schema") != 2
        or meta.get("operation") != "forecast_exec"
        or meta.get("basis") != "standard"
        or meta.get("ess_capacity_payment") is not False
    ):
        raise ValueError("시간별 결과의 실행·정산 기준이 맞지 않습니다.")
    if number(meta.get("capacity_mwh")) != float(row.ess_energy_mwh):
        raise ValueError("선택 설비와 저장된 ESS 용량이 다릅니다.")
    file = within(path.parent, path.parent / meta["file"])
    if hashlib.sha256(file.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("시간별 결과의 파일 검증값이 다릅니다.")
    frame = pd.read_parquet(file)
    replay.validate_hours(frame, float(row.ess_energy_mwh))
    replay.reconcile(meta["annual"], row)
    needed = {
        "committed_sell_mwh",
        "planned_charge_mwh",
        "planned_discharge_mwh",
        "planned_energy_end_mwh",
        "forecast_generation_mwh",
        "shortfall_mwh",
        "fixed_plan_supply_mwh",
        "fixed_plan_charge_mwh",
        "fixed_plan_discharge_mwh",
        "fixed_plan_energy_end_mwh",
        "fixed_plan_soc_end",
        "fixed_plan_shortfall_mwh",
        "expected_plan_vpp_income_krw",
        "fixed_plan_vpp_income_krw",
    }
    if needed - set(frame):
        raise ValueError("저장 결과에 확정 계획 비교 열이 없습니다.")
    if not np.isfinite(frame[list(needed)].to_numpy(float)).all():
        raise ValueError("확정 계획 비교 열에 결측이 있습니다.")
    annual = revenue_breakdown(row)
    for column, key in (("vpp_ess_krw", "ess"), ("vpp_solar_cp_krw", "solar_cp"),
                        ("vpp_penalty_krw", "penalty"), ("vpp_income_krw", "total")):
        if abs(float(frame[column].sum()) - annual[key]) > 0.04:
            raise ValueError("시간별 귀속 수입이 연간 보고값과 일치하지 않습니다.")
    for prefix in ("expected_plan_vpp", "fixed_plan_vpp", "vpp"):
        if not np.allclose(
            frame[f"{prefix}_income_krw"],
            frame[f"{prefix}_ess_krw"]
            + frame[f"{prefix}_solar_cp_krw"]
            + frame[f"{prefix}_penalty_krw"],
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError(f"{prefix} 시간별 수입 합계가 구성 항목과 다릅니다.")
    return frame, meta, path.parent.name


def day_frame(frame: pd.DataFrame, selected: date) -> pd.DataFrame:
    ts = pd.to_datetime(frame.ts)
    block = frame[ts.dt.date.eq(selected)].copy()
    if len(block) != 24 or pd.to_datetime(block.ts).dt.hour.tolist() != list(range(24)):
        raise ValueError("선택일의 00–23시 결과가 완전하지 않습니다.")
    return block.reset_index(drop=True)


def dates(frame: pd.DataFrame) -> list[str]:
    return sorted(pd.to_datetime(frame.ts).dt.strftime("%Y-%m-%d").unique().tolist())


def distribution_snapshot(root: Path, candidate: str, selected: date) -> dict:
    rows, _ = load_display(root)
    match = rows[rows.id.eq(candidate)]
    if match.empty:
        raise LookupError("저장된 결과에 없는 설비입니다.")
    row = match.iloc[0]
    hourly, meta, archive_id = load_archive(root, row)
    block = day_frame(hourly, selected)
    hours = []
    for i, r in block.iterrows():
        hours.append(
            {
                "hour": i,
                "forecastGeneration": float(r.forecast_generation_mwh),
                "actualGeneration": float(r.actual_generation_mwh),
                "plannedSupply": float(r.committed_sell_mwh),
                "actualSupply": float(r.supply_mwh),
                "plannedCharge": float(r.planned_charge_mwh),
                "plannedDischarge": float(r.planned_discharge_mwh),
                "charge": float(r.charge_mwh),
                "discharge": float(r.discharge_mwh),
                "plannedSoc": float(r.planned_energy_end_mwh / row.ess_energy_mwh * 100),
                "soc": float(r.soc_end * 100),
                "shortfall": float(r.shortfall_mwh),
                "fixedSupply": float(r.fixed_plan_supply_mwh),
                "fixedCharge": float(r.fixed_plan_charge_mwh),
                "fixedDischarge": float(r.fixed_plan_discharge_mwh),
                "fixedSoc": float(r.fixed_plan_soc_end * 100),
                "fixedShortfall": float(r.fixed_plan_shortfall_mwh),
                "plannedIncome": float(r.expected_plan_vpp_income_krw),
                "fixedIncome": float(r.fixed_plan_vpp_income_krw),
                "income": float(r.vpp_income_krw),
            }
        )
    ledger = []
    for label, suffix in (
        ("ESS 귀속 에너지 수입", "ess_krw"),
        ("태양광 용량정산금 · 50%", "solar_cp_krw"),
        ("임밸런스 페널티", "penalty_krw"),
    ):
        ledger.append(
            {
                "label": label,
                "planned": float(block[f"expected_plan_vpp_{suffix}"].sum()),
                "fixed": float(block[f"fixed_plan_vpp_{suffix}"].sum()),
                "actual": float(block[f"vpp_{suffix}"].sum()),
            }
        )
    cutoff = pd.to_datetime(hourly.ts).dt.date.le(selected)
    monthly = hourly[cutoff].copy()
    monthly["month"] = pd.to_datetime(monthly.ts).dt.strftime("%Y-%m")
    trend = [
        {
            "month": month,
            "planned": float(group.expected_plan_vpp_income_krw.sum()),
            "fixed": float(group.fixed_plan_vpp_income_krw.sum()),
            "actual": float(group.vpp_income_krw.sum()),
        }
        for month, group in monthly.groupby("month", sort=True)
    ]
    return {
        "id": f"distribution:{archive_id}:{selected}",
        "source": "distribution",
        "selection": candidate,
        "title": "배전망 ESS · 태양광 5.7 MW",
        "date": str(selected),
        "dates": dates(hourly),
        "resource": {
            "label": row.equipment,
            "capacity": float(row.ess_energy_mwh),
            "power": float(row.ess_power_mw),
            "basis": str(row.line_case),
        },
        "basis": "저장된 실행 백테스트 · forecast_exec / standard · 실제 정산 아님",
        "incomeLabel": "VPP 귀속 수입",
        "incomeNote": (
            "투자·유지·교체비 차감 전. 편차 정산은 에너지 수입에 포함되어 다시 차감하지 않습니다."
        ),
        "executionAvailable": True,
        "hours": hours,
        "ledger": ledger,
        "plannedIncome": float(block.expected_plan_vpp_income_krw.sum()),
        "fixedPlanIncome": float(block.fixed_plan_vpp_income_krw.sum()),
        "actualIncome": float(block.vpp_income_krw.sum()),
        "cumulativePlannedIncome": float(monthly.expected_plan_vpp_income_krw.sum()),
        "cumulativeFixedIncome": float(monthly.fixed_plan_vpp_income_krw.sum()),
        "cumulativeIncome": float(monthly.vpp_income_krw.sum()),
        "trend": trend,
        "warnings": [
            "계산 당시 입력·코드로 만든 과거 결과입니다. 현재 코드로 재계산한 결과가 아닙니다.",
            "고정계획 사후 수입은 계획 ESS 명령을 유지하고 "
            "물리 한도 위반만 최소 보정한 비교값입니다.",
        ],
        "provenance": {
            "snapshot": archive_id,
            "operation": "forecast_exec",
            "basis": "standard",
            "outputHash": meta["sha256"],
            "report": str(row.source_file),
        },
    }


def portfolio_snapshot(root: Path, selection: str, selected: date) -> dict:
    status, options = releases(root)
    release = next((r for r in options if r["id"] == selection), None)
    if release is None:
        raise LookupError("사용 가능한 결과 묶음이 아닙니다.")
    folder = resolve_project_path(root, release["backtest"])
    validation = read_json(folder / "validation.json")
    if validation.get("plan_checks_all_passed") is not True:
        raise ValueError("계획 검산을 통과한 결과만 표시합니다.")
    data = load_experiment(folder)
    payload = replay_payload(data, selected)
    p = payload["series"]["forecast"]
    actual = payload["series"].get("executed")
    plan = data["plan"] if data["plan"] is not None else data["forecast"]
    block = day_frame(plan, selected)
    rev = plan_revenue_breakdown(block, data["ess"])
    hours = []
    for i, point in enumerate(p):
        a = actual[i] if actual is not None else None
        hours.append(
            {
                "hour": i,
                "forecastGeneration": point["g"],
                "actualGeneration": a["g"] if a else None,
                "plannedSupply": point["q"],
                "actualSupply": a["q"] if a else None,
                "plannedCharge": point["c"],
                "plannedDischarge": point["d"],
                "charge": a["c"] if a else None,
                "discharge": a["d"] if a else None,
                "plannedSoc": point["soc"],
                "soc": a["soc"] if a else None,
                "shortfall": a["shortfall"] if a else None,
                "plannedIncome": number(point["profit"]),
                "fixedSupply": None,
                "fixedCharge": None,
                "fixedDischarge": None,
                "fixedSoc": None,
                "fixedShortfall": None,
                "fixedIncome": None,
                "income": number(a["profit"]) if a else None,
            }
        )
    execution = data.get("execution", {}).get("executed")
    ex = day_frame(execution, selected) if actual is not None else None

    def total(column):
        if ex is None or ex.settlement_basis.eq("price_missing").any():
            return None
        return number(ex[column].sum())

    sales = None if ex is None else total("committed_sales_krw")
    deviation = total("deviation_krw")
    ledger = [
        {
            "label": "판매 정산 · 편차 포함",
            "planned": rev["sales_revenue_krw"],
            "fixed": None,
            "actual": None if sales is None or deviation is None else sales + deviation,
        },
        {"label": "REC 수입", "planned": rev["rec_revenue_krw"], "fixed": None,
         "actual": total("rec_krw")},
        {
            "label": "열화비 차감",
            "planned": -rev["throughput_cost_krw"],
            "fixed": None,
            "actual": -total("wear_krw") if total("wear_krw") is not None else None,
        },
    ]
    trend = []
    period = plan[plan.ts.dt.date.le(selected)]
    for month, group in period.groupby(period.ts.dt.strftime("%Y-%m")):
        observed = execution[execution.ts.isin(group.ts)] if execution is not None else None
        complete = (
            observed is not None
            and len(observed) == len(group)
            and not observed.settlement_basis.eq("price_missing").any()
        )
        trend.append(
            {
                "month": month,
                "planned": plan_revenue_breakdown(group, data["ess"])["net_revenue_krw"],
                "fixed": None,
                "actual": number(observed.realized_net_krw.sum()) if complete else None,
            }
        )
    warnings = ["실행 수익은 실험용 정산 가정이며 실제 정산서가 아닙니다."]
    if status.get("state") in ("needs_data", "failed"):
        warnings.append("최신 계산에 필요한 입력이 부족합니다. 마지막 검산 완료 묶음을 표시합니다.")
    if not actual:
        warnings.append(
            "이 결과 묶음은 실측 실행 결과가 없습니다. 계획을 실적으로 대체하지 않습니다."
        )
    return {
        "id": f"portfolio:{release['id']}:{selected}",
        "source": "portfolio",
        "selection": selection,
        "title": "태양광·풍력 21곳 포트폴리오",
        "date": str(selected),
        "dates": dates(plan),
        "resource": {
            "label": "현재 결과 묶음의 ESS",
            "capacity": data["ess"].capacity_mwh,
            "power": data["ess"].discharge_power_mw,
            "basis": release["id"][:8],
        },
        "basis": "검산 완료 운영계획 · 실행 수익은 실험 가정",
        "incomeLabel": "운영 순수익",
        "incomeNote": "SMP + REC − 열화비. 설비 투자·금융비용은 제외됩니다.",
        "executionAvailable": actual is not None,
        "hours": hours,
        "ledger": ledger,
        "plannedIncome": number(rev["net_revenue_krw"]),
        "fixedPlanIncome": None,
        "actualIncome": total("realized_net_krw"),
        "cumulativePlannedIncome": sum(t["planned"] for t in trend),
        "cumulativeFixedIncome": None,
        "cumulativeIncome": sum(t["actual"] for t in trend)
        if all(t["actual"] is not None for t in trend)
        else None,
        "trend": trend,
        "warnings": warnings,
        "provenance": {
            "snapshot": release["id"],
            "operation": "executed" if actual else "planned",
            "basis": "experimental_settlement",
            "outputHash": "",
            "report": release["label"],
        },
    }


MAX_PERIOD_DAYS = 366


def period_frame(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Saved rows for [start, end]; every day must hold exactly 00–23h (no padding)."""
    if start > end:
        raise ValueError("시작일이 종료일보다 늦습니다.")
    if (end - start).days + 1 > MAX_PERIOD_DAYS:
        raise ValueError(f"기간은 최대 {MAX_PERIOD_DAYS}일까지 조회합니다.")
    ts = pd.to_datetime(frame.ts)
    block = frame[ts.dt.date.between(start, end)].copy()
    block["day"] = pd.to_datetime(block.ts).dt.strftime("%Y-%m-%d")
    expected = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    counts = block.groupby("day").size()
    missing = [d for d in expected if counts.get(d, 0) != 24]
    if missing:
        raise ValueError(f"저장 결과에 00–23시가 완전하지 않은 날이 있습니다: {missing[0]}")
    return block


def distribution_period(root: Path, candidate: str, start: date, end: date) -> dict:
    rows, _ = load_display(root)
    match = rows[rows.id.eq(candidate)]
    if match.empty:
        raise LookupError("저장된 결과에 없는 설비입니다.")
    row = match.iloc[0]
    hourly, _, archive_id = load_archive(root, row)
    block = period_frame(hourly, start, end)
    capacity = float(row.ess_energy_mwh)
    days = []
    for day, g in block.groupby("day", sort=True):
        days.append(
            {
                "date": day,
                "plannedSupply": float(g.committed_sell_mwh.sum()),
                "actualSupply": float(g.supply_mwh.sum()),
                "plannedCharge": float(g.planned_charge_mwh.sum()),
                "plannedDischarge": float(g.planned_discharge_mwh.sum()),
                "charge": float(g.charge_mwh.sum()),
                "discharge": float(g.discharge_mwh.sum()),
                "plannedSoc": float(g.planned_energy_end_mwh.iloc[-1] / capacity * 100),
                "soc": float(g.soc_end.iloc[-1] * 100),
                "shortfall": float(g.shortfall_mwh.sum()),
                "fixedSupply": float(g.fixed_plan_supply_mwh.sum()),
                "fixedCharge": float(g.fixed_plan_charge_mwh.sum()),
                "fixedDischarge": float(g.fixed_plan_discharge_mwh.sum()),
                "fixedSoc": float(g.fixed_plan_soc_end.iloc[-1] * 100),
                "fixedShortfall": float(g.fixed_plan_shortfall_mwh.sum()),
                "plannedIncome": float(g.expected_plan_vpp_income_krw.sum()),
                "fixedIncome": float(g.fixed_plan_vpp_income_krw.sum()),
                "income": float(g.vpp_income_krw.sum()),
            }
        )
    return {
        "source": "distribution",
        "selection": candidate,
        "start": str(start),
        "end": str(end),
        "dates": dates(hourly),
        "incomeLabel": "VPP 귀속 수입",
        "executionAvailable": True,
        "days": days,
        "snapshot": archive_id,
    }


def portfolio_period(root: Path, selection: str, start: date, end: date) -> dict:
    _, options = releases(root)
    release = next((r for r in options if r["id"] == selection), None)
    if release is None:
        raise LookupError("사용 가능한 결과 묶음이 아닙니다.")
    folder = resolve_project_path(root, release["backtest"])
    if read_json(folder / "validation.json").get("plan_checks_all_passed") is not True:
        raise ValueError("계획 검산을 통과한 결과만 표시합니다.")
    data = load_experiment(folder)
    plan = data["plan"] if data["plan"] is not None else data["forecast"]
    block = period_frame(plan, start, end)
    execution = (data.get("execution") or {}).get("executed")
    days = []
    for day, g in block.groupby("day", sort=True):
        ex = execution[execution.ts.isin(g.ts)] if execution is not None else None
        complete = (
            ex is not None and len(ex) == 24 and not ex.settlement_basis.eq("price_missing").any()
        )
        days.append(
            {
                "date": day,
                "plannedSupply": float(g.sell_mwh.sum()),
                "actualSupply": number(ex.supply_mwh.sum())
                if complete and "supply_mwh" in ex
                else None,
                "plannedCharge": float(g.charge_mwh.sum()),
                "plannedDischarge": float(g.discharge_mwh.sum()),
                "charge": None,
                "discharge": None,
                "plannedSoc": number(g.soc_end.iloc[-1] * 100),
                "soc": None,
                "shortfall": None,
                "fixedSupply": None,
                "fixedCharge": None,
                "fixedDischarge": None,
                "fixedSoc": None,
                "fixedShortfall": None,
                "plannedIncome": number(
                    plan_revenue_breakdown(g, data["ess"])["net_revenue_krw"]
                ),
                "fixedIncome": None,
                "income": number(ex.realized_net_krw.sum()) if complete else None,
            }
        )
    return {
        "source": "portfolio",
        "selection": selection,
        "start": str(start),
        "end": str(end),
        "dates": dates(plan),
        "incomeLabel": "운영 순수익",
        "executionAvailable": execution is not None,
        "days": days,
        "snapshot": release["id"],
    }


@router.get("/period")
def period(
    source: Literal["portfolio", "distribution"],
    selection: str = Query(min_length=1, max_length=180),
    start: date = date(2025, 1, 1),
    end: date = date(2025, 12, 31),
):
    """Daily totals of the saved result over a period, for the cumulative demo replay."""
    try:
        result = (portfolio_period if source == "portfolio" else distribution_period)(
            project_root(), selection, start, end
        )
        json.dumps(result, allow_nan=False)
        return result
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(422, f"저장 결과 확인 필요: {exc}") from exc


@router.get("/catalog")
def catalog():
    root = project_root()
    status, portfolio = releases(root)
    candidates = []
    warnings = []
    try:
        rows, _ = load_display(root)
        for _, r in rows.iterrows():
            ready = False
            archive_issue = None
            if archives(root, r.id):
                try:
                    # The demo must be able to open the result, not just find a manifest.
                    load_archive(root, r)
                    ready = True
                except (ValueError, KeyError, OSError) as exc:
                    archive_issue = str(exc)
            candidates.append(
                {
                    "id": r.id,
                    "label": r.equipment,
                    "line": r.line_case,
                    "capacity": float(r.ess_energy_mwh),
                    "power": float(r.ess_power_mw),
                    "npv": number(r.npv_krw),
                    "irr": number(r.irr),
                    "discountedPaybackYears": number(r.discounted_payback_years),
                    "neededAnnualSupport": number(r.needed_annual_support_krw),
                    "income": revenue_breakdown(r)["total"],
                    "hasArchive": ready,
                    "archiveIssue": archive_issue,
                }
            )
    except (OSError, ValueError) as exc:
        warnings.append(str(exc))
    return {
        "portfolio": [{"id": r["id"], "label": r["label"]} for r in portfolio],
        "candidates": candidates,
        "pipelineState": status.get("state", "not_run"),
        "warnings": warnings,
    }


@router.get("/economics.pdf")
def economics_pdf(selection: str = Query(min_length=1, max_length=180)):
    """Download one candidate's saved summary and 20-year cashflow as a PDF."""
    try:
        candidate, summary, cashflow, capacity = economics_result(project_root(), selection)
        content = build_economics_pdf(candidate, summary, cashflow, capacity)
        filename = (
            f"{candidate.line_case}_{candidate.battery}_"
            f"{candidate.ess_power_mw:g}MW_{candidate.ess_energy_mwh:g}MWh_economics.pdf"
        )
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(422, f"저장 결과 확인 필요: {exc}") from exc


@router.get("/snapshot")
def snapshot(
    source: Literal["portfolio", "distribution"],
    selection: str = Query(min_length=1, max_length=180),
    day: date = date(2025, 5, 1),
):
    try:
        result = (portfolio_snapshot if source == "portfolio" else distribution_snapshot)(
            project_root(), selection, day
        )
        # Reject non-JSON finite numbers before FastAPI's encoder sees them.
        json.dumps(result, allow_nan=False)
        return result
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(422, f"저장 결과 확인 필요: {exc}") from exc
