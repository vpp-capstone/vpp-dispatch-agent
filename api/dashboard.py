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
from fastapi import APIRouter, HTTPException, Query

from dashboard import distribution_replay as replay
from dashboard.distribution_display import load_display
from dashboard.distribution_operation import revenue_breakdown
from dashboard.operator_simulation import replay_payload
from dashboard.overview_data import load_experiment
from vpp.baselines import plan_revenue_breakdown

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


def project_root() -> Path:
    return Path(os.environ.get("VPP_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    return float(value) if value is not None and pd.notna(value) and np.isfinite(value) else None


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
        meta.get("schema") != 1
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
    }
    if needed - set(frame):
        raise ValueError("저장 결과에 확정 계획 비교 열이 없습니다.")
    if not np.isfinite(frame[list(needed)].to_numpy(float)).all():
        raise ValueError("확정 계획 비교 열에 결측이 있습니다.")
    annual = revenue_breakdown(row)
    for column, key in (
        ("vpp_ess_krw", "ess"),
        ("vpp_solar_cp_krw", "solar_cp"),
        ("vpp_penalty_krw", "penalty"),
        ("vpp_income_krw", "total"),
    ):
        if abs(float(frame[column].sum()) - annual[key]) > 0.04:
            raise ValueError("시간별 귀속 수입이 연간 보고값과 일치하지 않습니다.")
    if not np.allclose(
        frame.vpp_income_krw,
        frame.vpp_ess_krw + frame.vpp_solar_cp_krw + frame.vpp_penalty_krw,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("시간별 수입 합계가 구성 항목과 다릅니다.")
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
                "plannedIncome": None,
                "income": float(r.vpp_income_krw),
            }
        )
    ledger = [
        {"label": label, "planned": None, "actual": float(block[column].sum())}
        for label, column in (
            ("ESS 귀속 에너지 수입", "vpp_ess_krw"),
            ("태양광 용량정산금 · 50%", "vpp_solar_cp_krw"),
            ("임밸런스 페널티", "vpp_penalty_krw"),
        )
    ]
    cutoff = pd.to_datetime(hourly.ts).dt.date.le(selected)
    monthly = hourly[cutoff].copy()
    monthly["month"] = pd.to_datetime(monthly.ts).dt.strftime("%Y-%m")
    trend = [
        {"month": month, "planned": None, "actual": float(group.vpp_income_krw.sum())}
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
        "incomeNote": "투자·유지·교체비 차감 전. 편차 정산은 에너지 수입에 포함되어 다시 차감하지 않습니다.",
        "executionAvailable": True,
        "hours": hours,
        "ledger": ledger,
        "plannedIncome": None,
        "actualIncome": float(block.vpp_income_krw.sum()),
        "cumulativeIncome": float(monthly.vpp_income_krw.sum()),
        "trend": trend,
        "warnings": [
            "계산 당시 입력·코드로 만든 과거 결과입니다. 현재 코드로 재계산한 결과가 아닙니다.",
            "계획의 VPP 귀속 수입은 저장되지 않아 금액 차이를 계산하지 않습니다.",
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
    folder = within(
        root,
        Path(release["backtest"])
        if Path(release["backtest"]).is_absolute()
        else root / release["backtest"],
    )
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
            "actual": None if sales is None or deviation is None else sales + deviation,
        },
        {"label": "REC 수입", "planned": rev["rec_revenue_krw"], "actual": total("rec_krw")},
        {
            "label": "열화비 차감",
            "planned": -rev["throughput_cost_krw"],
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
        "actualIncome": total("realized_net_krw"),
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
