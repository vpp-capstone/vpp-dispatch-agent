"""담당: 최적화 — 계획일 D 의 ESS 운영계획 생성 (예측만으로 48시간을 풀고 D 24시간을 확정).

    python -m vpp.operating_plan --day 2024-06-05 [--ess config/ess_v0.yaml]
        [--degradation-config config/ess_v0_1.yaml] [--rec config/rec_v0.yaml]
        [--start-soc 0.5 | --start-energy-mwh 8.0 | --from-plan <이전 결과 폴더>]
        [--solar <parquet>] [--wind <parquet>] [--smp <csv>] [--plants <csv>]
        [--out data/processed/operating_plan] [--run-id <이름>] [--overwrite]

입력은 발전량 예측·SMP 예측·ESS(열화비 포함) 설정·시작 SOC 뿐이다. 실측 열이 없거나 비어 있어도
계획을 만든다. 실측은 쓰지 않는다 (성능 평가·완전예측 비교는 forecast_backtest 가 한다).

출력: <out>/<run_id>/
    plan_48h.csv        D 00시~D+1 23시 48행. status=committed(D 24시간) / provisional(D+1 24시간)
    committed_24h.csv   확정한 D 24시간
    summary.json        합계(판매수익·REC 수익·열화비·차감 후 계획 수익), 시작 SOC 와 출처,
                        입력 파일 식별정보, 코드 버전, 예측 발표시점, ESS·열화비·REC 설정, 종료 정책
    validation.json     제약 검산 (48시간 전체, 확정 24시간)

시작 SOC 는 계획 대상일 시작 시점의 **입력값**이다 (기본: ESS 설정의 soc_initial). --from-plan 으로
이전 운영계획의 확정 24시간 종료 **계획상** SOC 를 가져올 수 있는데, 그 계획의 계획일이 D-1 이
아니면 조용히 이월하지 않고 거부한다.

대시보드와 CLI 는 같은 run_operating_plan 을 호출한다. 수익은 예측이 맞았다고 가정한 계획 수익이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

import vpp
from vpp import forecast_inputs as fi
from vpp.backtest import _json_default, _write_json, new_run_dir
from vpp.forecast_inputs import PlanWindow
from vpp.optimization import EssParams, Terminal, WindowPlan, plan_window
from vpp.rec import RecParams, load_rec, rec_metadata
from vpp.validation import check_plan

MODE = "operating_plan_forecast_only"
SOLVE_HOURS = 48
COMMIT_HOURS = 24
START_SOURCES = ("config_soc_initial", "user_input", "previous_plan")

LIMITATIONS = [
    "계획 수익은 예측 발전량·예측 SMP 가 맞았다고 가정한 값이다. 실측 적용·정산·편차 페널티 없음.",
    "REC 는 --rec 를 줄 때만 가정값(판매 1 MWh 당 1회, 단일 가중치, 시나리오 단가)으로 넣는다. "
    "실제 발급 기준·계약 형태가 아니다.",
    "D+1 24시간은 잠정(provisional) 계획이며 다음 날 새 예측으로 다시 푼다. 합계에 넣지 않는다.",
    "시작 SOC 는 입력값(설정 또는 이전 계획의 계획상 값)이다. 실제 배터리 상태를 읽지 않는다.",
    "발전량 예측 파일에 공개시각 플래그가 없어 입찰 마감 전 가용 여부를 행 단위로 확인하지 못한다.",
    "ESS 가정값과 열화비(공개자료 proxy)는 설비 사양·확정 비용이 아니다.",
]


# ---------------------------------------------------------------- 시작 SOC


@dataclass(frozen=True)
class StartEnergy:
    """계획 대상일 시작 시점의 ESS 에너지와 그 출처."""

    energy_mwh: float
    source: str  # START_SOURCES
    note: str
    previous_run: str | None = None

    previous_final_energy_mwh: float | None = None  # previous_plan 일 때 이전 계획의 종료값 원본

    def to_dict(self, ess: EssParams) -> dict[str, Any]:
        out = {**asdict(self), "soc": self.energy_mwh / ess.capacity_mwh}
        if self.previous_final_energy_mwh is not None:
            # 이전 출력 → 다음 입력 전달 확인용. SOC 범위 보정이 없었다면 0 이어야 한다.
            out["carry_gap_mwh"] = self.energy_mwh - self.previous_final_energy_mwh
        return out


def resolve_ess(ess_path: str | Path, degradation_path: str | Path | None = None) -> EssParams:
    """ESS 설정 YAML 에 열화비 설정 YAML(없으면 ESS 파일 값)을 얹는다. 어느 YAML 도 고치지 않는다.

    열화비 설정은 degradation_model·throughput_basis·throughput_cost_krw_per_mwh 세 필드만 가져온다.
    화면의 '열화비 미적용 / 고정 열화비 적용' 선택이 이 인수로 들어온다.
    """
    ess = EssParams.from_yaml(ess_path)
    if degradation_path is None:
        return ess
    deg = EssParams.from_yaml(degradation_path)
    return EssParams(**{
        **ess.to_dict(),
        "degradation_model": deg.degradation_model,
        "throughput_basis": deg.throughput_basis,
        "throughput_cost_krw_per_mwh": deg.throughput_cost_krw_per_mwh,
    })


def resolve_start_energy(
    ess: EssParams,
    planning_day: date,
    *,
    start_soc: float | None = None,
    start_energy_mwh: float | None = None,
    from_plan: str | Path | None = None,
) -> StartEnergy:
    """시작 SOC 를 정한다. 셋 중 하나만 줄 수 있고, 아무것도 없으면 ESS 설정값."""
    given = [x for x in (start_soc, start_energy_mwh, from_plan) if x is not None]
    if len(given) > 1:
        raise ValueError(
            "시작 SOC 는 --start-soc, --start-energy-mwh, --from-plan 중 하나만 주세요."
        )
    if from_plan is not None:
        folder = Path(from_plan)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if summary.get("mode") != MODE:
            raise ValueError(f"운영계획 결과가 아닙니다: {folder} (mode={summary.get('mode')})")
        prev_day = date.fromisoformat(summary["planning_day"])
        if prev_day != planning_day - timedelta(days=1):
            raise ValueError(
                f"이전 계획의 계획일 {prev_day} 가 D-1({planning_day - timedelta(days=1)}) 이 "
                "아닙니다. 날짜가 끊긴 계획상 SOC 는 이월하지 않습니다. --start-soc 로 직접 주세요."
            )
        energy = float(summary["committed"]["final_energy_mwh"])
        return StartEnergy(
            ess.clip_energy(energy), "previous_plan",
            f"이전 운영계획({summary['run_id']}, 계획일 {prev_day})의 확정 24시간 종료 계획상 SOC. "
            "실측 SOC 아님.", previous_run=str(folder), previous_final_energy_mwh=energy,
        )
    if start_energy_mwh is not None:
        energy = float(start_energy_mwh)
    elif start_soc is not None:
        energy = float(start_soc) * ess.capacity_mwh
    else:
        return StartEnergy(ess.initial_energy_mwh, "config_soc_initial",
                           f"ESS 설정 soc_initial={ess.soc_initial} (입력값)")
    if not ess.energy_min_mwh - 1e-9 <= energy <= ess.energy_max_mwh + 1e-9:
        raise ValueError(
            f"시작 에너지 {energy:.3f} MWh 가 SOC 범위 "
            f"[{ess.energy_min_mwh:.3f}, {ess.energy_max_mwh:.3f}] 밖입니다."
        )
    return StartEnergy(ess.clip_energy(energy), "user_input", "사용자 입력값")


# ---------------------------------------------------------------- 입력 창 → 계획


def build_day_window(
    planning_day: date,
    *,
    solar_path: Path = fi.DEFAULT_SOLAR_PREDICTIONS,
    wind_path: Path = fi.DEFAULT_WIND_PREDICTIONS,
    smp_path: Path = fi.DEFAULT_SMP_FORECAST,
    plants_path: Path = fi.PLANT_MASTER_PATH,
) -> PlanWindow:
    """계획일 하나의 48시간 예측 창. 못 만들면 사유를 담은 예외."""
    pred = fi.load_generation_predictions(solar_path, wind_path, plants_path)
    portfolio = fi.aggregate_portfolio(pred, fi.expected_assets(plants_path))
    smp = fi.load_smp_forecast(smp_path)
    window, reason = fi.build_plan_window(portfolio, smp, planning_day)
    if window is None:
        raise ValueError(f"{planning_day} 운영계획을 만들 수 없습니다: {reason}")
    return window


def generate_plan(
    window: PlanWindow, ess: EssParams, start: StartEnergy, *,
    rec: RecParams | None = None, tol: float = 1e-6,
) -> WindowPlan:
    """48시간을 풀고 앞 24시간을 확정한다. 종료 제약 없음(운영). 실측은 쓰지 않는다.

    rec 를 주면 예측 창에 REC 가산 열을 붙여 판매 단가 가산으로 푼다.
    """
    forecast = window.forecast
    hours = float(forecast["dt_hours"].sum())
    if hours != SOLVE_HOURS:
        raise ValueError(f"운영계획은 48시간 창이 필요합니다: {hours}h")
    if rec is not None:
        forecast = rec.apply(forecast)
    return plan_window(
        forecast, ess, start.energy_mwh, commit_hours=COMMIT_HOURS, terminal=Terminal(), tol=tol,
    )


def plan_tables(plan: WindowPlan) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(48시간 표 with status, 확정 24시간 표)."""
    full = plan.hourly.copy()
    full["status"] = full["committed"].map({True: "committed", False: "provisional"})
    full = full.drop(columns="committed")
    committed = full[full["status"] == "committed"].drop(columns="status").reset_index(drop=True)
    return full, committed


def plan_validation(plan: WindowPlan, ess: EssParams, tol: float = 1e-6) -> dict[str, Any]:
    committed = plan.committed
    full_checks = check_plan(
        plan.hourly, ess, tol, initial_energy_mwh=plan.initial_energy_mwh,
        objective_krw=plan.totals(committed_only=False)["objective_krw"],
        solver_status=plan.solver_status,
    )
    committed_checks = check_plan(
        committed, ess, tol, initial_energy_mwh=plan.initial_energy_mwh,
        objective_krw=plan.totals(committed_only=True)["objective_krw"],
        solver_status=plan.solver_status,
    )
    structure = [
        {"item": "solved_hours_48", "passed": len(plan.hourly) == SOLVE_HOURS,
         "value": float(len(plan.hourly))},
        {"item": "committed_hours_24", "passed": len(committed) == COMMIT_HOURS,
         "value": float(len(committed))},
        {"item": "committed_is_first_24h",
         "passed": bool(plan.hourly["committed"].iloc[:COMMIT_HOURS].all()
                        and not plan.hourly["committed"].iloc[COMMIT_HOURS:].any()),
         "value": float(plan.committed_intervals)},
    ]
    all_checks = full_checks + committed_checks + structure
    return {
        "plan_checks_48h": full_checks,
        "plan_checks_committed_24h": committed_checks,
        "structure_checks": structure,
        "all_passed": all(c["passed"] for c in all_checks),
        "max_mip_gap": plan.mip_gap,
    }


# ---------------------------------------------------------------- 실행 메타데이터


def file_identity(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    stat = p.stat()
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": p.as_posix(), "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": digest.hexdigest(),
    }


def code_version() -> dict[str, Any]:
    """패키지 버전 + (있으면) git 커밋. git 은 읽기만 한다."""
    info: dict[str, Any] = {"package_version": vpp.__version__, "git_commit": None,
                            "git_dirty": None}
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True, timeout=5, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=root, capture_output=True, text=True, timeout=5, check=True)
        info["git_commit"] = head.stdout.strip()
        info["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


# ---------------------------------------------------------------- 실행


def run_operating_plan(
    planning_day: date,
    *,
    ess_path: str | Path = "config/ess_v0.yaml",
    degradation_path: str | Path | None = None,
    rec_path: str | Path | None = None,
    start_soc: float | None = None,
    start_energy_mwh: float | None = None,
    from_plan: str | Path | None = None,
    solar_path: Path = fi.DEFAULT_SOLAR_PREDICTIONS,
    wind_path: Path = fi.DEFAULT_WIND_PREDICTIONS,
    smp_path: Path = fi.DEFAULT_SMP_FORECAST,
    plants_path: Path = fi.PLANT_MASTER_PATH,
    out_dir: str | Path = "data/processed/operating_plan",
    run_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """운영계획을 만들고 저장한다. CLI 와 대시보드가 같이 쓴다. 결과 폴더 경로를 돌려준다."""
    ess = resolve_ess(ess_path, degradation_path)
    rec = load_rec(rec_path)
    start = resolve_start_energy(
        ess, planning_day, start_soc=start_soc, start_energy_mwh=start_energy_mwh,
        from_plan=from_plan,
    )
    window = build_day_window(
        planning_day, solar_path=solar_path, wind_path=wind_path, smp_path=smp_path,
        plants_path=plants_path,
    )
    plan = generate_plan(window, ess, start, rec=rec)
    full, committed = plan_tables(plan)
    validation = plan_validation(plan, ess)

    run_id = run_id or f"plan_{planning_day}_{datetime.now():%Y%m%d%H%M%S}"
    out = new_run_dir(out_dir, run_id, overwrite)
    full.to_csv(out / "plan_48h.csv", index=False, encoding="utf-8-sig")
    committed.to_csv(out / "committed_24h.csv", index=False, encoding="utf-8-sig")

    committed_totals = plan.totals(committed_only=True)
    # 잠정(D+1) 합계 = 48시간 전체 − 확정 24시간. 열화비도 같은 모델 비용 함수로 계산된 값의 차.
    full_totals = plan.totals(committed_only=False)
    prov_totals = {k: full_totals[k] - committed_totals[k] for k in full_totals}
    summary = {
        "mode": MODE,
        "run_id": run_id,
        "planning_day": planning_day.isoformat(),
        "created_at": datetime.now().astimezone().isoformat(),
        "code_version": code_version(),
        "window": {
            "start_ts": full["ts"].iloc[0].isoformat(), "end_ts": full["ts"].iloc[-1].isoformat(),
            "solved_hours": SOLVE_HOURS, "commit_hours": COMMIT_HOURS, "tz": fi.KST,
        },
        "terminal": plan.terminal.name,
        "start_energy": start.to_dict(ess),
        "committed": {
            **committed_totals,
            "final_energy_mwh": plan.final_committed_energy_mwh,
            "final_soc": plan.final_committed_energy_mwh / ess.capacity_mwh,
            "note": "D 24시간 확정 계획. 수익은 예측이 맞았다고 가정한 계획 수익",
        },
        "provisional_next_day": {
            **prov_totals,
            "final_energy_mwh": plan.final_solved_energy_mwh,
            "note": "D+1 24시간 잠정 계획. 다음 날 새 예측으로 다시 풀며 합계에 넣지 않는다",
        },
        "window_48h": full_totals,
        "forecast_sources": window.sources,
        "inputs": {
            "solar_predictions": file_identity(solar_path),
            "wind_predictions": file_identity(wind_path),
            "smp_forecast": file_identity(smp_path),
            "plants": file_identity(plants_path),
            "ess_config": file_identity(ess_path),
            "rec_config": None if rec_path is None else file_identity(rec_path),
            "actual_used": False,
        },
        "ess": {**ess.to_dict(), "degradation": plan.degradation, "source": str(ess_path),
                "degradation_source": str(degradation_path or ess_path)},
        "rec": rec_metadata(rec),
        # 실제 시작 잔량(--start-soc/--from-plan 반영) × 방전 효율 × 48시간 최고 가산. 보정값.
        "initial_energy_rec_upper_bound_krw": (
            0.0 if rec is None
            else rec.initial_energy_rec_upper_bound_krw(
                ess, full["ts"], initial_energy_mwh=start.energy_mwh)
        ),
        "limitations": LIMITATIONS,
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "validation.json", {"run_id": run_id, "mode": MODE, **validation,
                                          "limitations": LIMITATIONS})
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", type=date.fromisoformat, required=True, help="계획일 D (KST)")
    parser.add_argument("--ess", default="config/ess_v0.yaml")
    parser.add_argument("--degradation-config", default=None,
                        help="열화비 설정만 가져올 YAML (예: config/ess_v0_1.yaml). "
                             "없으면 --ess 값")
    parser.add_argument("--rec", default=None,
                        help="REC 수익 가정 YAML (예: config/rec_v0.yaml). 없으면 REC 미적용")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--start-soc", type=float, default=None, help="시작 SOC (0~1)")
    start.add_argument("--start-energy-mwh", type=float, default=None, help="시작 에너지 (MWh)")
    start.add_argument("--from-plan", default=None,
                       help="이전 운영계획 결과 폴더 (계획일이 D-1 이어야 함, 계획상 SOC 이월)")
    parser.add_argument("--solar", type=Path, default=fi.DEFAULT_SOLAR_PREDICTIONS)
    parser.add_argument("--wind", type=Path, default=fi.DEFAULT_WIND_PREDICTIONS)
    parser.add_argument("--smp", type=Path, default=fi.DEFAULT_SMP_FORECAST)
    parser.add_argument("--plants", type=Path, default=fi.PLANT_MASTER_PATH)
    parser.add_argument("--out", default="data/processed/operating_plan")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="같은 run-id 결과 폴더 덮어쓰기 허용")
    args = parser.parse_args(argv)
    out = run_operating_plan(
        args.day, ess_path=args.ess, degradation_path=args.degradation_config,
        rec_path=args.rec, start_soc=args.start_soc,
        start_energy_mwh=args.start_energy_mwh, from_plan=args.from_plan,
        solar_path=args.solar, wind_path=args.wind, smp_path=args.smp, plants_path=args.plants,
        out_dir=args.out, run_id=args.run_id, overwrite=args.overwrite,
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    print(f"결과 폴더: {out}")
    head = {k: summary[k] for k in ("planning_day", "start_energy", "committed",
                                     "provisional_next_day", "terminal")}
    head["ess"] = {k: summary["ess"][k] for k in ("degradation_model",
                                                   "throughput_cost_krw_per_mwh", "source")}
    head["rec"] = summary["rec"]
    print(json.dumps(head, ensure_ascii=False, indent=2, default=_json_default))
    if not validation["all_passed"]:
        print("❌ 제약 검산 실패")
        raise SystemExit(1)
    print("✅ 48시간 계산 · 24시간 확정 · 제약 검산 통과")


if __name__ == "__main__":
    main()
