"""담당: 최적화 — 예측 기반 ESS 계획과 완전예측(실측) 계획을 같은 조건으로 돌려 비교하는 CLI.

    python -m vpp.forecast_backtest --start 2024-06-05 --end 2024-06-11 \
        [--solar <parquet>] [--wind <parquet>] [--smp <csv>] [--ess config/ess_v0.yaml] \
        [--degradation-config config/ess_v0_1.yaml] [--rec config/rec_v0.yaml] \
        [--out data/processed/backtest] [--run-id <이름>] [--overwrite]

같은 run-id 폴더가 있으면 기본은 덮어쓰지 않는다 (--overwrite 로만 허용).
--rec 를 주면 REC 수익 가정(vpp.rec)이 두 시나리오 모두에 판매 단가 가산으로 들어가고, 요약에
판매수익(SMP)·REC 수익·열화비가 나뉘어 나온다. 'ESS 없음 대비 추가수익'은 같은 가산을 적용한
ESS 없음 수익과의 차이다.

출력: <out>/<실행ID>/
    hourly_forecast.csv, hourly_actual.csv   시나리오별 확정 24시간 계획 (계획일·구간 표시)
    daily_comparison.csv                      계획일별 두 시나리오 나란히 (계획 수익 차이 포함)
    excluded_days.csv                         제외한 계획일과 사유 (stage=plan: 예측 부족으로
                                              계획 불가, comparison: 실측 부족으로 비교만 불가)
    comparison.html                           SOC 이월·충방전 비교 그래프
    summary.json, validation.json             합계·입력·가정 / 비교 조건 검사·제약 검산

예측 기반 계획의 수익은 **예측 발전량·예측 가격이 맞았다고 가정한 계획 수익**이다. 두 시나리오의
차이를 실제 손실이나 개선으로 읽으면 안 된다. 실측 적용·정산·편차 페널티는 다음 단계.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vpp import forecast_inputs as fi
from vpp.backtest import _json_default, _write_json, new_run_dir
from vpp.forecast_rolling import (
    COMMIT_HOURS,
    SegmentResult,
    compare_daily,
    run_scenario,
    soc_trace,
)
from vpp.operating_plan import resolve_ess
from vpp.optimization import EssParams
from vpp.rec import RecParams, load_rec, rec_metadata
from vpp.validation import check_plan

MODE = "forecast_vs_perfect_foresight_plan"
DEFAULT_START = date(2024, 6, 5)
DEFAULT_END = date(2024, 6, 11)

LIMITATIONS = [
    "예측 기반 계획의 수익은 예측 발전량·예측 SMP가 맞았다고 가정한 계획 수익이다. "
    "실측 적용·정산을 하지 않았으므로 두 시나리오 차이는 실제 손익이 아니다.",
    "SOC 이월은 계획상 값이다. 실측에 따른 충·방전 실행 조정과 실제 SOC 재계산은 다음 단계.",
    "REC 는 --rec 를 줄 때만 가정값(판매 1 MWh 당 1회, 단일 가중치, 시나리오 단가)으로 넣는다. "
    "실제 발급 기준·계약 형태가 아니다. 구간 시작 저장분이 팔릴 때도 REC 가 붙는다(제외 미구현). "
    "그 보수적 상한을 initial_energy_rec_upper_bound_krw 에 따로 적는다.",
    "terminal_constraint_cost_krw 는 구간 마지막 날의 종료 제약(E[끝] ≥ 구간 시작 에너지)을 풀었을 "
    "때 늘어나는 계산상 수익이다. 남겨 둔 잔량을 못 판 기회비용이 포함되므로 실제 충전 비용이 "
    "아니며, 순수익에 더해 사이클 수익으로 해석하면 안 된다 (참고 지표).",
    "발전량 예측 파일에는 공개시각 플래그가 없어 D-2 18 UTC 발표분의 D-1 11:00 KST 이전 공개를 "
    "행 단위로 확인하지 못했다 (generation_availability_verified=false).",
    "2024년 발전량 예측은 모델 선택(조기 종료)에 쓴 검증 구간이라 성능이 실제보다 좋게 보일 수 "
    "있다.",
    "비교용 실측 발전량은 발전량 예측 파일의 정답 열(generation_kwh, label_is_valid 행만)"
    "이다. vpp.backtest 가 쓰는 원본 합산과 이상치 처리 기준이 다를 수 있다.",
    "완전예측 비교 계획도 입찰 낙찰·편차 페널티·열화(기본 k=0)·출력제어 상한을 반영하지 않는다.",
    "ESS 가정값(config/ess_v0.yaml)은 설비 사양이 아니다.",
]


# ---------------------------------------------------------------- 결과 합치기


def stack_hourly(results: list[SegmentResult]) -> pd.DataFrame:
    parts = [r.result.hourly.assign(segment_id=r.segment_id) for r in results]
    return pd.concat(parts, ignore_index=True)


def stack_daily(results: list[SegmentResult]) -> pd.DataFrame:
    parts = [r.result.daily.assign(segment_id=r.segment_id, scenario=r.scenario) for r in results]
    return pd.concat(parts, ignore_index=True)


def scenario_totals(results: list[SegmentResult]) -> dict[str, Any]:
    """시나리오 합계. 순수익 = 판매(SMP) + REC − 열화비. ESS 없음 수익은 같은 판매 가치
    (SMP+REC)로 같은 시간에 ESS 없이 팔았을 때이며, 그 차이가 'ESS 없음 대비 추가수익'이다."""
    daily = stack_daily(results)
    objective = float(daily["objective_krw"].sum())
    no_ess = float(daily["no_ess_same_curtailment_policy_krw"].sum())
    return {
        "days": int(len(daily)),
        "segments": [
            {"segment_id": r.segment_id, "start": r.start_day.isoformat(),
             "end": r.end_day.isoformat(), "days": int(len(r.result.daily)),
             "initial_energy_mwh": r.result.period_initial_energy_mwh,
             "final_energy_mwh": r.result.period_final_energy_mwh,
             "terminal_constraint_cost_krw": r.result.terminal_constraint_cost_krw}
            for r in results
        ],
        "planned_objective_krw": objective,
        "sales_revenue_krw": float(daily["sales_revenue_krw"].sum()),
        "rec_revenue_krw": float(daily["rec_revenue_krw"].sum()),
        "throughput_cost_krw": float(daily["throughput_cost_krw"].sum()),
        "no_ess_net_revenue_krw": no_ess,
        "ess_gain_vs_no_ess_krw": objective - no_ess,
        "terminal_constraint_cost_krw": float(sum(
            r.result.terminal_constraint_cost_krw or 0.0 for r in results
        )),
        "generation_mwh": float(daily["generation_mwh"].sum()),
        "sell_mwh": float(daily["sell_mwh"].sum()),
        "charge_mwh": float(daily["charge_mwh"].sum()),
        "discharge_mwh": float(daily["discharge_mwh"].sum()),
        "curtailment_mwh": float(daily["curtailment_mwh"].sum()),
        "max_mip_gap": float(daily["mip_gap"].max()),
    }


# ---------------------------------------------------------------- 검사


def _item(name: str, value: float, passed: bool) -> dict[str, Any]:
    return {"item": name, "passed": bool(passed), "value": float(value)}


def comparison_checks(
    forecast: list[SegmentResult],
    actual: list[SegmentResult],
    ess: EssParams,
    tol: float = 1e-6,
) -> list[dict[str, Any]]:
    """두 시나리오가 같은 조건(날짜·구간·시각·ESS·최초 SOC·종료 정책)으로 돌았는지 대조한다."""
    hf, ha = stack_hourly(forecast), stack_hourly(actual)
    df, da = stack_daily(forecast), stack_daily(actual)
    checks: list[dict[str, Any]] = []

    bounds_f = [(r.segment_id, r.start_day, r.end_day) for r in forecast]
    bounds_a = [(r.segment_id, r.start_day, r.end_day) for r in actual]
    checks.append(_item("same_segment_boundaries", abs(len(bounds_f) - len(bounds_a)),
                        bounds_f == bounds_a))

    days_f, days_a = set(df["planning_day"]), set(da["planning_day"])
    checks.append(_item("same_planning_days_symmetric_difference",
                        len(days_f ^ days_a), days_f == days_a))

    same_ts = len(hf) == len(ha) and bool((hf["ts"].to_numpy() == ha["ts"].to_numpy()).all())
    checks.append(_item("same_hourly_timestamps", abs(len(hf) - len(ha)), same_ts))

    if days_f == days_a and bounds_f == bounds_a:
        # 구간 첫날의 시작 에너지만 같아야 한다 (설정값). 이후는 각 시나리오의 계획을 따라 달라진다.
        firsts_f = df.groupby("segment_id")["initial_energy_mwh"].first().to_numpy()
        firsts_a = da.groupby("segment_id")["initial_energy_mwh"].first().to_numpy()
        diff_first = float(np.max(np.abs(firsts_f - firsts_a)))
        checks.append(_item("same_initial_energy_each_segment", diff_first, diff_first <= tol))
        off_config = float(np.max(np.abs(np.concatenate([firsts_f, firsts_a])
                                         - ess.initial_energy_mwh)))
        checks.append(_item("segment_initial_energy_equals_config", off_config, off_config <= tol))
    else:
        checks.append(_item("same_initial_energy_each_segment", float("nan"), False))
        checks.append(_item("segment_initial_energy_equals_config", float("nan"), False))

    pol_f = [(r.result.policy, r.result.horizon_hours, r.result.commit_hours) for r in forecast]
    pol_a = [(r.result.policy, r.result.horizon_hours, r.result.commit_hours) for r in actual]
    term_f = df.sort_values("planning_day")["terminal_constraint"].tolist()
    term_a = da.sort_values("planning_day")["terminal_constraint"].tolist()
    checks.append(_item("same_rolling_policy_and_terminal_constraints", 0.0,
                        pol_f == pol_a and term_f == term_a))

    solved_f = df.sort_values("planning_day")["solved_intervals"].tolist()
    solved_a = da.sort_values("planning_day")["solved_intervals"].tolist()
    checks.append(_item("same_window_lengths", 0.0, solved_f == solved_a))

    committed_ok = (df["committed_intervals"] == COMMIT_HOURS).all() and (
        da["committed_intervals"] == COMMIT_HOURS
    ).all()
    checks.append(_item("committed_24h_per_day_both", float(COMMIT_HOURS), bool(committed_ok)))
    return checks


def comparison_warnings(
    forecast: list[SegmentResult], actual: list[SegmentResult]
) -> list[dict[str, Any]]:
    """실패는 아니지만 확인할 신호. 예측 입력이 모든 시간에 실측과 같으면 실측이 섞였을 수 있다."""
    hf, ha = stack_hourly(forecast), stack_hourly(actual)
    warnings: list[dict[str, Any]] = []
    if len(hf) == len(ha):
        gen_same = hf["generation_mwh"].to_numpy() == ha["generation_mwh"].to_numpy()
        price_same = hf["price_krw_per_mwh"].to_numpy() == ha["price_krw_per_mwh"].to_numpy()
        differ = int((~gen_same | ~price_same).sum())
        warnings.append({
            "item": "forecast_inputs_identical_to_actual",
            "flagged": differ == 0,
            "value": float(len(hf) - differ),
            "note": f"예측 입력과 실측이 같은 시간 {len(hf) - differ}/{len(hf)}. "
                    "전부 같으면 예측 창에 실측이 들어갔는지 확인한다.",
        })
    return warnings


def plan_checks(results: list[SegmentResult], ess: EssParams) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        # objective_krw 를 넘기면 check_plan 이 열화모델 비용 함수로 목적함수를 다시 계산해 대조한다
        # (집계 수익 = 판매 − 모델 열화비 인지 검증).
        checks = check_plan(r.result.hourly, ess,
                            initial_energy_mwh=r.result.period_initial_energy_mwh,
                            objective_krw=r.result.objective_krw)
        out.append({"scenario": r.scenario, "segment_id": r.segment_id,
                    "plan_checks": checks, "rolling_checks": r.result.checks,
                    "all_passed": all(c["passed"] for c in checks + r.result.checks)})
    return out


# ---------------------------------------------------------------- 그래프


def segment_initials(results: list[SegmentResult]) -> dict[int, float]:
    return {r.segment_id: r.result.period_initial_energy_mwh for r in results}


def comparison_figure(forecast: list[SegmentResult], actual: list[SegmentResult],
                      ess: EssParams) -> go.Figure:
    fig = go.Figure()
    for name, results, color in (("예측 기반", forecast, "#8B5CF6"),
                                 ("완전예측(실측)", actual, "#0F766E")):
        frame = stack_hourly(results)
        trace = soc_trace(frame, segment_initials(results))
        fig.add_scatter(x=trace["ts"], y=trace["energy_mwh"], name=f"{name} · 잔량 MWh",
                        mode="lines+markers", line=dict(color=color, width=2),
                        marker=dict(size=4))
        fig.add_bar(x=frame["ts"], y=frame["discharge_mwh"], name=f"{name} · 방전",
                    marker_color=color, opacity=0.35, yaxis="y2", offset=0,
                    width=3_600_000 * frame["dt_hours"])
        fig.add_bar(x=frame["ts"], y=-frame["charge_mwh"], name=f"{name} · 충전",
                    marker_color=color, opacity=0.2, yaxis="y2", offset=0,
                    width=3_600_000 * frame["dt_hours"])
    for level, label in ((ess.energy_max_mwh, "상한"), (ess.energy_min_mwh, "하한")):
        fig.add_hline(y=level, line_dash="dot", line_color="gray",
                      annotation_text=f"{label} {level:.1f} MWh")
    fig.update_layout(
        title="ESS 잔량(SOC) 이월 비교 — 예측 기반 vs 완전예측 계획 "
              "(점: 구간 시작 잔량과 각 시간 종료 잔량, 확정 24시간만)",
        barmode="relative", height=520, plot_bgcolor="#fff",
        xaxis=dict(title="시간 (KST)"),
        yaxis=dict(title="잔량 MWh", range=[0, ess.capacity_mwh * 1.08]),
        yaxis2=dict(title="충(-)·방전(+) MWh", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ---------------------------------------------------------------- 실행


def run_forecast_backtest(
    start: date,
    end: date,
    *,
    solar_path: Path = fi.DEFAULT_SOLAR_PREDICTIONS,
    wind_path: Path = fi.DEFAULT_WIND_PREDICTIONS,
    smp_path: Path = fi.DEFAULT_SMP_FORECAST,
    plants_path: Path = fi.PLANT_MASTER_PATH,
    ess_path: str | Path = "config/ess_v0.yaml",
    degradation_path: str | Path | None = None,
    rec_path: str | Path | None = None,
    out_dir: str | Path = "data/processed/backtest",
    run_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """예측·실측 두 시나리오를 같은 창·ESS·열화비·REC 설정으로 풀고 결과 폴더 경로를 돌려준다."""
    ess = resolve_ess(ess_path, degradation_path)  # 운영계획과 같은 설정 해석
    rec: RecParams | None = load_rec(rec_path)
    pred = fi.load_generation_predictions(solar_path, wind_path, plants_path)
    portfolio = fi.aggregate_portfolio(pred, fi.expected_assets(plants_path))
    smp = fi.load_smp_forecast(smp_path)
    all_windows, plan_excluded = fi.build_plan_windows(portfolio, smp, start, end)
    if not all_windows:
        raise RuntimeError(f"만든 창이 없습니다. 제외 사유: {plan_excluded.to_dict('records')}")
    # 비교는 예측·실측이 모두 있는 날만, 두 시나리오를 같은 날짜 집합으로 돈다.
    # 실측만 없는 날(계획은 가능)은 stage=comparison 으로 제외표에 남긴다.
    windows = fi.comparable_windows(all_windows)
    comparison_excluded = fi.comparison_exclusions(all_windows)
    excluded = pd.concat([
        plan_excluded.assign(stage="plan"),
        comparison_excluded.assign(stage="comparison"),
    ], ignore_index=True).sort_values("planning_day").reset_index(drop=True)
    if not windows:
        raise RuntimeError(
            f"비교할 수 있는 창이 없습니다. 제외 사유: {excluded.to_dict('records')}"
        )

    forecast = run_scenario(windows, ess, "forecast", rec=rec)
    actual = run_scenario(windows, ess, "actual", rec=rec)
    hourly_f, hourly_a = stack_hourly(forecast), stack_hourly(actual)
    daily = compare_daily(forecast, actual)
    compare = comparison_checks(forecast, actual, ess)
    warnings = comparison_warnings(forecast, actual)
    plans = plan_checks(forecast, ess) + plan_checks(actual, ess)

    run_id = run_id or f"forecast_{start}_{end}"
    out = new_run_dir(out_dir, run_id, overwrite)
    hourly_f.to_csv(out / "hourly_forecast.csv", index=False, encoding="utf-8-sig")
    hourly_a.to_csv(out / "hourly_actual.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out / "daily_comparison.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(out / "excluded_days.csv", index=False, encoding="utf-8-sig")
    comparison_figure(forecast, actual, ess).write_html(out / "comparison.html",
                                                        include_plotlyjs="cdn")

    totals = {"forecast": scenario_totals(forecast), "actual": scenario_totals(actual)}
    summary = {
        "mode": MODE,
        "run_id": run_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": totals["forecast"]["days"],
        "excluded_days": int(len(plan_excluded)),
        "comparison_excluded_days": int(len(comparison_excluded)),
        "plan_only_days_note": "comparison_excluded_days 는 예측은 있어 계획은 가능하지만 "
                               "실측이 없어 비교에서 뺀 날",
        "policy": "rolling48",
        "horizon_hours": 48.0,
        "commit_hours": float(COMMIT_HOURS),
        "terminal": "구간 마지막 계획일은 24시간만 풀고 E[끝] ≥ 구간 시작 에너지",
        "scenarios": totals,
        "planned_objective_diff_krw": (
            totals["forecast"]["planned_objective_krw"] - totals["actual"]["planned_objective_krw"]
        ),
        "planned_objective_diff_note": "예측이 맞았다고 가정한 계획 수익의 차이. 실제 손익 아님",
        "rec": rec_metadata(rec),
        # 구간 시작 저장분을 전부 팔았을 때 붙을 수 있는 REC 의 보수적 상한 (구간마다 실제 시작
        # 잔량 × 그 구간의 최고 가산). 보정값.
        "initial_energy_rec_upper_bound_krw": (
            0.0 if rec is None
            else float(sum(rec.initial_energy_rec_upper_bound_krw(
                ess, r.result.hourly["ts"],
                initial_energy_mwh=r.result.period_initial_energy_mwh) for r in actual))
        ),
        "inputs": {
            "solar_predictions": str(solar_path), "wind_predictions": str(wind_path),
            "smp_forecast": str(smp_path), "plants": str(plants_path), "ess": str(ess_path),
            "degradation_config": str(degradation_path or ess_path),
            "rec_config": None if rec_path is None else str(rec_path),
            "generation_forecast_issue": "D-2 18 UTC (GFS), forecast_day_offset 0·1",
            "smp_forecast_issue": "D-1 11:00 KST, horizon 1·2",
            "window_sources": [s for r in forecast for s in r.sources],
        },
        "ess": asdict(ess),
        "limitations": LIMITATIONS,
    }
    _write_json(out / "summary.json", summary)
    validation = {
        "run_id": run_id,
        "mode": MODE,
        "comparison_checks": compare,
        "comparison_all_passed": all(c["passed"] for c in compare),
        "comparison_warnings": warnings,
        "plan_checks": plans,
        "plan_checks_all_passed": all(p["all_passed"] for p in plans),
        "data_checks": {
            "windows_built": int(len(all_windows)),
            "comparable_windows": int(len(windows)),
            "excluded_days": excluded.assign(
                planning_day=excluded["planning_day"].astype(str)
            ).to_dict("records"),
            "prediction_rows": int(len(pred)),
            "assets": {k: len(v) for k, v in fi.expected_assets(plants_path).items()},
        },
        "limitations": LIMITATIONS,
    }
    _write_json(out / "validation.json", validation)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument("--solar", type=Path, default=fi.DEFAULT_SOLAR_PREDICTIONS)
    parser.add_argument("--wind", type=Path, default=fi.DEFAULT_WIND_PREDICTIONS)
    parser.add_argument("--smp", type=Path, default=fi.DEFAULT_SMP_FORECAST)
    parser.add_argument("--plants", type=Path, default=fi.PLANT_MASTER_PATH)
    parser.add_argument("--ess", default="config/ess_v0.yaml")
    parser.add_argument("--degradation-config", default=None,
                        help="열화비 설정만 가져올 YAML (예: config/ess_v0_1.yaml). "
                             "없으면 --ess 값")
    parser.add_argument("--rec", default=None,
                        help="REC 수익 가정 YAML (예: config/rec_v0.yaml). 없으면 REC 미적용")
    parser.add_argument("--out", default="data/processed/backtest")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="같은 run-id 결과 폴더 덮어쓰기 허용")
    args = parser.parse_args(argv)
    out = run_forecast_backtest(
        args.start, args.end, solar_path=args.solar, wind_path=args.wind, smp_path=args.smp,
        plants_path=args.plants, ess_path=args.ess, degradation_path=args.degradation_config,
        rec_path=args.rec, out_dir=args.out, run_id=args.run_id, overwrite=args.overwrite,
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    print(f"결과 폴더: {out}")
    head = {k: v for k, v in summary.items() if k not in ("inputs", "limitations", "ess")}
    print(json.dumps(head, ensure_ascii=False, indent=2, default=_json_default))
    for w in validation.get("comparison_warnings", []):
        if w["flagged"]:
            print(f"⚠️ 확인 필요: {w['item']} — {w['note']}")
    failed = [c for c in validation["comparison_checks"] if not c["passed"]]
    if not validation["plan_checks_all_passed"]:
        failed.append({"item": "plan_checks_all_passed", "passed": False})
    if failed:
        print("❌ 비교 조건·검산 불일치:")
        for c in failed:
            print(f"  - {c['item']}: {c.get('value')}")
        raise SystemExit(1)
    print("✅ 비교 조건 검사·제약 검산 모두 통과")


if __name__ == "__main__":
    main()
