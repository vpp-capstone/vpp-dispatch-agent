"""담당: 최적화·대시보드팀 — ESS v0 완전예측 백테스트 CLI.

    python -m vpp.backtest --raw-dir <원본 폴더> --start 2024-06-01 --end 2025-12-31 \
        --ess config/ess_v0.yaml [--policy rolling48|free|equal_initial] \
        [--out data/processed/backtest]

출력: <out>/<실행ID>/hourly_plan.csv, daily_summary.csv, baselines_hourly.csv,
summary.json, validation.json.
실측 하루전 SMP·실측 발전량을 그대로 넣는 완전예측 실험이며 실제 입찰 성과가 아니다.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp.baselines import no_ess, plan_revenue_krw, rule_based
from vpp.evaluation import summarize_backtest
from vpp.optimization import TERMINAL_POLICIES, EssParams, RollingResult, run_rolling
from vpp.portfolio_data import build_portfolio_hourly
from vpp.validation import check_plan, check_properties

LIMITATIONS = [
    "완전예측: 실측 하루전 SMP와 실측 발전량을 그대로 입력했다. "
    "실제 입찰·예측 오차 대응 성과가 아니다.",
    "전량 낙찰 가정: 계획한 판매량 전부가 입력 가격에 팔린다고 본다. 낙찰 모형이 아니다.",
    "REC·실시간 정산(입찰 편차 페널티)·열화 미반영(기본 k=0 원/MWh). "
    "실제 비용이 0이라는 뜻이 아니다.",
    "출력제어 상한 없음: 판매 상한(export cap)을 두지 않았고, "
    "실측 발전량에 이미 포함된 출력제어를 되돌리지 않았다.",
    "가상 단일 포트폴리오: 태양광 18곳 + 풍력 3곳을 네트워크 제약 없이 합산하고 "
    "ESS 1대가 전체에 접근한다고 가정했다.",
    "ESS 가정값(config/ess_v0.yaml)은 설비 사양이 아니다 (status: assumption_not_equipment_spec).",
    "풍력 원본 단위(성산 kWh/kW 표기 모순, 상명 Wh→kWh 배율 전환)는 기관 확인 전 잠정 해석이다.",
    "용량 초과 시간은 플래그만 남기고 기본은 그대로 사용했다 (--clip-to-capacity 로만 클리핑).",
    "결측으로 제외한 날의 앞뒤는 연속된 것처럼 이어 붙여 풀었다 "
    "(잔량 이월과 48시간 창이 제외된 날을 건너뛴다).",
]


def _json_default(value: Any):
    if isinstance(value, pd.Timestamp | datetime):
        return value.isoformat()
    if isinstance(value, np.integer | np.floating | np.bool_):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(text + "\n", encoding="utf-8")


def _daily_table(
    plan: RollingResult, baselines: dict[str, pd.DataFrame], ess: EssParams
) -> pd.DataFrame:
    """창(window)별 요약 + 같은 날짜 범위의 기준 계획 수익."""
    daily = plan.daily.copy()
    daily["date"] = pd.to_datetime(daily["start_ts"]).dt.date
    for name, frame in baselines.items():
        dates = pd.to_datetime(frame["ts"]).dt.date
        per_day = frame.groupby(dates, sort=True).apply(
            lambda f: plan_revenue_krw(f, ess), include_groups=False
        )
        daily[f"{name}_net_revenue_krw"] = daily["date"].map(per_day).astype(float)
        daily[f"milp_gain_vs_{name}_krw"] = (
            daily["objective_krw"] - daily[f"{name}_net_revenue_krw"]
        )
    return daily


def run_backtest(
    raw_dir: str | Path,
    start: str,
    end: str,
    ess_path: str | Path,
    *,
    policy: str = "rolling48",
    horizon_hours: float = 48,
    commit_hours: float = 24,
    clip_to_capacity: bool = False,
    market_path: str | Path | None = None,
    out_dir: str | Path = "data/processed/backtest",
    run_id: str | None = None,
) -> Path:
    """백테스트 전체를 실행하고 결과 폴더 경로를 반환."""
    ess = EssParams.from_yaml(ess_path)
    frame, data_checks = build_portfolio_hourly(
        raw_dir, start, end, clip_to_capacity=clip_to_capacity,
        **({} if market_path is None else {"market_path": market_path}),
    )
    if frame.empty:
        raise RuntimeError(f"포함된 날이 없습니다. 제외 사유: {data_checks['excluded_days']}")
    plan = run_rolling(
        frame, ess, horizon_hours=horizon_hours, commit_hours=commit_hours, policy=policy
    )
    baselines = {"no_ess": no_ess(frame), "rule_based": rule_based(frame, ess)}
    summary = summarize_backtest(plan, baselines, ess)
    daily = _daily_table(plan, baselines, ess)

    run_id = run_id or f"{start}_{end}_{policy}_{datetime.now():%Y%m%d%H%M%S}"
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    plan.hourly.to_csv(out / "hourly_plan.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out / "daily_summary.csv", index=False, encoding="utf-8-sig")
    merged = frame[["ts", "generation_mwh", "price_krw_per_mwh"]].copy()
    for name, b in baselines.items():
        for col in ("sell_mwh", "charge_mwh", "discharge_mwh", "curtailment_mwh",
                    "energy_end_mwh"):
            merged[f"{name}_{col}"] = b[col].to_numpy()
    merged.to_csv(out / "baselines_hourly.csv", index=False, encoding="utf-8-sig")

    summary.update({
        "run_id": run_id, "start": start, "end": end,
        "horizon_hours": plan.horizon_hours, "commit_hours": plan.commit_hours,
        "excluded_days": len(data_checks["excluded_days"]),
    })
    _write_json(out / "summary.json", summary)
    validation = {
        "run_id": run_id,
        "mode": summary["mode"],
        "assumptions": {**ess.to_dict(), "source": str(ess_path), "policy": policy},
        "limitations": LIMITATIONS,
        "data_checks": data_checks,
        "plan_checks": plan.checks,
        "windows_solved": int(len(plan.daily)),
        "max_mip_gap": float(plan.daily["mip_gap"].max()),
        "plan_checks_full_period": check_plan(
            plan.hourly, ess, initial_energy_mwh=plan.period_initial_energy_mwh,
            objective_krw=plan.objective_krw,
        ),
        "properties": check_properties(summary),
    }
    _write_json(out / "validation.json", validation)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ESS v0 완전예측 백테스트 (실제 입찰 성과 아님)")
    parser.add_argument("--raw-dir", default="data/raw/portfolio", help="원본 CSV 4개가 있는 폴더")
    parser.add_argument("--start", default="2024-06-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--ess", default="config/ess_v0.yaml")
    parser.add_argument("--policy", default="rolling48", choices=TERMINAL_POLICIES)
    parser.add_argument("--horizon-hours", type=float, default=48)
    parser.add_argument("--commit-hours", type=float, default=24)
    parser.add_argument("--clip-to-capacity", action="store_true",
                        help="설비용량 초과 시간을 용량으로 클리핑")
    parser.add_argument("--market", default=None,
                        help="가격 대조용 market_hourly.csv (기본: data/processed/market/)")
    parser.add_argument("--out", default="data/processed/backtest")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    out = run_backtest(
        args.raw_dir, args.start, args.end, args.ess, policy=args.policy,
        horizon_hours=args.horizon_hours, commit_hours=args.commit_hours,
        clip_to_capacity=args.clip_to_capacity, market_path=args.market,
        out_dir=args.out, run_id=args.run_id,
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(f"결과 폴더: {out}")
    head = {k: v for k, v in summary.items() if k != "baselines"}
    print(json.dumps(head, ensure_ascii=False, indent=2))
    print(json.dumps(summary["baselines"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
