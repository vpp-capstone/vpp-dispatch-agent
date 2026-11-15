"""기존 실행 결과의 독립 사후 정산: 1시간 DA/RT 가격과 임밸런스 페널티.

기존 settlement API와 실행·최적화 호출은 변경하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp.imbalance import hourly_imbalance
from vpp.optimization import EssParams
from vpp.rec import REC_COLUMN
from vpp.settlement_inputs import (
    RULE_EFFECTIVE_DATE,
    HourlySettlementParams,
    prepare_hourly_inputs,
)

HOURLY_REVENUE_COLUMNS = (
    "committed_sales_krw", "deviation_krw", "energy_revenue_krw", "rec_krw", "wear_krw",
    "net_before_penalty_krw", "imbalance_penalty_krw", "realized_net_krw",
)


def settle_hourly(
    execution: pd.DataFrame,
    market: pd.DataFrame,
    ess: EssParams,
    *,
    params: HourlySettlementParams,
    dispatch: pd.DataFrame | None = None,
    fcr: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """시간별 사후 정산. 판매/급전 편차를 분리하고 페널티를 비용으로 한 번 차감한다.

    market: ts, smp_da, smp_rt_mean (원/kWh). fcr는 별도 시간별 예비력 가격이다.
    dispatch 입력은 settlement_inputs 참고.
    계획·실행을 재계산하지 않는다. 입력 시각이 없으면 가격/급전 값을 보간하지 않는다.
    """
    out = prepare_hourly_inputs(execution, market, dispatch, params, fcr=fcr)
    # 기존 정산 결과를 입력받아도 예전 단일 SMP를 새 정산 가격으로 오해하지 않게 한다.
    if "settlement_price_krw_per_mwh" in out:
        out = out.rename(columns={
            "settlement_price_krw_per_mwh": "legacy_settlement_price_krw_per_mwh",
        })
    penalties = pd.DataFrame([
        hourly_imbalance(**{name: row[name] for name in (
            "supply_mwh", "resource_capacity_mw", "set_point_mw", "available_capacity_mw",
            "realtime_price_krw_per_mwh", "fcr_price_krw_per_mwh",
            "min_offer_price_krw_per_mwh", "offer_floor_krw_per_mwh",
        )})
        for row in out.to_dict("records")
    ])
    for col in penalties:
        out[col] = penalties[col]
    out["sales_deviation_mwh"] = out["supply_mwh"] - out["committed_sell_mwh"]
    out["committed_sales_krw"] = out["committed_sell_mwh"] * out["day_ahead_price_krw_per_mwh"]
    out["deviation_krw"] = out["sales_deviation_mwh"] * out["realtime_price_krw_per_mwh"]
    out["energy_revenue_krw"] = out["committed_sales_krw"] + out["deviation_krw"]
    out["rec_krw"] = (
        out["supply_mwh"] * out[REC_COLUMN] if params.rec_mode == "supplied_mwh_once" else 0.0
    )
    out["wear_krw"] = [
        ess.degradation_cost_krw(np.array([c]), np.array([d]))
        for c, d in zip(out["charge_mwh"], out["discharge_mwh"], strict=True)
    ]
    out["net_before_penalty_krw"] = out["energy_revenue_krw"] + out["rec_krw"] - out["wear_krw"]
    out["realized_net_krw"] = out["net_before_penalty_krw"] - out["imbalance_penalty_krw"]
    out["energy_computable"] = out["energy_revenue_krw"].notna()
    out["settlement_computable"] = out["realized_net_krw"].notna()
    out["settlement_missing_inputs"] = [
        ",".join(filter(None, [
            "smp_da" if pd.isna(row.smp_da) else "",
            "smp_rt_mean" if pd.isna(row.smp_rt_mean) else "",
            row.imbalance_missing_inputs,
        ]))
        for row in out.itertuples()
    ]
    out["settlement_basis"] = "hourly_da_rt_with_imbalance"
    out["settlement_status"] = "hourly_approximation_not_official_settlement"
    out["rule_application"] = np.where(
        out["ts"] < pd.Timestamp(RULE_EFFECTIVE_DATE, tz="Asia/Seoul"),
        "current_rule_counterfactual", "effective_rule_hourly_approximation",
    )
    out["day_ahead_loss_factor"] = params.day_ahead_loss_factor
    out["realtime_loss_factor"] = params.realtime_loss_factor
    out["loss_factor_source"] = params.loss_factor_source
    out["loss_factor_note"] = params.loss_factor_note
    out["rec_basis"] = params.rec_mode
    return out


def hourly_settlement_totals(settled: pd.DataFrame) -> dict[str, Any]:
    """일부 누락을 skipna 합계로 숨기지 않는다. 각 항목별 계산 가능 합계를 반환한다."""
    if settled.empty:
        raise ValueError("정산 결과가 비었습니다.")
    out = {
        "hours": len(settled),
        "computable": bool(settled["settlement_computable"].all()),
        "price_missing_hours": int((~settled["energy_computable"]).sum()),
        "fcr_price_missing_hours": int(settled["fcr_price_krw_per_kwh"].isna().sum()),
        "penalty_missing_hours": int(settled["imbalance_status"].ne("computed").sum()),
        "penalty_applied_hours": int(settled["imbalance_penalty_krw"].gt(0).sum()),
        "penalty_zero_hours": int(settled["imbalance_penalty_krw"].eq(0).sum()),
        "counterfactual_hours": int(
            settled["rule_application"].eq("current_rule_counterfactual").sum()
        ),
        "dispatch_source_counts": settled["dispatch_source"].value_counts().to_dict(),
        "basis": "hourly_da_rt_with_imbalance",
        "status": "hourly_approximation_not_official_settlement",
    }
    for col in HOURLY_REVENUE_COLUMNS:
        out[col] = float(settled[col].sum()) if settled[col].notna().all() else None
    times = settled["ts"]
    expected = int((times.max() - times.min()) / pd.Timedelta(hours=1)) + 1
    out["missing_execution_hours_between_start_end"] = expected - len(settled)
    out["start"] = times.min().isoformat()
    out["end"] = times.max().isoformat()
    return out


def hourly_settlement_daily(settled: pd.DataFrame) -> pd.DataFrame:
    """KST 일자별 집계. 계산 가능한 일부 시간만 있는 날은 full_day_coverage=False."""
    rows = []
    for day, group in settled.groupby(settled["ts"].dt.date, sort=True):
        total = hourly_settlement_totals(group)
        total.pop("dispatch_source_counts")
        rows.append({"date": day.isoformat(), "full_day_coverage": len(group) == 24, **total})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="1시간 DA/RT·임밸런스 독립 사후 정산")
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--market", required=True, type=Path)
    parser.add_argument("--dispatch", type=Path)
    parser.add_argument("--fcr", type=Path, help="FCR 시간 가격 CSV (급전 자료와 별도)")
    parser.add_argument("--scenario", help="실행 CSV에 여러 시나리오가 있으면 하나를 선택")
    parser.add_argument(
        "--ess", type=Path,
        help="명시적 ESS 설정. 생략하면 실행 CSV 옆 summary.json의 ess를 재사용",
    )
    parser.add_argument("--config", type=Path, default=Path("config/settlement_jeju_hourly.yaml"))
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args(argv)
    # 다른 실행의 결과를 덮어쓰지 않는다. 계산 오류 시에는 출력 폴더를 만들지 않는다.
    if args.outdir.exists() and (not args.outdir.is_dir() or any(args.outdir.iterdir())):
        parser.error("--outdir은 비어 있는 폴더여야 합니다.")
    params = HourlySettlementParams.from_yaml(args.config)
    source_summary = None
    if args.ess:
        ess = EssParams.from_yaml(args.ess)
        ess_source = str(args.ess.resolve())
    else:
        source_summary_path = args.execution.with_name("summary.json")
        if not source_summary_path.is_file():
            parser.error(
                "기존 ESS 설정이 없습니다. 실행 CSV 옆 summary.json 또는 --ess가 필요합니다."
            )
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        if not isinstance(source_summary, dict) or not isinstance(source_summary.get("ess"), dict):
            parser.error("기존 summary.json에 ess 설정이 없습니다. --ess를 지정하세요.")
        ess = EssParams(**source_summary["ess"])
        ess_source = str(source_summary_path.resolve()) + "#ess"
    execution = pd.read_csv(args.execution)
    if args.scenario:
        if "scenario" not in execution:
            parser.error("--scenario를 지정했지만 실행 CSV에 scenario 열이 없습니다.")
        execution = execution.loc[execution["scenario"].eq(args.scenario)]
    result = settle_hourly(
        execution, pd.read_csv(args.market), ess, params=params,
        dispatch=pd.read_csv(args.dispatch) if args.dispatch else None,
        fcr=pd.read_csv(args.fcr) if args.fcr else None,
    )
    summary = {
        "parameters": params.metadata(), "ess": ess.to_dict(),
        "ess_source": ess_source,
        "source_rec_metadata": source_summary.get("rec") if source_summary else None,
        "inputs": {name: str(getattr(args, name).resolve()) if getattr(args, name) else None
                   for name in ("execution", "market", "dispatch", "fcr", "ess", "config")},
        "scenario": args.scenario,
        "totals": hourly_settlement_totals(result),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.outdir / "hourly_settlement.csv", index=False)
    hourly_settlement_daily(result).to_csv(args.outdir / "daily_settlement.csv", index=False)
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
