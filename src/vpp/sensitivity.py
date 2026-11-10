"""ESS 용량·출력·열화비 민감도 분석.

기존 ``run_rolling``을 여러 ESS 파라미터 조합에 반복 적용하고, 같은 입력 기간에서
순수익·열화비·충방전량·기준 방식 대비 차이를 하나의 표로 저장한다.

이 모듈의 결과는 완전예측 백테스트 또는 전달받은 입력 프레임에 대한 시나리오 비교다.
실제 입찰·낙찰·실시간 정산 성과로 해석하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp.baselines import no_ess, rule_based
from vpp.evaluation import summarize_backtest
from vpp.optimization import TERMINAL_POLICIES, EssParams, run_rolling


@dataclass(frozen=True)
class SensitivityConfig:
    """반복할 ESS 시나리오 축."""

    capacities_mwh: tuple[float, ...]
    charge_powers_mw: tuple[float, ...]
    discharge_powers_mw: tuple[float, ...]
    throughput_costs_krw_per_mwh: tuple[float, ...]

    def __post_init__(self) -> None:
        axes = {
            "capacities_mwh": self.capacities_mwh,
            "charge_powers_mw": self.charge_powers_mw,
            "discharge_powers_mw": self.discharge_powers_mw,
            "throughput_costs_krw_per_mwh": self.throughput_costs_krw_per_mwh,
        }
        for name, values in axes.items():
            if not values:
                raise ValueError(f"{name}은 하나 이상이어야 합니다.")
            if not all(np.isfinite(values)):
                raise ValueError(f"{name}에는 유한한 수만 넣을 수 있습니다.")
            if name != "throughput_costs_krw_per_mwh" and not all(v > 0 for v in values):
                raise ValueError(f"{name}은 0보다 커야 합니다.")
            if name == "throughput_costs_krw_per_mwh" and not all(v >= 0 for v in values):
                raise ValueError("throughput_costs_krw_per_mwh은 0 이상이어야 합니다.")

    @property
    def scenario_count(self) -> int:
        return (
            len(self.capacities_mwh)
            * len(self.charge_powers_mw)
            * len(self.discharge_powers_mw)
            * len(self.throughput_costs_krw_per_mwh)
        )


def _tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not out:
        raise ValueError(f"{name}은 하나 이상이어야 합니다.")
    return out


def config_from_base(
    base_ess: EssParams,
    *,
    capacities_mwh: Iterable[float],
    charge_powers_mw: Iterable[float] | None = None,
    discharge_powers_mw: Iterable[float] | None = None,
    throughput_costs_krw_per_mwh: Iterable[float] = (0.0,),
) -> SensitivityConfig:
    """기본 ESS에서 민감도 분석 축을 만든다."""
    return SensitivityConfig(
        capacities_mwh=_tuple(capacities_mwh, "capacities_mwh"),
        charge_powers_mw=_tuple(
            (base_ess.charge_power_mw,)
            if charge_powers_mw is None else charge_powers_mw,
            "charge_powers_mw",
        ),
        discharge_powers_mw=_tuple(
            (base_ess.discharge_power_mw,)
            if discharge_powers_mw is None else discharge_powers_mw,
            "discharge_powers_mw",
        ),
        throughput_costs_krw_per_mwh=_tuple(
            throughput_costs_krw_per_mwh, "throughput_costs_krw_per_mwh"
        ),
    )


def _scenario_ess(base: EssParams, capacity: float, charge_power: float,
                  discharge_power: float, throughput_cost: float) -> EssParams:
    """기본 ESS의 운전 가정은 유지하고 민감도 축만 덮어쓴다."""
    return replace(
        base,
        capacity_mwh=capacity,
        charge_power_mw=charge_power,
        discharge_power_mw=discharge_power,
        throughput_cost_krw_per_mwh=throughput_cost,
    )


def _row(
    summary: dict[str, Any],
    ess: EssParams,
    *,
    scenario_id: int,
    solver_checks_passed: bool,
) -> dict[str, Any]:
    no_ess_summary = summary["baselines"].get("no_ess", {})
    rule_summary = summary["baselines"].get("rule_based", {})
    net = float(summary["net_revenue_krw"])
    no_ess_net = float(no_ess_summary.get("net_revenue_krw", np.nan))
    rule_net = float(rule_summary.get("net_revenue_krw", np.nan))
    row = {
        "scenario_id": scenario_id,
        "capacity_mwh": ess.capacity_mwh,
        "charge_power_mw": ess.charge_power_mw,
        "discharge_power_mw": ess.discharge_power_mw,
        "soc_min": ess.soc_min,
        "soc_max": ess.soc_max,
        "charge_efficiency": ess.charge_efficiency,
        "discharge_efficiency": ess.discharge_efficiency,
        "degradation_model": ess.degradation_model,
        "throughput_basis": ess.throughput_basis,
        "throughput_cost_krw_per_mwh": ess.throughput_cost_krw_per_mwh,
        "days": summary["days"],
        "hours": summary["hours"],
        "sales_revenue_krw": summary["sales_revenue_krw"],
        "throughput_cost_krw": summary["throughput_cost_krw"],
        "net_revenue_krw": net,
        "gain_vs_no_ess_krw": net - no_ess_net,
        "gain_vs_no_ess_pct": summary["baselines"].get("no_ess", {}).get("milp_gain_pct"),
        "gain_vs_rule_based_krw": net - rule_net,
        "gain_vs_rule_based_pct": summary["baselines"].get("rule_based", {}).get("milp_gain_pct"),
        "sold_mwh": summary["sold_mwh"],
        "charge_mwh": summary["charge_mwh"],
        "discharge_mwh": summary["discharge_mwh"],
        "curtailment_mwh": summary["curtailment_mwh"],
        "cycles_per_day": summary["cycles_per_day"],
        "period_initial_energy_mwh": summary["period_initial_energy_mwh"],
        "period_final_energy_mwh": summary["period_final_energy_mwh"],
        "max_mip_gap": float(plan_gap := summary.get("max_mip_gap", np.nan)),
        "solver_checks_passed": solver_checks_passed,
    }
    # summarize_backtest does not carry daily MIP gap; keep this field nullable for
    # callers that add it later without making the normal path depend on it.
    del plan_gap
    return row


def run_sensitivity(
    frame: pd.DataFrame,
    base_ess: EssParams,
    config: SensitivityConfig,
    *,
    policy: str = "rolling48",
    horizon_hours: float = 48,
    commit_hours: float = 24,
    export_cap_col: str | None = None,
) -> pd.DataFrame:
    """입력 프레임에 모든 ESS 시나리오를 적용하고 결과 표를 반환한다."""
    if config.scenario_count > 500:
        raise ValueError("시나리오가 500개를 초과합니다. 축을 줄여 실행 시간을 관리하세요.")
    if frame.empty:
        raise ValueError("민감도 분석 입력 프레임이 비어 있습니다.")
    if policy not in TERMINAL_POLICIES:
        raise ValueError(f"policy는 {TERMINAL_POLICIES} 중 하나여야 합니다.")

    rows: list[dict[str, Any]] = []
    scenario_id = 0
    for capacity in config.capacities_mwh:
        for charge_power in config.charge_powers_mw:
            for discharge_power in config.discharge_powers_mw:
                for throughput_cost in config.throughput_costs_krw_per_mwh:
                    ess = _scenario_ess(
                        base_ess, capacity, charge_power, discharge_power, throughput_cost
                    )
                    plan = run_rolling(
                        frame,
                        ess,
                        horizon_hours=horizon_hours,
                        commit_hours=commit_hours,
                        policy=policy,
                        export_cap_col=export_cap_col,
                    )
                    baselines = {
                        "no_ess": no_ess(frame),
                        "rule_based": rule_based(frame, ess),
                    }
                    summary = summarize_backtest(plan, baselines, ess)
                    summary["max_mip_gap"] = float(plan.daily["mip_gap"].max())
                    checks_passed = all(check["passed"] for check in plan.checks)
                    rows.append(
                        _row(
                            summary,
                            ess,
                            scenario_id=scenario_id,
                            solver_checks_passed=checks_passed,
                        )
                    )
                    scenario_id += 1
    return pd.DataFrame(rows)


def load_input_csv(path: str | Path) -> pd.DataFrame:
    """일반 입력 CSV를 ESS 입력 계약으로 읽는다."""
    frame = pd.read_csv(path)
    required = {"generation_mwh", "price_krw_per_mwh"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"입력 CSV에 컬럼이 없습니다: {missing}")
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"])
    else:
        frame.insert(0, "ts", pd.RangeIndex(len(frame)))
    if "dt_hours" not in frame.columns:
        frame["dt_hours"] = 1.0
    return frame


def save_results(
    results: pd.DataFrame,
    out_dir: str | Path,
    *,
    input_path: str | Path,
    ess_path: str | Path,
    config: SensitivityConfig,
    policy: str,
    horizon_hours: float,
    commit_hours: float,
    source_note: str | None = None,
) -> Path:
    """민감도 결과 CSV와 재현 메타데이터를 저장한다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "sensitivity_results.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "input_path": str(input_path),
        "ess_path": str(ess_path),
        "policy": policy,
        "horizon_hours": horizon_hours,
        "commit_hours": commit_hours,
        "scenario_count": int(len(results)),
        "scenario_axes": asdict(config),
        "source_note": source_note,
        "limitations": [
            "결과는 입력 프레임에 대한 완전예측 또는 시나리오 비교이며 "
            "실제 입찰·정산 성과가 아니다.",
            "ESS 비용·용량·출력은 실제 설비 사양이 아닌 설정값일 수 있다.",
        ],
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESS 용량·열화비 민감도 분석")
    parser.add_argument("--input", required=True, help="generation_mwh/price_krw_per_mwh CSV")
    parser.add_argument("--ess", default="config/ess_v0_1.yaml")
    parser.add_argument("--capacity-mwh", nargs="+", type=float, default=[8, 16, 32])
    parser.add_argument("--charge-power-mw", nargs="+", type=float, default=None)
    parser.add_argument("--discharge-power-mw", nargs="+", type=float, default=None)
    parser.add_argument("--throughput-cost", nargs="+", type=float, default=[0.0])
    parser.add_argument("--policy", choices=TERMINAL_POLICIES, default="rolling48")
    parser.add_argument("--horizon-hours", type=float, default=48)
    parser.add_argument("--commit-hours", type=float, default=24)
    parser.add_argument("--out", default="data/processed/sensitivity")
    parser.add_argument("--source-note", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    frame = load_input_csv(args.input)
    base = EssParams.from_yaml(args.ess)
    config = config_from_base(
        base,
        capacities_mwh=args.capacity_mwh,
        charge_powers_mw=args.charge_power_mw,
        discharge_powers_mw=args.discharge_power_mw,
        throughput_costs_krw_per_mwh=args.throughput_cost,
    )
    results = run_sensitivity(
        frame,
        base,
        config,
        policy=args.policy,
        horizon_hours=args.horizon_hours,
        commit_hours=args.commit_hours,
    )
    out = save_results(
        results,
        args.out,
        input_path=args.input,
        ess_path=args.ess,
        config=config,
        policy=args.policy,
        horizon_hours=args.horizon_hours,
        commit_hours=args.commit_hours,
        source_note=args.source_note,
    )
    best = results.loc[results["net_revenue_krw"].idxmax()]
    print(f"결과 폴더: {out}")
    print(f"시나리오 수: {len(results)}")
    print(
        "최고 순수익: "
        f"{best['net_revenue_krw']:,.0f}원 "
        f"(용량 {best['capacity_mwh']:g} MWh, "
        f"열화비 {best['throughput_cost_krw_per_mwh']:,.2f}원/MWh)"
    )


if __name__ == "__main__":
    main()
