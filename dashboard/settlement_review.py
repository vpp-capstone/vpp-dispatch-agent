"""저장된 1시간 정산 결과만 조회한다. 운영 파이프라인·최적화는 실행하지 않는다."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from vpp.hourly_settlement import hourly_settlement_totals

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "data/processed/settlement/stage1_existing_execution_fcr"
EXAMPLE_RESULT = ROOT / "data/processed/settlement/stage1_sample_view"
BRANCH_LABELS = {
    "normal": "양수 가격 · 급전 목표 ≥ 공급가능용량",
    "dispatch_reduction": "양수 가격 · 급전 목표 < 공급가능용량",
    "nonpositive_rt": "실시간 정산가격 ≤ 0",
    "unknown": "급전 입력 부족 · 판단 대기",
}


def money(value):
    return "미계산" if value is None or pd.isna(value) else f"{value:,.0f} 원"


def number(value, unit):
    return "미입력" if pd.isna(value) else f"{value:,.6g} {unit}"


def render():
    st.set_page_config(page_title="1시간 정산 확인", layout="wide")
    st.title("1시간 정산 확인")
    st.caption("저장된 실행 결과의 하루전·실시간 정산과 FCR 가격 연결을 확인합니다.")
    mode = st.sidebar.radio("확인할 결과", ["기존 실행 결과", "조건별 검산 예제"])
    default_folder = EXAMPLE_RESULT if mode == "조건별 검산 예제" else DEFAULT_RESULT
    folder = Path(st.sidebar.text_input(
        "정산 결과 폴더", str(default_folder), key=f"result_folder_{mode}",
    )).expanduser()
    try:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        hourly = pd.read_csv(folder / "hourly_settlement.csv")
        hourly["ts"] = pd.to_datetime(hourly["ts"], utc=True).dt.tz_convert("Asia/Seoul")
        hourly_settlement_totals(hourly)  # 기존 백테스트 폴더를 새 정산으로 오해하지 않는다.
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"1시간 정산 결과를 읽을 수 없습니다: {exc}")
        st.info("hourly_settlement 명령으로 생성한 결과 폴더를 선택하세요.")
        return
    if mode == "조건별 검산 예제":
        st.info(
            "가격·급전값이 모두 가상인 기존 7시간 손계산 예제입니다. 실제 시장 실적은 아닙니다."
        )

    dates = hourly["ts"].dt.date
    period = st.sidebar.date_input(
        "조회 기간", value=(dates.min(), dates.max()), min_value=dates.min(), max_value=dates.max(),
    )
    if len(period) != 2:
        st.info("시작일과 종료일을 선택하세요.")
        return
    block = hourly.loc[dates.between(*period)].copy()
    if block.empty:
        st.info("선택 기간에 저장된 실행 결과가 없습니다.")
        return
    totals = hourly_settlement_totals(block)
    a, b, c = st.columns(3)
    linked_hours = len(block) - totals["fcr_price_missing_hours"]
    a.metric("FCR 가격 연결", f"{linked_hours:,}/{len(block):,} 시간")
    b.metric("에너지 정산액", money(totals["energy_revenue_krw"]))
    c.metric("페널티 차감 전 수익", money(totals["net_before_penalty_krw"]))
    a, b = st.columns(2)
    a.metric("임밸런스 페널티", money(totals["imbalance_penalty_krw"]))
    b.metric("최종 순수익", money(totals["realized_net_krw"]))
    if totals["penalty_missing_hours"]:
        st.warning(
            f"{totals['penalty_missing_hours']:,}시간의 페널티를 계산할 입력이 부족합니다. "
            "FCR 가격 연결과 별도로 급전 목표·공급가능용량 등의 입력이 필요합니다."
        )
    if totals["counterfactual_hours"]:
        st.caption(
            "규칙 시행 전 자료에 현행 규칙을 적용한 1시간 근사이며, 당시 공식 정산액은 아닙니다."
        )
    if totals["missing_execution_hours_between_start_end"]:
        missing_hours = totals["missing_execution_hours_between_start_end"]
        st.caption(
            f"조회 범위 중 원래 실행 결과가 없는 {missing_hours:,}"
            "시간은 계산에서 제외되어 있습니다."
        )

    st.subheader("가격 비교")
    st.line_chart(block.set_index("ts")[["smp_da", "smp_rt_mean", "fcr_price_krw_per_kwh"]].rename(
        columns={
            "smp_da": "하루전 SMP", "smp_rt_mean": "실시간 SMP", "fcr_price_krw_per_kwh": "FCR",
        },
    ))
    st.caption("KST 시간 시작 기준. SMP 원/kWh, FCR 원/kW-h의 시간 평균 가격입니다.")
    st.subheader("확정 판매량과 실제 공급량")
    st.line_chart(block.set_index("ts")[["committed_sell_mwh", "supply_mwh"]].rename(
        columns={"committed_sell_mwh": "확정 판매량 (MWh)", "supply_mwh": "실제 공급량 (MWh)"},
    ))
    st.subheader("선택한 시간의 적용 가격과 계산")
    selected_time = st.selectbox(
        "확인할 시간", block["ts"].tolist(), format_func=lambda ts: ts.strftime("%Y-%m-%d %H:%M"),
    )
    row = block.loc[block["ts"].eq(selected_time)].iloc[0]
    st.write(
        f"확정 판매 정산: {number(row.committed_sell_mwh, 'MWh')} × "
        f"{number(row.day_ahead_price_krw_per_mwh / 1000, '원/kWh')} × 1,000 "
        f"= {money(row.committed_sales_krw)}"
    )
    st.write(
        f"편차 정산: ({number(row.supply_mwh, 'MWh')} − "
        f"{number(row.committed_sell_mwh, 'MWh')}) × "
        f"{number(row.realtime_price_krw_per_mwh / 1000, '원/kWh')} × 1,000 "
        f"= {money(row.deviation_krw)}"
    )
    st.caption("위 정산 단가는 시장 SMP에 설정의 손실계수를 적용한 값입니다.")
    a, b = st.columns(2)
    a.metric("적용 조건", BRANCH_LABELS.get(row.imbalance_branch, row.imbalance_branch))
    b.metric("허용오차율", "판단 대기" if pd.isna(row.tolerance_ratio)
             else f"{row.tolerance_ratio * 100:g}%")
    st.write(
        f"급전 목표: {number(row.set_point_mw, 'MW')} / "
        f"공급가능용량: {number(row.available_capacity_mw, 'MW')} / "
        f"발전설비 용량: {number(row.resource_capacity_mw, 'MW')}"
    )
    if row.imbalance_status == "computed":
        st.write(
            f"급전 편차: {number(row.dispatch_deviation_mwh, 'MWh')} / "
            f"허용 편차: {number(row.allowed_deviation_mwh, 'MWh')}"
        )
        if row.penalized_mwh == 0:
            st.write("허용범위 이내이므로 페널티가 없습니다.")
        elif row.dispatch_deviation_mwh < 0 and row.imbalance_branch != "normal":
            st.write("미달 발전 면제 조건이므로 페널티 단가는 0입니다.")
        elif row.imbalance_branch == "normal":
            st.write(f"페널티 단가: FCR 가격 {number(row.fcr_price_krw_per_kwh, '원/kWh')}")
        elif row.imbalance_branch == "dispatch_reduction":
            st.write(
                "페널티 단가: max(실시간 정산가격 "
                f"{number(row.realtime_price_krw_per_mwh / 1000, '')} − 최소입찰가격 "
                f"({number(row.min_offer_price_krw_per_kwh, '')}), 0) + FCR "
                f"{number(row.fcr_price_krw_per_kwh, '')} = "
                f"{number(row.penalty_price_krw_per_mwh / 1000, '원/kWh')}"
            )
        elif row.imbalance_branch == "nonpositive_rt":
            st.write(
                "페널티 단가: max(−입찰하한가격 "
                f"({number(row.offer_floor_krw_per_kwh, '')}), 0) × 2.5 + FCR "
                f"{number(row.fcr_price_krw_per_kwh, '')} = "
                f"{number(row.penalty_price_krw_per_mwh / 1000, '원/kWh')}"
            )
        st.write(
            f"페널티: 허용범위 밖 {number(row.penalized_mwh, 'MWh')} × "
            f"{number(row.penalty_price_krw_per_mwh / 1000, '원/kWh')} × 1,000 "
            f"= {money(row.imbalance_penalty_krw)}"
        )
        st.caption("면제 조건이면 허용범위 밖 물량이 있어도 적용 단가가 0이 됩니다.")
    else:
        st.info(f"페널티 계산에 부족한 입력: {row.imbalance_missing_inputs}")
    st.subheader("시간별 정산과 누락 입력")
    cols = [
        "ts", "smp_da", "smp_rt_mean", "fcr_price_krw_per_kwh", "fcr_price_source",
        "sales_deviation_mwh", "committed_sales_krw", "deviation_krw", "rec_krw", "wear_krw",
        "set_point_mw", "available_capacity_mw", "imbalance_branch", "tolerance_ratio",
        "dispatch_deviation_mwh", "allowed_deviation_mwh", "penalized_mwh",
        "penalty_price_krw_per_mwh",
        "net_before_penalty_krw", "imbalance_penalty_krw", "realized_net_krw",
        "settlement_missing_inputs",
    ]
    st.dataframe(block[cols], hide_index=True, use_container_width=True)
    st.download_button(
        "선택 기간 CSV 내려받기", block.to_csv(index=False).encode("utf-8-sig"),
        file_name="hourly_settlement_selected.csv", mime="text/csv",
    )
    with st.expander("정산 입력과 적용 설정"):
        st.json({"inputs": summary["inputs"], "parameters": summary["parameters"]})


if __name__ == "__main__":
    render()
