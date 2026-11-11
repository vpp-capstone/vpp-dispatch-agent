"""One verified forecast/result release powers every operator screen."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.operator_simulation import render_simulator, replay_payload
from dashboard.overview_data import load_experiment
from dashboard.signal_replay import render_signal

from ui import ROOT
from vpp import operator_pipeline as pipeline

CSS_PATH = Path(__file__).parent / "assets/operator_console.css"


def kpis(items):
    for col, (label, value, hint) in zip(st.columns(len(items)), items, strict=True):
        col.markdown(
            f'<div class="oc-kpi"><small>{escape(label)}</small>'
            f"<strong>{escape(str(value))}</strong><p>{escape(hint)}</p></div>",
            unsafe_allow_html=True,
        )


def money(value):
    if abs(value) >= 1e8:
        return f"{value / 1e8:,.2f}억 원"
    return f"{value / 10000:,.1f}만 원"


def state_key(state):
    return json.dumps(
        {k: state.get(k) for k in ("state", "revision", "active", "stages")},
        sort_keys=True,
        default=str,
    )


@st.fragment(run_every="30s")
def monitor():
    cfg = pipeline.config(ROOT)
    state = pipeline.status(ROOT)
    if st.session_state.get("operator_auto", cfg.get("auto_refresh", True)):
        state = pipeline.launch(ROOT)
    current = state_key(state)
    if current != st.session_state.get("operator_state_key"):
        st.session_state["operator_state_key"] = current
        st.rerun()
    with st.sidebar:
        st.caption("자동 갱신 · 30초 간격" if st.session_state.get("operator_auto", True)
                   else "자동 갱신 일시 중지")


@st.cache_data(show_spinner=False)
def read_report(folder: str, stamp: int):
    return load_experiment(Path(folder))


def flow(state):
    stages = state.get("stages", [])
    files_ok = all(r["ready"] for r in pipeline.requirements(pipeline.config(ROOT)))
    running = state.get("state") in ("running", "queued")
    blocks = [
        ("01 · 공유 파일 확인", "Drive 에서 받은 예측 파일·설정" if files_ok else "없는 파일 있음",
         files_ok),
        ("02 · MILP 운영계획", "파일 내용이 바뀌면 재계산", bool(state.get("active"))),
        ("03 · 화면 갱신", "모든 화면에 같은 결과 묶음",
         bool(state.get("active")) and bool(stages)),
    ]
    st.markdown(
        '<div class="oc-flow">'
        + "".join(
            f'<div class="oc-step"><b><i class="oc-dot {"run" if running else ""} '
            f'{"" if ok else "warn"}"></i>{title}</b><small>{caption}</small></div>'
            for title, caption, ok in blocks
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def model_cards(release):
    """이 결과 묶음이 실제로 읽은 예측 파일. 모델 설명 대신 파일·열·기간을 보여준다."""
    columns = release.get("forecast_columns", [])
    used = ("improved_generation_kwh" if "improved_generation_kwh" in columns
            else ", ".join(columns) or "미기록")
    sources = release.get("source_paths", {})
    rows = [
        {"입력": "태양광 예측", "파일": Path(sources.get("solar", "")).name or "미기록",
         "사용한 열": used, "담당": "발전량"},
        {"입력": "풍력 예측", "파일": Path(sources.get("wind", "")).name or "미기록",
         "사용한 열": used, "담당": "발전량"},
        {"입력": "SMP 예측", "파일": Path(sources.get("smp", "")).name or "미기록",
         "사용한 열": "smp_hat_krw_per_mwh · trained_until은 학습 종료일", "담당": "SMP"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"계획 기간 {release.get('start')} ~ {release.get('end')} · "
               f"{release.get('days')}일 · 계산 결과 묶음 {release.get('id', '')[:8]}")


def plan_frame(data, scenario="forecast"):
    """화면의 '계획'. 실행 결과가 있으면 실행 SOC 에서 시작한 확정 계획, 없으면 예측 계획."""
    if scenario == "forecast":
        plan = data.get("plan")
        return data["forecast"] if plan is None else plan
    return data[scenario]


def day_slice(data, day, scenario="forecast"):
    frame = plan_frame(data, scenario)
    return frame[frame.ts.dt.date == day].reset_index(drop=True)


def day_revenue(data, day, scenario="forecast"):
    """선택일 확정 24시간의 계획 수익 분해 (CLI 와 같은 함수)."""
    from vpp.baselines import plan_revenue_breakdown

    return plan_revenue_breakdown(day_slice(data, day, scenario), data["ess"])


def process_strip(data, day, release, checks_ok):
    """선택일의 핵심 수치만 표시한다. 기간 합계는 수익 화면에서 제공한다."""
    plan = day_slice(data, day)
    rev = day_revenue(data, day)
    steps = ["예측 입력", "운영계획·판매계획", "예상 수익", "실적·정산"]
    st.markdown(
        '<p class="oc-process">'
        + ' <span>→</span> '.join(f"<b>{s}</b>" if i == 1 else s for i, s in enumerate(steps))
        + "</p>",
        unsafe_allow_html=True,
    )
    kpis([
        ("하루 예측 발전량", f"{plan.generation_mwh.sum():,.1f} MWh", "태양광·풍력 21곳 합산"),
        ("하루 판매 계획", f"{plan.sell_mwh.sum():,.1f} MWh", "ESS 충방전을 반영한 판매량"),
        ("하루 예상 순수익", money(rev["net_revenue_krw"]), "SMP + REC − 열화비 · 예측 기준"),
    ])
    st.caption(
        f"하루 ESS 충전 {plan.charge_mwh.sum():,.1f} MWh · 방전 {plan.discharge_mwh.sum():,.1f} MWh"
        f" · 계획 검산 {'통과' if checks_ok else '실패'}"
    )


def bid_table(plan):
    """판매계획 관점의 시간별 표: 판매 전력량, 예측 SMP, REC 가산, ESS 동작."""
    out = pd.DataFrame({
        "시각": plan.ts.dt.strftime("%H:%M"),
        "예측 발전량 MWh": plan.generation_mwh.round(2),
        "판매계획 MWh": plan.sell_mwh.round(2),
        "ESS 충전 MWh": plan.charge_mwh.round(2),
        "ESS 방전 MWh": plan.discharge_mwh.round(2),
        "감축 MWh": plan.curtailment_mwh.round(2),
        "예측 SMP 원/kWh": (plan.price_krw_per_mwh / 1000).round(1),
        "종료 SOC %": (plan.soc_end * 100).round(1),
    })
    if "rec_krw_per_mwh" in plan:
        out.insert(7, "REC 가산 원/kWh", (plan.rec_krw_per_mwh / 1000).round(1))
    return out


def show_settlement(data, day):
    """실적·정산 화면: 계산된 것과 아직 계산하지 않는 것을 한 표에 적는다."""
    has_actual = data["actual"] is not None
    st.subheader(f"{day} 실적·정산 · 계산 범위")
    rev = day_revenue(data, day)
    block = exec_day(data, day)
    t = exec_totals(block) if block is not None else None
    ok = bool(t and t["computable"])

    def realized(key):
        return money(t[key]) if ok else "—"

    rows = [
        {"정산 항목": "확정 판매 정산액 (확정 판매량 × 하루전 가격)",
         "선택일 계획값": money(rev["sales_revenue_krw"]),
         "선택일 실행값": realized("committed_sales_krw"),
         "상태": "가정 기반 실행 수익" if ok else "계획값만",
         "비고": "하루전 가격 자료가 없어 실측 SMP 로 대체 (실험 가정)"},
        {"정산 항목": "편차 정산 (초과 지급 · 부족 환수)", "선택일 계획값": "—",
         "선택일 실행값": realized("deviation_krw"),
         "상태": "가정 기반 실행 수익" if ok else "미계산",
         "비고": "편차 가격 = 실측 SMP, 배수 1.0 · 벌칙 0 (대칭 기본값, 근거 없음)"},
        {"정산 항목": "REC 수익", "선택일 계획값": money(rev["rec_revenue_krw"]),
         "선택일 실행값": realized("rec_krw"),
         "상태": "실험 가정", "비고": "실제 공급 1 MWh당 1회 · 가중치 1.0 · 시나리오 단가"},
        {"정산 항목": "열화비 (차감)", "선택일 계획값": money(-rev["throughput_cost_krw"]),
         "선택일 실행값": money(-t["wear_krw"]) if ok else "—",
         "상태": "실행 충방전 기준" if ok else "계획값만", "비고": "충전+방전 MWh × 단가"},
        {"정산 항목": "실행 순수익", "선택일 계획값": money(rev["net_revenue_krw"]),
         "선택일 실행값": realized("realized_net_krw"),
         "상태": "가정 기반 실행 수익" if ok else "미계산",
         "비고": "계획값 열은 예측이 맞았다고 가정한 예상 순수익"},
        {"정산 항목": "발전량 예측 정산금", "선택일 계획값": "—", "선택일 실행값": "—",
         "상태": "미구현",
         "비고": "예측제도 오차율 구간(4%·6%)별 원/kWh. 설비용량 기준 오차율 계산 필요"},
        {"정산 항목": "용량정산금", "선택일 계획값": "—", "선택일 실행값": "—",
         "상태": "미연결", "비고": "최적화팀 모듈(#13) 머지 후 같은 방식으로 연결"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if block is None:
        note = (data.get("execution_summary") or {}).get("note") or "실행 결과 없음"
        st.info(f"실행값 미계산 · {note}")
    elif not ok:
        st.info("이 날은 정산 가격이 없는 시간이 있어 실행 수익을 계산하지 않았습니다.")
    summary = data.get("execution_summary") or {}
    if summary.get("available"):
        sc = summary["scenarios"]
        labels = {"no_ess_executed": "ESS 미사용", "legacy_plan_executed": "기존 계획 방식",
                  "executed": "수정 계획 + 실행 운전"}
        table = pd.DataFrame([
            {"방식": labels[k], "실행 순수익": money(v["realized_net_krw"]) if v["realized_net_krw"]
             is not None else "계산 불가",
             "부족 MWh": round(v["shortfall_mwh"], 1), "초과 MWh": round(v["surplus_mwh"], 1),
             "충전 MWh": round(v["charge_mwh"], 1), "방전 MWh": round(v["discharge_mwh"], 1),
             "열화비": money(v["wear_krw"]) if v["wear_krw"] is not None else "—",
             "종료 잔량 부족 MWh": round(v["executed_end_shortfall_mwh"], 1)}
            for k, v in sc.items() if k in labels
        ])
        st.markdown("**기간 전체 · 같은 실측·정산 조건의 세 방식 비교**")
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("종료 잔량 부족 = 구간 시작보다 비운 잔량. 양수면 배터리를 비워 번 돈이 섞인 "
                   "것이라 순수익만으로 우열을 가리지 않습니다. 실측을 미리 알고 최적화한 비교 "
                   "계획은 예상 수익 화면에만 있습니다 (실제 운영 불가).")
    if not has_actual:
        st.info("실측이 없어 오차와 완전예측 비교는 비어 있습니다. 발전량 parquet 의 "
                "generation_kwh 열과 SMP 파일의 smp_actual 열이 채워지면 아래가 켜집니다.")
        return
    m = data["metrics"]
    kpis(
        [
            ("기간 전체 · 발전량 평균 오차", f"{m['generation_mae_mwh']:.2f} MWh",
             "21곳 합산 · 시간당 절대 오차"),
            ("기간 전체 · SMP 평균 오차", f"{m['smp_mae_krw_per_kwh']:.2f}원/kWh",
             "낮을수록 실제와 가까운 예측"),
            ("0원 이하 가격 적중", f"{m['forecast_low_hits']} / {m['actual_low_hours']}시간",
             "동일 비교 기간의 0원 이하 실제 가격"),
        ]
    )
    st.caption("2024년은 모델 선택에도 사용된 기간입니다. 독립적인 최종 시험 성능과 구분하세요.")


def exec_day(data, day, scenario="executed"):
    """선택일의 실행 운전·정산 결과 (없으면 None). 예측값으로 채우지 않는다."""
    frame = (data.get("execution") or {}).get(scenario)
    if frame is None:
        return None
    block = frame[frame.ts.dt.date == day].reset_index(drop=True)
    return block if len(block) == 24 else None


def exec_totals(block):
    """실행 결과 묶음의 합계 (정산 항목 + 물량). 가격 없는 시간이 있으면 금액 None."""
    computable = bool((block["settlement_basis"] != "price_missing").all())
    money_cols = ("committed_sales_krw", "deviation_krw", "rec_krw", "wear_krw",
                  "realized_net_krw")
    out = {c: (float(block[c].sum()) if computable else None) for c in money_cols}
    out.update({
        "computable": computable,
        "committed_sell_mwh": float(block["committed_sell_mwh"].sum()),
        "supply_mwh": float(block["supply_mwh"].sum()),
        "shortfall_mwh": float(block["shortfall_mwh"].sum()),
        "surplus_mwh": float(block["surplus_mwh"].sum()),
        "charge_mwh": float(block["charge_mwh"].sum()),
        "discharge_mwh": float(block["discharge_mwh"].sum()),
        "planned_charge_mwh": float(block["planned_charge_mwh"].sum()),
        "planned_discharge_mwh": float(block["planned_discharge_mwh"].sum()),
        "start_soc": float(block["energy_start_mwh"].iloc[0]),
        "end_soc": float(block["energy_end_mwh"].iloc[-1]),
        "planned_end_soc": float(block["planned_energy_end_mwh"].iloc[-1]),
    })
    return out


def show_execution_day(data, day):
    """운영계획 화면 하단: 확정 계획 vs 실행 (충방전·공급·SOC·이월). 실측 없으면 사유만."""
    st.markdown("#### 계획 대비 실행")
    block = exec_day(data, day)
    if block is None:
        note = (data.get("execution_summary") or {}).get("note") or "실행 결과 없음"
        st.info(f"미계산 · {note}")
        return
    ess = data["ess"]
    t = exec_totals(block)
    cap = ess.capacity_mwh
    kpis([
        ("확정 판매 → 실제 공급", f"{t['committed_sell_mwh']:,.1f} → {t['supply_mwh']:,.1f} MWh",
         f"부족 {t['shortfall_mwh']:,.1f} · 초과 {t['surplus_mwh']:,.1f} MWh"),
        ("충전 계획 → 실행", f"{t['planned_charge_mwh']:,.1f} → {t['charge_mwh']:,.1f} MWh",
         "계획에 없던 충전은 팔 수 없는 전력에만"),
        ("방전 계획 → 실행", f"{t['planned_discharge_mwh']:,.1f} → {t['discharge_mwh']:,.1f} MWh",
         "부족 시 충전 축소 후 추가 방전"),
        ("종료 SOC 계획 → 실행", f"{t['planned_end_soc'] / cap:.0%} → {t['end_soc'] / cap:.0%}",
         f"오늘 시작 {t['start_soc'] / cap:.0%} · 실행 종료 잔량이 내일 시작값"),
    ])
    daily = data.get("execution_daily")
    if daily is not None:
        rows = daily[(daily.scenario == "executed")].sort_values("planning_day")
        today = rows[rows.planning_day == day]
        if len(today):
            row = today.iloc[0]
            if bool(row.get("segment_start", False)):
                reason = row.get("segment_start_reason", "")
                st.caption(f"오늘은 새 계산 구간의 시작입니다 ({reason}). "
                           "전날 실행 SOC 를 이월하지 않고 설정값에서 시작합니다.")
            else:
                prev = rows[rows.planning_day < day].iloc[-1]
                st.caption(
                    f"전날 실행 종료 SOC {prev.executed_final_energy_mwh / cap:.1%} → 오늘 시작 "
                    f"{row.exec_start_energy_mwh / cap:.1%} · 전날 계획 종료 SOC 는 "
                    f"{prev.planned_final_energy_mwh / cap:.1%} 였습니다."
                )
    fig = go.Figure([
        go.Bar(x=block.ts, y=block.committed_sell_mwh, name="확정 판매량", marker_color="#c9d6de"),
        go.Bar(x=block.ts, y=block.supply_mwh, name="실제 공급량", marker_color="#1f8f7a"),
        go.Scatter(x=block.ts, y=block.planned_energy_end_mwh / cap * 100, name="계획 SOC %",
                   yaxis="y2", mode="lines", line=dict(color="#5f7481", dash="dot")),
        go.Scatter(x=block.ts, y=block.soc_end * 100, name="실행 SOC %", yaxis="y2",
                   mode="lines", line=dict(color="#173a44", width=2)),
    ])
    fig.update_layout(
        barmode="group", height=320, margin=dict(t=30, b=10), paper_bgcolor="#fff",
        plot_bgcolor="#fff", xaxis_tickformat="%H시", yaxis_title="MWh",
        yaxis2=dict(title="SOC %", overlaying="y", side="right", range=[0, 100],
                    showgrid=False),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(gridcolor="#e1e8ed")
    st.plotly_chart(fig, width="stretch")
    with st.expander("시간별 계획 vs 실행 표"):
        table = pd.DataFrame({
            "시각": block.ts.dt.strftime("%H:%M"),
            "예측 발전량": block.forecast_generation_mwh.round(2),
            "실측 발전량": block.actual_generation_mwh.round(2),
            "확정 판매량": block.committed_sell_mwh.round(2),
            "실제 공급량": block.supply_mwh.round(2),
            "부족": block.shortfall_mwh.round(2), "초과": block.surplus_mwh.round(2),
            "충전 계획": block.planned_charge_mwh.round(2), "충전 실행": block.charge_mwh.round(2),
            "방전 계획": block.planned_discharge_mwh.round(2),
            "방전 실행": block.discharge_mwh.round(2),
            "SOC 계획 %": (block.planned_energy_end_mwh / cap * 100).round(1),
            "SOC 실행 %": (block.soc_end * 100).round(1),
            "적용 규칙": block.get("actions", pd.Series("", index=block.index)).fillna(""),
        })
        st.dataframe(table, hide_index=True, width="stretch")
    st.caption("실행 운전은 판매계획을 고정한 규칙 방식(수익 최적 아님)입니다. "
               "정산은 실측 SMP 를 하루전·편차 가격으로 쓴 실험 가정입니다.")


def period_revenue(data, frame):
    """임의 구간의 계획 수익 분해 + ESS 없이 즉시 판매한 경우와의 차이."""
    from vpp.baselines import no_ess, plan_revenue_breakdown

    rev = plan_revenue_breakdown(frame, data["ess"])
    base = plan_revenue_breakdown(no_ess(frame), data["ess"])
    rev["ess_gain_krw"] = rev["net_revenue_krw"] - base["net_revenue_krw"]
    return rev


def show_summary(data, day, checks_ok):
    """홈: 오늘 계획 · 어제 실적 · 이번 달 누적 · 서비스 효과. 없는 값은 비워 둔다."""
    import datetime as dt

    plan = day_slice(data, day)
    today = day_revenue(data, day)
    whole = plan_frame(data)
    dates = whole.ts.dt.date
    month = whole[(dates >= day.replace(day=1)) & (dates <= day)]
    mtd = period_revenue(data, month)
    yesterday = day - dt.timedelta(days=1)
    y_exec = exec_day(data, yesterday)
    if y_exec is not None and exec_totals(y_exec)["computable"]:
        ty = exec_totals(y_exec)
        y_value = money(ty["realized_net_krw"])
        y_hint = (f"실행 수익 (가정 기반 정산) · 공급 {ty['supply_mwh']:,.0f} / 확정 "
                  f"{ty['committed_sell_mwh']:,.0f} MWh · 부족 {ty['shortfall_mwh']:,.1f} MWh")
    elif data["actual"] is not None and (dates == yesterday).any():
        f = day_slice(data, yesterday)
        a = day_slice(data, yesterday, "actual")
        err = float((f.generation_mwh - a.generation_mwh).abs().mean())
        y_value = f"오차 {err:.1f} MWh/시간"
        y_hint = "실측 대비 발전량 예측 오차 · 실행 수익은 정산 가격 없음"
    else:
        y_value, y_hint = "실측 대기", "실측 발전량 파일이 오면 실행 수익과 오차를 계산"
    gain = mtd["ess_gain_krw"]
    gain_hint = (("ESS 없이 즉시 판매보다 감소" if gain < -0.01 else "ESS 없이 즉시 판매 대비")
                 + " · 계획 기준")
    daily = data.get("execution_daily")
    if daily is not None:
        month_mask = (daily.planning_day >= day.replace(day=1)) & (daily.planning_day <= day)
        exe = daily[(daily.scenario == "executed") & month_mask]
        base = daily[(daily.scenario == "no_ess_executed") & month_mask]
        if len(exe) and len(base) and exe["computable"].all() and base["computable"].all():
            gain = float(exe["realized_net_krw"].sum() - base["realized_net_krw"].sum())
            gain_hint = (("ESS 없이 운전했을 때보다 감소" if gain < -0.01
                          else "ESS 없이 운전했을 때 대비") + " · 실행 수익 기준")
    st.subheader(f"{day:%m월 %d일} 요약")
    kpis([
        ("오늘 판매계획", f"{plan.sell_mwh.sum():,.1f} MWh",
         f"예상 순수익 {money(today['net_revenue_krw'])} · "
         f"검산 {'통과' if checks_ok else '실패'}"),
        ("어제 실적", y_value, y_hint),
        ("이번 달 누적 예상 수익", money(mtd["net_revenue_krw"]),
         f"{day:%m/01}~{day:%m/%d} 계획 기준 · REC {money(mtd['rec_revenue_krw'])} 포함"),
        ("서비스 효과 (ESS)", money(gain), gain_hint + " · 예측 효과는 기준선 확정 후"),
    ])
    fig = go.Figure([
        go.Bar(x=plan.ts, y=plan.generation_mwh, name="예측 발전량", marker_color="#c9d6de"),
        go.Scatter(x=plan.ts, y=plan.sell_mwh, name="판매계획", mode="lines+markers",
                   line=dict(color="#173a44", width=2)),
    ])
    fig.update_layout(title="오늘 시간별 판매계획", height=280, yaxis_title="MWh",
                      margin=dict(t=50, b=10), paper_bgcolor="#fff", plot_bgcolor="#fff",
                      xaxis_tickformat="%H시", legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(gridcolor="#e1e8ed")
    st.plotly_chart(fig, width="stretch")
    st.caption("시간별 ESS 동작과 CSV는 운영계획, 금액 분해는 예상 수익, "
               "계산 범위는 실적·정산 화면에 있습니다.")


def show_forecasts(data, day, release):
    st.subheader(f"{day} 예측 입력 · 발전량·SMP")
    plan = day_slice(data, day)
    if data["actual"] is None:
        left, right = st.columns(2)
        for col, column, title, scale, unit in (
            (left, "generation_mwh", "예측 발전량 (21곳 합산)", 1, "MWh"),
            (right, "price_krw_per_mwh", "예측 SMP", 0.001, "원/kWh"),
        ):
            fig = go.Figure(go.Scatter(x=plan.ts, y=plan[column] * scale, mode="lines+markers",
                                       line=dict(color="#173a44")))
            fig.update_layout(title=title, height=260, yaxis_title=unit, margin=dict(t=50, b=10),
                              paper_bgcolor="#fff", plot_bgcolor="#fff", xaxis_tickformat="%H시")
            fig.update_yaxes(gridcolor="#e1e8ed")
            col.plotly_chart(fig, width="stretch")
        with st.expander("입력 상세 · 파일과 예측 열"):
            model_cards(release)
        return
    st.caption("재생을 누르면 예측선과 실측선을 같은 시간으로 따라갑니다.")
    choice = st.radio(
        "비교할 예측",
        ["전체 발전량", "태양광", "풍력", "SMP 가격"],
        horizontal=True,
        key="oc_signal",
    )
    f, a = data["forecast"], data["actual"]
    f, a = f[f.ts.dt.date == day], a[a.ts.dt.date == day]
    if choice in ("태양광", "풍력"):
        from vpp import forecast_inputs as fi
        from vpp import forecast_metrics as fm

        sources = release["inputs"]
        pred = fm.load_generation_eval_frame(
            Path(sources["solar"]), Path(sources["wind"]), Path(sources["plants"])
        )
        series = fm.generation_timeseries(
            pred,
            fi.expected_assets(Path(sources["plants"])),
            level="energy_type",
            group="solar" if choice == "태양광" else "wind",
            forecast_day_offset=0,
            start=day,
            end=day,
        )
        rows = [
            {"hour": r.ts.strftime("%H:%M"), "f": r.lightgbm_kwh / 1000, "a": r.actual_kwh / 1000}
            for r in series.itertuples()
            if pd.notna(r.lightgbm_kwh) and pd.notna(r.actual_kwh)
        ]
        unit = "MWh"
    else:
        column, divisor, unit = (
            ("price_krw_per_mwh", 1000, "원/kWh")
            if choice == "SMP 가격"
            else ("generation_mwh", 1, "MWh")
        )
        rows = [
            {"hour": t.strftime("%H:%M"), "f": float(p / divisor), "a": float(q / divisor)}
            for t, p, q in zip(f.ts, f[column], a[column], strict=True)
        ]
    render_signal(rows, title=f"{choice} · 예측과 실측", day=day, unit=unit)
    m = data["metrics"]
    day_err = float((plan.generation_mwh - day_slice(data, day, "actual").generation_mwh)
                    .abs().mean())
    kpis(
        [
            ("선택일 · 발전량 오차", f"{day_err:.2f} MWh", "시간당 절대 오차 평균"),
            ("기간 전체 · 발전량 평균 오차", f"{m['generation_mae_mwh']:.2f} MWh",
             "21곳 합산 · 시간당 절대 오차"),
            ("기간 전체 · SMP 평균 오차", f"{m['smp_mae_krw_per_kwh']:.2f}원/kWh",
             "낮을수록 실제와 가까운 예측"),
        ]
    )
    st.caption("2024년은 모델 선택에도 사용된 기간입니다. 독립적인 최종 시험 성능과 구분하세요.")
    with st.expander("입력 상세 · 파일과 예측 열"):
        model_cards(release)


def show_profit(data, day):
    from vpp.baselines import no_ess, plan_revenue_breakdown

    st.subheader("예상 수익")
    scope = st.radio("집계 기간", ["선택한 하루", "전체 기간"], horizontal=True,
                     key="oc_profit_scope")
    scenario = "forecast"
    if data["actual"] is not None:
        options = ["예측으로 세운 계획의 예상 수익", "확정 계획을 실측에 적용한 실행 수익",
                   "실측을 미리 알고 세운 비교 계획"]
        if not (data.get("execution") or {}).get("executed") is not None:
            options.remove("확정 계획을 실측에 적용한 실행 수익")
        mode = st.radio("계산 기준", options, horizontal=True, key="oc_profit_basis")
        if mode.startswith("확정"):
            show_realized_profit(data, day, scope)
            return
        scenario = "actual" if mode.startswith("실측") else "forecast"
    if scope == "선택한 하루":
        plan = day_slice(data, day, scenario)
        rev = day_revenue(data, day, scenario)
        base = plan_revenue_breakdown(no_ess(plan), data["ess"])
        rev["ess_gain_krw"] = rev["net_revenue_krw"] - base["net_revenue_krw"]
        period = f"{day} · 00:00–24:00"
    else:
        plan = plan_frame(data, scenario)
        rev = (period_revenue(data, plan) if scenario == "forecast"
               else data["revenues"][scenario])
        period = (f"{plan.ts.min():%Y-%m-%d} ~ {plan.ts.max():%Y-%m-%d}"
                  f" · {plan.ts.dt.date.nunique()}일")
    basis = "예측 기준" if scenario == "forecast" else "실측을 미리 안 경우"
    st.markdown(f"**{scope} · {period} · {basis}**")
    kpis([
        (f"{scope} 예상 순수익", money(rev["net_revenue_krw"]), "SMP + REC − 열화비"),
        (f"{scope} ESS 추가수익", money(rev["ess_gain_krw"]),
         "ESS 미사용보다 계획 수익 감소" if rev["ess_gain_krw"] < -0.01
         else "ESS 없이 판매했을 때와 비교"),
    ])
    scale, unit = (1e8, "억 원") if scope == "전체 기간" else (10000, "만 원")
    fig = go.Figure(go.Waterfall(
        x=["SMP 판매", "REC", "열화비", "예상 순수익"],
        measure=["relative", "relative", "relative", "total"],
        y=[rev["sales_revenue_krw"] / scale, rev["rec_revenue_krw"] / scale,
           -rev["throughput_cost_krw"] / scale, 0],
        text=[money(rev["sales_revenue_krw"]), money(rev["rec_revenue_krw"]),
              money(-rev["throughput_cost_krw"]), money(rev["net_revenue_krw"])],
        textposition="outside",
        increasing={"marker": {"color": "#458a79"}},
        decreasing={"marker": {"color": "#b66a48"}},
        totals={"marker": {"color": "#173a44"}},
    ))
    fig.update_layout(height=330, yaxis_title=unit, margin=dict(t=40, b=20),
                      paper_bgcolor="#fff", plot_bgcolor="#fff")
    st.plotly_chart(fig, width="stretch")
    if scope == "선택한 하루":
        st.markdown("**시간별 누적 수익**")
        replay = replay_payload(data, day)
        if data["actual"] is not None:
            rows = [{"hour": f["hour"], "f": f["cumulative"] / 10000,
                     "a": a["cumulative"] / 10000}
                    for f, a in zip(replay["series"]["forecast"], replay["series"]["actual"],
                                    strict=True)]
            render_signal(rows, title="하루 누적 계획 수익", day=day, unit="만 원",
                          kind="revenue", forecast_label="예측이 맞았다고 가정",
                          actual_label="실측을 미리 알고 계획")
        else:
            rows = replay["series"]["forecast"]
            fig = go.Figure(go.Scatter(x=[r["hour"] for r in rows],
                                      y=[r["cumulative"] / 10000 for r in rows],
                                      mode="lines", line=dict(color="#173a44", width=2)))
            fig.update_layout(height=240, yaxis_title="만 원", margin=dict(t=10, b=10),
                              paper_bgcolor="#fff", plot_bgcolor="#fff")
            st.plotly_chart(fig, width="stretch")
    st.caption("REC는 ESS 없이 판매해도 받습니다. "
               "실제 실행·편차 정산 수익은 아직 계산하지 않습니다.")
    with st.expander("정확한 금액과 계산 범위"):
        table = pd.DataFrame({
            "항목": ["SMP 판매", "REC", "열화비 (차감)", "순수익", "ESS 추가수익"],
            "금액 (원)": [rev["sales_revenue_krw"], rev["rec_revenue_krw"],
                        -rev["throughput_cost_krw"], rev["net_revenue_krw"],
                        rev["ess_gain_krw"]],
        })
        st.dataframe(table, hide_index=True, width="stretch")
        note = ("두 계획의 수익 차이는 실제 손익이 아닙니다. " if data["actual"] is not None
                else "이 금액은 계획값이며 실제 손익이 아닙니다. ")
        st.write(note + "편차 정산·예측 정산금·용량정산금의 반영 상태는 "
                 "실적·정산 화면에서 확인하세요.")


def show_realized_profit(data, day, scope):
    """실행 수익 (확정 계획 × 실측). 계획 항목과 나란히 놓고 차이를 적는다."""
    frame = (data.get("execution") or {}).get("executed")
    if scope == "선택한 하루":
        block = exec_day(data, day)
        plan = day_slice(data, day)
        period = f"{day} · 00:00–24:00"
    else:
        block = frame
        plan = data["forecast"]
        period = f"{frame.ts.min():%Y-%m-%d} ~ {frame.ts.max():%Y-%m-%d}"
    if block is None:
        note = (data.get("execution_summary") or {}).get("note") or "실행 결과 없음"
        st.info(f"미계산 · {note}")
        return
    t = exec_totals(block)
    if not t["computable"]:
        st.info("정산 가격이 없는 시간이 있어 실행 수익을 계산하지 않았습니다.")
        return
    from vpp.baselines import plan_revenue_breakdown

    plan = plan[plan.ts.isin(block.ts)]
    rev = plan_revenue_breakdown(plan, data["ess"])
    st.markdown(f"**{scope} · {period} · 가정 기반 실행 수익 (하루전·편차 가격 = 실측 SMP)**")
    kpis([
        ("실행 순수익", money(t["realized_net_krw"]),
         "확정 판매 정산 + 편차 정산 + REC − 실행 열화비"),
        ("예상 순수익 (같은 계획)", money(rev["net_revenue_krw"]),
         "예측이 맞았다고 가정한 계획 수익"),
        ("차이 (실행 − 예상)", money(t["realized_net_krw"] - rev["net_revenue_krw"]),
         f"부족 {t['shortfall_mwh']:,.1f} · 초과 {t['surplus_mwh']:,.1f} MWh"),
    ])
    scale, unit = (1e8, "억 원") if scope == "전체 기간" else (10000, "만 원")
    fig = go.Figure(go.Waterfall(
        x=["확정 판매 정산", "편차 정산", "REC", "열화비", "실행 순수익"],
        measure=["relative", "relative", "relative", "relative", "total"],
        y=[t["committed_sales_krw"] / scale, t["deviation_krw"] / scale, t["rec_krw"] / scale,
           -t["wear_krw"] / scale, 0],
        text=[money(t["committed_sales_krw"]), money(t["deviation_krw"]), money(t["rec_krw"]),
              money(-t["wear_krw"]), money(t["realized_net_krw"])],
        textposition="outside",
        increasing={"marker": {"color": "#458a79"}},
        decreasing={"marker": {"color": "#b66a48"}},
        totals={"marker": {"color": "#173a44"}},
    ))
    fig.update_layout(height=330, yaxis_title=unit, margin=dict(t=40, b=20),
                      paper_bgcolor="#fff", plot_bgcolor="#fff")
    st.plotly_chart(fig, width="stretch")
    rows = [
        ("SMP 판매 ↔ 확정 판매 정산 + 편차 정산", rev["sales_revenue_krw"],
         t["committed_sales_krw"] + t["deviation_krw"]),
        ("REC", rev["rec_revenue_krw"], t["rec_krw"]),
        ("열화비 (차감)", -rev["throughput_cost_krw"], -t["wear_krw"]),
        ("순수익", rev["net_revenue_krw"], t["realized_net_krw"]),
    ]
    st.dataframe(pd.DataFrame([
        {"항목": n, "예상 (계획)": money(a), "실행": money(b), "차이": money(b - a)}
        for n, a, b in rows
    ]), hide_index=True, width="stretch")
    st.caption("실측을 미리 알고 최적화한 비교 계획의 수익은 '실측을 미리 알고 세운 비교 계획'에서 "
               "따로 봅니다. 실제 운영에서는 불가능한 기준입니다.")


def show_execution(state, release):
    st.subheader("실행 상태 · 공유 파일 → MILP → 화면")
    flow(state)
    labels = {"complete": "완료", "running": "실행 중", "blocked": "파일 대기", "failed": "오류"}
    for step in state.get("stages", []):
        with st.container(border=True):
            st.markdown(f"**{step['name']} · {labels.get(step['state'], step['state'])}**")
            for missing in step.get("missing", []):
                st.caption("없는 파일")
                st.code(missing, language="text")
            if step.get("error"):
                st.error(step["error"])
    if state.get("error"):
        st.error(state["error"])
    st.markdown("**동작 방식**")
    st.write(
        "팀원이 모델을 바꾸면 결과 파일을 Drive 로 공유합니다. 받은 파일을 아래 위치에 같은 "
        "이름으로 두면, 화면이 열려 있는 동안 30초마다 파일 내용·ESS·REC 설정·최적화 코드 변경을 "
        "확인하고 "
        "바뀐 경우에만 MILP(비교 백테스트 + 48시간 운영계획)를 다시 풉니다. 검산을 통과한 결과만 "
        "모든 화면의 기본 결과가 됩니다."
    )
    st.caption(
        "모델 학습·추론은 여기서 하지 않습니다. 앱이 꺼져 있으면 확인하지 않고, 같은 내용을 반복 "
        "계산하지 않습니다."
    )
    st.dataframe(
        pd.DataFrame(pipeline.requirements(pipeline.config(ROOT))).rename(
            columns={
                "name": "공유 파일",
                "ready": "파일 있음",
                "path": "두는 위치",
                "purpose": "출처",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("정확한 버전 확인"):
        st.code(
            f"입력·코드 지문: {state.get('revision', '미기록')}\n"
            f"결과 묶음: {release.get('id', '없음')} · {release.get('label', '')}"
        )
        st.write("사용한 발전량 예측 열:", ", ".join(release.get("forecast_columns", [])) or "없음")
        st.json(release.get("hashes", {}))
    st.markdown("**계산에 아직 없는 부분**")
    st.write(
        "확정 판매계획을 실측에 적용한 실행 운전·정산, 실제 SOC 이월은 아직 미구현입니다. "
        "종료 SOC 정책은 변경하지 않았습니다. "
        "REC 초기 저장분 제외도 미구현이며 실험 가정을 유지합니다. "
        "팀원 모델의 학습·추론은 각 담당자가 돌리고 결과 파일만 공유받습니다."
    )


def render():
    st.markdown("<style>" + CSS_PATH.read_text() + "</style>", unsafe_allow_html=True)
    st.markdown(
        '<div class="oc-brand"><div class="oc-word">JEJU FLOW'
        '<div class="oc-sub">발전 예측부터 ESS 운영계획까지</div></div>'
        '<div class="oc-badge">운영 시뮬레이션 · 실제 입찰·정산 아님</div></div>',
        unsafe_allow_html=True,
    )
    cfg = pipeline.config(ROOT)
    with st.sidebar:
        st.subheader("실행 설정")
        st.toggle(
            "코드·입력 변경 시 자동 실행", value=cfg.get("auto_refresh", True), key="operator_auto"
        )
        if st.button("지금 다시 계산", type="primary", width="stretch"):
            pipeline.launch(ROOT, force=True)
            st.rerun()
        st.caption("1년치 기준 수 분 걸립니다. 진행 상태는 실행 상태 화면에서 확인합니다.")
        st.caption("공유 파일 경로와 ESS·REC 가정은 config/operator_pipeline.yaml 에서 정합니다.")
        st.code(
            f"ESS: {Path(cfg['ess']).name}\n"
            f"REC: {Path(cfg['rec']).name if cfg.get('rec') else '미적용'}"
        )
    monitor()
    state = pipeline.status(ROOT)
    release = state.get("active") or {}
    screens = ["요약", "운영계획", "예측 입력", "예상 수익", "실적·정산", "실행 상태"]
    legacy_screens = {"운영 관제": "요약", "2 운영계획·입찰안": "운영계획",
                      "1 예측 입력": "예측 입력", "3 예상 수익": "예상 수익",
                      "4 실적·정산": "실적·정산"}
    current = st.session_state.get("operator_screen", "요약")
    current = legacy_screens.get(current, current)
    st.session_state["operator_screen"] = current if current in screens else "요약"
    nav = st.radio(
        "화면",
        screens,
        horizontal=True,
        label_visibility="collapsed",
        key="operator_screen",
    )
    if state.get("state") in ("running", "queued"):
        st.info("새 코드·입력으로 실행 중입니다. 완료 전까지 아래는 이전 검산 완료 결과입니다.")
    if state.get("state") in ("needs_data", "failed"):
        st.markdown(
            '<div class="oc-callout">마지막 계산이 끝나지 않았거나 실패했습니다. '
            "아래는 마지막 검산 완료 결과이며, 사용 예측은 바로 아래에 표시됩니다. "
            "‘실행 상태’에서 없는 파일과 오류를 확인하세요.</div>",
            unsafe_allow_html=True,
        )
    if nav == "실행 상태":
        show_execution(state, release)
        return
    if not release:
        st.info("첫 실행 결과를 준비하고 있습니다. 실행 상태에서 진행 상황을 확인하세요.")
        flow(state)
        return
    if not release.get("backtest"):
        st.info("예측 창이 없어 운영계획을 만들지 못했습니다. "
                "실행 상태에서 입력 파일을 확인하세요.")
        return
    folder = Path(release["backtest"])
    try:
        data = read_report(str(folder), (folder / "summary.json").stat().st_mtime_ns)
    except (ValueError, OSError) as exc:
        st.error(f"저장 결과 검증 실패: {exc}")
        return
    has_actual = data["actual"] is not None
    days = sorted(data["forecast"].ts.dt.date.unique())
    if st.session_state.get("operator_day") not in days:
        st.session_state["operator_day"] = days[0]
    date_col, status_col = st.columns([1, 2])
    with date_col:
        day = st.selectbox("운영일", days, key="operator_day")
    with status_col:
        status = "예측·실측 비교 가능" if has_actual else "예측 기반 계획 · 실측 대기"
        updated = pd.Timestamp(release["created_at"]).tz_convert("Asia/Seoul")
        st.markdown(f'<div class="oc-data-status"><b>{status}</b>'
                    f'<span>결과 갱신 {updated:%m/%d %H:%M} KST</span></div>',
                    unsafe_allow_html=True)
    with st.expander("사용 데이터 · 기간과 출처"):
        st.caption(f"{release['label']} · {release['start']} ~ {release['end']} · "
                   f"{release['days']}일")
        st.caption("현재 실측 파일이 없어 오차 비교는 제공하지 않습니다." if not has_actual
                   else "예측과 실측을 같은 시각에 맞춰 비교합니다.")
    checks_ok = bool(pipeline.read_json(folder / "validation.json").get("plan_checks_all_passed"))
    if nav == "요약":
        show_summary(data, day, checks_ok)
    elif nav == "예측 입력":
        show_forecasts(data, day, release)
    elif nav == "예상 수익":
        show_profit(data, day)
    elif nav == "실적·정산":
        show_settlement(data, day)
    elif nav == "운영계획":
        st.subheader(f"{day:%m월 %d일} 운영계획")
        process_strip(data, day, release, checks_ok)
        render_simulator(data, day)
        if data["metrics"].get("execution_available"):
            show_execution_day(data, day)
        selected = day_slice(data, day)
        st.markdown("#### 시간별 판매계획")
        st.caption(
            "하루전시장에 낼 시간대별 판매 전력량의 계획값입니다. 발전량 예측과 ESS 충·방전을 합친 "
            "결과이며, 가격 입찰과 낙찰 결과는 이 도구 범위 밖입니다. 실제 실행 기록이 아닙니다."
        )
        st.dataframe(bid_table(selected), hide_index=True, width="stretch")
        st.download_button(
            "판매계획 CSV 내려받기",
            selected.to_csv(index=False).encode("utf-8-sig"),
            f"sales_plan_{day}.csv",
            "text/csv",
        )
        with st.expander("최근 운영일의 48시간 계획 (D 확정 + D+1 잠정)"):
            st.caption(release["start_soc_note"])
            plan48 = pd.read_csv(Path(release["plan"]) / "plan_48h.csv")
            plan48["ts"] = pd.to_datetime(plan48["ts"], utc=True).dt.tz_convert("Asia/Seoul")
            table48 = bid_table(plan48)
            table48.insert(0, "날짜", plan48.ts.dt.strftime("%m/%d"))
            table48.insert(2, "구분", plan48.forecast_day_offset.map({0: "D 확정", 1: "D+1 잠정"}))
            st.dataframe(table48, hide_index=True, width="stretch")
