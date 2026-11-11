"""탭 4 — ESS 백테스트 비교. 예측 기반 계획 vs 완전예측(실측) 계획, SOC 이월, 충·방전·판매·출력제어·
계획 수익. 실행은 vpp.forecast_backtest.run_forecast_backtest (CLI 와 같은 함수).

예전 완전예측 결과(vpp.backtest, 실측 기반만)도 같은 탭 안의 하위 탭으로 연다.
탭 구성은 바뀌지 않는다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from legacy_views import (
    render_economics_tab,
    render_plan_tab,
    render_resources_tab,
    render_validation_tab,
)
from settings import BACKTEST_OUT_DIR, Settings, rec_diff, settings_chips
from ui import (
    FORECAST_MODE,
    MAN,
    checks_table,
    daily_compare_figure,
    kpi,
    list_runs,
    load_forecast_run,
    load_run,
    run_mode,
    run_stamp,
    soc_compare_figure,
    timeline_figure,
    unit,
)
from view_operating_plan import revenue_caption
from vpp.forecast_backtest import run_forecast_backtest


def render_forecast_run(run: dict[str, Any], day) -> None:
    s, v, ess = run["summary"], run["validation"], run["ess"]
    tf, ta = s["scenarios"]["forecast"], s["scenarios"]["actual"]
    compare = v.get("comparison_checks", [])
    plan_ok = v.get("plan_checks_all_passed", False)
    cmp_ok = v.get("comparison_all_passed", False)
    diff = s["planned_objective_diff_krw"]
    rec_applied = (s.get("rec") or {}).get("enabled", False)

    plan_only = s.get("comparison_excluded_days", 0)
    plan_only_text = f" · 실측 없어 비교 제외 {plan_only}일" if plan_only else ""
    st.markdown(f'<span class="jf-tag">계획일 {s["days"]}일 · 예측 부족 제외 {s["excluded_days"]}일'
                f'{plan_only_text} · {s["start"]} ~ {s["end"]} · '
                f'{"REC 적용(가정)" if rec_applied else "REC 미적용"}</span>',
                unsafe_allow_html=True)
    man = unit("만 원")
    n_pass = sum(chk["passed"] for chk in compare)
    all_ok = cmp_ok and plan_ok
    cards = [
        kpi("예측 기반 계획 수익", f"{tf['planned_objective_krw'] / MAN:,.0f}{man}",
            revenue_caption(tf, digits=0) + "만 원", "flat", "🔮", "#F5F3FF"),
        kpi("완전예측 계획 수익", f"{ta['planned_objective_krw'] / MAN:,.0f}{man}",
            revenue_caption(ta, digits=0) + "만 원", "flat", "🎯"),
        kpi("계획 수익 차이", f"{diff / MAN:+,.0f}{man}", "실제 손익 아님",
            "up" if diff >= 0 else "down", "Δ", "#FFF7ED"),
        kpi("비교 조건·검산", f"{n_pass} / {len(compare)}",
            "제약 검산 " + ("통과" if plan_ok else "실패"), "up" if all_ok else "down",
            "✅" if all_ok else "❌", "#ECFDF5" if all_ok else "#FEF2F2"),
    ]
    for col, card in zip(st.columns(4), cards, strict=True):
        col.markdown(card, unsafe_allow_html=True)
    if "ess_gain_vs_no_ess_krw" in ta:
        # ESS 한계이익: 같은 판매 가치(SMP+REC)로 ESS 없이 팔았을 때와의 차이. 라이언님이 물은
        # '정답값으로 운영했을 때 ESS 가 더 벌어주는 돈'이 완전예측 쪽 값이다.
        g_a, g_f = ta["ess_gain_vs_no_ess_krw"], tf["ess_gain_vs_no_ess_krw"]
        term = ta.get("terminal_constraint_cost_krw", 0.0) or 0.0
        rec_note = (" REC 는 ESS 없이 팔아도 받으므로 전체 REC 수익이 곧 ESS 추가수익은 아니다 "
                    "(저장 손실·감축량 변화만큼만 달라진다)."
                    if rec_applied else "")
        st.markdown(
            f'<div class="note">🔋 <b>ESS 없음 대비 추가수익</b> · 완전예측(실측) '
            f'<b>{g_a / MAN:+,.0f}만 원</b> / 예측 기반 {g_f / MAN:+,.0f}만 원 — '
            f'ESS 없음 수익 {ta["no_ess_net_revenue_krw"] / MAN:,.0f}만 원(실측 기준)과의 차이. '
            f'참고: 구간 끝 종료 조건을 풀면 계산상 {term / MAN:,.1f}만 원 더 나온다 '
            f'(못 판 잔량의 기회비용 포함, 실제 충전 비용 아님).{rec_note}</div>',
            unsafe_allow_html=True)
    if not cmp_ok:
        bad = ", ".join(chk["item"] for chk in compare if not chk["passed"])
        st.error(f"두 시나리오의 비교 조건이 일치하지 않습니다: {bad}")
    if not plan_ok:
        st.error("확정 계획의 제약 검산(check_plan)에 실패한 시나리오·구간이 있습니다.")
    for w in v.get("comparison_warnings", []):
        if w.get("flagged"):
            st.warning(f"확인 필요 · {w['item']}: {w['note']}")

    seg_initials = {
        key: {seg["segment_id"]: seg["initial_energy_mwh"]
              for seg in s["scenarios"][key]["segments"]}
        for key in ("forecast", "actual")
    }
    st.plotly_chart(soc_compare_figure(run["forecast"], run["actual"], seg_initials, ess,
                                       "ESS 잔량(SOC) 이월 비교 · 기간 전체 "
                                       "(◆ 구간 시작 잔량, ● 각 시간 종료 잔량)"),
                    width="stretch")
    st.plotly_chart(daily_compare_figure(run["daily"]), width="stretch")

    daily = run["daily"]
    row = daily[daily["planning_day"].dt.date == day].iloc[0]
    reset = bool(row["segment_start_forecast"])
    st.markdown(f"#### 이 날의 계획 · {day:%Y-%m-%d}"
                + (" · 구간 시작 (설정값 SOC 로 초기화)" if reset else " · 전날 종료 잔량 이월"))
    f_day = run["forecast"][run["forecast"]["date"] == day]
    a_day = run["actual"][run["actual"]["date"] == day]
    m = st.columns(4)
    m[0].metric("시작 잔량 (예측 기반)", f"{row['initial_energy_mwh_forecast']:.2f} MWh",
                None if reset else f"전날 종료 대비 {row['carry_gap_mwh_forecast']:+.3f}")
    m[1].metric("종료 잔량 (예측 기반)", f"{row['final_energy_mwh_forecast']:.2f} MWh")
    m[2].metric("시작 잔량 (완전예측)", f"{row['initial_energy_mwh_actual']:.2f} MWh",
                None if reset else f"전날 종료 대비 {row['carry_gap_mwh_actual']:+.3f}")
    m[3].metric("종료 잔량 (완전예측)", f"{row['final_energy_mwh_actual']:.2f} MWh")
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(timeline_figure(f_day, ess, " · 예측 기반"), width="stretch")
    with g2:
        st.plotly_chart(timeline_figure(a_day, ess, " · 완전예측(실측)"), width="stretch")
    day_initials = {
        "forecast": {int(f_day["segment_id"].iloc[0]): float(row["initial_energy_mwh_forecast"])},
        "actual": {int(a_day["segment_id"].iloc[0]): float(row["initial_energy_mwh_actual"])},
    }
    st.plotly_chart(soc_compare_figure(f_day, a_day, day_initials, ess,
                                       "이 날 SOC · ◆ 시작 잔량, ● 각 시간 종료 잔량"),
                    width="stretch")

    t1, t2 = st.columns([1.4, 1])
    with t1:
        st.markdown("#### 계획일별 비교 (잔량 MWh · 수익 원)")
        cols = {
            "계획일": daily["planning_day"].dt.date,
            "구간": daily["segment_id_forecast"],
            "구간 시작(초기화)": daily["segment_start_forecast"].map({True: "◆ 초기화", False: ""}),
            "시작 잔량(예측)": daily["initial_energy_mwh_forecast"],
            "종료 잔량(예측)": daily["final_energy_mwh_forecast"],
            "전날 종료 대비(예측)": daily["carry_gap_mwh_forecast"],
            "시작 잔량(실측)": daily["initial_energy_mwh_actual"],
            "종료 잔량(실측)": daily["final_energy_mwh_actual"],
            "전날 종료 대비(실측)": daily["carry_gap_mwh_actual"],
            "계획 수익(예측)": daily["objective_krw_forecast"],
            "계획 수익(실측)": daily["objective_krw_actual"],
            "수익 차이": daily["planned_objective_diff_krw"],
        }
        if "throughput_cost_krw_forecast" in daily:
            cols["열화비(예측)"] = daily["throughput_cost_krw_forecast"]
            cols["열화비(실측)"] = daily["throughput_cost_krw_actual"]
        if "rec_revenue_krw_actual" in daily and daily["rec_revenue_krw_actual"].abs().sum() > 0:
            cols["REC 수익(예측)"] = daily["rec_revenue_krw_forecast"]
            cols["REC 수익(실측)"] = daily["rec_revenue_krw_actual"]
        if "no_ess_same_curtailment_policy_krw_actual" in daily:
            cols["ESS 없음 대비(실측)"] = (daily["objective_krw_actual"]
                                      - daily["no_ess_same_curtailment_policy_krw_actual"])
        cols.update({
            "충전(예측)": daily["charge_mwh_forecast"], "충전(실측)": daily["charge_mwh_actual"],
            "방전(예측)": daily["discharge_mwh_forecast"],
            "방전(실측)": daily["discharge_mwh_actual"],
            "감축(예측)": daily["curtailment_mwh_forecast"],
            "감축(실측)": daily["curtailment_mwh_actual"],
        })
        st.dataframe(pd.DataFrame(cols).round(2), width="stretch", hide_index=True)
        st.caption("'전날 종료 대비'가 0 이면 계획상 SOC 가 그대로 이월된 것. 구간 첫날은 설정값 "
                   "SOC 로 초기화되어 비어 있다.")
        if len(run["excluded"]):
            st.markdown("#### 제외한 계획일")
            st.dataframe(run["excluded"], width="stretch", hide_index=True)
    with t2:
        st.markdown("#### 비교 조건 검사")
        st.dataframe(checks_table(compare), width="stretch", hide_index=True)
        st.markdown("#### 제약 검산 (시나리오·구간별)")
        rows = [{"scenario": p["scenario"], "segment": p["segment_id"],
                 "결과": "✅" if p["all_passed"] else "❌"} for p in v.get("plan_checks", [])]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("#### 한계")
        for line in s.get("limitations", []):
            st.markdown(f"- {line}")


def applied_vs_current(summary: dict[str, Any], settings: Settings) -> list[str]:
    diffs = []
    if (summary.get("start"), summary.get("end")) != (str(settings.start), str(settings.end)):
        diffs.append(f"기간 {summary.get('start')}~{summary.get('end')} → "
                     f"{settings.start}~{settings.end}")
    cur = settings.ess().to_dict()
    for key in ("capacity_mwh", "charge_power_mw", "soc_min", "soc_max", "soc_initial",
                "charge_efficiency", "degradation_model", "throughput_basis",
                "throughput_cost_krw_per_mwh"):
        if summary.get("ess", {}).get(key) != cur[key]:
            diffs.append(f"{key} {summary.get('ess', {}).get(key)} → {cur[key]}")
    return diffs + rec_diff(summary, settings)


def render(settings: Settings, bundle: dict[str, Any] | None) -> None:
    st.markdown('<div class="jf-h1">예측 기반 계획과 완전예측 계획을 같은 조건으로 '
                '비교합니다</div>', unsafe_allow_html=True)
    st.markdown('<div class="jf-lead">같은 계획일의 발전량 예측(D-2 18 UTC)·SMP 예측(D-1 11시) '
                '48시간으로 앞 24시간만 확정하고 계획상 SOC 를 이월합니다. 실측 기반 계획도 같은 '
                'ESS·열화비·시작 SOC·종료 정책으로 풉니다. 수익은 <b>계획 수익</b>이며 실제 손익이 '
                '아닙니다. 실측 적용·정산은 범위 밖.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(settings_chips(settings, show_period=True), unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        c1.caption("기간은 사이드바에서. 실측이 없는 날은 계획만 가능하므로 비교에서 빠지고 "
                   "`excluded_days.csv` 에 stage=comparison 으로 남습니다.")
        can_run = bundle is not None and settings.start is not None
        clicked = c2.button("▶ 이 설정으로 백테스트 실행", type="primary", key="bt_run",
                            disabled=not can_run, width="stretch")
    if clicked and can_run:
        run_id = f"forecast_{settings.start}_{settings.end}_{datetime.now():%Y%m%d%H%M%S}"
        with st.spinner("계획일마다 예측·실측 두 시나리오를 푸는 중"):
            try:
                out = run_forecast_backtest(
                    settings.start, settings.end, solar_path=settings.solar,
                    wind_path=settings.wind, smp_path=settings.smp, plants_path=settings.plants,
                    ess_path=settings.ess_path, degradation_path=settings.degradation_path,
                    rec_path=settings.rec_path, out_dir=BACKTEST_OUT_DIR, run_id=run_id,
                )
                st.session_state["bt_folder"] = str(out)
                st.session_state["bt_run_folder"] = out  # 방금 만든 결과를 바로 표시
                st.success(f"저장: {out}")
            except (ValueError, RuntimeError, FileExistsError) as exc:
                st.error(f"백테스트를 실행하지 못했습니다: {exc}")

    runs = list_runs(BACKTEST_OUT_DIR)
    if not runs:
        st.info("백테스트 결과가 없습니다. 위 버튼을 누르거나 터미널에서 "
                "`python -m vpp.forecast_backtest --start YYYY-MM-DD --end YYYY-MM-DD` 를 "
                "실행하세요. "
                "다른 탭은 결과 없이도 쓸 수 있습니다.")
        return
    current = st.session_state.get("bt_folder")
    index = next((i for i, p in enumerate(runs) if str(p) == current), 0)
    top = st.columns([2, 1.2, 3])
    choice = top[0].selectbox("백테스트 결과", runs, index=index, format_func=lambda p: p.name,
                              key="bt_run_folder")
    if run_mode(choice) == FORECAST_MODE:
        try:
            run = load_forecast_run(str(choice), run_stamp(choice))
        except Exception as exc:  # 깨진 결과 폴더가 있어도 탭은 떠야 한다
            st.error(f"결과 폴더를 읽지 못했습니다: {exc}")
            return
        days = sorted(run["forecast"]["date"].unique())
        day = top[1].selectbox("날짜", days, index=0, key="bt_day_forecast")
        diffs = applied_vs_current(run["summary"], settings)
        if diffs:
            top[2].warning("현재 설정이 표시 중인 결과와 다릅니다 (미실행): " + " · ".join(diffs))
        else:
            top[2].success("표시 중인 결과는 현재 설정으로 계산된 것입니다.")
        render_forecast_run(run, day)
        return
    try:
        run = load_run(str(choice), run_stamp(choice))
    except Exception as exc:
        st.error(f"결과 폴더를 읽지 못했습니다: {exc}")
        return
    days = sorted(run["hourly"]["date"].unique())
    day = top[1].selectbox("날짜", days, index=len(days) - 1, key="bt_day_legacy")
    top[2].info("예전 완전예측 결과(vpp.backtest · 실측 기반만). 예측 기반 비교는 없습니다.")
    sub = st.tabs(["운영계획 (실측 기반)", "경제성 비교", "운영 자원", "검증"])
    with sub[0]:
        render_plan_tab(run, day)
    with sub[1]:
        render_economics_tab(run, day)
    with sub[2]:
        render_resources_tab(run, day)
    with sub[3]:
        render_validation_tab(run, day)
