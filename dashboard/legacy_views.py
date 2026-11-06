"""예전 완전예측 백테스트(vpp.backtest 결과)의 날짜별 화면.

백테스트 비교 탭 안의 하위 탭으로 쓴다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ui import (
    ACTION_STYLE,
    EOK,
    EPS,
    MAN,
    action_card,
    actions_from_plan,
    baseline_day,
    checks_table,
    compare_bar_figure,
    contributions,
    day_slice,
    degradation_choices,
    kpi,
    monthly_table,
    net_revenue,
    plan_stats,
    recompute_day,
    soc_figure,
    timeline_figure,
    unit,
    waterfall_figure,
)
from vpp.optimization import EssParams
from vpp.validation import check_plan, check_properties


def sim_settings_panel(run: dict[str, Any], day) -> tuple[EssParams | None, bool]:
    """왼쪽 '시뮬레이션 설정' 패널. (ESS 설정, 실행 버튼 눌림)을 돌려준다."""
    stored = run["ess"]
    st.markdown("#### 시뮬레이션 설정")
    st.caption("ESS 사양을 바꿔 이 날(48시간 창)을 다시 풉니다. 가격·발전량은 실측 그대로입니다.")
    if st.button("↻ 초기화", key="reset_sim"):
        for k in list(st.session_state):
            if k.startswith("sim_"):
                del st.session_state[k]
    cap = st.slider("ESS 용량 (MWh)", 1.0, 64.0, float(stored.capacity_mwh), 1.0, key="sim_cap")
    power = st.slider("충·방전 출력 (MW)", 0.5, 20.0, float(stored.charge_power_mw), 0.5,
                      key="sim_power")
    soc_min, soc_max = st.slider("SOC 범위 (%)", 0, 100,
                                 (int(stored.soc_min * 100), int(stored.soc_max * 100)), 5,
                                 key="sim_soc")
    eff = st.slider("충·방전 효율 (%)", 70, 100, int(round(stored.charge_efficiency * 100)), 1,
                    key="sim_eff")
    choices = degradation_choices()
    labels = list(choices)
    default = next((i for i, k in enumerate(labels)
                    if choices[k]["throughput_cost_krw_per_mwh"]
                    == stored.throughput_cost_krw_per_mwh), 0)
    wear_label = st.selectbox("열화비", labels, index=default, key="sim_wear",
                              help="config/ess_v0.yaml(미적용)·ess_v0_1.yaml(고정 열화비) 값. "
                                   "선택해도 YAML 은 바뀌지 않으며, 실행 시 최적화부터 다시 푼다.")
    wear = choices[wear_label]
    hourly = run["hourly"]
    i0 = int(hourly.index[hourly["date"] == day][0])
    prev_energy = (stored.initial_energy_mwh if i0 == 0
                   else float(hourly.iloc[i0 - 1]["energy_end_mwh"]))
    soc0 = st.slider("시작 SOC (%)", 0, 100,
                     int(round(prev_energy / stored.capacity_mwh * 100)), 5,
                     key="sim_soc0", help="기본값은 저장된 계획의 전날 종료 잔량")
    if soc_min >= soc_max:
        st.error("SOC 최소값은 최대값보다 작아야 합니다.")
        return None, False
    ess = EssParams(
        capacity_mwh=cap, charge_power_mw=power, discharge_power_mw=power,
        soc_min=soc_min / 100, soc_max=soc_max / 100,
        soc_initial=min(max(soc0 / 100, soc_min / 100), soc_max / 100),
        charge_efficiency=eff / 100, discharge_efficiency=eff / 100,
        throughput_cost_krw_per_mwh=float(wear["throughput_cost_krw_per_mwh"]),
        degradation_model=wear["degradation_model"], throughput_basis=wear["throughput_basis"],
    )
    clicked = st.button("▶ 이 설정으로 시뮬레이션 실행", key="run_sim", type="primary",
                        width="stretch")
    st.caption("왼쪽 변수를 조정하면 우측 결과가 저장 계획과 나란히 비교됩니다.")
    return ess, clicked


def render_plan_tab(run: dict[str, Any], day) -> None:
    ess = run["ess"]
    left, right = st.columns([1, 3.4], gap="large")
    with left, st.container(border=True):
        sim_ess, clicked = sim_settings_panel(run, day)
    if clicked and sim_ess is not None:
        with st.spinner("MILP 풀이 중"):
            try:
                st.session_state["sim_result"] = {
                    "day": day, "run": run["folder"].name,
                    "out": recompute_day(run, day, sim_ess, sim_ess.initial_energy_mwh),
                }
            except RuntimeError as exc:
                st.session_state.pop("sim_result", None)
                st.error(f"시뮬레이션 실패: {exc}")
    sim = st.session_state.get("sim_result")
    if sim and (sim["day"] != day or sim["run"] != run["folder"].name):
        sim = None

    stored_plan = day_slice(run, day)
    no_ess = baseline_day(run, day, "no_ess")
    rule = baseline_day(run, day, "rule_based")
    plan = sim["out"]["plan"] if sim else stored_plan
    plan_ess = sim["out"]["ess"] if sim else ess
    stats = plan_stats(plan, plan_ess)
    base_stats = plan_stats(no_ess, ess) if no_ess is not None else None
    rule_stats = plan_stats(rule, ess) if rule is not None else None
    actions = actions_from_plan(plan)
    label = "시뮬레이션 계획" if sim else "저장된 MILP 계획"
    gain = stats["순수익 (만 원)"] - base_stats["순수익 (만 원)"] if base_stats else None

    with right:
        st.markdown(f'<span class="jf-tag">{day:%Y-%m-%d} 운영계획 · {label}</span>',
                    unsafe_allow_html=True)
        gain_txt = f" (ESS 없음 대비 {gain:+,.1f}만 원)" if gain is not None else ""
        st.markdown(f'<div class="jf-h1">이 날 할 일 {len(actions)}가지로 순수익 '
                    f'{stats["순수익 (만 원)"]:,.1f}만 원{gain_txt}</div>', unsafe_allow_html=True)
        st.markdown('<div class="jf-lead">실측 하루전 SMP와 실측 발전량으로 ESS 충·방전·감축을 '
                    '정한 결과입니다. 왼쪽에서 ESS 사양을 바꿔 다시 풀 수 있습니다.</div>',
                    unsafe_allow_html=True)
        c = st.columns(4)
        if base_stats:
            base_net = base_stats["순수익 (만 원)"]
            pct = gain / base_net * 100 if base_net else 0
            c[0].markdown(kpi("중개사업자 순수익 (ESS 없음 대비)",
                              f"{gain:+,.1f}{unit('만 원')}", f"{pct:+.2f}%",
                              "up" if gain >= 0 else "down", "💰"), unsafe_allow_html=True)
            curt = stats["버려지는 전력 (MWh)"] - base_stats["버려지는 전력 (MWh)"]
            c[3].markdown(kpi("버려지는 전력 감소", f"{curt:+,.1f}{unit(' MWh')}",
                              f"ESS 없음 {base_stats['버려지는 전력 (MWh)']:.1f} MWh",
                              "down" if curt <= 0 else "up", "🌿"), unsafe_allow_html=True)
        if rule_stats:
            g2 = stats["순수익 (만 원)"] - rule_stats["순수익 (만 원)"]
            c[1].markdown(kpi("규칙 계획 대비", f"{g2:+,.1f}{unit('만 원')}",
                              f"규칙 계획 {rule_stats['순수익 (만 원)']:,.1f}만 원",
                              "up" if g2 >= 0 else "down", "📊", "#EFF6FF"),
                          unsafe_allow_html=True)
        cycles = stats["ESS 방전량 (MWh)"] / max(plan_ess.usable_capacity_mwh, EPS)
        c[2].markdown(kpi("ESS 방전량 / 충전량",
                          f"{stats['ESS 방전량 (MWh)']:.1f} / {stats['ESS 충전량 (MWh)']:.1f}"
                          f"{unit(' MWh')}",
                          f"사용가능 {plan_ess.usable_capacity_mwh:.1f} MWh · {cycles:.2f} 사이클",
                          "flat", "🔋", "#F5F3FF"), unsafe_allow_html=True)

        st.plotly_chart(timeline_figure(plan, plan_ess, f" · {label}"), width="stretch")
        chips = "".join(
            f'<span class="chip" style="background:{ACTION_STYLE[a["name"]][0]}">'
            f'{a["name"]} {a["time"]}</span>' for a in actions
        )
        st.markdown(chips or '<span class="jf-sub">이 날은 ESS 조치가 없습니다.</span>',
                    unsafe_allow_html=True)

        st.markdown(f"#### 이 날 할 일 · {len(actions)}건")
        if not actions:
            return
        names = [f"{a['name']} {a['time']}" for a in actions]
        pick = st.radio("상세 볼 조치", names, horizontal=True, label_visibility="collapsed",
                        key=f"pick_{day}_{bool(sim)}")
        sel = names.index(pick)
        cols = st.columns(min(len(actions), 5))
        for k, a in enumerate(actions):
            cols[k % len(cols)].markdown(action_card(a, k == sel), unsafe_allow_html=True)
        a = actions[sel]
        i, j = a["rows"]
        block = plan.iloc[i:j + 1]
        base_block = no_ess.iloc[i:j + 1] if no_ess is not None else None
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"#### 선택 조치 상세 · {a['name']} {a['time']}")
            if base_block is not None:
                base_col = [0.0, 0.0, float(base_block["sell_mwh"].sum()),
                            float(base_block["curtailment_mwh"].sum()), np.nan,
                            float(base_block["price_krw_per_mwh"] @ base_block["sell_mwh"]) / MAN]
            else:
                base_col = [np.nan] * 6
            detail = pd.DataFrame({
                "항목": ["ESS 충전 (MWh)", "ESS 방전 (MWh)", "판매 (MWh)", "감축 (MWh)",
                       "구간 끝 SOC (%)", "판매수익 (만 원)"],
                "ESS 없음": base_col,
                label: [float(block["charge_mwh"].sum()), float(block["discharge_mwh"].sum()),
                        float(block["sell_mwh"].sum()), float(block["curtailment_mwh"].sum()),
                        float(block["soc_end"].iloc[-1] * 100),
                        float(block["price_krw_per_mwh"] @ block["sell_mwh"]) / MAN],
            })
            detail["변화"] = detail[label] - detail["ESS 없음"]
            st.dataframe(detail.round(2), width="stretch", hide_index=True)
        with d2:
            st.markdown("#### 시나리오 비교 (이 날)")
            table = {}
            if base_stats:
                table["ESS 없음"] = base_stats
            if rule_stats:
                table["규칙 계획"] = rule_stats
            table["저장 MILP"] = plan_stats(stored_plan, ess)
            if sim:
                table["시뮬레이션"] = stats
            st.dataframe(pd.DataFrame(table).round(2), width="stretch")
            if base_stats:
                curt = stats["버려지는 전력 (MWh)"] - base_stats["버려지는 전력 (MWh)"]
                st.markdown(f'<div class="note">💡 {label}은 ESS 없음 대비 순수익이 '
                            f'<b>{gain:+,.1f}만 원</b>, 버려지는 전력이 <b>{curt:+,.1f} MWh</b> '
                            f'달라집니다.</div>', unsafe_allow_html=True)
        if sim:
            out = sim["out"]
            diff = out["objective_recomputed_krw"] - out["objective_stored_krw"]
            passed = sum(chk["passed"] for chk in out["checks"])
            st.info(f"시뮬레이션: {out['solved_hours']}시간 창을 풀고 앞 "
                    f"{out['committed_hours']}시간만 비교. "
                    f"시작 {out['initial_energy_mwh']:.2f} MWh · 저장 계획 대비 "
                    f"{diff / MAN:+,.1f}만 원 · 검산 {passed}/{len(out['checks'])} 통과. "
                    f"종료 잔량 {float(out['plan']['energy_end_mwh'].iloc[-1]):.2f} MWh "
                    f"(저장 계획 {float(stored_plan['energy_end_mwh'].iloc[-1]):.2f} MWh) — "
                    "다음 날을 위해 남긴 에너지의 가치는 이 날 수익에 포함되지 않습니다.")


def render_economics_tab(run: dict[str, Any], day) -> None:
    ess, s = run["ess"], run["summary"]
    st.markdown(f'<span class="jf-tag">{day:%Y-%m-%d} · 기간 {s["start"]} ~ {s["end"]} '
                f'({s["days"]}일)</span>', unsafe_allow_html=True)
    st.markdown('<div class="jf-h1">ESS 계획을 쓰면 중개사업자 수익이 얼마나 달라지는가</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="jf-lead">ESS 없음 · 규칙 계획 · MILP 계획의 순수익을 하루와 '
                '기간 전체로 비교합니다. 완전예측 결과이므로 상한에 가깝습니다.</div>',
                unsafe_allow_html=True)

    plan = day_slice(run, day)
    no_ess = baseline_day(run, day, "no_ess")
    rule = baseline_day(run, day, "rule_based")
    day_vals = {}
    if no_ess is not None:
        day_vals["ESS 없음"] = net_revenue(no_ess, ess)
    if rule is not None:
        day_vals["규칙 계획"] = net_revenue(rule, ess)
    day_vals["MILP 계획"] = net_revenue(plan, ess)
    base_total = s["baselines"].get("no_ess", {})
    rule_total = s["baselines"].get("rule_based", {})
    c = st.columns(4)
    if no_ess is not None:
        g = day_vals["MILP 계획"] - day_vals["ESS 없음"]
        pct = f"{g / day_vals['ESS 없음'] * 100:+.2f}%" if day_vals["ESS 없음"] else ""
        c[0].markdown(kpi("이 날 순수익 증가 (vs ESS 없음)", f"{g / MAN:+,.1f}{unit('만 원')}", pct,
                          "up" if g >= 0 else "down", "💰"), unsafe_allow_html=True)
    if base_total:
        c[1].markdown(kpi("기간 순수익 증가 (vs ESS 없음)",
                          f"{base_total['milp_gain_krw'] / EOK:+,.2f}{unit('억 원')}",
                          f"{base_total.get('milp_gain_pct') or 0:+.2f}% · {s['days']}일", "up",
                          "📊", "#EFF6FF"), unsafe_allow_html=True)
        per_day = base_total["milp_gain_krw"] / max(s["days"], 1)
        c[2].markdown(kpi("연간 환산 증가 (일평균 ×365)",
                          f"{per_day * 365 / EOK:,.2f}{unit('억 원')}",
                          f"일평균 {per_day / MAN:,.1f}만 원", "up", "⚡", "#FFF7ED"),
                      unsafe_allow_html=True)
    if rule_total:
        c[3].markdown(kpi("기간 규칙 계획 대비",
                          f"{rule_total['milp_gain_krw'] / EOK:+,.2f}{unit('억 원')}",
                          f"{rule_total.get('milp_gain_pct') or 0:+.2f}%", "up", "🧭", "#F5F3FF"),
                      unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    g1.plotly_chart(compare_bar_figure(day_vals), width="stretch")
    g2.plotly_chart(waterfall_figure(contributions(plan, no_ess, ess)), width="stretch")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### 기간 손익 상세")
        rows = [("판매수익 (억 원)", s["sales_revenue_krw"] / EOK),
                ("열화비 (억 원)", -s["throughput_cost_krw"] / EOK),
                ("순수익 (억 원)", s["net_revenue_krw"] / EOK),
                ("판매량 (MWh)", s["sold_mwh"]), ("ESS 방전량 (MWh)", s["discharge_mwh"]),
                ("버려지는 전력 (MWh)", s["curtailment_mwh"]),
                ("일평균 사이클", s.get("cycles_per_day") or 0)]
        table = pd.DataFrame(rows, columns=["항목", "MILP 계획"])
        if base_total:
            table["ESS 없음"] = [base_total["net_revenue_krw"] / EOK, 0.0,
                               base_total["net_revenue_krw"] / EOK, base_total["sold_mwh"], 0.0,
                               base_total["curtailment_mwh"], 0.0]
            table["변화"] = table["MILP 계획"] - table["ESS 없음"]
        st.dataframe(table.round(2), width="stretch", hide_index=True)
    with t2:
        st.markdown("#### 월별 순수익")
        monthly = monthly_table(run)
        if monthly is not None:
            st.dataframe(monthly.round(1), width="stretch", hide_index=True)
        else:
            st.caption("daily_summary.csv가 없어 월별 표를 만들 수 없습니다.")
    with st.expander("비용·손실·수익 계산 가정"):
        for line in run["validation"].get("limitations", []):
            st.markdown(f"- {line}")


def render_resources_tab(run: dict[str, Any], day) -> None:
    ess, v = run["ess"], run["validation"]
    src = v.get("data_checks", {}).get("sources", {})
    plan = day_slice(run, day)
    st.markdown('<div class="jf-h1">같은 설비로 비교합니다</div>', unsafe_allow_html=True)
    st.markdown('<div class="jf-lead">백테스트에 쓴 포트폴리오·ESS 가정값·제약조건입니다. '
                'ESS 값은 설비 사양이 아닌 가정값이며, "운영계획" 탭의 시뮬레이션 설정으로 '
                '바꿔 볼 수 있습니다.</div>', unsafe_allow_html=True)
    pv_info = src.get("pv", {})
    pv_kw = pv_info.get("capacity_kw", 14_081.22)
    wind_kw = 0.0
    for k in ("seongsan", "hangyeong", "sangmyeong"):
        info = src.get(k, {})
        if "capacity_kw_by_stage" in info:
            wind_kw += sum(info["capacity_kw_by_stage"].values())
        elif "capacity_kw" in info:
            wind_kw += info["capacity_kw"]
    pv_day = float(plan["pv_kwh"].sum() / 1000) if "pv_kwh" in plan else float("nan")
    wind_day = float(plan["wind_kwh"].sum() / 1000) if "wind_kwh" in plan else float("nan")
    price = plan["price_krw_per_mwh"]
    c = st.columns(4)
    pv_unit = unit(f" MW · {pv_info.get('plants', 18)}개소")
    c[0].markdown(kpi("태양광", f"{pv_kw / 1000:,.1f}{pv_unit}",
                      f"이 날 발전 {pv_day:,.1f} MWh", "flat", "☀️", "#FFFBEB"),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("풍력 (성산·한경·상명)", f"{wind_kw / 1000:,.0f}{unit(' MW · 3개 단지')}",
                      f"이 날 발전 {wind_day:,.1f} MWh", "flat", "🌀", "#EFF6FF"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("ESS (가정값)", f"{ess.charge_power_mw:g} MW / {ess.capacity_mwh:g}"
                      f"{unit(' MWh')}",
                      f"SOC {ess.soc_min * 100:.0f}→{ess.soc_max * 100:.0f}% · "
                      f"효율 {ess.charge_efficiency:.2f}", "flat", "🔋", "#F5F3FF"),
                  unsafe_allow_html=True)
    c[3].markdown(kpi("하루전 SMP (이 날)",
                      f"{price.min() / 1000:,.0f}~{price.max() / 1000:,.0f}{unit(' 원/kWh')}",
                      f"음수 가격 {int((price < 0).sum())}시간", "flat", "💱", "#ECFDF5"),
                  unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("#### ESS 가정값")
        st.caption(f"status: {ess.status} · 출처 {v['assumptions'].get('source', '')}")
        st.table(pd.DataFrame({"항목": list(ess.to_dict()),
                               "값": [str(x) for x in ess.to_dict().values()]}))
    with b:
        st.markdown("#### 제약조건 (docs/formulation.md §2)")
        st.markdown(
            "- 전력균형 `G + d = q + c + r` (계통 충전 금지)\n"
            "- SOC 전이 `E[t] = E[t-1] + η_c·c − d/η_d`, 범위 `SOC_min·Cap ≤ E ≤ SOC_max·Cap`\n"
            "- 출력 한도 `c ≤ min(G, P_ch·dt)`, `d ≤ P_dis·dt`, 동시 충방전 금지\n"
            f"- 풀이 방식 `{v['assumptions'].get('policy')}`: 48시간을 보고 앞 24시간만 확정, "
            "마지막 창은 기간 시작 잔량 이상으로 종료\n"
            "- 목적함수 `max Σ p·q − k·Σ(c + d)`"
        )
        st.markdown("#### 계획과 제어의 범위")
        st.markdown(
            "- ✅ 실측 발전량·하루전 SMP로 시간별 판매·충전·방전·감축을 정한다\n"
            "- ✅ ESS 없음·규칙 계획과 같은 조건으로 비교한다\n"
            "- ⛔ 발전량 예측·입찰 편차 정산·REC·출력제어 지시는 포함하지 않는다\n"
            "- ⛔ 실제 입찰·설비 제어는 하지 않는다 (백테스트)"
        )
    dc = v.get("data_checks", {})
    with st.expander(f"데이터 검사 · 제외한 날 {len(dc.get('excluded_days', []))}일 · "
                     f"용량 초과 플래그 {dc.get('over_capacity_hours_in_range', {})}"):
        if dc.get("excluded_days"):
            st.dataframe(pd.DataFrame(dc["excluded_days"]), width="stretch", hide_index=True)
        st.json({k: val for k, val in dc.items() if k not in ("excluded_days", "sources")})
        st.json(src)


def render_validation_tab(run: dict[str, Any], day) -> None:
    v, s, ess = run["validation"], run["summary"], run["ess"]
    plan_checks = v.get("plan_checks_full_period") or check_plan(
        run["hourly"], ess, objective_krw=s["net_revenue_krw"]
    )
    rolling_checks = v.get("plan_checks", [])
    props = v.get("properties") or check_properties(s)
    all_checks = plan_checks + rolling_checks + props
    passed = sum(chk["passed"] for chk in all_checks)
    excluded = len(v.get("data_checks", {}).get("excluded_days", []))
    st.markdown(f'<span class="jf-tag">기간 전체 {s["days"]}일 · 사후 검산</span>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="jf-h1">계획의 제약 검산 {passed}/{len(all_checks)}개가 '
                '통과했습니다</div>', unsafe_allow_html=True)
    st.markdown('<div class="jf-lead">저장된 시간별 계획을 다시 검산하고(전력균형·SOC·출력 한도·'
                '동시 충방전·목적함수·롤링 연속), 기준 계획 대비 속성을 확인합니다. '
                '예측이 없으므로 계획 = 실측 적용입니다.</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(kpi("검산 통과", f"{passed} / {len(all_checks)}", "제약·속성 전체",
                      "up" if passed == len(all_checks) else "down", "✅"),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("푼 창 수", f"{v.get('windows_solved', 0)}", f"{s['policy']} · 전부 optimal",
                      "flat", "🧮", "#EFF6FF"), unsafe_allow_html=True)
    c[2].markdown(kpi("최대 MIP gap", f"{v.get('max_mip_gap', float('nan')):.1e}", "목표 1e-8",
                      "flat", "🎯", "#F5F3FF"), unsafe_allow_html=True)
    c[3].markdown(kpi("제외한 날 (결측)", f"{excluded}{unit('일')}", "보간 없이 제외", "flat", "⚠️",
                      "#FFF7ED"), unsafe_allow_html=True)

    g1, g2 = st.columns([1.2, 1])
    with g1:
        st.plotly_chart(soc_figure(day_slice(run, day), ess), width="stretch")
    with g2:
        st.markdown(f"#### 이 날 다시 계산 · {day:%Y-%m-%d}")
        st.caption("저장된 ESS 사양으로 48시간 창을 즉석에서 다시 풀어 저장 계획과 비교합니다.")
        if st.button("이 날 다시 계산", key="recompute"):
            with st.spinner("MILP 풀이 중"):
                out = recompute_day(run, day)
            diff = out["objective_recomputed_krw"] - out["objective_stored_krw"]
            m = st.columns(3)
            m[0].metric("다시 계산한 순수익", f"{out['objective_recomputed_krw'] / MAN:,.1f} 만원")
            m[1].metric("저장된 순수익", f"{out['objective_stored_krw'] / MAN:,.1f} 만원",
                        f"차이 {diff:+,.0f} 원")
            m[2].metric("창 길이", f"{out['solved_hours']}h (확정 {out['committed_hours']}h)",
                        f"시작 {out['initial_energy_mwh']:.2f} MWh")
            st.dataframe(checks_table(out["checks"]), width="stretch", hide_index=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### 제약 검산 (check_plan · 기간 전체)")
        st.dataframe(checks_table(plan_checks + rolling_checks), width="stretch", hide_index=True)
    with t2:
        st.markdown("#### 속성 검사 (check_properties)")
        st.dataframe(checks_table(props), width="stretch", hide_index=True)
        st.markdown("#### 한계")
        for line in v.get("limitations", []):
            st.markdown(f"- {line}")

