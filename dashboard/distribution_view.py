"""Distribution ESS selection screens; session choices never change operating inputs."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.distribution_display import (
    BATTERIES, OPERATING, REPORT, RESULTS, SIZING, line_label, load_display,
    selected_id, visible_candidates,
)

CSS = Path(__file__).parent / "assets/distribution_view.css"
STEPS = ("선로 조건", "설비 비교", "참여 판단")
DEFAULTS = {"de_step": 0, "de_kind": "cap", "de_line": "cap_2.50MW",
            "de_battery": "all", "de_size": "all", "de_chosen": None}


def _money(value, digits=1):
    return "결과 없음" if pd.isna(value) else f"{float(value) / 1e8:,.{digits}f} 억원"


def _go(step):
    st.session_state.de_step = step


def _sync_choice(field):
    st.session_state[field] = st.session_state["_" + field]
    if field == "de_kind":
        st.session_state.de_line = None
    if field in ("de_kind", "de_line"):
        st.session_state.de_battery = "all"
        st.session_state.de_size = "all"
        st.session_state.de_chosen = None


def _choice(label, options, field, format_func=str):
    # Durable choices live outside widget keys, so they survive step navigation.
    if st.session_state[field] not in options:
        st.session_state[field] = options[0]
    st.session_state["_" + field] = st.session_state[field]
    return st.selectbox(label, options, format_func=format_func, key="_" + field,
                        on_change=_sync_choice, args=(field,))


def _heading(step, title, description):
    st.markdown(f'<div class="de-heading"><span>STEP {step:02d} / 03</span>'
                f'<h2>{escape(title)}</h2><p>{escape(description)}</p></div>',
                unsafe_allow_html=True)


def _facts(items):
    st.markdown('<div class="de-facts">' + ''.join(
        f'<div><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>'
        for label, value in items) + '</div>', unsafe_allow_html=True)


def _metrics(items):
    st.markdown('<div class="de-metrics">' + ''.join(
        f'<div><span>{escape(label)}</span><strong>{escape(str(value))}</strong>'
        f'<small>{escape(note)}</small></div>' for label, value, note in items)
        + '</div>', unsafe_allow_html=True)


def _line_screen(rows):
    _heading(1, "어떤 선로에서 검토하나요?", "저장된 선로 조건을 선택해 배터리와 용량을 비교합니다.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 선로 조건")
        kind = _choice("상한 형태", ["cap", "dyn"], "de_kind",
                       lambda k: "낮 시간 고정 상한" if k == "cap" else "발전량에 따른 동적 상한")
        lines = sorted(rows.loc[rows.line_case.str.startswith(kind + "_"), "line_case"].unique())
        line = _choice("선로 여유", lines, "de_line", line_label)
        _facts([("모집 태양광", "5.7 MW"), ("결과 기준", "2025 제주 · SMP 예측 운전"),
                ("참여 판단 관점", "표준 수익 배분 · VPP")])
    with right:
        st.markdown("#### 이 선로의 운전 조건")
        cap = float(line.split("_")[1][:-2])
        if kind == "cap":
            _facts([("00–08시", "상한 없음 · 시나리오 가정"),
                    ("08–18시", f"{cap:.2f} MW 판매 상한"),
                    ("18–24시", "상한 없음 · 시나리오 가정")])
        else:
            _facts([("최대 일사 때", f"{cap:.2f} MW 여유"),
                    ("일사가 줄어들 때", "선로 여유 증가"),
                    ("기존 태양광 가정", "같은 선로 5 MW")])
        st.markdown('<div class="de-note">한전의 실제 시간별 허용범위가 아닌 시나리오 조건입니다. '
                    '이 선택은 저장 결과의 조회 범위만 바꿉니다.</div>', unsafe_allow_html=True)
    candidates = visible_candidates(rows, line)
    st.caption(f"이 선로의 저장된 설비 후보 {len(candidates)}개 · "
               f"사업성 보고값 {candidates.npv_krw.notna().sum()}개")
    st.button("배터리·용량 비교하기 →", type="primary", on_click=_go, args=(1,))


def _equipment_screen(rows):
    _heading(2, "어떤 배터리와 용량을 선택할까요?", "같은 선로의 저장 결과를 비교하고 검토할 설비를 선택하세요.")
    line = st.session_state.de_line
    all_rows = rows[rows.line_case == line]
    left, right = st.columns(2)
    with left:
        _choice("배터리 비용·수명 가정", ["all", *BATTERIES], "de_battery",
                lambda k: "전체 배터리" if k == "all" else BATTERIES[k])
    sizes = sorted({f"{r.ess_power_mw:g}:{r.ess_energy_mwh:g}" for r in all_rows.itertuples()},
                   key=lambda v: tuple(reversed(tuple(map(float, v.split(":"))))))
    with right:
        _choice("정격 용량", ["all", *sizes], "de_size",
                lambda v: "전체 용량" if v == "all" else v.replace(":", " MW / ") + " MWh")
    candidates = visible_candidates(rows, line, st.session_state.de_battery, st.session_state.de_size)
    if candidates.empty:
        st.session_state.de_chosen = None
        st.info("이 배터리·용량 조합의 저장 결과가 없습니다. 필터를 바꾸면 다른 후보를 볼 수 있습니다.")
    else:
        st.session_state.de_chosen = selected_id(candidates, st.session_state.de_chosen)
        labels = candidates.set_index("id").equipment.to_dict()
        _choice("검토할 설비", candidates.id.tolist(), "de_chosen", labels.__getitem__)
        selected = candidates[candidates.id == st.session_state.de_chosen].iloc[0]
        table = pd.DataFrame({
            "검토": ["선택" if key == selected.id else "" for key in candidates.id],
            "배터리 / 용량": candidates.equipment,
            "ESS 방전 매출 / 년": candidates.ess_discharge_revenue_krw.map(_money),
            "20년 NPV · 보고값": candidates.npv_krw.map(_money),
            "사업성 보고값": candidates.npv_krw.map(lambda v: "결과 없음" if pd.isna(v) else "저장됨"),
        })
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("보고된 NPV가 큰 순서입니다. ESS 방전 매출은 비용 차감 전이며 VPP 순수익이 아닙니다. "
                   "보고서에 없는 NPV는 계산하거나 보간하지 않습니다.")
        st.markdown(f'<div class="de-selected"><span>검토할 설비</span>'
                    f'<strong>{escape(selected.equipment)}</strong>'
                    f'<span>20년 NPV</span><strong>{_money(selected.npv_krw)}</strong></div>',
                    unsafe_allow_html=True)
    back, forward = st.columns(2)
    back.button("← 선로 조건", on_click=_go, args=(0,))
    forward.button("선택한 설비의 참여 판단 →", type="primary", disabled=candidates.empty,
                   on_click=_go, args=(2,))


def _decision_screen(rows):
    candidates = visible_candidates(rows, st.session_state.de_line,
                                    st.session_state.de_battery, st.session_state.de_size)
    chosen = selected_id(candidates, st.session_state.de_chosen)
    st.session_state.de_chosen = chosen
    if chosen is None:
        st.info("현재 조건의 저장 결과가 없습니다. 설비 비교에서 다른 조합을 선택하세요.")
        st.button("← 설비 비교", on_click=_go, args=(1,))
        return
    row = candidates[candidates.id == chosen].iloc[0]
    _heading(3, "이 조건에서 참여를 검토할 수 있나요?", row.equipment + " · " + line_label(row.line_case))
    known = pd.notna(row.npv_krw)
    positive = known and row.npv_krw > 0
    verdict = "참여 검토" if positive else "경제성 재검토" if known else "판단 보류"
    detail = ("보고된 20년 NPV가 양수입니다. 실제 선로·견적·계약 요건을 확인한 뒤 참여를 검토하세요."
              if positive else "보고된 20년 NPV가 0 이하입니다. 현재 조건의 투자 경제성을 다시 검토할 필요가 있습니다."
              if known else "이 조합의 사업성 보고값이 없습니다. 운영 매출만으로 참여 여부를 판단하지 않습니다.")
    tone = "positive" if positive else "caution"
    st.markdown(f'<div class="de-verdict {tone}"><span>저장 결과 기준</span>'
                f'<h3>{verdict}</h3><p>{detail}</p></div>', unsafe_allow_html=True)
    _metrics([("20년 NPV", _money(row.npv_krw), "투자·지원·유지·교체비 반영 · 보고값"),
              ("ESS 방전 매출 / 년", _money(row.ess_discharge_revenue_krw), "비용 차감 전 · 운영 CSV"),
              ("추가 연간 보상 필요액", _money(row.needed_annual_support_krw, digits=2), "NPV 0 기준 · 보고값이 있을 때만")])
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 저장된 운전 결과")
        _facts([("충전량", f"{row.ess_charge_mwh:,.1f} MWh / 년"),
                ("방전량", f"{row.ess_discharge_mwh:,.1f} MWh / 년"),
                ("결과 기간", f"{row.days:g}일"),
                ("NPV 출처", row.npv_source if known else "해당 조합의 보고값 없음")])
    with right:
        st.markdown("#### 실제 참여 전 확인")
        _facts([("대상 선로·허용범위", "실제 한전 자료 확인 필요"),
                ("설비 비용", "국내 견적·보증 조건 확인 필요"),
                ("정산·배분", "계약상 귀속 조건 확인 필요"),
                ("예측 오차·페널티", "이 결과에는 발전량 예측 오차 미반영")])
    peers = rows[(rows.battery == row.battery) & (rows.ess_power_mw == row.ess_power_mw)
                 & (rows.ess_energy_mwh == row.ess_energy_mwh)].sort_values("line_case")
    with st.expander("같은 설비를 다른 선로에서도 비교"):
        st.dataframe(pd.DataFrame({"선로 조건": peers.line_case.map(line_label),
                                   "ESS 방전 매출 / 년": peers.ess_discharge_revenue_krw.map(_money),
                                   "20년 NPV · 보고값": peers.npv_krw.map(_money)}),
                     hide_index=True, width="stretch")
    st.caption("경제성 기준의 검토 결과이며 실제 사업 참여 승인 여부는 아닙니다. "
               "선택한 배전망 설비의 시간별 운전 원본은 현재 운영계획 화면과 연결되지 않았습니다.")
    back, restart = st.columns(2)
    back.button("← 다른 설비 비교", on_click=_go, args=(1,))
    restart.button("선로 조건 다시 선택", on_click=_go, args=(0,))


def render(root: Path):
    st.markdown("<style>" + CSS.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    try:
        rows, report = load_display(root)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"배전망 ESS 저장 결과를 표시할 수 없습니다: {exc}")
        return
    with st.container(key="distribution_ess_view"):
        st.markdown("### 배전망 ESS 참여 검토")
        st.caption("2026-11-15 저장 결과 · 제주 시장을 사용한 시연 · 실제 대상 선로의 승인 결과 아님")
        for i, column in enumerate(st.columns(3)):
            column.button(f"{i + 1}  {STEPS[i]}", key=f"de_nav_{i}", width="stretch",
                          type="primary" if st.session_state.de_step == i else "secondary",
                          on_click=_go, args=(i,))
        line = st.session_state.de_line
        if line not in rows.line_case.values:
            matching = rows.loc[rows.line_case.str.startswith(st.session_state.de_kind + "_"), "line_case"]
            st.session_state.de_line = line = sorted(matching.unique())[0]
        st.markdown(f'<div class="de-context"><strong>{escape(line_label(line))}</strong>'
                    '<span>태양광 5.7 MW</span><span>기준 · 연 0%</span>'
                    '<span>ESS 용량정산금 제외</span><span>표준 수익 배분</span></div>',
                    unsafe_allow_html=True)
        (_line_screen, _equipment_screen, _decision_screen)[st.session_state.de_step](rows)
        with st.expander("조회 기준과 원본 결과"):
            st.write("운영 수치는 기존 CSV, NPV와 보상 필요액은 기존 결과 보고서의 표를 표시합니다. "
                     "사업성은 기준 시나리오·SMP 예측 운전·표준 수익 배분·ESS 용량정산금 제외 기준입니다. "
                     "할인율 7%, 20년, 구축비 50% 지원 가정이며 NPV는 보고서의 반올림 정밀도를 유지합니다.")
            st.caption("선택과 필터는 이 화면에만 적용됩니다. 운영 설정·계산 결과·현재 포트폴리오 선택은 바뀌지 않습니다.")
            for name, label in ((OPERATING, "운영 결과 CSV"), (SIZING, "용량 비교 CSV")):
                st.download_button(label, (root / RESULTS / name).read_bytes(), name, "text/csv")
            if report:
                st.download_button("사업성 결과 보고서", report, REPORT, "text/markdown")
