"""담당: 대시보드 통합. 실행: python -m streamlit run dashboard/app.py

탭 4개는 어떤 결과 파일을 고르든 바뀌지 않는다:
    운영계획 생성 · 발전량 예측 성능 · SMP 예측 성능 · ESS 백테스트 비교
사이드바의 공통 설정(데이터 묶음, 날짜·기간, ESS 설정, 열화비 설정)은 파일에서 읽기만 한다.
최적화는 각 탭의 '이 설정으로 실행' 버튼을 눌렀을 때만 돌고, CLI 와 같은 함수를 호출한다
(vpp.operating_plan.run_operating_plan, vpp.forecast_backtest.run_forecast_backtest).
백테스트 결과가 없어도 운영계획 생성·예측 성능 화면은 쓸 수 있다.
화면의 수익은 모두 예측이 맞았다고 가정한 계획 수익이며 실제 입찰·정산이 아니다.
"""

from __future__ import annotations

import streamlit as st

import view_backtest
import view_forecast_perf
import view_operating_plan
from settings import render_sidebar
from ui import CSS

TABS = ("운영계획 생성", "발전량 예측 성능", "SMP 예측 성능", "ESS 백테스트 비교")


def page_header(right: str) -> None:
    st.markdown(
        '<div class="jf-header"><div class="jf-logo">J</div>'
        '<span class="jf-title">JEJU FLOW</span><span class="jf-sub">제주 VPP 중개사업자용</span>'
        '<span class="jf-badge">계획 수익 · 실제 입찰·정산 아님</span>'
        f'<span class="jf-right">{right}</span></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="JEJU FLOW · 제주 VPP", layout="wide", page_icon="🔋")
    st.markdown(CSS, unsafe_allow_html=True)
    settings, bundle = render_sidebar()
    right = (f"계획일 {settings.planning_day} · 기간 {settings.start} ~ {settings.end} · "
             f"{settings.ess_path.name} · {settings.degradation_label}"
             if settings.planning_day else "데이터 묶음을 읽지 못했습니다")
    page_header(right)
    tabs = st.tabs(list(TABS))
    views = (
        view_operating_plan.render,
        view_forecast_perf.render_generation,
        view_forecast_perf.render_smp,
        view_backtest.render,
    )
    for tab, view in zip(tabs, views, strict=True):
        with tab:
            try:
                view(settings, bundle)
            except Exception as exc:  # 한 탭이 깨져도 다른 탭은 떠야 한다
                st.exception(exc)


main()
