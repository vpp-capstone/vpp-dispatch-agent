"""Display annual receipts and real hourly execution from the existing engine.

The fixed report uses standard shares (ESS energy 100%, solar CP 50%, penalty 100%)
and excludes ESS CP. The display only adds those published annual components.
It never synthesizes hourly dispatch, SOC, cashflows, or missing NPV values.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

TEMPLATE = Path(__file__).parent / "assets" / "distribution_operation.html"


def revenue_breakdown(row: pd.Series) -> dict[str, float]:
    """Annual VPP receipts under the saved report's standard/no-ESS-CP basis."""
    if row.get("operation") != "forecast_exec" or row.get("days") != 365:
        raise ValueError("예측 → 실행 → 정산 기준의 365일 결과가 필요합니다.")
    columns = ("ess_discharge_revenue_krw", "solar_capacity_payment_krw", "imbalance_penalty_krw")
    if any(pd.isna(row.get(c)) or not math.isfinite(float(row[c])) for c in columns):
        raise ValueError("수익 구성 항목이 누락되어 VPP 수입을 표시할 수 없습니다.")
    parts = {
        "ess": float(row.ess_discharge_revenue_krw),
        "solar_cp": float(row.solar_capacity_payment_krw) * 0.5,
        "penalty": -float(row.imbalance_penalty_krw),
    }
    # Energy receipts already contain deviation settlement. Do not subtract it again.
    return {**parts, "total": sum(parts.values())}


def replay_html(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    raw = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return TEMPLATE.read_text(encoding="utf-8").replace("__REPLAY_DATA__", raw)


def render_replay(root: Path, row: pd.Series) -> None:
    from dashboard import distribution_replay as replay

    try:
        cached = replay.load(root, row)
    except (OSError, ValueError, KeyError) as exc:
        st.info(str(exc))
        cached = None
    if cached is None:
        st.info(
            "선택한 설비의 시간별 실행·정산 결과를 준비합니다. 최초 한 번은 2025년 전체를 "
            "기존 계산 함수로 실행하고 연간 저장 결과와 대조합니다."
        )
        if not st.button("시간별 결과 연결", key="de_replay_build", type="primary"):
            return
        try:
            with st.spinner("365일 실행·정산 중 · 날짜 간 배터리 잔량을 이어 계산합니다…"):
                cached = replay.build(root, row)
        except (OSError, ValueError, KeyError) as exc:
            st.error(str(exc))
            return
    hourly, meta = cached
    dates = sorted(pd.to_datetime(hourly.ts).dt.date.unique())
    # A new candidate receives its own widget state and never reuses another's hourly result.
    date = st.selectbox(
        "재생 날짜",
        dates,
        index=dates.index(pd.Timestamp("2025-05-01").date()),
        key="de_replay_date_" + str(row.id),
        format_func=lambda d: d.strftime("%Y년 %m월 %d일"),
    )
    html = replay_html(replay.day_payload(hourly, meta, str(date)))
    if hasattr(st, "iframe"):
        st.iframe(html, height="content", alt="배전망 ESS 시간별 충방전·배터리 잔량·누적 수입 재생")
    else:
        components.html(html, height=860, scrolling=True)
    with st.expander("시간별 원본과 연결 기준"):
        st.caption(
            "forecast_exec / standard · 8,760시간 연간 원본 대조 통과 · "
            "실측 발전량으로 실행한 백테스트이며 실시간 계측 화면은 아닙니다. "
            "하루가 바뀌어도 배터리 잔량은 이어지고, "
            "화면의 누적 수입은 선택한 날짜 0시에 시작합니다."
        )
        st.download_button(
            "시간별 실행·정산 CSV",
            hourly.to_csv(index=False).encode("utf-8-sig"),
            "distribution_forecast_exec_standard_hourly.csv",
            "text/csv",
            key="de_replay_download",
        )
