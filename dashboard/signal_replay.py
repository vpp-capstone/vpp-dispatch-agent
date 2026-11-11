"""Animated comparisons based exclusively on the selected saved input/result bundle."""

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def render_signal(
    rows: list[dict],
    *,
    title: str,
    day,
    unit: str,
    kind="forecast",
    forecast_label="예측",
    actual_label="실측",
):
    if not rows:
        st.info("선택한 날의 비교 가능한 값이 없습니다.")
        return
    data = dict(
        rows=rows,
        title=title,
        day=str(day),
        unit=unit,
        kind=kind,
        forecastLabel=forecast_label,
        actualLabel=actual_label,
        deltaLabel="계획 금액 차이 · 실제 손익 아님" if kind == "revenue" else "예측 − 실측",
        note="실측을 미리 안 계획은 비교용 재최적화 결과입니다. 실제 운전·정산은 미구현입니다."
        if kind == "revenue"
        else "같은 시각의 저장된 예측과 실측을 비교합니다. 빈 시각을 0으로 채우지 않습니다.",
    )
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = (
        (Path(__file__).parent / "assets/signal_replay.html")
        .read_text()
        .replace("__SIGNAL_DATA__", payload)
    )
    if hasattr(st, "iframe"):
        st.iframe(html, height="content", alt=title)
    else:
        components.html(html, height=660, scrolling=True)
