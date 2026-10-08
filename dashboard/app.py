"""담당: 대시보드팀. 실행: python -m streamlit run dashboard/app.py

지금은 data/sample/hourly.csv를 읽어 그래프만 그린다.
예측·최적화·평가 모듈이 완성되면 pipeline.run_day_ahead 결과를 여기에 연결한다.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SAMPLE = Path(__file__).resolve().parents[1] / "data/sample/hourly.csv"

st.set_page_config(page_title="제주 VPP", layout="wide")
st.title("제주 VPP · 다음 날 수요예측")
st.caption("샘플 데이터")

frame = pd.read_csv(SAMPLE, parse_dates=["timestamp"])
fig = go.Figure(go.Scatter(x=frame["timestamp"], y=frame["demand_mw"], name="수요"))
fig.update_layout(xaxis_title="시간 (KST)", yaxis_title="전력 (MW)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("연결 예정")
st.write("수요예측 → ESS 운영계획 → 예측 평가 → 운영비용 평가")
