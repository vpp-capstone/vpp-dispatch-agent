"""Operator-facing entry point. Local jobs are owned by vpp.operator_pipeline."""

import streamlit as st

from operator_console import render

st.set_page_config(
    page_title="JEJU FLOW · 운영계획",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed",
)
render()
