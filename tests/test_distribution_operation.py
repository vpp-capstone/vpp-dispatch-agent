"""The operating screen must preserve the saved execution and standard revenue basis."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from dashboard.distribution_display import load_display
from dashboard.distribution_operation import replay_html, revenue_breakdown
from streamlit.testing.v1 import AppTest

from vpp.project_economics import EconomicsConfig, _standard_view

ROOT = Path(__file__).resolve().parents[1]


def test_annual_receipts_match_existing_standard_economics_for_every_candidate():
    rows, _ = load_display(ROOT)
    cfg = EconomicsConfig.from_yaml(ROOT / "config/project_economics_v0.yaml")
    before = rows.copy(deep=True)
    for _, row in rows.iterrows():
        parts = revenue_breakdown(row)
        expected = _standard_view(cfg, row, row.ess_charge_mwh + row.ess_discharge_mwh)
        assert parts["total"] == pytest.approx(expected.op.market_revenue_krw, rel=0, abs=1e-6)
        assert parts["total"] + expected.owner_fixed_krw == pytest.approx(
            row.market_revenue_krw, abs=0.02
        )
    pd.testing.assert_frame_equal(rows, before)


def test_deviation_and_ess_capacity_payment_are_not_added_again():
    rows, _ = load_display(ROOT)
    row = rows[rows.id.eq("cap_2.50MW|kr_notice_example|2:8")].iloc[0].copy()
    parts = revenue_breakdown(row)
    assert parts["total"] == pytest.approx(176211537.23, rel=0, abs=1e-6)
    row["deviation_krw"] = -1e10
    row["ess_capacity_payment_krw"] = 1e10
    row["solar_direct_revenue_krw"] = 1e10
    assert revenue_breakdown(row) == parts


@pytest.mark.parametrize(
    "column,value",
    [
        ("operation", "forecast"),
        ("days", 364),
        ("solar_capacity_payment_krw", float("nan")),
        ("imbalance_penalty_krw", float("inf")),
    ],
)
def test_incomplete_or_other_operation_results_are_not_presented_as_income(column, value):
    rows, _ = load_display(ROOT)
    row = rows.iloc[0].copy()
    row[column] = value
    with pytest.raises(ValueError):
        revenue_breakdown(row)


def test_hourly_replay_escapes_embedded_data():
    payload = {"candidate": '</script><script>alert("test")</script>', "hours": []}
    html = replay_html(payload)
    raw = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html).group(
        1
    )
    data = json.loads(raw)
    assert data == payload
    assert payload["candidate"] not in html
    assert "시간별 충·방전" in html


def make_app(line="cap_2.50MW", chosen="cap_2.50MW|kr_notice_example|2:8"):
    return AppTest.from_string(f"""
from pathlib import Path
import streamlit as st
from dashboard.distribution_view import render
st.session_state.setdefault('de_step', 3)
st.session_state.setdefault('de_line', {line!r})
st.session_state.setdefault('de_chosen', {chosen!r})
st.session_state.setdefault('operator_day', 'untouched')
render(Path({str(ROOT)!r}))
""").run(timeout=10)


def test_income_screen_follows_selected_equipment_and_preserves_other_dashboard_state():
    app = make_app()
    assert not app.exception
    page = "\n".join(m.value for m in app.markdown)
    assert "STEP 04 / 04" in page
    assert "1.76 억원" in page
    assert "-1.5 억원" in page
    app.button(key="de_operation_back").click().run()
    app.button(key="de_operation_next").click().run()
    assert app.session_state["de_chosen"] == "cap_2.50MW|kr_notice_example|2:8"
    app.button(key="de_operation_compare").click().run()
    app.selectbox(key="_de_chosen").select("cap_2.50MW|new_2025_bnef|2:8").run()
    app.button(key="de_nav_3").click().run()
    assert not app.exception
    assert app.session_state["de_chosen"] == "cap_2.50MW|new_2025_bnef|2:8"
    assert any("9.0 억원" in m.value for m in app.markdown)
    assert app.session_state["operator_day"] == "untouched"


def test_income_does_not_imply_profit_when_npv_is_missing():
    app = make_app("cap_1.75MW", "cap_1.75MW|new_2025_bnef|4:20")
    assert not app.exception
    assert any("투자 경제성 판단은 보류" in m.value for m in app.markdown)


def test_empty_equipment_filter_cannot_show_a_previous_candidates_income():
    app = make_app()
    app.session_state["de_line"] = "cap_4.00MW"
    app.session_state["de_size"] = "2:8"
    app.run()
    assert not app.exception
    assert app.session_state["de_chosen"] is None
    assert any("저장 결과가 없습니다" in message.value for message in app.info)
    assert not any("VPP 귀속 수입 / 년" in m.value for m in app.markdown)
