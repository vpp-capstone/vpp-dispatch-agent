"""Display-only navigation must preserve saved values and keep missing results missing."""

from pathlib import Path
from shutil import copyfile

import pandas as pd
import pytest
from dashboard.distribution_display import (
    OPERATING,
    RESULTS,
    SIZING,
    load_display,
    selected_id,
    visible_candidates,
)
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_reported_values_and_unreported_combinations():
    rows, _ = load_display(ROOT)
    assert len(rows) == 135
    assert rows.id.is_unique
    assert rows.operation.eq("forecast_exec").all()
    indexed = rows.set_index("id")
    small = indexed.loc["cap_2.50MW|kr_notice_example|2:8"]
    large = indexed.loc["cap_2.50MW|kr_notice_example|4:20"]
    assert small.npv_krw == -1.5e8  # Exact published precision; no cashflow recalculation.
    assert large.npv_krw == -26.4e8
    assert large.needed_annual_support_krw == pytest.approx(2.49e8, rel=0, abs=1e-6)
    dynamic = indexed.loc["dyn_1.00MW|second_life_example|4:20"]
    assert dynamic.npv_krw == 2.5e8
    assert dynamic.npv_source == "결과 보고서 §5 · 소수 1자리 억원"
    assert small.needed_annual_support_krw > 0
    assert pd.notna(small.irr)
    profitable = indexed.loc["cap_2.50MW|new_2025_bnef|2:8"]
    assert profitable.irr > 0
    assert 0 < profitable.discounted_payback_years < 20
    assert profitable.needed_annual_support_krw == 0
    assert pd.isna(indexed.loc["cap_1.75MW|new_2025_bnef|4:20"].npv_krw)
    # Forecast-only sizing rows must not be substituted for execution results.
    assert "cap_2.50MW|kr_notice_example|3:15" not in indexed.index
    assert "cap_2.50MW|second_life_example|2:8" not in indexed.index


def test_operating_money_and_energy_are_copied_without_rescaling():
    rows, _ = load_display(ROOT)
    sources = {name: pd.read_csv(ROOT / RESULTS / name) for name in (OPERATING, SIZING)}
    for row in rows.itertuples():
        source = sources[row.source_file]
        stored = source[(source["case"] == row.case) & (source.battery == row.battery)
                        & (source.operation == "forecast_exec")].iloc[0]
        for column in ("ess_charge_mwh", "ess_discharge_mwh", "ess_discharge_revenue_krw",
                       "market_revenue_krw", "solar_direct_revenue_krw", "days"):
            assert getattr(row, column) == stored[column]


def test_missing_report_keeps_operating_rows_without_inventing_economics(tmp_path):
    target = tmp_path / RESULTS
    target.mkdir(parents=True)
    for name in (OPERATING, SIZING):
        copyfile(ROOT / RESULTS / name, target / name)
    rows, report = load_display(tmp_path)
    assert not report
    assert len(rows) == 135
    assert rows.npv_krw.isna().all()
    assert rows.needed_annual_support_krw.isna().all()
    assert rows.irr.isna().all()
    assert rows.discounted_payback_years.isna().all()


def test_empty_filter_never_reuses_a_different_equipment():
    rows, _ = load_display(ROOT)
    empty = visible_candidates(rows, "cap_4.00MW", "new_2025_bnef", "2:8")
    assert selected_id(empty, "cap_2.50MW|new_2025_bnef|2:8") is None


def test_step_navigation_preserves_choices_and_line_change_clears_filters():
    app = AppTest.from_string(f"""
from pathlib import Path
import streamlit as st
from dashboard.distribution_view import render
st.session_state.setdefault('operator_day', 'untouched')
render(Path({str(ROOT)!r}))
""").run(timeout=10)
    assert not app.exception
    app.button(key="de_nav_1").click().run()
    app.selectbox(key="_de_battery").select("new_2025_bnef").run()
    app.selectbox(key="_de_size").select("3:12").run()
    chosen = "cap_2.50MW|new_2025_bnef|3:12"
    assert app.session_state["de_chosen"] == chosen
    app.button(key="de_nav_2").click().run()
    assert not app.exception
    assert app.session_state["de_chosen"] == chosen
    assert any('class="de-verdict positive"' in m.value for m in app.markdown)
    app.button(key="de_nav_1").click().run()
    assert app.selectbox(key="_de_battery").value == "new_2025_bnef"
    assert app.selectbox(key="_de_size").value == "3:12"
    app.button(key="de_nav_0").click().run()
    app.selectbox(key="_de_line").select("cap_4.00MW").run()
    app.button(key="de_nav_1").click().run()
    assert not app.exception
    assert app.session_state["de_battery"] == "all"
    assert app.session_state["de_size"] == "all"
    assert app.session_state["de_chosen"].startswith("cap_4.00MW|")
    assert app.session_state["operator_day"] == "untouched"


def test_missing_npv_is_a_pending_decision():
    app = AppTest.from_string(f"""
from pathlib import Path
import streamlit as st
from dashboard.distribution_view import render
st.session_state.de_step = 2
st.session_state.de_line = 'cap_1.75MW'
st.session_state.de_chosen = 'cap_1.75MW|new_2025_bnef|4:20'
render(Path({str(ROOT)!r}))
""").run(timeout=10)
    assert not app.exception
    assert any("판단 보류" in m.value for m in app.markdown)
    assert not any("de-verdict positive" in m.value for m in app.markdown)
