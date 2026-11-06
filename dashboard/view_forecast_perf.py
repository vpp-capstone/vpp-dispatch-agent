"""탭 2·3 — 발전량 / SMP 예측 성능. 계산은 vpp.forecast_metrics 만 호출한다 (모델 내부 무관)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from settings import Settings
from ui import COLORS, kpi, unit
from vpp import forecast_metrics as fm

OFFSET_LABEL = {0: "1일차 (D)", 1: "2일차 (D+1)"}
HORIZON_LABEL = {1: "내일 (h=1)", 2: "모레 (h=2)"}
MODEL_LABEL = {"lightgbm": "LightGBM 예측", "baseline": "기준 예측 (기후값)",
               "model": "SMP 모델", "b1": "기준 예측 (B1)"}


def _months(start: date, end: date) -> list[tuple[str, date, date]]:
    out = []
    for p in pd.period_range(start, end, freq="M"):
        s, e = max(p.start_time.date(), start), min(p.end_time.date(), end)
        out.append((str(p), s, e))
    return out


@st.cache_data(show_spinner=False)
def gen_metrics(_pred: pd.DataFrame, _expected: dict, stamp: tuple, level: str,
                start: date, end: date) -> pd.DataFrame:
    return fm.generation_metrics(_pred, _expected, level=level, start=start, end=end)


@st.cache_data(show_spinner=False)
def gen_series(_pred: pd.DataFrame, _expected: dict, stamp: tuple, level: str,
               group: str | None, offset: int, start: date, end: date) -> pd.DataFrame:
    return fm.generation_timeseries(_pred, _expected, level=level, group=group,
                                    forecast_day_offset=offset, start=start, end=end)


@st.cache_data(show_spinner=False)
def smp_metrics(_smp: pd.DataFrame, stamp: tuple, start: date, end: date) -> pd.DataFrame:
    return fm.smp_metrics(_smp, start=start, end=end)


def _gen_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in table.sort_values(["model", "forecast_day_offset"]).iterrows():
        rows.append({
            "모델": MODEL_LABEL[r["model"]], "일차": OFFSET_LABEL[int(r["forecast_day_offset"])],
            "nMAE (% of 설비용량)": r["nmae_pct"], "MAE (kWh/h)": r["mae_kwh"],
            "RMSE (kWh/h)": r["rmse_kwh"], "편향 (% of 설비용량)": r["bias_pct"],
            "평가 시각 수": r["n_evaluated"], "발전소 누락 제외": r["n_excluded_incomplete_assets"],
            "실측 결측": r["n_missing_actual"], "예측 결측": r["n_missing_forecast"],
        })
    return pd.DataFrame(rows)


def render_generation(settings: Settings, bundle: dict[str, Any] | None) -> None:
    st.markdown('<div class="jf-h1">발전량 예측이 실측과 얼마나 맞는가</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="jf-lead">예측·실측·기준 예측(기후값)을 같은 시각으로 맞춰 MAE·'
                'nMAE(설비용량 기준)·편향을 계산합니다. 1일차와 2일차는 같은 대상 시각을 '
                '두 번 세지 않도록 항상 따로 봅니다. 발전소가 빠진 시각은 합산하지 않고 '
                '제외 건수로 표시합니다. 모델 코드는 바꾸지 않았고 저장된 예측 파일만 '
                '평가합니다.</div>', unsafe_allow_html=True)
    if bundle is None or settings.start is None:
        st.warning("데이터 묶음을 읽지 못해 성능을 계산할 수 없습니다. 사이드바를 확인하세요.")
        return
    pred, expected, stamp = bundle["pred"], bundle["expected"], settings.stamp()
    start, end = settings.start, settings.end
    c1, c2, c3 = st.columns([1, 1.4, 1])
    level = c1.radio("범위", ["portfolio", "energy_type", "asset"],
                     format_func={"portfolio": "포트폴리오 (21개소 합)", "energy_type": "발전원",
                                  "asset": "발전소"}.get, key="gen_level", horizontal=True)
    group = None
    if level == "energy_type":
        group = c2.radio("발전원", ["solar", "wind"], key="gen_type", horizontal=True,
                         format_func={"solar": "태양광", "wind": "풍력"}.get)
    elif level == "asset":
        assets = sorted(pred["asset_id"].unique())
        group = c2.selectbox("발전소", assets, key="gen_asset")
    offset = c3.radio("시계열 일차", [0, 1], format_func=OFFSET_LABEL.get, key="gen_offset",
                      horizontal=True)

    table = gen_metrics(pred, expected, stamp, level, start, end)
    if group is not None:
        table = table[table["group"] == group]
    if table.empty or table["n_evaluated"].sum() == 0:
        st.warning("이 범위·기간에는 예측과 실측이 함께 있는 시각이 없습니다 "
                   "(실측 열이 없는 파일이면 운영계획은 만들 수 있지만 성능 평가는 할 수 "
                   "없습니다).")
        st.dataframe(_gen_table(table), width="stretch", hide_index=True)
        return
    lg0 = table[(table["model"] == "lightgbm") & (table["forecast_day_offset"] == 0)].iloc[0]
    bl0 = table[(table["model"] == "baseline") & (table["forecast_day_offset"] == 0)].iloc[0]
    lg1 = table[(table["model"] == "lightgbm") & (table["forecast_day_offset"] == 1)]
    lg1 = lg1.iloc[0] if len(lg1) else None
    cards = [
        kpi("1일차 nMAE (LightGBM)", f"{lg0['nmae_pct']:.2f}{unit(' %')}",
            f"기준 예측 {bl0['nmae_pct']:.2f}% → 개선 {bl0['nmae_pct'] - lg0['nmae_pct']:+.2f}p",
            "up" if lg0["nmae_pct"] < bl0["nmae_pct"] else "down", "☀️", "#FFFBEB"),
        kpi("1일차 편향", f"{lg0['bias_pct']:+.2f}{unit(' % of 용량')}",
            "양수 = 과대예측 · 음수 = 과소예측", "flat", "⚖️", "#EFF6FF"),
        kpi("2일차 nMAE (LightGBM)",
            f"{lg1['nmae_pct']:.2f}{unit(' %')}" if lg1 is not None else "—",
            f"1일차 대비 {lg1['nmae_pct'] - lg0['nmae_pct']:+.2f}p" if lg1 is not None else "",
            "flat", "📅", "#F5F3FF"),
        kpi("평가 시각 수 (1일차)", f"{int(lg0['n_evaluated']):,}{unit(' 시간')}",
            f"누락 제외 {int(lg0['n_excluded_incomplete_assets'])} · 실측 결측 "
            f"{int(lg0['n_missing_actual'])}", "flat", "🧮"),
    ]
    for col, card in zip(st.columns(4), cards, strict=True):
        col.markdown(card, unsafe_allow_html=True)

    series = gen_series(pred, expected, stamp, level, group, offset, start, end)
    fig = go.Figure()
    fig.add_scatter(x=series["ts"], y=series["actual_kwh"] / 1000, name="실측", mode="lines",
                    line=dict(color="#111827", width=1.5))
    fig.add_scatter(x=series["ts"], y=series["lightgbm_kwh"] / 1000, name="LightGBM 예측",
                    mode="lines", line=dict(color=COLORS["accent"], width=1.5))
    fig.add_scatter(x=series["ts"], y=series["baseline_kwh"] / 1000, name="기준 예측 (기후값)",
                    mode="lines", line=dict(color="#9CA3AF", width=1, dash="dot"))
    fig.update_layout(height=360, margin=dict(t=40, b=30, l=10, r=10), plot_bgcolor="#fff",
                      title=dict(text=f"시계열 · {OFFSET_LABEL[offset]} · 빠진 시각은 끊어 그림",
                                 font=dict(size=15)),
                      yaxis=dict(title="MWh/h"), legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(gridcolor="#F3F4F6")
    st.plotly_chart(fig, width="stretch")

    t1, t2 = st.columns([1.3, 1])
    with t1:
        st.markdown("#### 지표 (모델 × 일차)")
        st.dataframe(_gen_table(table).round(3), width="stretch", hide_index=True)
        st.caption("nMAE = 100 × Σ|예측−실측| / Σ설비용량 "
                   "(generation_forecast.model.error_metrics 와 같은 식). "
                   "편향 = 100 × Σ(예측−실측) / Σ설비용량.")
    with t2:
        st.markdown("#### 월별 nMAE · 편향 (LightGBM)")
        rows = []
        for label, s, e in _months(start, end):
            m = gen_metrics(pred, expected, stamp, level, s, e)
            if group is not None:
                m = m[m["group"] == group]
            if m.empty:
                continue
            lg = m[m["model"] == "lightgbm"].set_index("forecast_day_offset")
            if lg.empty or lg["n_evaluated"].sum() == 0:
                continue
            rows.append({"월": label,
                         "1일차 nMAE %": lg["nmae_pct"].get(0),
                         "2일차 nMAE %": lg["nmae_pct"].get(1),
                         "1일차 편향 %": lg["bias_pct"].get(0),
                         "평가 시각": int(lg["n_evaluated"].get(0, 0))})
        st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)

    if level != "asset":
        st.markdown("#### 발전소별 1일차 nMAE (LightGBM)")
        assets = gen_metrics(pred, expected, stamp, "asset", start, end)
        a0 = assets[(assets["model"] == "lightgbm") & (assets["forecast_day_offset"] == 0)]
        a0 = a0.sort_values("nmae_pct", ascending=False)
        bar = go.Figure(go.Bar(x=a0["group"], y=a0["nmae_pct"], marker_color=COLORS["accent"],
                               text=a0["nmae_pct"].round(1), textposition="outside"))
        bar.update_layout(height=300, margin=dict(t=20, b=30, l=10, r=10), plot_bgcolor="#fff",
                          yaxis=dict(title="nMAE %", gridcolor="#F3F4F6"))
        st.plotly_chart(bar, width="stretch")


def render_smp(settings: Settings, bundle: dict[str, Any] | None) -> None:
    st.markdown('<div class="jf-h1">SMP 예측이 실측과 얼마나 맞는가</div>', unsafe_allow_html=True)
    st.markdown('<div class="jf-lead">단위 원/kWh. 예측(모델)·기준 예측(B1)·실측을 같은 대상 '
                '시각으로 '
                '맞춰 MAE·RMSE·편향을, 그리고 ESS 충·방전 시각에 중요한 <b>고가 4시간 적중률</b>'
                '(일별 예측 상위 4시간 ∩ 실측 상위 4시간 / 4, 동률은 이른 시각 우선)을 계산합니다. '
                '24시간이 모두 있는 날만 적중률에 넣습니다.</div>', unsafe_allow_html=True)
    if bundle is None or settings.start is None:
        st.warning("데이터 묶음을 읽지 못해 성능을 계산할 수 없습니다. 사이드바를 확인하세요.")
        return
    smp, stamp = bundle["smp_eval"], settings.stamp()
    start, end = settings.start, settings.end
    table = smp_metrics(smp, stamp, start, end)
    if table.empty or table["n_evaluated"].sum() == 0:
        st.warning("이 기간에는 SMP 예측과 실측이 함께 있는 시각이 없습니다.")
        return
    m1 = table[(table["horizon"] == 1) & (table["model"] == "model")].iloc[0]
    b1 = table[(table["horizon"] == 1) & (table["model"] == "b1")].iloc[0]
    cards = [
        kpi("내일 MAE (모델)", f"{m1['mae']:.1f}{unit(' 원/kWh')}",
            f"B1 {b1['mae']:.1f} → 개선 {b1['mae'] - m1['mae']:+.1f}",
            "up" if m1["mae"] < b1["mae"] else "down", "💱"),
        kpi("내일 고가 4시간 적중률", f"{m1['top4_hit_rate'] * 100:.1f}{unit(' %')}",
            f"B1 {b1['top4_hit_rate'] * 100:.1f}% · {int(m1['n_days_scored'])}일 평가",
            "up" if m1["top4_hit_rate"] >= b1["top4_hit_rate"] else "down", "🎯", "#F5F3FF"),
        kpi("내일 편향", f"{m1['bias']:+.1f}{unit(' 원/kWh')}", "양수 = 과대예측", "flat", "⚖️",
            "#EFF6FF"),
        kpi("평가 시간 수", f"{int(m1['n_evaluated']):,}{unit(' 시간')}",
            f"실측 결측 {int(m1['n_missing_actual'])} · 불완전 일 "
            f"{int(m1['n_days_excluded_incomplete'])}", "flat", "🧮", "#FFF7ED"),
    ]
    for col, card in zip(st.columns(4), cards, strict=True):
        col.markdown(card, unsafe_allow_html=True)

    horizon = st.radio("시계열 horizon", [1, 2], format_func=HORIZON_LABEL.get, key="smp_h",
                       horizontal=True)
    series = fm.smp_timeseries(smp, horizon=horizon, start=start, end=end)
    fig = go.Figure()
    fig.add_scatter(x=series["ts"], y=series["smp_actual"], name="실측", mode="lines",
                    line=dict(color="#111827", width=1.5))
    fig.add_scatter(x=series["ts"], y=series["smp_hat"], name="SMP 모델", mode="lines",
                    line=dict(color=COLORS["smp"], width=1.5))
    fig.add_scatter(x=series["ts"], y=series["smp_b1"], name="기준 예측 (B1)", mode="lines",
                    line=dict(color="#9CA3AF", width=1, dash="dot"))
    fig.update_layout(height=360, margin=dict(t=40, b=30, l=10, r=10), plot_bgcolor="#fff",
                      title=dict(text=f"시계열 · {HORIZON_LABEL[horizon]}", font=dict(size=15)),
                      yaxis=dict(title="원/kWh"), legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(gridcolor="#F3F4F6")
    st.plotly_chart(fig, width="stretch")

    t1, t2 = st.columns([1.3, 1])
    with t1:
        st.markdown("#### 지표 (horizon × 모델)")
        show = table.assign(horizon=table["horizon"].map(HORIZON_LABEL),
                            model=table["model"].map(MODEL_LABEL))
        show = show.rename(columns={
            "horizon": "대상", "model": "모델", "mae": "MAE", "rmse": "RMSE", "bias": "편향",
            "top4_hit_rate": "고가 4시간 적중률", "n_evaluated": "평가 시간",
            "n_days_scored": "평가 일수",
            "n_missing_actual": "실측 결측", "n_days_excluded_incomplete": "불완전 일 제외",
        })[["대상", "모델", "MAE", "RMSE", "편향", "고가 4시간 적중률", "평가 시간", "평가 일수",
            "실측 결측", "불완전 일 제외"]]
        st.dataframe(show.round(3), width="stretch", hide_index=True)
        unit_diff = table.get("unit_check_max_abs_diff_krw_per_mwh")
        if unit_diff is not None:
            st.caption(f"단위 대조: smp_hat_krw_per_mwh 와 smp_hat×1000 의 최대 차이 "
                       f"{float(unit_diff.iloc[0]):.2f} 원/MWh (파일 반올림 수준이면 정상).")
    with t2:
        st.markdown("#### 월별 (내일, 모델 vs B1)")
        rows = []
        for label, s, e in _months(start, end):
            m = smp_metrics(smp, stamp, s, e)
            if m.empty:
                continue
            mm = m[(m["horizon"] == 1) & (m["model"] == "model")]
            bb = m[(m["horizon"] == 1) & (m["model"] == "b1")]
            if mm.empty or bb.empty:
                continue
            rows.append({"월": label, "모델 MAE": mm["mae"].iloc[0], "B1 MAE": bb["mae"].iloc[0],
                         "모델 적중률": mm["top4_hit_rate"].iloc[0],
                         "B1 적중률": bb["top4_hit_rate"].iloc[0],
                         "평가 일수": int(mm["n_days_scored"].iloc[0])})
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
