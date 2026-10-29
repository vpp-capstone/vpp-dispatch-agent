"""담당: 대시보드팀. 실행: python -m streamlit run dashboard/app.py

JEJU FLOW 시안(캡스톤디자인/대시보드/*.png)의 구성을 따른다. 발전량 예측은 아직 없으므로
예측 구간·예측 오차 패널티·"어제 결과(계획 vs 실제)"는 빼고, 그 자리에 "검증" 탭을 둔다.
데이터는 `python -m vpp.backtest` 결과 폴더(data/processed/backtest/<실행ID>)를 읽는다.
완전예측 백테스트(실측 하루전 SMP·실측 발전량 입력)이며 실제 입찰 성과가 아니다.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vpp.optimization import EssParams, solve_window
from vpp.validation import check_plan, check_properties

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = ROOT / "data/processed/backtest"
SAMPLE = ROOT / "data/sample/hourly.csv"
EPS = 1e-6
MAN = 1e4  # 만 원
EOK = 1e8  # 억 원

COLORS = {
    "pv": "#F2C14E", "wind": "#9DC3E6", "ess": "#8B5CF6", "ess_light": "#C4B5FD",
    "curtail": "#F97066", "sell": "#111827", "smp": "#15803D", "accent": "#0F766E",
    "no_ess": "#D1D5DB", "rule": "#9CA3AF",
}
ACTION_STYLE = {
    "ESS 충전": ("#3B82F6", "🔋"), "ESS 방전": ("#8B5CF6", "⚡"), "출력 감축": ("#F97066", "🚫"),
}

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
.jf-header {display:flex; align-items:center; gap:12px; padding:6px 0 10px 0;
            border-bottom:1px solid #E5E7EB; margin-bottom:8px;}
.jf-logo {background:#0F766E; color:#fff; font-weight:800; border-radius:8px;
          width:34px; height:34px; display:flex; align-items:center; justify-content:center;}
.jf-title {font-weight:800; font-size:20px; letter-spacing:1px; color:#111827;}
.jf-sub {color:#6B7280; font-size:13px;}
.jf-badge {border:1px solid #F59E0B; color:#B45309; border-radius:14px; padding:2px 10px;
           font-size:12px; background:#FFFBEB;}
.jf-right {margin-left:auto; color:#6B7280; font-size:13px;}
.jf-h1 {font-size:28px; font-weight:800; color:#111827; margin:4px 0 2px 0;}
.jf-lead {color:#6B7280; font-size:14px; margin-bottom:12px;}
.jf-tag {display:inline-block; border-left:4px solid #0F766E; padding-left:8px;
         color:#0F766E; font-size:13px; font-weight:600;}
.kpi {background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:14px 16px;
      display:flex; gap:12px; align-items:flex-start; min-height:104px;}
.kpi .icon {width:40px; height:40px; border-radius:50%; display:flex; align-items:center;
            justify-content:center; font-size:20px; flex-shrink:0;}
.kpi .label {color:#6B7280; font-size:13px;}
.kpi .value {font-size:24px; font-weight:800; color:#111827; line-height:1.3;}
.kpi .unit {font-size:14px; font-weight:600; color:#374151;}
.kpi .delta {font-size:13px; font-weight:600;}
.up {color:#0F9D58;} .down {color:#2563EB;} .flat {color:#6B7280;}
.act {background:#fff; border:1px solid #E5E7EB; border-left:5px solid #ccc; border-radius:10px;
      padding:10px 12px; min-height:150px;}
.act.sel {box-shadow:0 0 0 2px #0F766E inset;}
.act .time {color:#6B7280; font-size:12px; display:flex; justify-content:space-between;}
.act .pill {background:#0F766E; color:#fff; border-radius:10px; padding:0 8px; font-size:11px;}
.act .name {font-weight:800; font-size:16px; margin:4px 0;}
.act .desc {color:#6B7280; font-size:12px; line-height:1.4;}
.act .eff {margin-top:6px; font-size:13px; font-weight:700; color:#0F766E;}
.note {background:#ECFDF5; border-radius:10px; padding:10px 14px; color:#065F46; font-size:13px;}
.chip {display:inline-block; border-radius:6px; padding:3px 12px; color:#fff; font-size:12px;
       font-weight:600; margin:2px 4px 2px 0;}
</style>
"""


# =============================== 결과 로딩 (순수 함수) ===============================

def list_runs(base: Path = BACKTEST_DIR) -> list[Path]:
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if (p / "summary.json").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_ts_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert("Asia/Seoul")
    return frame


def ess_from_validation(validation: dict[str, Any]) -> EssParams:
    names = {f.name for f in fields(EssParams)}
    return EssParams(**{k: v for k, v in validation["assumptions"].items() if k in names})


@st.cache_data(show_spinner=False)
def load_run(folder_str: str) -> dict[str, Any]:
    folder = Path(folder_str)
    hourly = _read_ts_csv(folder / "hourly_plan.csv")
    hourly["date"] = hourly["ts"].dt.date
    run: dict[str, Any] = {
        "folder": folder,
        "hourly": hourly,
        "summary": json.loads((folder / "summary.json").read_text(encoding="utf-8")),
        "validation": json.loads((folder / "validation.json").read_text(encoding="utf-8")),
        "daily": None,
        "baselines": None,
    }
    if (folder / "daily_summary.csv").exists():
        run["daily"] = pd.read_csv(folder / "daily_summary.csv", parse_dates=["date"])
    if (folder / "baselines_hourly.csv").exists():
        base = _read_ts_csv(folder / "baselines_hourly.csv")
        base["date"] = base["ts"].dt.date
        run["baselines"] = base
    run["ess"] = ess_from_validation(run["validation"])
    return run


def day_slice(run: dict[str, Any], day) -> pd.DataFrame:
    hourly = run["hourly"]
    return hourly[hourly["date"] == day].reset_index(drop=True)


def baseline_day(run: dict[str, Any], day, name: str) -> pd.DataFrame | None:
    """baselines_hourly.csv에서 그날의 기준 계획을 hourly 계획 컬럼명으로 돌려준다."""
    base = run["baselines"]
    if base is None:
        return None
    rows = base[base["date"] == day].reset_index(drop=True)
    if rows.empty:
        return None
    out = rows[["ts", "generation_mwh", "price_krw_per_mwh"]].copy()
    for col in ("sell_mwh", "charge_mwh", "discharge_mwh", "curtailment_mwh", "energy_end_mwh"):
        out[col] = rows[f"{name}_{col}"].to_numpy()
    return out


def net_revenue(plan: pd.DataFrame, ess: EssParams) -> float:
    throughput = float(plan["charge_mwh"].sum() + plan["discharge_mwh"].sum())
    return float(plan["price_krw_per_mwh"] @ plan["sell_mwh"]) - (
        ess.throughput_cost_krw_per_mwh * throughput
    )


def plan_stats(plan: pd.DataFrame, ess: EssParams) -> dict[str, float]:
    return {
        "순수익 (만 원)": net_revenue(plan, ess) / MAN,
        "판매량 (MWh)": float(plan["sell_mwh"].sum()),
        "ESS 충전량 (MWh)": float(plan["charge_mwh"].sum()),
        "ESS 방전량 (MWh)": float(plan["discharge_mwh"].sum()),
        "버려지는 전력 (MWh)": float(plan["curtailment_mwh"].sum()),
    }


def actions_from_plan(plan: pd.DataFrame) -> list[dict[str, Any]]:
    """계획에서 연속 구간(충전·방전·감축)을 뽑아 '할 일' 카드로 만든다."""
    kinds = (("ESS 충전", "charge_mwh"), ("ESS 방전", "discharge_mwh"),
             ("출력 감축", "curtailment_mwh"))
    actions = []
    for name, col in kinds:
        active = plan[col].to_numpy() > EPS
        i = 0
        while i < len(plan):
            if not active[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(plan) and active[j + 1]:
                j += 1
            block = plan.iloc[i:j + 1]
            price = block["price_krw_per_mwh"]
            amount = float(block[col].sum())
            if name == "ESS 방전":
                effect = float(price @ block[col])
                desc = "저장한 전기를 고가 시간에 판매"
            elif name == "ESS 충전":
                effect = -float(price.clip(lower=0) @ block[col])
                desc = "발전 중 일부를 저장 (저가·음수 가격 시간, 판매 보류)"
            else:
                effect = float((-price).clip(lower=0) @ block[col])
                desc = "음수 가격 시간 판매 대신 감축 (손실 회피)"
            start, end = block["ts"].iloc[0], block["ts"].iloc[-1] + pd.Timedelta(hours=1)
            actions.append({
                "name": name, "start": start, "end": end,
                "time": f"{start:%H:%M}–{end:%H:%M}", "hours": int(j - i + 1),
                "mwh": amount, "effect_krw": effect, "desc": desc,
                "smp": f"SMP {price.min() / 1000:,.0f}~{price.max() / 1000:,.0f}원/kWh",
                "rows": (i, j),
            })
            i = j + 1
    return sorted(actions, key=lambda a: a["start"])


def contributions(plan: pd.DataFrame, no_ess: pd.DataFrame | None,
                  ess: EssParams) -> list[tuple]:
    """ESS 없음 → MILP 순수익을 조치별로 나눈 워터폴 항목. 잔차는 '기타'로 둔다."""
    base = net_revenue(no_ess, ess) if no_ess is not None else float(
        plan["price_krw_per_mwh"].clip(lower=0) @ plan["generation_mwh"]
    )
    price = plan["price_krw_per_mwh"]
    discharge = float(price @ plan["discharge_mwh"])
    charge = -float(price.clip(lower=0) @ plan["charge_mwh"])
    wear = -ess.throughput_cost_krw_per_mwh * float(
        plan["charge_mwh"].sum() + plan["discharge_mwh"].sum()
    )
    total = net_revenue(plan, ess)
    other = total - base - discharge - charge - wear
    items = [("ESS 없음", base, "absolute"), ("ESS 충전\n(판매 보류)", charge, "relative"),
             ("ESS 방전\n(피크 판매)", discharge, "relative")]
    if abs(wear) > 1:
        items.append(("열화비", wear, "relative"))
    if abs(other) > 1:
        items.append(("기타", other, "relative"))
    items.append(("MILP 계획", total, "total"))
    return items


def recompute_day(run: dict[str, Any], day, ess: EssParams | None = None,
                  initial_energy_mwh: float | None = None,
                  horizon_hours: int = 48) -> dict[str, Any]:
    """선택한 날부터 48시간 창을 다시 풀고 앞 24시간을 돌려준다.

    ESS 사양을 바꿔 넘기면 시뮬레이션이 된다.
    """
    hourly = run["hourly"]
    stored_ess = run["ess"]
    ess = ess or stored_ess
    idx = hourly.index[hourly["date"] == day]
    if len(idx) == 0:
        raise ValueError("선택한 날짜가 결과에 없습니다.")
    i0 = int(idx[0])
    window = hourly.iloc[i0:i0 + horizon_hours]
    n = len(idx)
    if initial_energy_mwh is None:
        initial_energy_mwh = (
            stored_ess.initial_energy_mwh if i0 == 0
            else float(hourly.iloc[i0 - 1]["energy_end_mwh"])
        )
    initial = ess.clip_energy(initial_energy_mwh)
    is_last = i0 + n >= len(hourly)
    policy = run["validation"]["assumptions"].get("policy", "rolling48")
    final_lo = ess.initial_energy_mwh if (policy == "rolling48" and is_last) else None
    result = solve_window(
        window["generation_mwh"].to_numpy(), window["price_krw_per_mwh"].to_numpy(),
        window["dt_hours"].to_numpy(), ess, initial, final_energy_min_mwh=final_lo,
    )
    frame = result.to_frame(window["generation_mwh"], window["price_krw_per_mwh"],
                            window["dt_hours"])
    checks = check_plan(frame, ess, initial_energy_mwh=initial,
                        objective_krw=result.objective_krw, final_energy_min_mwh=final_lo)
    committed = window.iloc[:n].copy().reset_index(drop=True)
    for col in ("sell_mwh", "charge_mwh", "discharge_mwh", "curtailment_mwh",
                "energy_end_mwh", "soc_end"):
        committed[col] = frame[col].to_numpy()[:n]
    stored = day_slice(run, day)
    return {
        "plan": committed, "checks": checks, "ess": ess,
        "solved_hours": int(len(window)), "committed_hours": n,
        "initial_energy_mwh": initial, "mip_gap": result.mip_gap,
        "objective_recomputed_krw": net_revenue(committed, ess),
        "objective_stored_krw": net_revenue(stored, stored_ess),
    }


def checks_table(checks: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(checks)
    if table.empty:
        return table
    table["결과"] = table["passed"].map({True: "✅", False: "❌"})
    return table[["결과", "item", "value"]].rename(columns={"item": "항목", "value": "값"})


# =============================== 그래프 ===============================

def timeline_figure(plan: pd.DataFrame, ess: EssParams, title_suffix: str = "") -> go.Figure:
    x = plan["ts"]
    dt = plan["dt_hours"]
    pv = plan["pv_kwh"] / 1000 / dt if "pv_kwh" in plan else plan["generation_mwh"] / dt
    wind = plan["wind_kwh"] / 1000 / dt if "wind_kwh" in plan else 0 * pv
    fig = go.Figure()
    fig.add_bar(x=x, y=pv, name="태양광", marker_color=COLORS["pv"])
    fig.add_bar(x=x, y=wind, name="풍력", marker_color=COLORS["wind"])
    fig.add_bar(x=x, y=plan["discharge_mwh"] / dt, name="ESS 방전", marker_color=COLORS["ess"])
    fig.add_bar(x=x, y=-plan["charge_mwh"] / dt, name="ESS 충전",
                marker_color=COLORS["ess_light"])
    fig.add_bar(x=x, y=-plan["curtailment_mwh"] / dt, name="감축", marker_color=COLORS["curtail"])
    fig.add_scatter(x=x, y=plan["sell_mwh"] / dt, name="판매량 (계획)", mode="lines",
                    line=dict(color=COLORS["sell"], width=2.5, shape="hv"))
    fig.add_scatter(x=x, y=plan["price_krw_per_mwh"] / 1000, name="하루전 SMP (원/kWh)",
                    yaxis="y2", mode="lines",
                    line=dict(color=COLORS["smp"], width=1.5, dash="dash"))
    fig.update_layout(
        barmode="relative", height=400, margin=dict(t=40, b=30, l=10, r=10),
        title=dict(text=f"24시간 타임라인{title_suffix}", font=dict(size=15)),
        xaxis=dict(title="시간 (KST)", tickformat="%H"),
        yaxis=dict(title="MW (시간 평균 출력)"),
        yaxis2=dict(title="원/kWh", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12, x=0), plot_bgcolor="#fff",
    )
    fig.update_yaxes(gridcolor="#F3F4F6")
    return fig


def soc_figure(plan: pd.DataFrame, ess: EssParams) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=plan["ts"], y=plan["energy_end_mwh"], name="구간 끝 에너지 (MWh)",
                    mode="lines+markers", line=dict(color=COLORS["ess"], shape="hv"))
    for level, label in ((ess.energy_max_mwh, "상한"), (ess.energy_min_mwh, "하한")):
        fig.add_hline(y=level, line_dash="dot", line_color="gray",
                      annotation_text=f"{label} {level:.1f} MWh")
    fig.update_layout(height=280, margin=dict(t=30, b=30, l=10, r=10),
                      xaxis=dict(tickformat="%H"),
                      yaxis=dict(title="MWh", range=[0, ess.capacity_mwh * 1.08]),
                      plot_bgcolor="#fff", title=dict(text="ESS 잔량 (SOC)", font=dict(size=15)))
    return fig


def compare_bar_figure(values: dict[str, float]) -> go.Figure:
    names = list(values)
    vals = [values[n] / MAN for n in names]
    colors = [COLORS["no_ess"], COLORS["rule"], COLORS["accent"], "#F59E0B"][: len(names)]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:,.1f}" for v in vals], textposition="outside"))
    fig.update_layout(height=260, margin=dict(t=30, b=20, l=10, r=40), plot_bgcolor="#fff",
                      xaxis=dict(title="만 원", gridcolor="#F3F4F6"),
                      title=dict(text="계획별 순수익", font=dict(size=15)))
    return fig


def waterfall_figure(items: list[tuple]) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        x=[i[0] for i in items], y=[i[1] / MAN for i in items], measure=[i[2] for i in items],
        text=[f"{i[1] / MAN:+,.1f}" if i[2] == "relative" else f"{i[1] / MAN:,.1f}"
              for i in items],
        textposition="outside", connector=dict(line=dict(color="#9CA3AF", dash="dot")),
        increasing=dict(marker_color="#3B82F6"), decreasing=dict(marker_color="#F97316"),
        totals=dict(marker_color=COLORS["accent"]),
    ))
    fig.update_layout(height=320, margin=dict(t=30, b=20, l=10, r=10), plot_bgcolor="#fff",
                      yaxis=dict(title="만 원", gridcolor="#F3F4F6"),
                      title=dict(text="조치별 기여도 (순수익)", font=dict(size=15)))
    return fig


def monthly_table(run: dict[str, Any]) -> pd.DataFrame | None:
    daily = run["daily"]
    if daily is None or "no_ess_net_revenue_krw" not in daily:
        return None
    month = daily["date"].dt.to_period("M").astype(str)
    grouped = daily.groupby(month).agg(
        days=("date", "count"), no_ess=("no_ess_net_revenue_krw", "sum"),
        rule=("rule_based_net_revenue_krw", "sum"), milp=("objective_krw", "sum"),
    )
    out = pd.DataFrame({
        "월": grouped.index, "일수": grouped["days"].to_numpy(),
        "ESS 없음 (만 원)": grouped["no_ess"].to_numpy() / MAN,
        "규칙 계획 (만 원)": grouped["rule"].to_numpy() / MAN,
        "MILP (만 원)": grouped["milp"].to_numpy() / MAN,
    })
    out["증가액 vs ESS 없음 (만 원)"] = out["MILP (만 원)"] - out["ESS 없음 (만 원)"]
    out["증가율 (%)"] = out["증가액 vs ESS 없음 (만 원)"] / out["ESS 없음 (만 원)"] * 100
    return out


# =============================== HTML 조각 ===============================

def kpi(label: str, value: str, delta: str = "", direction: str = "flat", icon: str = "●",
        bg: str = "#ECFDF5") -> str:
    arrow = {"up": "▲", "down": "▼", "flat": ""}[direction]
    return (
        f'<div class="kpi"><div class="icon" style="background:{bg}">{icon}</div><div>'
        f'<div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="delta {direction}">{arrow} {delta}</div></div></div>'
    )


def unit(text: str) -> str:
    return f"<span class='unit'>{text}</span>"


def action_card(a: dict[str, Any], selected: bool) -> str:
    color, icon = ACTION_STYLE[a["name"]]
    sign = "+" if a["effect_krw"] >= 0 else "−"
    label = "정산" if a["name"] != "ESS 충전" else "판매 보류"
    return (
        f'<div class="act{" sel" if selected else ""}" style="border-left-color:{color}">'
        f'<div class="time"><span>{a["time"]}</span><span class="pill">반영</span></div>'
        f'<div class="name">{icon} {a["name"]} {a["mwh"]:.1f} MWh</div>'
        f'<div class="desc">{a["desc"]}<br>{a["smp"]} · {a["hours"]}시간</div>'
        f'<div class="eff">{label} {sign}{abs(a["effect_krw"]) / MAN:,.1f}만 원</div></div>'
    )


def header(run: dict[str, Any] | None, day) -> None:
    right = ""
    if run is not None:
        s = run["summary"]
        right = (f"{day:%Y.%m.%d (%a)} · 백테스트 {s['start']} ~ {s['end']} · "
                 f"{s['policy']} · {run['folder'].name}")
    st.markdown(
        '<div class="jf-header"><div class="jf-logo">J</div>'
        '<span class="jf-title">JEJU FLOW</span><span class="jf-sub">제주 VPP 중개사업자용</span>'
        '<span class="jf-badge">완전예측 백테스트 · 실제 입찰 아님</span>'
        f'<span class="jf-right">{right}</span></div>',
        unsafe_allow_html=True,
    )


# =============================== 탭 ===============================

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
    wear = st.number_input("열화비 (원/MWh)", 0, 200_000,
                           int(stored.throughput_cost_krw_per_mwh), 1000, key="sim_wear")
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
        throughput_cost_krw_per_mwh=float(wear),
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


def render_no_results() -> None:
    st.info(
        "백테스트 결과 폴더가 없습니다. 원본 CSV 4개가 있는 폴더를 지정해 먼저 실행하세요:\n\n"
        "`python -m vpp.backtest --raw-dir <원본 폴더> --start 2024-06-01 --end 2025-12-31`\n\n"
        f"결과는 `{BACKTEST_DIR.relative_to(ROOT)}/<실행ID>/` 에 생깁니다."
    )
    if SAMPLE.exists():
        st.caption("참고: 샘플 수요 데이터 (data/sample/hourly.csv)")
        frame = pd.read_csv(SAMPLE, parse_dates=["timestamp"])
        fig = go.Figure(go.Scatter(x=frame["timestamp"], y=frame["demand_mw"], name="수요"))
        fig.update_layout(xaxis_title="시간 (KST)", yaxis_title="전력 (MW)", height=300)
        st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(page_title="JEJU FLOW · 제주 VPP", layout="wide", page_icon="🔋")
    st.markdown(CSS, unsafe_allow_html=True)
    runs = list_runs()
    if not runs:
        header(None, None)
        render_no_results()
        return
    top = st.columns([2, 1.2, 4])
    choice = top[0].selectbox("백테스트 결과", runs, format_func=lambda p: p.name,
                              key="run_folder")
    try:
        run = load_run(str(choice))
    except Exception as exc:  # 깨진 결과 폴더가 있어도 화면은 떠야 한다
        header(None, None)
        st.error(f"결과 폴더를 읽지 못했습니다: {exc}")
        return
    days = sorted(run["hourly"]["date"].unique())
    day = top[1].selectbox("날짜", days, index=len(days) - 1, key="day")
    header(run, day)
    tab_plan, tab_econ, tab_res, tab_check = st.tabs(
        ["운영계획", "경제성 비교", "운영 자원", "검증"]
    )
    with tab_plan:
        render_plan_tab(run, day)
    with tab_econ:
        render_economics_tab(run, day)
    with tab_res:
        render_resources_tab(run, day)
    with tab_check:
        render_validation_tab(run, day)


main()
