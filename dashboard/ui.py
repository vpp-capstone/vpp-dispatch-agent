"""대시보드 공통: 상수·CSS·결과 로더·그래프·HTML 조각. 계산은 vpp 패키지 함수만 호출한다."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vpp.forecast_rolling import soc_trace
from vpp.optimization import EssParams, Terminal, plan_window
from vpp.validation import check_plan

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = ROOT / "data/processed/backtest"
CONFIG_DIR = ROOT / "config"
# 화면의 열화비 선택지는 이 두 YAML 에서 읽는다 (선택해도 YAML 은 덮어쓰지 않는다).
DEGRADATION_CONFIGS = (CONFIG_DIR / "ess_v0.yaml", CONFIG_DIR / "ess_v0_1.yaml")
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
.block-container {padding-top: 5rem; max-width: 1500px;}
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


def run_stamp(folder: Path) -> float:
    """같은 실행ID 로 다시 돌리면 파일이 바뀌므로 캐시 키에 수정시각을 넣는다."""
    files = [p for p in folder.iterdir() if p.is_file()]
    return max((p.stat().st_mtime for p in files), default=0.0)


@st.cache_data(show_spinner=False)
def load_run(folder_str: str, stamp: float = 0.0) -> dict[str, Any]:
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


def degradation_choices() -> dict[str, dict[str, Any]]:
    """열화비 선택지: {표시 이름: EssParams 열화 필드}. 값은 config YAML 에서 읽는다."""
    out: dict[str, dict[str, Any]] = {}
    for path in DEGRADATION_CONFIGS:
        if not path.exists():
            continue
        ess = EssParams.from_yaml(path)
        cost = ess.throughput_cost_krw_per_mwh
        if ess.degradation_model == "none" or cost == 0:
            label = f"열화비 미적용 ({ess.degradation_model}, 0원/MWh)"
        else:
            basis = "충전+방전" if ess.throughput_basis == "charge_plus_discharge" else "방전만"
            label = f"고정 열화비 적용 ({ess.degradation_model}, {cost:,.2f}원/MWh, {basis} 기준)"
        out[label] = {
            "degradation_model": ess.degradation_model,
            "throughput_basis": ess.throughput_basis,
            "throughput_cost_krw_per_mwh": cost,
            "source": str(path.relative_to(ROOT)),
        }
    return out


def net_revenue(plan: pd.DataFrame, ess: EssParams) -> float:
    """판매수익 − 열화비. 열화비는 ESS 열화모델의 비용 함수로만 계산한다."""
    return float(plan["price_krw_per_mwh"] @ plan["sell_mwh"]) - ess.degradation_cost_krw(
        plan["charge_mwh"], plan["discharge_mwh"]
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
    wear = -ess.degradation_cost_krw(plan["charge_mwh"], plan["discharge_mwh"])
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
    terminal = (Terminal.period_end_ge_initial(ess.initial_energy_mwh)
                if policy == "rolling48" and is_last else Terminal())
    # CLI 와 같은 공통 엔진(plan_window)으로 푼다. 대시보드에 최적화 계산을 따로 두지 않는다.
    plan = plan_window(
        window, ess, initial, commit_hours=float(window["dt_hours"].iloc[:n].sum()),
        terminal=terminal,
    )
    checks = check_plan(plan.hourly, ess, initial_energy_mwh=initial,
                        objective_krw=plan.totals(committed_only=False)["objective_krw"],
                        final_energy_min_mwh=terminal.final_energy_min_mwh)
    committed = plan.committed.drop(columns="committed").reset_index(drop=True)
    stored = day_slice(run, day)
    return {
        "plan": committed, "checks": checks, "ess": ess,
        "solved_hours": int(len(window)), "committed_hours": n,
        "initial_energy_mwh": initial, "mip_gap": plan.mip_gap,
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
    fig = go.Figure()
    if "pv_kwh" in plan and "wind_kwh" in plan:
        fig.add_bar(x=x, y=plan["pv_kwh"] / 1000 / dt, name="태양광", marker_color=COLORS["pv"])
        fig.add_bar(x=x, y=plan["wind_kwh"] / 1000 / dt, name="풍력", marker_color=COLORS["wind"])
    else:  # 발전원 구분이 없는 포트폴리오 합계 (예측 비교 실행)
        fig.add_bar(x=x, y=plan["generation_mwh"] / dt, name="전체 발전량 (태양광+풍력)",
                    marker_color=COLORS["pv"])
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


def header(run: dict[str, Any] | None, day,
           badge: str = "완전예측 백테스트 · 실제 입찰 아님") -> None:
    right = ""
    if run is not None:
        s = run["summary"]
        right = (f"{day:%Y.%m.%d (%a)} · 백테스트 {s['start']} ~ {s['end']} · "
                 f"{s['policy']} · {run['folder'].name}")
    st.markdown(
        '<div class="jf-header"><div class="jf-logo">J</div>'
        '<span class="jf-title">JEJU FLOW</span><span class="jf-sub">제주 VPP 중개사업자용</span>'
        f'<span class="jf-badge">{badge}</span>'
        f'<span class="jf-right">{right}</span></div>',
        unsafe_allow_html=True,
    )


# =============================== 예측 vs 완전예측 결과 ===============================

FORECAST_MODE = "forecast_vs_perfect_foresight_plan"


def run_mode(folder: Path) -> str:
    try:
        return json.loads((folder / "summary.json").read_text(encoding="utf-8")).get("mode", "")
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def load_forecast_run(folder_str: str, stamp: float = 0.0) -> dict[str, Any]:
    folder = Path(folder_str)
    run: dict[str, Any] = {
        "folder": folder,
        "summary": json.loads((folder / "summary.json").read_text(encoding="utf-8")),
        "validation": json.loads((folder / "validation.json").read_text(encoding="utf-8")),
        "daily": pd.read_csv(folder / "daily_comparison.csv", parse_dates=["planning_day"]),
        "excluded": pd.read_csv(folder / "excluded_days.csv"),
    }
    for name in ("forecast", "actual"):
        frame = _read_ts_csv(folder / f"hourly_{name}.csv")
        frame["date"] = frame["ts"].dt.date
        run[name] = frame
    names = {f.name for f in fields(EssParams)}
    run["ess"] = EssParams(**{k: v for k, v in run["summary"]["ess"].items() if k in names})
    return run


def soc_compare_figure(forecast: pd.DataFrame, actual: pd.DataFrame,
                       initials: dict[str, dict[int, float]], ess: EssParams,
                       title: str) -> go.Figure:
    """시작 잔량은 구간 첫 시각에, 종료 잔량은 각 시간의 끝(ts+dt)에 놓는다 (soc_trace)."""
    fig = go.Figure()
    for key, name, frame, color in (("forecast", "예측 기반", forecast, COLORS["ess"]),
                                    ("actual", "완전예측(실측)", actual, COLORS["accent"])):
        trace = soc_trace(frame, initials[key])
        fig.add_scatter(x=trace["ts"], y=trace["energy_mwh"], name=name, mode="lines+markers",
                        line=dict(color=color, width=2), marker=dict(size=5))
        starts = trace[trace["point"] == "start"]
        fig.add_scatter(x=starts["ts"], y=starts["energy_mwh"], name=f"{name} · 시작 잔량",
                        mode="markers", marker=dict(color=color, size=11, symbol="diamond"),
                        showlegend=False)
    for level, label in ((ess.energy_max_mwh, "상한"), (ess.energy_min_mwh, "하한")):
        fig.add_hline(y=level, line_dash="dot", line_color="gray",
                      annotation_text=f"{label} {level:.1f} MWh")
    fig.update_layout(height=320, margin=dict(t=40, b=30, l=10, r=10), plot_bgcolor="#fff",
                      title=dict(text=title, font=dict(size=15)),
                      yaxis=dict(title="MWh", range=[0, ess.capacity_mwh * 1.08]),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(gridcolor="#F3F4F6")
    return fig


def daily_compare_figure(daily: pd.DataFrame) -> go.Figure:
    x = daily["planning_day"]  # 날짜형 그대로 (문자열로 바꾸면 plotly 가 다른 해로 해석할 수 있다)
    fig = go.Figure()
    fig.add_bar(x=x, y=daily["objective_krw_forecast"] / MAN, name="예측 기반 계획 수익",
                marker_color=COLORS["ess"])
    fig.add_bar(x=x, y=daily["objective_krw_actual"] / MAN, name="완전예측 계획 수익",
                marker_color=COLORS["accent"])
    fig.add_scatter(x=x, y=daily["charge_mwh_forecast"], name="충전 MWh (예측)", yaxis="y2",
                    mode="lines+markers", line=dict(color=COLORS["ess_light"], dash="dot"))
    fig.add_scatter(x=x, y=daily["charge_mwh_actual"], name="충전 MWh (실측)", yaxis="y2",
                    mode="lines+markers", line=dict(color="#99F6E4", dash="dot"))
    fig.update_layout(barmode="group", height=340, margin=dict(t=40, b=30, l=10, r=10),
                      plot_bgcolor="#fff",
                      title=dict(text="계획일별 계획 수익과 충전량", font=dict(size=15)),
                      xaxis=dict(type="date", tickformat="%Y-%m-%d", dtick=86_400_000),
                      yaxis=dict(title="만 원"),
                      yaxis2=dict(title="MWh", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(gridcolor="#F3F4F6")
    return fig
