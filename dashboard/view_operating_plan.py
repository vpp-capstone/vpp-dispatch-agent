"""탭 1 — 운영계획 생성. 계획일·설정 → '이 설정으로 운영계획 생성' → 48시간 계획과 확정 24시간.

계산은 vpp.operating_plan.run_operating_plan (CLI 와 같은 함수) 만 호출하고, 화면은 저장된 결과
폴더를 읽어 그린다. 새로고침으로는 다시 계산하지 않는다.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from settings import OPERATING_PLAN_DIR, Settings, settings_chips
from ui import COLORS, MAN, checks_table, kpi, run_stamp, timeline_figure, unit
from vpp.operating_plan import MODE as PLAN_MODE
from vpp.operating_plan import run_operating_plan
from vpp.optimization import EssParams

START_LABEL = {"config_soc_initial": "시작: ESS 설정값", "user_input": "시작: 직접 입력",
               "previous_plan": "시작: 이전 계획 이월(계획상)"}


def list_plan_runs(base: Path = OPERATING_PLAN_DIR) -> list[Path]:
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if (p / "summary.json").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _summary(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "summary.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_plan_run(folder_str: str, stamp: float = 0.0) -> dict[str, Any]:
    folder = Path(folder_str)
    plan = pd.read_csv(folder / "plan_48h.csv")
    plan["ts"] = pd.to_datetime(plan["ts"], utc=True).dt.tz_convert("Asia/Seoul")
    summary = _summary(folder)
    ess_fields = {k: v for k, v in summary["ess"].items()
                  if k in EssParams.__dataclass_fields__}
    return {
        "folder": folder, "plan": plan, "summary": summary,
        "validation": json.loads((folder / "validation.json").read_text(encoding="utf-8")),
        "ess": EssParams(**ess_fields),
    }


def previous_day_runs(day: date) -> list[Path]:
    """계획일이 D-1 인 운영계획 결과만 (이월 후보). 날짜가 끊긴 계획은 후보에서 뺀다."""
    out = []
    for p in list_plan_runs():
        try:
            s = _summary(p)
        except Exception:
            continue
        prev_day = (day - timedelta(days=1)).isoformat()
        if s.get("mode") == PLAN_MODE and s.get("planning_day") == prev_day:
            out.append(p)
    return out


def applied_vs_current(summary: dict[str, Any], settings: Settings) -> list[str]:
    """표시 중인 결과에 적용된 설정과 현재 사이드바 설정의 차이 목록 (비어 있으면 같음)."""
    diffs = []
    if summary["planning_day"] != str(settings.planning_day):
        diffs.append(f"계획일 {summary['planning_day']} → {settings.planning_day}")
    cur = settings.ess().to_dict()
    for key in ("capacity_mwh", "charge_power_mw", "discharge_power_mw", "soc_min", "soc_max",
                "soc_initial", "charge_efficiency", "discharge_efficiency",
                "degradation_model", "throughput_basis", "throughput_cost_krw_per_mwh"):
        if summary["ess"].get(key) != cur[key]:
            diffs.append(f"{key} {summary['ess'].get(key)} → {cur[key]}")
    for name, path in settings.files.items():
        used = summary["inputs"].get({"solar": "solar_predictions", "wind": "wind_predictions",
                                      "smp": "smp_forecast", "plants": "plants"}[name], {})
        if Path(used.get("path", "")).resolve() != path.resolve():
            diffs.append(f"{name} 파일 {used.get('path')} → {path}")
        elif path.exists() and used.get("size_bytes") != path.stat().st_size:
            diffs.append(f"{name} 파일 내용 변경됨 (크기 {used.get('size_bytes')} → "
                         f"{path.stat().st_size})")
    return diffs


def plan48_figure(plan: pd.DataFrame, ess: EssParams) -> go.Figure:
    fig = timeline_figure(plan, ess, " · 48시간 (앞 24시간 확정, 뒤 24시간 잠정)")
    split = plan["ts"].iloc[24] if len(plan) > 24 else None
    if split is not None:
        fig.add_vrect(x0=split, x1=plan["ts"].iloc[-1] + pd.Timedelta(hours=1),
                      fillcolor="#9CA3AF", opacity=0.12, line_width=0,
                      annotation_text="D+1 잠정", annotation_position="top left")
    fig.update_xaxes(tickformat="%m-%d %H시", title="시간 (KST)")
    return fig


def soc48_figure(plan: pd.DataFrame, ess: EssParams, start_energy: float) -> go.Figure:
    """시작 잔량은 첫 시각에, 각 시간의 종료 잔량은 그 시간 끝(ts+1h)에 놓는다."""
    ts_end = plan["ts"] + pd.to_timedelta(plan["dt_hours"], unit="h")
    xs = [plan["ts"].iloc[0], *ts_end.tolist()]
    ys = [start_energy, *plan["energy_end_mwh"].tolist()]
    fig = go.Figure()
    fig.add_scatter(x=xs[:25], y=ys[:25], name="확정 24시간", mode="lines+markers",
                    line=dict(color=COLORS["ess"], width=2))
    fig.add_scatter(x=xs[24:], y=ys[24:], name="잠정 D+1", mode="lines+markers",
                    line=dict(color=COLORS["ess_light"], width=2, dash="dot"))
    fig.add_scatter(x=[xs[0]], y=[ys[0]], name="시작 잔량 (입력값)", mode="markers",
                    marker=dict(color=COLORS["ess"], size=12, symbol="diamond"))
    for level, label in ((ess.energy_max_mwh, "상한"), (ess.energy_min_mwh, "하한")):
        fig.add_hline(y=level, line_dash="dot", line_color="gray",
                      annotation_text=f"{label} {level:.1f} MWh")
    fig.update_layout(height=300, margin=dict(t=40, b=30, l=10, r=10), plot_bgcolor="#fff",
                      title=dict(text="ESS 잔량 (계획상 SOC) · ◆ 시작, ● 각 시간 종료",
                                 font=dict(size=15)),
                      xaxis=dict(tickformat="%m-%d %H시"),
                      yaxis=dict(title="MWh", range=[0, ess.capacity_mwh * 1.08]),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(gridcolor="#F3F4F6")
    return fig


def render(settings: Settings, bundle: dict[str, Any] | None) -> None:
    st.markdown('<div class="jf-h1">계획일 D 의 48시간을 예측만으로 풀고 D 24시간을 확정합니다'
                '</div>', unsafe_allow_html=True)
    st.markdown('<div class="jf-lead">입력은 발전량 예측(D-2 18 UTC)·SMP 예측(D-1 11시)·ESS 및 '
                '열화비 설정·시작 SOC 뿐입니다. 실측은 쓰지 않습니다. 수익은 예측이 맞았다고 '
                '가정한 '
                '<b>계획 수익</b>이며 실제 정산이 아닙니다. D+1 24시간은 잠정 계획입니다.</div>',
                unsafe_allow_html=True)

    day = settings.planning_day
    ess = settings.ess()
    with st.container(border=True):
        st.markdown(settings_chips(settings, show_day=True), unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        mode = c1.radio("시작 SOC", ["ESS 설정값", "직접 입력", "이전 운영계획에서 이월"],
                        key="op_soc_mode", horizontal=False)
        start_soc = from_plan = None
        can_run = bundle is not None and day is not None
        if mode == "ESS 설정값":
            c2.metric("시작 SOC (입력값)",
                      f"{ess.soc_initial:.0%} · {ess.initial_energy_mwh:.2f} MWh",
                      help=f"{settings.ess_path.name} 의 soc_initial")
        elif mode == "직접 입력":
            start_soc = c2.slider("시작 SOC (%)", int(ess.soc_min * 100), int(ess.soc_max * 100),
                                  int(round(ess.soc_initial * 100)), 1, key="op_soc_value") / 100
            c2.caption(f"= {start_soc * ess.capacity_mwh:.2f} MWh")
        else:
            prev = previous_day_runs(day) if day else []
            if not prev:
                c2.warning(f"계획일 {day - timedelta(days=1) if day else '-'} 의 운영계획 결과가 "
                           "없어 이월할 수 없습니다. 날짜가 끊긴 계획은 이월하지 않습니다.")
                can_run = False
            else:
                from_plan = c2.selectbox("이월할 이전 계획 (D-1)", prev, key="op_from_plan",
                                         format_func=lambda p: p.name)
                prev_final = _summary(from_plan)["committed"]["final_energy_mwh"]
                c2.caption(f"이전 계획 확정 종료 잔량 {prev_final:.3f} MWh → 오늘 시작값으로 "
                           "그대로 전달 (계획상 SOC, 실측 아님)")
        clicked = c3.button("▶ 이 설정으로 운영계획 생성", type="primary", key="op_run",
                            disabled=not can_run, width="stretch")
        c3.caption("누를 때만 최적화를 다시 계산합니다. 결과는 폴더로 저장됩니다.")

    if clicked and can_run:
        run_id = f"plan_{day}_{datetime.now():%Y%m%d%H%M%S}"
        with st.spinner("48시간 MILP 풀이 중"):
            try:
                out = run_operating_plan(
                    day, ess_path=settings.ess_path, degradation_path=settings.degradation_path,
                    start_soc=start_soc, from_plan=from_plan,
                    solar_path=settings.solar, wind_path=settings.wind, smp_path=settings.smp,
                    plants_path=settings.plants, out_dir=OPERATING_PLAN_DIR, run_id=run_id,
                )
                st.session_state["op_plan_folder"] = str(out)
                st.session_state["op_plan_view"] = out  # 방금 만든 결과를 바로 표시
                st.success(f"저장: {out}")
            except (ValueError, RuntimeError, FileExistsError) as exc:
                st.error(f"운영계획을 만들지 못했습니다: {exc}")

    runs = list_plan_runs()
    if not runs:
        st.info("아직 생성한 운영계획이 없습니다. 위 버튼을 누르거나 터미널에서 "
                "`python -m vpp.operating_plan --day YYYY-MM-DD` 를 실행하세요.")
        return
    current = st.session_state.get("op_plan_folder")
    index = next((i for i, p in enumerate(runs) if str(p) == current), 0)
    choice = st.selectbox("결과 보기", runs, index=index, format_func=lambda p: p.name,
                          key="op_plan_view")
    run = load_plan_run(str(choice), run_stamp(choice))
    s, v, plan, run_ess = run["summary"], run["validation"], run["plan"], run["ess"]

    diffs = applied_vs_current(s, settings)
    if diffs:
        st.warning("현재 사이드바 설정이 **표시 중인 결과에 적용된 설정과 다릅니다** "
                   "(아직 실행 안 함): " + " · ".join(diffs))
    else:
        st.success("표시 중인 결과는 현재 설정으로 계산된 것입니다.")

    c = s["committed"]
    se = s["start_energy"]
    man = unit("만 원")
    cards = [
        kpi("확정 24시간 계획 수익", f"{c['objective_krw'] / MAN:,.1f}{man}",
            f"판매 {c['sales_revenue_krw'] / MAN:,.1f} − 열화비 "
            f"{c['throughput_cost_krw'] / MAN:,.1f}", "flat", "💰"),
        kpi("ESS 방전 / 충전 (확정)",
            f"{c['discharge_mwh']:.2f} / {c['charge_mwh']:.2f}{unit(' MWh')}",
            f"감축 {c['curtailment_mwh']:.2f} MWh", "flat", "🔋", "#F5F3FF"),
        kpi("시작 → 종료 잔량",
            f"{se['energy_mwh']:.2f} → {c['final_energy_mwh']:.2f}{unit(' MWh')}",
            START_LABEL[se["source"]], "flat", "⚡", "#EFF6FF"),
        kpi("D+1 잠정 계획 수익", f"{s['provisional_next_day']['objective_krw'] / MAN:,.1f}{man}",
            "합계에 넣지 않음 · 내일 다시 품", "flat", "🕒", "#FFF7ED"),
    ]
    for col, card in zip(st.columns(4), cards, strict=True):
        col.markdown(card, unsafe_allow_html=True)
    if se["source"] == "previous_plan":
        gap = se.get("carry_gap_mwh", 0.0)
        st.info(f"이월 확인 · 이전 계획 종료 {se['previous_final_energy_mwh']:.3f} MWh → 오늘 시작 "
                f"{se['energy_mwh']:.3f} MWh (차이 {gap:+.3f}). {se['note']}")
    ok = v.get("all_passed", False)
    (st.success if ok else st.error)(
        f"제약 검산 {'통과' if ok else '실패'} · 48시간 계산 {s['window']['solved_hours']}h · "
        f"확정 {s['window']['commit_hours']}h · 종료 정책 {s['terminal']} · "
        f"열화 {s['ess']['degradation']['name']} "
        f"{s['ess']['throughput_cost_krw_per_mwh']:,.0f}원/MWh"
    )

    g1, g2 = st.columns([1.5, 1])
    g1.plotly_chart(plan48_figure(plan, run_ess), width="stretch")
    g2.plotly_chart(soc48_figure(plan, run_ess, se["energy_mwh"]), width="stretch")

    t1, t2 = st.columns([1.6, 1])
    with t1:
        st.markdown("#### 48시간 계획")
        table = pd.DataFrame({
            "시각": plan["ts"].dt.strftime("%m-%d %H시"), "구분": plan["status"].map(
                {"committed": "확정", "provisional": "잠정"}),
            "발전량 예측 (MWh)": plan["generation_mwh"],
            "SMP 예측 (원/kWh)": plan["price_krw_per_mwh"] / 1000,
            "판매 (MWh)": plan["sell_mwh"], "충전 (MWh)": plan["charge_mwh"],
            "방전 (MWh)": plan["discharge_mwh"], "감축 (MWh)": plan["curtailment_mwh"],
            "종료 잔량 (MWh)": plan["energy_end_mwh"], "SOC": plan["soc_end"],
        })
        st.dataframe(table.round(3), width="stretch", hide_index=True, height=420)
    with t2:
        st.markdown("#### 제약 검산")
        st.dataframe(checks_table(v.get("structure_checks", []) + v.get("plan_checks_48h", [])),
                     width="stretch", hide_index=True, height=420)
    with st.expander("실행 기록 · 입력 파일 식별 · 코드 버전"):
        st.json({k: s[k] for k in ("run_id", "created_at", "code_version", "start_energy",
                                   "forecast_sources", "inputs", "ess", "terminal")})
        st.markdown("#### 한계")
        for line in s.get("limitations", []):
            st.markdown(f"- {line}")
