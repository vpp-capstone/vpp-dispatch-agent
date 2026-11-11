"""User-facing overview: one saved experiment, one comparison period, explicit gaps."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.operator_simulation import connection_inventory, render_simulator
from dashboard.overview_data import experiments, load_experiment

from settings import BACKTEST_OUT_DIR, Settings, rec_diff
from ui import ROOT, kpi

TEAL = "#0F766E"
INK = "#334155"
ORANGE = "#C56B25"


def _money(value: float) -> str:
    return f"{value / 10000:+,.1f}만 원"


def _line(frame_f, frame_a, column, title, unit, scale=1.0):
    fig = go.Figure()
    fig.add_scatter(
        x=frame_a["ts"], y=frame_a[column] * scale, name="실측", line=dict(color=INK, width=2)
    )
    fig.add_scatter(
        x=frame_f["ts"],
        y=frame_f[column] * scale,
        name="당시 예측",
        line=dict(color=TEAL, width=2, dash="dot"),
    )
    fig.update_layout(
        title=title,
        height=270,
        margin=dict(l=10, r=10, t=55, b=15),
        yaxis_title=unit,
        hovermode="x unified",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.2),
        xaxis_tickformat="%H시",
    )
    fig.update_yaxes(gridcolor="#EEF2F6")
    return fig


@st.cache_data(show_spinner=False)
def _load(folder: str, stamp: tuple) -> dict:
    return load_experiment(Path(folder))


def _model_context(summary: dict) -> None:
    inventory = connection_inventory(ROOT)
    st.markdown("### 지금 어떤 예측이 연결돼 있나요?")
    st.info(
        "현재 발전량 입력은 기존 LightGBM 예측 열을 읽습니다. 개선 모델은 코드가 들어와 있지만 "
        "현재 입력 경로에 연결되지 않았습니다. SMP는 저장 파일에 모델 버전·기상 입력 사용 여부가 "
        "없어 개선 모델 반영 여부를 확인할 수 없습니다."
    )
    for col, name, description, status in zip(
        st.columns(3),
        ("☀ 태양광 18곳", "풍력 3곳", "₩ SMP 가격"),
        (
            "일사·구름·기온 예보 → LightGBM → 시간별 발전량",
            "풍속·풍향 등 기상예보 → LightGBM → 시간별 발전량",
            "과거 가격·수요·달력 정보 → LightGBM → 시간별 가격",
        ),
        (
            "개선안: 발전소별 모델 + 전체 발전량 모델 결합. "
            + (
                "학습 파일 있음 · 현재 연결은 별도 확인"
                if inventory["solar_bundle"]
                else "개선 모델의 학습 완료 파일이 현재 로컬에 없음"
            ),
            "개선안: 관측 기상과 예보의 차이를 반영한 특징 추가. "
            + (
                "학습 파일 있음 · 현재 연결은 별도 확인"
                if inventory["wind_bundle"]
                else "개선 모델의 학습 완료 파일이 현재 로컬에 없음"
            ),
            "개선안: 기상예보 특징 추가 가능. 이 실험 파일이 기상을 사용했는지는 기록 없음",
        ),
        strict=True,
    ):
        with col.container(border=True):
            st.markdown(f"**{name}**")
            st.write(description)
            st.caption(status)
    with st.expander("어떤 데이터와 모델로 예측하나요?", expanded=False):
        st.markdown("**프로젝트의 개발 기준**과 **이 실험에 실제 사용된 예측 파일**을 구분합니다.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "항목": "발전량",
                        "예측 방법": "태양광: 일사·구름·기온 / 풍력: 풍속·풍향 등 기상예보",
                        "개발 데이터": "2023년 학습 → 2024년 모델 비교·개선 → 2025년 최종 평가용",
                        "계획에 전달": "태양광 18곳 + 풍력 3곳, 시간별 48시간 발전량",
                    },
                    {
                        "항목": "SMP",
                        "예측 방법": "최근 가격·수요·요일·기상예보로 가격 예측 (LightGBM)",
                        "개발 데이터": "현재 문서 기준 2021년부터 당시 공개 자료로 매월 재학습",
                        "계획에 전달": "예측 발행 시각 기준 내일·모레 시간별 가격 48개",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "발전량 최신 개선: 태양광 결합 모델·풍력 개선 LightGBM. 이 저장 실험이 최신 모델로 "
            "생성됐는지는 모델 식별 기록이 있어야 확인할 수 있습니다. 2024년은 모델 선택에도 "
            "사용되어 독립적인 최종 평가가 아닙니다. 위 내용은 프로젝트 문서 기준입니다."
        )
        rows = []
        for key, label in (
            ("solar_predictions", "태양광"),
            ("wind_predictions", "풍력"),
            ("smp_forecast", "SMP"),
        ):
            source = summary.get("inputs", {}).get(key)
            recorded = source.get("path", "") if isinstance(source, dict) else source
            p = Path(recorded) if recorded else None
            if p is not None and not p.is_absolute():
                p = ROOT / p
            rows.append(
                {
                    "입력": label,
                    "실험에 기록된 파일": recorded or "기록 없음",
                    "현재 파일 수정일": datetime.fromtimestamp(p.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    if p is not None and p.exists()
                    else "파일 없음",
                    "모델 버전 일치": "확인 필요",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "수정일은 현재 로컬 파일의 정보입니다. 저장 실험을 만든 모델 버전이나 학습 완료일을 "
            "증명하지 않습니다. 아래 지표는 현재 파일을 섞지 않고 저장된 실험 CSV로 계산합니다."
        )


def render(settings: Settings, bundle: dict[str, Any] | None) -> None:
    st.markdown(
        '<div class="jf-h1">제주 VPP 운영 관제실</div>',
        unsafe_allow_html=True,
    )
    st.caption("하루를 재생하면서 발전량, 판매량, 배터리 잔량과 계획 수익을 함께 확인하세요.")
    runs = experiments(BACKTEST_OUT_DIR)
    if not runs:
        st.info(
            "아직 비교할 저장 실험이 없습니다. ‘ESS 백테스트 비교’ 탭에서 기간과 REC·열화비를 "
            "선택해 실행하면 이 화면에 연결됩니다. 없는 결과는 0원으로 표시하지 않습니다."
        )
        return
    options = {str(path): summary for path, summary in runs}

    def label(path):
        s = options[path]
        rec = "REC 적용" if (s.get("rec") or {}).get("enabled") else "REC 미적용"
        return f"{s.get('start')} ~ {s.get('end')} · {s.get('days')}일 · {rec} · {Path(path).name}"

    choice = st.selectbox("확인할 저장 실험", list(options), format_func=label, key="overview_run")
    folder = Path(choice)
    st.caption(
        "이 화면은 선택한 저장 실험 전체를 보여줍니다. 사이드바의 기간·ESS·REC 변경은 "
        "새로 실행할 때 적용됩니다. 기존 결과를 자동으로 다시 계산하지 않습니다."
    )
    try:
        stamp = tuple(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(folder.iterdir())
            if p.suffix in (".json", ".csv")
        )
        data = _load(choice, stamp)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"이 실험을 비교할 수 없습니다: {exc}")
        return
    s, f, a, m = data["summary"], data["forecast"], data["actual"], data["metrics"]
    rev = data["revenues"]
    rec_enabled = bool((s.get("rec") or {}).get("enabled"))
    mismatch = rec_diff(s, settings)
    ess = data["ess"]
    current_ess = settings.ess()
    if any(
        getattr(ess, name) != getattr(current_ess, name)
        for name in (
            "capacity_mwh",
            "charge_power_mw",
            "discharge_power_mw",
            "soc_initial",
            "soc_min",
            "soc_max",
            "charge_efficiency",
            "discharge_efficiency",
            "degradation_model",
            "throughput_basis",
            "throughput_cost_krw_per_mwh",
        )
    ):
        mismatch.append("ESS 또는 열화비 설정이 다름")
    st.markdown(
        f"**분석 대상: {a.ts.min():%Y-%m-%d} ~ {a.ts.max():%Y-%m-%d} · "
        f"{m['days']}일 / {m['hours']:,}시간 · 연속 구간 {m['segments']}개**"
    )
    st.caption(
        f"저장 조건: ESS {ess.capacity_mwh:g} MWh / {ess.charge_power_mw:g} MW · "
        f"열화비 {ess.throughput_cost_krw_per_mwh:,.0f}원/MWh ({ess.throughput_basis}) · "
        f"{'REC 적용 — 실험 가정' if rec_enabled else 'REC 미적용'}"
    )
    if mismatch:
        st.info("현재 사이드바 설정과 저장 결과가 다릅니다: " + ", ".join(mismatch))
    days = sorted(f.ts.dt.date.unique())
    day = st.selectbox("살펴볼 하루", days, key=f"overview_day_{folder.name}")
    fd, ad = f[f.ts.dt.date == day], a[a.ts.dt.date == day]
    render_simulator(data, day)
    _model_context(s)

    st.markdown("### 1. 같은 시간의 예측과 실제 값")
    cards = [
        kpi(
            "발전량 평균 오차 · 21곳 합계",
            f"{m['generation_mae_mwh']:.2f} MWh",
            "한 시간 발전량이 평균적으로 빗나간 양",
            "flat",
            "☀",
            "#FFFBEB",
        ),
        kpi(
            "SMP 평균 오차",
            f"{m['smp_mae_krw_per_kwh']:.2f} 원/kWh",
            "예측 가격과 실제 가격의 평균 절대 차이",
            "flat",
            "₩",
            "#EFF6FF",
        ),
        kpi(
            "ESS 추가수익 · 실측을 미리 안 경우",
            _money(rev["actual"]["ess_gain_krw"]),
            "같은 실측 조건의 ESS 없음과 비교",
            "flat",
            "↗",
            "#ECFDF5",
        ),
        kpi(
            "예측 계획의 실제 운전 수익",
            "아직 미계산",
            "계획 고정 → 실제 운전·정산 단계 필요",
            "flat",
            "…",
            "#FFF7ED",
        ),
    ]
    for col, card in zip(st.columns(4), cards, strict=True):
        col.markdown(card, unsafe_allow_html=True)
    st.caption(
        "저장된 확정 24시간만 한 번씩 비교합니다. 모레의 잠정 계획을 중복 집계하지 않습니다. "
        "오차가 작을수록 좋으며 ‘100 − 오차’를 정확도로 읽으면 안 됩니다."
    )
    st.write(
        f"**0원 이하 가격:** 실제 {m['actual_low_hours']}시간 중 예측이 잡은 시간은 "
        f"{m['forecast_low_hits']}시간, 실제 양수인데 0원 이하로 예측한 시간은 "
        f"{m['forecast_false_alarms']}시간입니다."
    )

    left, right = st.columns(2)
    left.plotly_chart(
        _line(fd, ad, "generation_mwh", "발전량 · 예측과 실측", "MWh"), width="stretch"
    )
    right.plotly_chart(
        _line(fd, ad, "price_krw_per_mwh", "SMP · 예측과 실측", "원/kWh", 0.001), width="stretch"
    )

    st.markdown("### 2. 그 예측으로 어떤 운영계획을 세웠나?")
    st.caption(
        "‘실측을 미리 안 계획’도 비교를 위해 다시 최적화한 계획입니다. "
        "실제 ESS 운전 기록이 아닙니다."
    )
    chart = go.Figure()
    for plan, name, color, dash in (
        (fd, "예측 계획", TEAL, "solid"),
        (ad, "실측을 미리 안 계획", ORANGE, "dot"),
    ):
        chart.add_scatter(
            x=plan.ts,
            y=plan.discharge_mwh - plan.charge_mwh,
            name=name,
            mode="lines+markers",
            line=dict(color=color, dash=dash),
        )
    chart.add_hline(y=0, line_color="#CBD5E1")
    chart.update_layout(
        title="충방전 시간 · 위쪽은 방전, 아래쪽은 충전",
        height=270,
        yaxis_title="MWh",
        hovermode="x unified",
        xaxis_tickformat="%H시",
        margin=dict(t=55, b=15, l=10, r=10),
        legend=dict(orientation="h", y=1.2),
    )
    left, right = st.columns(2)
    left.plotly_chart(chart, width="stretch")
    soc = go.Figure()
    for plan, name, color in ((fd, "예측 계획", TEAL), (ad, "실측을 미리 안 계획", ORANGE)):
        soc.add_scatter(x=plan.ts, y=plan.soc_end * 100, name=name, line=dict(color=color))
    soc.update_layout(
        title="시간별 종료 배터리 잔량",
        yaxis_title="%",
        height=270,
        hovermode="x unified",
        xaxis_tickformat="%H시",
        yaxis_range=[0, 100],
        margin=dict(t=55, b=15, l=10, r=10),
        legend=dict(orientation="h", y=1.2),
    )
    right.plotly_chart(soc, width="stretch")
    ops = []
    for col, title in (
        ("generation_mwh", "발전량"),
        ("sell_mwh", "판매 계획량"),
        ("charge_mwh", "충전량"),
        ("discharge_mwh", "방전량"),
        ("curtailment_mwh", "판매하지 않고 줄인 양"),
    ):
        ops.append(
            {
                "선택한 하루 · MWh": title,
                "예측 계획": fd[col].sum(),
                "실측을 미리 안 계획": ad[col].sum(),
                "차이 (예측−실측계획)": fd[col].sum() - ad[col].sum(),
            }
        )
    st.dataframe(pd.DataFrame(ops).round(3), hide_index=True, width="stretch")

    st.markdown("### 3. 그래서 얼마를 벌었나? — 실험 전체")
    st.warning(
        "아래 두 수익은 서로 다른 발전량·가격으로 계산한 계획 수익입니다. 두 금액의 차이를 "
        "예측 때문에 잃은 돈이나 실제 벌어들인 돈으로 해석할 수 없습니다."
    )
    rows = []
    for key, title in (
        ("sales_revenue_krw", "전기 판매 수익 (SMP)"),
        ("rec_revenue_krw", "REC 수익"),
        ("throughput_cost_krw", "열화비 (차감)"),
        ("net_revenue_krw", "계획 순수익"),
        ("no_ess_net_krw", "ESS 없이 판매한 수익"),
        ("ess_gain_krw", "ESS 없음 대비 추가수익"),
    ):
        rows.append(
            {
                "항목 · 만 원": title,
                "예측이 그대로 맞았다고 가정": rev["forecast"][key] / 10000,
                "실측을 미리 알고 계획": rev["actual"][key] / 10000,
            }
        )
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    st.info(
        "실제 운전·정산 수익: 아직 미계산. 확정 판매계획을 유지하면서 발전량 부족·초과에 대응하고, "
        "그때의 ESS 잔량·실제 정산을 계산한 결과가 필요합니다. 현재는 이 칸을 비워 둡니다."
    )
    st.caption(
        "REC는 ESS 없이 판매해도 받습니다. 전체 REC 수익이 곧 ESS 추가수익은 아닙니다. "
        "순수익은 표시된 SMP + REC − 열화비이며 설비 투자비를 차감한 사업 이익은 아닙니다."
    )

    st.markdown("### 4. 지금 비어 있는 부분 / 다음에 확인할 것")
    terminal = s.get("scenarios", {}).get("actual", {}).get("terminal_constraint_cost_krw")
    excluded = s.get("excluded_days")
    gaps = [
        {
            "확인 항목": "예측 계획을 실제 상황에 적용한 수익",
            "상태": "미계산",
            "필요한 작업": "판매계획 고정 → 충방전 대응 → 실제 정산 → 다음 날 잔량 이월",
        },
        {
            "확인 항목": "최신 예측모델과 이번 수익의 연결",
            "상태": "확인 필요",
            "필요한 작업": "모델 버전·생성 시각이 기록된 새 예측 결과로 같은 기간 재평가",
        },
        {
            "확인 항목": "데이터가 이어지는가",
            "상태": f"{m['segments']}개 연속 구간",
            "필요한 작업": f"제외 {excluded if excluded is not None else '미기록'}일. "
            "제외 사유 확인 후 연속 구간별로 해석",
        },
        {
            "확인 항목": "종료 잔량 조건의 영향",
            "상태": "정책 검토 필요",
            "필요한 작업": (
                f"조건 해제 시 계산상 {_money(terminal)}. 실제 충전 비용이 아님"
                if terminal is not None
                else "이 저장 결과에는 조건 해제 비교 금액이 없음"
            ),
        },
        {
            "확인 항목": "REC 조건",
            "상태": "실험 가정" if rec_enabled else "이 실험은 미적용",
            "필요한 작업": "실제 단가·인정 기준 확인. 초기 저장 전력의 REC 제외는 미구현",
        },
    ]
    st.dataframe(pd.DataFrame(gaps), hide_index=True, width="stretch")
    with st.expander("제외된 날짜와 저장 조건 확인"):
        if not data["excluded"].empty:
            st.dataframe(data["excluded"], hide_index=True, width="stretch")
        else:
            st.caption("제외 날짜 목록이 없거나 비어 있습니다.")
        st.write("저장된 종료 정책:", s.get("terminal", "기록 없음"))
        st.json(
            {
                "REC": s.get("rec", {"enabled": False}),
                "ESS": s.get("ess"),
                "초기 저장분 REC 참고 상한": s.get("initial_energy_rec_upper_bound_krw"),
                "결과 폴더": str(folder),
            }
        )
    export = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "이 실험의 수익 비교표 내려받기",
        export,
        "plan_revenue_comparison.csv",
        mime="text/csv",
        key="overview_download",
    )
