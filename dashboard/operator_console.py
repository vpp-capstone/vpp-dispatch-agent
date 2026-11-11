"""One verified forecast/result release powers every operator screen."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.operator_simulation import render_simulator, replay_payload
from dashboard.overview_data import load_experiment
from dashboard.signal_replay import render_signal

from ui import ROOT
from vpp import operator_pipeline as pipeline

CSS = "<style>" + (Path(__file__).parent / "assets/operator_console.css").read_text() + "</style>"


def kpis(items):
    for col, (label, value, hint) in zip(st.columns(len(items)), items, strict=True):
        col.markdown(
            f'<div class="oc-kpi"><small>{escape(label)}</small>'
            f"<strong>{escape(str(value))}</strong><p>{escape(hint)}</p></div>",
            unsafe_allow_html=True,
        )


def money(value):
    return f"{value / 10000:,.1f}만 원"


def state_key(state):
    return json.dumps(
        {k: state.get(k) for k in ("state", "revision", "active", "stages")},
        sort_keys=True,
        default=str,
    )


@st.fragment(run_every="30s")
def monitor():
    cfg = pipeline.config(ROOT)
    state = pipeline.status(ROOT)
    if st.session_state.get("operator_auto", cfg.get("auto_refresh", True)):
        state = pipeline.launch(ROOT)
    current = state_key(state)
    if current != st.session_state.get("operator_state_key"):
        st.session_state["operator_state_key"] = current
        st.rerun()
    running = state.get("state") in ("running", "queued")
    st.caption(
        (
            "● 자동 확인 중 · 30초마다 코드·입력 변경 확인 · "
            + ("계산 실행 중" if running else "변경된 경우에만 다시 계산")
        )
        if st.session_state.get("operator_auto", True)
        else "자동 갱신 일시 중지"
    )


@st.cache_data(show_spinner=False)
def read_report(folder: str, stamp: int):
    return load_experiment(Path(folder))


def flow(state):
    stages = state.get("stages", [])
    files_ok = all(r["ready"] for r in pipeline.requirements(pipeline.config(ROOT)))
    running = state.get("state") in ("running", "queued")
    blocks = [
        ("01 · 공유 파일 확인", "Drive 에서 받은 예측 파일·설정" if files_ok else "없는 파일 있음",
         files_ok),
        ("02 · MILP 운영계획", "파일 내용이 바뀌면 재계산", bool(state.get("active"))),
        ("03 · 화면 갱신", "모든 화면에 같은 결과 묶음",
         bool(state.get("active")) and bool(stages)),
    ]
    st.markdown(
        '<div class="oc-flow">'
        + "".join(
            f'<div class="oc-step"><b><i class="oc-dot {"run" if running else ""} '
            f'{"" if ok else "warn"}"></i>{title}</b><small>{caption}</small></div>'
            for title, caption, ok in blocks
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def model_cards(release):
    upgraded = "improved_generation_kwh" in release.get("forecast_columns", [])
    for col, title, recipe, status in zip(
        st.columns(3),
        ("태양광", "풍력", "SMP"),
        (
            "기상예보 → 발전소별 LightGBM + 전체 발전량 모델 → 시간별 발전량",
            "풍속·풍향 + 기상 오차 보정 → LightGBM (PR #14) → 시간별 발전량",
            "최근 가격·수요·요일 + 기상예보 → LightGBM → 내일·모레 가격",
        ),
        (
            "공유 파일 · PR #14 결과 열 사용" if upgraded
            else "공유 파일 · PR #14 이전 LightGBM 열 (새 parquet 받으면 자동 전환)",
            "공유 파일 · PR #14 결과 열 사용" if upgraded
            else "공유 파일 · PR #14 이전 LightGBM 열 (새 parquet 받으면 자동 전환)",
            "공유 파일 · 모델 버전은 trained_until 열로 확인",
        ),
        strict=True,
    ):
        col.markdown(
            f'<div class="oc-model"><small>MODEL → FORECAST</small><h4>{title}</h4>'
            f"<p>{recipe}</p><p><b>{status}</b></p></div>",
            unsafe_allow_html=True,
        )


def show_forecasts(data, day, release):
    st.subheader("예측은 실제와 얼마나 달랐나요?")
    model_cards(release)
    if data["actual"] is None:
        st.info("실측이 없어 비교할 수 없습니다. 발전량 parquet 의 generation_kwh 열과 SMP 파일의 "
                "smp_actual 열이 채워지면 이 화면이 켜집니다.")
        return
    st.caption("재생을 누르면 예측선과 실측선을 같은 시간으로 따라갑니다.")
    choice = st.radio(
        "비교할 예측",
        ["전체 발전량", "태양광", "풍력", "SMP 가격"],
        horizontal=True,
        key="oc_signal",
    )
    f, a = data["forecast"], data["actual"]
    f, a = f[f.ts.dt.date == day], a[a.ts.dt.date == day]
    if choice in ("태양광", "풍력"):
        from vpp import forecast_inputs as fi
        from vpp import forecast_metrics as fm

        sources = release["inputs"]
        pred = fm.load_generation_eval_frame(
            Path(sources["solar"]), Path(sources["wind"]), Path(sources["plants"])
        )
        series = fm.generation_timeseries(
            pred,
            fi.expected_assets(Path(sources["plants"])),
            level="energy_type",
            group="solar" if choice == "태양광" else "wind",
            forecast_day_offset=0,
            start=day,
            end=day,
        )
        rows = [
            {"hour": r.ts.strftime("%H:%M"), "f": r.lightgbm_kwh / 1000, "a": r.actual_kwh / 1000}
            for r in series.itertuples()
            if pd.notna(r.lightgbm_kwh) and pd.notna(r.actual_kwh)
        ]
        unit = "MWh"
    else:
        column, divisor, unit = (
            ("price_krw_per_mwh", 1000, "원/kWh")
            if choice == "SMP 가격"
            else ("generation_mwh", 1, "MWh")
        )
        rows = [
            {"hour": t.strftime("%H:%M"), "f": float(p / divisor), "a": float(q / divisor)}
            for t, p, q in zip(f.ts, f[column], a[column], strict=True)
        ]
    render_signal(rows, title=f"{choice} · 예측과 실측", day=day, unit=unit)
    m = data["metrics"]
    kpis(
        [
            (
                "기간 전체 · 발전량 평균 오차",
                f"{m['generation_mae_mwh']:.2f} MWh",
                "21곳 합산 · 시간당 절대 오차",
            ),
            (
                "기간 전체 · SMP 평균 오차",
                f"{m['smp_mae_krw_per_kwh']:.2f}원/kWh",
                "낮을수록 실제와 가까운 예측",
            ),
            (
                "0원 이하 가격 적중",
                f"{m['forecast_low_hits']} / {m['actual_low_hours']}시간",
                "동일 비교 기간의 0원 이하 실제 가격",
            ),
        ]
    )
    st.caption("2024년은 모델 선택에도 사용된 기간입니다. 독립적인 최종 시험 성능과 구분하세요.")


def show_profit(data, day):
    st.subheader("판매 수익에서 무엇이 더해지고 빠졌나요?")
    has_actual = data["actual"] is not None
    replay = replay_payload(data, day)
    if has_actual:
        rows = [
            {"hour": f["hour"], "f": f["cumulative"] / 10000, "a": a["cumulative"] / 10000}
            for f, a in zip(replay["series"]["forecast"], replay["series"]["actual"],
                            strict=True)
        ]
        render_signal(
            rows,
            title="하루 누적 계획 수익",
            day=day,
            unit="만 원",
            kind="revenue",
            forecast_label="예측이 맞았다고 가정",
            actual_label="실측을 미리 알고 계획",
        )
        mode = st.radio(
            "전체 기간 수익 기준",
            ["실측을 미리 알고 계획", "예측이 맞았다고 가정"],
            horizontal=True,
            key="oc_revenue_mode",
        )
    else:
        series = replay["series"]["forecast"]
        st.caption(f"{day} 하루 누적 계획 수익(예측 기준) · 마지막 시각 "
                   f"{series[-1]['cumulative'] / 10000:,.1f}만 원. "
                   "실측 비교는 실측이 들어오면 표시.")
        mode = "예측이 맞았다고 가정"
    rev = data["revenues"]["actual" if mode.startswith("실측") else "forecast"]
    kpis(
        [
            ("전체 기간 순수익", money(rev["net_revenue_krw"]), "SMP + REC − 열화비"),
            ("ESS 없음 대비 추가수익", money(rev["ess_gain_krw"]), "같은 입력·REC 조건으로 비교"),
            ("실제로 운전한 뒤의 수익", "미계산", "예측 오차 대응·편차 정산 단계 필요"),
        ]
    )
    fig = go.Figure(
        go.Waterfall(
            x=["SMP 판매", "REC", "열화비", "계획 순수익"],
            measure=["relative", "relative", "relative", "total"],
            y=[
                rev["sales_revenue_krw"] / 1e8,
                rev["rec_revenue_krw"] / 1e8,
                -rev["throughput_cost_krw"] / 1e8,
                0,
            ],
            increasing={"marker": {"color": "#218d81"}},
            decreasing={"marker": {"color": "#cc805b"}},
            totals={"marker": {"color": "#203e55"}},
        )
    )
    fig.update_layout(
        height=310,
        yaxis_title="억 원",
        margin=dict(t=20, b=20),
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
    )
    st.plotly_chart(fig, width="stretch")
    st.info(
        "REC는 ESS 없이 판매해도 받습니다. "
        "예측 조건과 실측 조건의 수익 차이를 실제 손실로 읽으면 안 됩니다."
    )


def show_execution(state, release):
    st.subheader("코드부터 화면까지, 지금 어디까지 실행됐나요?")
    flow(state)
    labels = {"complete": "완료", "running": "실행 중", "blocked": "파일 대기", "failed": "오류"}
    for step in state.get("stages", []):
        with st.container(border=True):
            st.markdown(f"**{step['name']} · {labels.get(step['state'], step['state'])}**")
            for missing in step.get("missing", []):
                st.caption("없는 파일")
                st.code(missing, language="text")
            if step.get("error"):
                st.error(step["error"])
    if state.get("error"):
        st.error(state["error"])
    st.markdown("**동작 방식**")
    st.write(
        "팀원이 모델을 바꾸면 결과 파일을 Drive 로 공유합니다. 받은 파일을 아래 위치에 같은 "
        "이름으로 두면, 화면이 열려 있는 동안 30초마다 파일 내용·ESS·REC 설정·최적화 코드 변경을 "
        "확인하고 "
        "바뀐 경우에만 MILP(비교 백테스트 + 48시간 운영계획)를 다시 풉니다. 검산을 통과한 결과만 "
        "모든 화면의 기본 결과가 됩니다."
    )
    st.caption(
        "모델 학습·추론은 여기서 하지 않습니다. 앱이 꺼져 있으면 확인하지 않고, 같은 내용을 반복 "
        "계산하지 않습니다."
    )
    st.dataframe(
        pd.DataFrame(pipeline.requirements(pipeline.config(ROOT))).rename(
            columns={
                "name": "공유 파일",
                "ready": "파일 있음",
                "path": "두는 위치",
                "purpose": "출처",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("정확한 버전 확인"):
        st.code(
            f"입력·코드 지문: {state.get('revision', '미기록')}\n"
            f"결과 묶음: {release.get('id', '없음')} · {release.get('label', '')}"
        )
        st.write("사용한 발전량 예측 열:", ", ".join(release.get("forecast_columns", [])) or "없음")
        st.json(release.get("hashes", {}))
    st.markdown("**계산에 아직 없는 부분**")
    st.write(
        "확정 판매계획을 실측에 적용한 실행 운전·정산, 실제 SOC 이월은 아직 미구현입니다. "
        "종료 SOC 정책은 변경하지 않았습니다. "
        "REC 초기 저장분 제외도 미구현이며 실험 가정을 유지합니다. "
        "팀원 모델의 학습·추론은 각 담당자가 돌리고 결과 파일만 공유받습니다."
    )


def render():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="oc-brand"><div class="oc-word">JEJU FLOW'
        '<div class="oc-sub">발전 예측부터 ESS 운영계획까지</div></div>'
        '<div class="oc-badge">운영 시뮬레이션 · 실제 입찰·정산 아님</div></div>',
        unsafe_allow_html=True,
    )
    cfg = pipeline.config(ROOT)
    with st.sidebar:
        st.subheader("실행 설정")
        st.toggle(
            "코드·입력 변경 시 자동 실행", value=cfg.get("auto_refresh", True), key="operator_auto"
        )
        if st.button("지금 다시 계산", type="primary", width="stretch"):
            pipeline.launch(ROOT, force=True)
            st.rerun()
        st.caption("188일 기준 수 분 걸립니다. 진행 상태는 실행 상태 화면에서 확인합니다.")
        st.caption("공유 파일 경로와 ESS·REC 가정은 config/operator_pipeline.yaml 에서 정합니다.")
        st.code(
            f"ESS: {Path(cfg['ess']).name}\n"
            f"REC: {Path(cfg['rec']).name if cfg.get('rec') else '미적용'}"
        )
    monitor()
    state = pipeline.status(ROOT)
    release = state.get("active") or {}
    nav = st.radio(
        "화면",
        ["운영 관제", "예측 비교", "운영 계획", "수익 분석", "실행 상태"],
        horizontal=True,
        label_visibility="collapsed",
        key="operator_screen",
    )
    if state.get("state") in ("running", "queued"):
        st.info("새 코드·입력으로 실행 중입니다. 완료 전까지 아래는 이전 검산 완료 결과입니다.")
    if state.get("state") in ("needs_data", "failed"):
        st.markdown(
            '<div class="oc-callout">마지막 계산이 끝나지 않았거나 실패했습니다. '
            "아래는 마지막 검산 완료 결과이며, 사용 예측은 바로 아래에 표시됩니다. "
            "‘실행 상태’에서 없는 파일과 오류를 확인하세요.</div>",
            unsafe_allow_html=True,
        )
    if nav == "실행 상태":
        show_execution(state, release)
        return
    if not release:
        st.info("첫 실행 결과를 준비하고 있습니다. 실행 상태에서 진행 상황을 확인하세요.")
        flow(state)
        return
    st.caption(
        f"같은 결과 묶음 사용 · {release['label']} · {release['start']} ~ {release['end']} · "
        f"{release['days']}일 · "
        f"계산 {pd.Timestamp(release['created_at']).tz_convert('Asia/Seoul'):%m/%d %H:%M} KST"
    )
    if not release.get("backtest"):
        st.info("예측 창이 없어 운영계획을 만들지 못했습니다. "
                "실행 상태에서 입력 파일을 확인하세요.")
        return
    folder = Path(release["backtest"])
    try:
        data = read_report(str(folder), (folder / "summary.json").stat().st_mtime_ns)
    except (ValueError, OSError) as exc:
        st.error(f"저장 결과 검증 실패: {exc}")
        return
    has_actual = data["actual"] is not None
    if not has_actual:
        st.info(
            "이 기간은 실측이 아직 없습니다. 예측으로 세운 운영계획과 계획 수익만 보여주고, "
            "예측 비교와 실측 기준 수익은 실측 파일이 들어오면 같은 자리에 채워집니다."
        )
    days = sorted(data["forecast"].ts.dt.date.unique())
    if st.session_state.get("operator_day") not in days:
        st.session_state["operator_day"] = days[0]
    day = st.selectbox("운영일 · 모든 화면 공통", days, key="operator_day")
    if nav == "예측 비교":
        show_forecasts(data, day, release)
    elif nav == "수익 분석":
        show_profit(data, day)
    elif nav == "운영 계획":
        st.subheader("언제 저장하고, 언제 판매하나요?")
        render_simulator(data, day)
        f = data["forecast"]
        selected = f[f.ts.dt.date == day]
        st.caption(
            "예측으로 세운 확정 24시간 계획입니다. 낙찰받은 판매량이나 실제 실행 기록은 아닙니다."
        )
        display = selected[
            ["ts", "generation_mwh", "sell_mwh", "charge_mwh", "discharge_mwh", "soc_end"]
        ].copy()
        display.columns = [
            "시각",
            "발전량 MWh",
            "판매 MWh",
            "충전 MWh",
            "방전 MWh",
            "종료 SOC 비율",
        ]
        st.dataframe(display, hide_index=True, width="stretch")
        st.download_button(
            "하루 계획 CSV",
            selected.to_csv(index=False).encode("utf-8-sig"),
            f"plan_{day}.csv",
            "text/csv",
        )
        with st.expander("최근 운영일의 48시간 계획"):
            st.caption(release["start_soc_note"])
            st.dataframe(pd.read_csv(Path(release["plan"]) / "plan_48h.csv"), hide_index=True)
    else:
        st.subheader("오늘의 흐름을 한 번에 확인하세요")
        m = data["metrics"]
        if has_actual:
            rev = data["revenues"]["actual"]
            kpis(
                [
                    (
                        "발전량 평균 오차",
                        f"{m['generation_mae_mwh']:.2f} MWh",
                        "전체 비교 기간 · 시간당",
                    ),
                    ("SMP 평균 오차", f"{m['smp_mae_krw_per_kwh']:.2f}원/kWh", "전체 비교 기간"),
                    ("ESS 추가수익", money(rev["ess_gain_krw"]), "기간 전체 · 실측을 미리 안 경우"),
                    ("실제 실행 수익", "미계산", "계획 고정 후 오차 대응 필요"),
                ]
            )
        else:
            rev = data["revenues"]["forecast"]
            kpis(
                [
                    ("계획 수익 (예측 기준)", money(rev["net_revenue_krw"]),
                     f"{m['days']}일 · SMP + REC − 열화비"),
                    ("ESS 추가수익 (예측 기준)", money(rev["ess_gain_krw"]),
                     "같은 예측으로 ESS 없이 팔 때와의 차이"),
                    ("예측 오차", "실측 없음", "실측 파일이 들어오면 표시"),
                    ("실제 실행 수익", "미계산", "계획 고정 후 오차 대응 필요"),
                ]
            )
        st.write("")
        render_simulator(data, day)
        flow(state)
        st.caption(f"연속 구간 {m['segments']}개 · 모든 화면이 같은 입력과 조건을 사용합니다.")
