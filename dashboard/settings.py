"""사이드바 공통 설정: 데이터 묶음 · 날짜/기간 · ESS 설정 · 열화비 설정.

값은 파일(예측 결과, config YAML)에서 읽기만 하고 어떤 파일도 고치지 않는다. 설정을 바꿔도 계산은
각 탭의 '이 설정으로 실행' 버튼을 눌러야 돈다. 여기서 하는 계산은 파일 읽기·창 구성뿐이며 캐시된다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

from ui import CONFIG_DIR, ROOT, degradation_choices
from vpp import forecast_inputs as fi
from vpp import forecast_metrics as fm
from vpp.operating_plan import resolve_ess
from vpp.optimization import EssParams

OPERATING_PLAN_DIR = Path(os.environ.get("VPP_OPERATING_PLAN_DIR",
                                         ROOT / "data/processed/operating_plan"))
BACKTEST_OUT_DIR = Path(os.environ.get("VPP_BACKTEST_DIR", ROOT / "data/processed/backtest"))


@dataclass(frozen=True)
class Settings:
    solar: Path
    wind: Path
    smp: Path
    plants: Path
    ess_path: Path
    degradation_path: Path | None
    degradation_label: str
    planning_day: date | None
    start: date | None
    end: date | None

    @property
    def files(self) -> dict[str, Path]:
        return {"solar": self.solar, "wind": self.wind, "smp": self.smp, "plants": self.plants}

    def missing_files(self) -> list[str]:
        return [f"{k}: {p}" for k, p in self.files.items() if not p.exists()]

    def stamp(self) -> tuple[float, ...]:
        return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in self.files.values())

    def ess(self) -> EssParams:
        return resolve_ess(self.ess_path, self.degradation_path)

    def signature(self) -> dict[str, Any]:
        """표시 중인 결과에 적용된 설정과 비교하기 위한 요약."""
        return {
            "inputs": {k: {"path": str(p), "size_bytes": p.stat().st_size if p.exists() else None}
                       for k, p in self.files.items()},
            "ess_path": str(self.ess_path),
            "degradation_path": str(self.degradation_path or self.ess_path),
            "ess": self.ess().to_dict(),
        }


@st.cache_data(show_spinner="예측 파일을 읽고 계획일 창을 만드는 중")
def load_bundle(solar: str, wind: str, smp: str, plants: str,
                stamp: tuple[float, ...]) -> dict[str, Any]:
    """데이터 묶음 한 번 읽기. 파일 수정시각(stamp)이 바뀌면 다시 읽는다."""
    expected = fi.expected_assets(Path(plants))
    pred = fm.load_generation_eval_frame(Path(solar), Path(wind), Path(plants))
    portfolio = fi.aggregate_portfolio(pred, expected)
    smp_raw = fi.load_smp_forecast(Path(smp))
    days = sorted(portfolio["planning_day"].unique())
    windows, excluded = fi.build_plan_windows(portfolio, smp_raw, days[0], days[-1])
    return {
        "expected": expected,
        "pred": pred,
        "smp_eval": fm.load_smp_eval_frame(Path(smp)),
        "plan_days": [w.planning_day for w in windows],
        "comparable_days": [w.planning_day for w in windows if w.comparable],
        "excluded": excluded,
        "comparison_excluded": fi.comparison_exclusions(windows),
        "data_days": (days[0], days[-1]),
    }


def _short(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def _file_caption(path: Path) -> str:
    if not path.exists():
        return "❌ 없음"
    mtime = date.fromtimestamp(path.stat().st_mtime)
    return f"✓ {path.stat().st_size / 1e6:.1f} MB · {mtime}"


def render_sidebar() -> tuple[Settings, dict[str, Any] | None]:
    """사이드바를 그리고 (설정, 데이터 묶음 또는 None) 을 돌려준다."""
    with st.sidebar:
        st.markdown("### 공통 설정")
        with st.expander("데이터 묶음 (예측 결과 파일)", expanded=False):
            st.caption("기본값은 README '데이터 준비'의 위치. 바꾸면 그 파일로 모든 탭이 계산한다.")
            solar = Path(st.text_input("태양광 예측 parquet", str(fi.DEFAULT_SOLAR_PREDICTIONS),
                                       key="set_solar"))
            st.caption(_file_caption(solar))
            wind = Path(st.text_input("풍력 예측 parquet", str(fi.DEFAULT_WIND_PREDICTIONS),
                                      key="set_wind"))
            st.caption(_file_caption(wind))
            smp = Path(st.text_input("SMP 예측 csv", str(fi.DEFAULT_SMP_FORECAST), key="set_smp"))
            st.caption(_file_caption(smp))
            plants = Path(st.text_input("발전소 기준정보 csv", str(fi.PLANT_MASTER_PATH),
                                        key="set_plants"))
            st.caption(_file_caption(plants))

        ess_files = sorted(CONFIG_DIR.glob("ess*.yaml"))
        ess_path = st.selectbox("ESS 설정 (YAML)", ess_files, format_func=lambda p: p.name,
                                key="set_ess",
                                help="용량·출력·SOC 범위·효율·시작 SOC. 파일은 고치지 않는다.")
        choices = degradation_choices()
        labels = list(choices)
        own = EssParams.from_yaml(ess_path)
        default = next((i for i, k in enumerate(labels)
                        if choices[k]["throughput_cost_krw_per_mwh"]
                        == own.throughput_cost_krw_per_mwh), 0)
        deg_label = st.radio("열화비", labels, index=default, key="set_deg",
                             help="config/ess_v0.yaml(미적용)·ess_v0_1.yaml(고정 열화비) 값. "
                                  "바꾼 뒤 각 탭의 '이 설정으로 실행'을 눌러야 "
                                  "최적화부터 다시 계산.")
        degradation_path = ROOT / choices[deg_label]["source"]
        ess = resolve_ess(ess_path, degradation_path)
        st.caption(f"ESS {ess.capacity_mwh:g} MWh · {ess.charge_power_mw:g} MW · "
                   f"SOC {ess.soc_min:.0%}~{ess.soc_max:.0%} · 시작 {ess.soc_initial:.0%} · "
                   f"효율 {ess.charge_efficiency:.2f} · 열화 {ess.degradation_model} "
                   f"{ess.throughput_cost_krw_per_mwh:,.0f}원/MWh")

        draft = Settings(solar, wind, smp, plants, Path(ess_path), degradation_path, deg_label,
                         None, None, None)
        missing = draft.missing_files()
        bundle = None
        if missing:
            st.error("데이터 묶음 파일이 없습니다:\n\n" + "\n".join(f"- {m}" for m in missing))
        else:
            try:
                bundle = load_bundle(str(solar), str(wind), str(smp), str(plants), draft.stamp())
            except Exception as exc:  # 파일이 깨져도 사이드바와 탭 뼈대는 떠야 한다
                st.error(f"예측 파일을 읽지 못했습니다: {exc}")

        planning_day = start = end = None
        if bundle is not None and bundle["plan_days"]:
            days = bundle["plan_days"]
            st.markdown("### 날짜 · 기간")
            planning_day = st.selectbox(
                "계획일 D (운영계획 생성)", days, index=len(days) - 1, key="set_day",
                help="예측 48시간이 갖춰진 날만 보인다. 실측 없이도 계획은 만들 수 있다.",
            )
            first, last = bundle["data_days"]  # 예측 파일이 있는 전체 범위 (성능 평가 가능)
            period = st.date_input("기간 (예측 성능 · 백테스트)", (days[0], days[-1]),
                                   min_value=first, max_value=last, key="set_period",
                                   help="기본은 계획 가능한 날짜 범위. 예측 성능만 보려면 "
                                        f"{first} 까지 넓힐 수 있다.")
            if isinstance(period, tuple | list) and len(period) == 2:
                start, end = period
            else:  # 날짜 하나만 고른 중간 상태
                start = end = period if isinstance(period, date) else first
            n_cmp = len(bundle["comparable_days"])
            st.caption(f"계획 가능 {len(days)}일 · 실측까지 있는 날 {n_cmp}일 · "
                       f"예측 부족 제외 {len(bundle['excluded'])}일")
        st.caption(f"결과 저장: `{_short(OPERATING_PLAN_DIR)}` · `{_short(BACKTEST_OUT_DIR)}`")

    settings = Settings(solar, wind, smp, plants, Path(ess_path), degradation_path, deg_label,
                        planning_day, start, end)
    return settings, bundle


def settings_chips(settings: Settings, *, show_day: bool = False,
                   show_period: bool = False) -> str:
    parts = []
    if show_day and settings.planning_day:
        parts.append(f"계획일 {settings.planning_day}")
    if show_period and settings.start and settings.end:
        parts.append(f"기간 {settings.start} ~ {settings.end}")
    parts += [f"ESS {settings.ess_path.name}", f"열화비 {settings.degradation_label}",
              f"예측 {settings.solar.name} · {settings.wind.name} · {settings.smp.name}"]
    return "".join(f'<span class="chip" style="background:#374151">{p}</span>' for p in parts)
