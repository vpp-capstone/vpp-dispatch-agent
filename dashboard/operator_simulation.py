"""Replay saved energy flows. No live dispatch, prediction, or settlement is performed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from vpp.rec import REC_COLUMN

TEMPLATE = Path(__file__).parent / "assets" / "operator_simulator.html"


def replay_payload(data: dict, day) -> dict:
    ess = data["ess"]
    series = {}
    daily_soc = {}
    for mode in ("forecast", "actual"):
        frame = data[mode]
        block = frame[frame.ts.dt.date == day].reset_index(drop=True)
        actual = data["actual"]
        actual = actual[actual.ts.dt.date == day].reset_index(drop=True)
        if block.empty or not block.ts.equals(actual.ts):
            raise ValueError("선택한 하루의 예측·실측 시각이 일치하지 않습니다.")
        if len(block) != 24 or block.ts.dt.hour.tolist() != list(range(24)):
            raise ValueError("하루 재생에는 00시부터 23시까지 빠짐없는 데이터가 필요합니다.")
        rec = block.get(REC_COLUMN, pd.Series(0.0, index=block.index))
        cumulative = 0.0
        records = []
        for i, row in block.iterrows():
            charge, discharge = float(row.charge_mwh), float(row.discharge_mwh)
            initial = (
                float(row.energy_end_mwh)
                - ess.charge_efficiency * charge
                + discharge / ess.discharge_efficiency
            )
            wear = ess.degradation_cost_krw(np.array([charge]), np.array([discharge]))
            sales = float(row.sell_mwh * row.price_krw_per_mwh)
            certificate = float(row.sell_mwh * rec.iloc[i])
            profit = sales + certificate - wear
            cumulative += profit
            records.append(
                {
                    "hour": row.ts.strftime("%H:%M"),
                    "g": float(row.generation_mwh),
                    "c": charge,
                    "d": discharge,
                    "q": float(row.sell_mwh),
                    "r": float(row.curtailment_mwh),
                    "soc": float(row.soc_end * 100),
                    "startSoc": initial / ess.capacity_mwh * 100,
                    "e": float(row.energy_end_mwh),
                    "p": float(row.price_krw_per_mwh / 1000),
                    "rec": float(rec.iloc[i] / 1000),
                    "wear": wear,
                    "profit": profit,
                    "cumulative": cumulative,
                    "actualG": float(actual.iloc[i].generation_mwh),
                    "actualP": float(actual.iloc[i].price_krw_per_mwh / 1000),
                }
            )
        series[mode] = records
        # Only exact adjacent boundaries count as carry-over. Never bridge missing days.
        previous = frame[frame.ts == block.iloc[0].ts - pd.Timedelta(hours=1)]
        following = frame[frame.ts == block.iloc[-1].ts + pd.Timedelta(hours=1)]

        def boundary(neighbor, current, *, before, has_segments, target):
            if neighbor.empty:
                return {"soc": None, "status": "missing"}
            row = neighbor.iloc[0]
            energy = float(row.energy_end_mwh)
            if not before:
                energy += (
                    float(row.discharge_mwh) / ess.discharge_efficiency
                    - ess.charge_efficiency * float(row.charge_mwh)
                )
            soc = energy / ess.capacity_mwh * 100
            same_segment = not has_segments or row.segment_id == current.segment_id
            status = "reset" if not same_segment else (
                "matched" if abs(soc - target) <= 1e-5 else "mismatch"
            )
            return {"soc": soc, "status": status}

        has_segments = "segment_id" in frame
        daily_soc[mode] = {
            "start": records[0]["startSoc"],
            "end": records[-1]["soc"],
            "previous": boundary(previous, block.iloc[0], before=True,
                                 has_segments=has_segments, target=records[0]["startSoc"]),
            "next": boundary(following, block.iloc[-1], before=False,
                             has_segments=has_segments, target=records[-1]["soc"]),
        }
    return {
        "day": str(day),
        "capacity": ess.capacity_mwh,
        "minSoc": ess.soc_min * 100,
        "maxSoc": ess.soc_max * 100,
        "series": series,
        "dailySoc": daily_soc,
    }


def simulator_html(data: dict, day) -> str:
    # JSON is data, never executable HTML from filenames or inputs.
    payload = json.dumps(replay_payload(data, day), ensure_ascii=False, allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return TEMPLATE.read_text(encoding="utf-8").replace("__REPLAY_DATA__", payload)


def render_simulator(data: dict, day) -> None:
    html = simulator_html(data, day)
    if hasattr(st, "iframe"):
        st.iframe(html, height="content", alt="시간별 발전·판매·ESS 운영계획 재생")
    else:
        components.html(html, height=980, scrolling=True)


def connection_inventory(root: Path) -> dict:
    """Artifact availability is not evidence of which model generated a saved CSV."""
    return {
        "generation_code": (root / "src/vpp/generation_forecast/improved_model.py").exists(),
        "solar_bundle": (root / "artifacts/models/solar_improved_production.joblib").exists(),
        "wind_bundle": (root / "artifacts/models/wind_improved_production.joblib").exists(),
        "smp_code": (root / "src/vpp/smp_weather_features.py").exists(),
    }
