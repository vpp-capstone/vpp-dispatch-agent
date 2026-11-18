import numpy as np
import pandas as pd
import pytest

from vpp import spo_forecast as so

LP = so.DayLP()
SOLAR = np.r_[
    np.zeros(7), np.linspace(0.5, 4, 5), np.full(3, 4.2), np.linspace(4, 0.5, 4), np.zeros(5)
]  # fmt: skip


def _prices(day: float, evening: float) -> np.ndarray:
    p = np.full(24, 100.0)
    p[9:16] = day
    p[17:22] = evening
    return p


def test_day_lp_respects_cap_and_shifts_to_evening():
    tm = LP.template(SOLAR)
    sell = LP.sell(tm, _prices(20.0, 200.0))
    day = np.arange(24)
    assert (sell[(day >= 8) & (day < 18)] <= LP.cap_mw + 1e-6).all()
    assert sell[17:22].sum() > 0  # 낮에 담아 저녁에 판다
    reverse = LP.sell(tm, _prices(200.0, 20.0))  # 저녁이 더 싸면 낮에 판다
    assert sell[17:22].sum() > reverse[17:22].sum()


def test_spo_grad_is_zero_when_prediction_is_exact():
    tm = LP.template(SOLAR)
    p = _prices(20.0, 200.0)
    assert np.allclose(so.spo_plus_grad(LP, tm, p, p), 0.0)


def test_spo_grad_restores_compressed_spread():
    """실제 가격 차가 큰데 예측이 평평하면, 저녁 예측을 올리는 방향(기울기 음수)으로 고친다."""
    tm = LP.template(SOLAR)
    true = _prices(20.0, 200.0)
    flat = np.full(24, true.mean())
    g = so.spo_plus_grad(LP, tm, flat, true)
    assert g[17:22].sum() < 0
    assert g[17:22].sum() < g[9:16].sum()


def test_fit_spo_runs_and_keeps_shape(monkeypatch):
    monkeypatch.setattr(
        so.sf, "REG_PARAMS", {**so.sf.REG_PARAMS, "n_estimators": 20, "min_child_samples": 5}
    )
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-05-01", periods=40, freq="D")
    rows = []
    for d in days:
        for h in range(24):
            spread = 80.0 if 17 <= h < 22 else (-60.0 if 9 <= h < 16 else 0.0)
            rows.append({"target_date": d, "hour": h, "x_hour": h, "x_noise": rng.normal(),
                         "y": 100.0 + spread + rng.normal(0, 5)})  # fmt: skip
    df = pd.DataFrame(rows)
    train, test = df[df.target_date < days[30]], df[df.target_date >= days[30]]
    solar = pd.Series(
        np.tile(SOLAR, 366), index=pd.date_range("2025-01-01", periods=24 * 366, freq="h")
    )
    l1, spo = so.fit_spo(
        train.reset_index(drop=True), test.reset_index(drop=True), ["x_hour", "x_noise"],
        solar, rounds=5, folds=2,
    )  # fmt: skip
    assert l1.shape == spo.shape == (len(test),)
    assert np.isfinite(spo).all()


def test_replace_point_only_touches_point_forecast():
    base = pd.DataFrame(
        {
            "issue_date": ["2025-01-01"] * 2,
            "horizon": [1, 1],
            "target_time": ["2025-01-02T00:00:00+09:00", "2025-01-02T01:00:00+09:00"],
            "smp_hat": [100.0, 110.0],
            "smp_hat_krw_per_mwh": [100000.0, 110000.0],
            "smp_p10": [80.0, 90.0],
        }
    )
    preds = pd.DataFrame(
        {"issue_date": ["2025-01-01"], "horizon": [1],
         "target_ts": [pd.Timestamp("2025-01-02 01:00")], "smp_spo": [123.4567]}
    )  # fmt: skip
    out = so.replace_point(base, preds)
    assert out["smp_hat"].tolist() == [100.0, pytest.approx(123.457)]
    assert out["smp_hat_krw_per_mwh"].iloc[1] == pytest.approx(123456.7)
    assert out["smp_p10"].tolist() == [80.0, 90.0]
