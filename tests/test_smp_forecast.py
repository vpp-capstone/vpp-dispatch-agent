"""SMP 예측(vpp.smp_forecast) 테스트: 발행 시점 이후 정보가 섞이지 않는지 확인한다."""

from __future__ import annotations

import pandas as pd
import pytest

from vpp import smp_forecast as sf


def make_hourly(days: int = 40, start: str = "2024-01-01") -> pd.DataFrame:
    """가격 = 날짜 번호 x 100 + 시간. 값만 보고 어느 날·시간인지 알 수 있게 만든다."""
    ts = pd.date_range(start, periods=days * 24, freq="h")
    day_no = (ts.normalize() - pd.Timestamp(start)).days
    return pd.DataFrame(
        {
            "ts": ts,
            "smp_da": day_no * 100 + ts.hour,
            "demand_mwh": 10_000 + day_no * 100 + ts.hour,
            "fcst_jeju": 20_000 + day_no * 100 + ts.hour,
            "smp_land": 50.0,
        }
    )


def code(day: int, hour: int) -> int:
    return day * 100 + hour


@pytest.mark.parametrize("h", [1, 2])
def test_samples_use_only_prices_published_by_issue_time(h):
    s = sf.build_samples(make_hourly(), horizons=(h,), use_fcst=True).set_index(
        ["issue_date", "hour"]
    )
    issue = pd.Timestamp("2024-01-20")  # 날짜 번호 19
    row = s.loc[(issue, 13)]

    assert row["target_ts"] == issue + pd.Timedelta(days=h, hours=13)
    assert row["issue_ts"] == issue + pd.Timedelta(hours=11)
    assert row["y"] == code(19 + h, 13)  # 정답은 대상일 가격
    assert row["p_last"] == code(19, 13)  # 발행일 자신의 가격 (전날 18시 공표)
    assert row["p_last2"] == code(18, 13)
    assert row["p_week"] == code(19 + h - 7, 13)  # 대상일 7일 전
    assert row["demand_prev"] == 10_000 + code(18, 13)  # 수요 실적은 발행 전날까지만
    assert row["fcst_jeju"] == 20_000 + code(19 + h, 13)  # 옵션일 때만 대상일 예측값
    assert row["p_last_max"] == code(19, 23)

    # 모든 행에서 가격 특징은 발행일(날짜 번호 n)까지의 값, 즉 n*100+23 이하여야 한다
    price_feats = ["p_last", "p_last2", "p_week", "p_last_mean", "p_last_min", "p_last_max"]
    issue_no = (s.index.get_level_values("issue_date") - pd.Timestamp("2024-01-01")).days
    latest_allowed = pd.Series(issue_no * 100 + 23, index=s.index)
    assert (
        s[price_feats].max(axis=1).dropna() <= latest_allowed[s[price_feats].max(axis=1).notna()]
    ).all()


def test_default_samples_exclude_kpx_demand_forecast():
    s = sf.build_samples(make_hourly(), horizons=(1,))
    assert "fcst_jeju" not in s.columns
    assert "fcst_jeju" not in sf.BASE_FEATURES


def test_weather_uses_latest_forecast_available_before_issue():
    s = sf.build_samples(make_hourly(), horizons=(1,))
    target = pd.Timestamp("2024-01-21 13:00")  # 발행일 2024-01-20 11시
    weather = pd.DataFrame(
        {
            "issued_at": ["2024-01-19 21:00", "2024-01-20 03:00", "2024-01-20 09:00"],
            "available_at": ["2024-01-20 01:00", "2024-01-20 07:00", "2024-01-20 13:00"],
            "target_ts": [str(target)] * 3,
            "solar_index": [1.0, 2.0, 3.0],  # 세 번째는 11시 이후 이용 가능 → 쓰면 안 됨
            "coverage_ratio": [0.84] * 3,  # 품질 정보는 입력에서 제외
        }
    )
    for c in ("issued_at", "available_at", "target_ts"):
        weather[c] = pd.to_datetime(weather[c])

    merged, cols = sf.attach_weather(s, weather)
    row = merged.set_index("target_ts").loc[target]

    assert cols == ["wx_solar_index"]
    assert row["wx_solar_index"] == 2.0
    assert row["wx_available_at"] == pd.Timestamp("2024-01-20 07:00")


def test_weather_older_than_tolerance_is_not_used():
    s = sf.build_samples(make_hourly(), horizons=(1,))
    target = pd.Timestamp("2024-01-21 13:00")
    weather = pd.DataFrame(
        {"available_at": [pd.Timestamp("2024-01-17 07:00")], "target_ts": [target],
         "solar_index": [9.0]}
    )  # fmt: skip
    merged, _ = sf.attach_weather(s, weather)
    assert pd.isna(merged.set_index("target_ts").loc[target, "wx_solar_index"])


def test_walk_forward_trains_only_on_published_targets(monkeypatch):
    s = sf.build_samples(make_hourly(days=120), horizons=(1, 2))
    seen = []

    class Spy:
        def __init__(self, **_):
            pass

        def fit(self, x, y):
            seen.append(s.loc[x.index, "target_date"].max())  # 학습에 쓴 가장 늦은 대상일
            return self

        def predict(self, x):
            return [0.0] * len(x)

        def predict_proba(self, x):
            import numpy as np

            return np.tile([0.9, 0.1], (len(x), 1))

    monkeypatch.setattr(sf.lgb, "LGBMRegressor", Spy)
    monkeypatch.setattr(sf.lgb, "LGBMClassifier", Spy)
    pred = sf.walk_forward(
        s, "2024-03-01", "2024-04-15", sf.BASE_FEATURES, train_start="2024-01-01"
    )

    # 발행일 02-29 ~ 04-14, 발행일마다 내일·모레 48시간
    assert pred["issue_date"].min() == pd.Timestamp("2024-02-29")
    assert pred["issue_date"].max() == pd.Timestamp("2024-04-14")
    assert (pred.groupby("issue_date").size() == 48).all()
    assert pred["target_ts"].min() == pd.Timestamp("2024-03-01 00:00")
    assert pred["target_ts"].max() == pd.Timestamp("2024-04-16 23:00")  # 마지막 발행일의 모레
    # 각 달 모델은 그 달 1일까지 공표된 정답으로만 학습한다
    assert (pred["trained_until"] == pred["issue_date"].dt.to_period("M").dt.to_timestamp()).all()
    months = sorted(pred["trained_until"].unique())
    # 달 x 내일·모레 x (회귀 + 분위수 2개). 이 자료엔 0원 이하가 1시간뿐이라 분류기는 학습하지 않는다
    assert len(seen) == len(months) * 2 * 3
    for i, month in enumerate(months):
        assert all(t <= month for t in seen[i * 6 : i * 6 + 6])
    assert (pred["p_low"] < 0.01).all() and (pred["p_neg"] == 0).all()  # 사건 비율로 대체


def test_event_probability_falls_back_to_rate_when_events_are_rare():
    x = pd.DataFrame({"a": range(100)})
    rare = pd.Series([1] * 3 + [0] * 97)
    assert (sf.event_probability(x, rare, x.head(5)) == 0.03).all()
    common = pd.Series([1] * 30 + [0] * 70)
    p = sf.event_probability(x, common, x)
    assert p.shape == (100,) and p[:30].mean() > p[30:].mean()  # 분류기가 사건 쪽에 높은 확률


def test_quantile_band_brackets_point_forecast():
    s = sf.build_samples(make_hourly(days=60), horizons=(1, 2))
    out = sf.to_output(
        sf.walk_forward(s, "2024-02-10", "2024-02-20", sf.BASE_FEATURES, train_start="2024-01-01")
    )
    assert (out["smp_p10"] <= out["smp_hat"] + 1e-9).all()
    assert (out["smp_hat"] <= out["smp_p90"] + 1e-9).all()
    assert ((out["smp_p90_krw_per_mwh"] - out["smp_p90"] * 1000).abs() <= 0.5).all()
    for c in ("p_low", "p_neg"):
        assert out[c].between(0, 1).all()


def test_window_for_issue_gives_48_hours_for_optimizer():
    s = sf.build_samples(make_hourly(days=60), horizons=(1, 2))
    pred = sf.walk_forward(
        s, "2024-02-10", "2024-02-20", sf.BASE_FEATURES, train_start="2024-01-01"
    )
    out = sf.to_output(pred)
    win = sf.window_for_issue(out, "2024-02-12")

    assert len(win) == 48
    assert str(win["ts"].dt.tz) == sf.KST
    assert win["ts"].iloc[0] == pd.Timestamp("2024-02-13 00:00", tz=sf.KST)
    assert list(win["horizon"].unique()) == [1, 2]
    assert win["lead_hour"].tolist() == list(range(13, 61))  # 11시 발행 → 내일 00시 ~ 모레 23시


def test_output_uses_generation_forecast_column_names():
    s = sf.build_samples(make_hourly(days=60), horizons=(1, 2))
    pred = sf.walk_forward(
        s, "2024-02-10", "2024-02-12", sf.BASE_FEATURES, train_start="2024-01-01"
    )
    out = sf.to_output(pred)

    for c in ("issue_time", "target_time", "lead_hour", "smp_hat"):
        assert c in out.columns
    expected = (out["target_time"] - out["issue_time"]) / pd.Timedelta(hours=1)
    assert (out["lead_hour"] == expected).all()
    assert (out["issue_time"].dt.hour == 11).all()


def test_new_features_use_only_published_prices():
    s = sf.build_samples(make_hourly(), horizons=(1,)).set_index(["issue_date", "hour"])
    row = s.loc[(pd.Timestamp("2024-01-20"), 13)]  # 토요일(날짜 번호 19) 11시 발행, 대상 일요일

    assert row["p_med3"] == code(18, 13)  # 19·18·17일의 중앙값
    assert row["p_med7"] == code(16, 13)  # 19~13일의 중앙값
    assert row["p_sametype"] == code(13, 13)  # 대상이 일요일 → 가장 최근 일요일(1/14)
    for c in sf.EXTRA_FEATURES:
        assert c in s.columns


def test_issue_at_18_uses_next_day_price_but_not_newer_demand():
    s = sf.build_samples(make_hourly(), horizons=(2,), issue_hour=18).set_index(
        ["issue_date", "hour"]
    )
    issue = pd.Timestamp("2024-01-20")  # 날짜 번호 19, 18시엔 20번(1/21) 가격이 공표됨
    row = s.loc[(issue, 13)]

    assert row["issue_ts"] == issue + pd.Timedelta(hours=18)
    assert row["y"] == code(21, 13)  # 대상: 모레(1/22)
    assert row["p_last"] == code(20, 13)  # 막 공표된 내일 가격 = B1
    assert row["p_last2"] == code(19, 13)
    assert row["p_last_max"] == code(20, 23)
    assert row["p_med3"] == code(19, 13)
    assert row["p_sametype"] == code(18, 13)  # 대상 월요일 → 공표된 날 중 가장 최근 평일(1/19)
    assert row["demand_prev"] == 10_000 + code(18, 13)  # 수요는 11시와 같이 발행 전날까지

    with pytest.raises(ValueError):
        sf.build_samples(make_hourly(), horizons=(1,), issue_hour=18)  # 내일은 이미 공표


def test_forecast_at_18_fills_tomorrow_with_published_price():
    hourly = make_hourly(days=60)
    pred = sf.forecast(hourly, "2024-02-10", "2024-02-12", issue_hour=18)
    out = sf.to_output(pred)

    for _, g in out.groupby("issue_date"):
        assert len(g) == 48
        assert g["lead_hour"].tolist() == list(range(6, 54))
    tomorrow = out[out["horizon"] == 1]
    assert (tomorrow["smp_hat"] == tomorrow["smp_actual"]).all()
    assert (tomorrow["smp_p10"] == tomorrow["smp_actual"]).all()  # 공표된 가격은 불확실성 없음
    assert (tomorrow["smp_p90"] == tomorrow["smp_actual"]).all()
    assert (out["issue_time"].dt.hour == 18).all()


def test_price_variants_only_swap_the_price():
    hourly = make_hourly(days=60)
    out = sf.to_output(sf.forecast(hourly, "2024-02-10", "2024-02-11"))

    b1, actual = sf.price_variant(out, "b1"), sf.price_variant(out, "actual")
    assert (b1["smp_hat"] == out["smp_b1"]).all()
    assert (actual["smp_hat"] == out["smp_actual"]).all()
    assert (actual["smp_hat_krw_per_mwh"] == out["smp_actual"] * 1000).all()
    assert (actual["smp_p10"] == actual["smp_hat"]).all() and (b1["smp_p90"] == b1["smp_hat"]).all()
    keep = ["issue_date", "target_time", "lead_hour", "smp_actual"]
    assert b1[keep].equals(out[keep])
    with pytest.raises(ValueError):
        sf.price_variant(out, "oracle")
