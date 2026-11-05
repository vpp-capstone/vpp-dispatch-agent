"""제주 하루전 SMP 예측 — D-1 11시에 내일(D)·모레(D+1) 48시간을 예측한다.

담당: 데이터&수요예측(가격). 최적화(`optimization.run_rolling`)가 48시간 창을 풀고 앞 24시간만
확정하므로, 입찰 마감(D-1 11시)에 D와 D+1 가격이 함께 필요하다.

정보 시점 규칙 (D-1 = 발행일 I, 발행 시각 I 11:00)
    - 하루전 SMP: 날짜 X의 가격은 X-1 18시에 공표 → 발행일 I 자신의 가격까지 사용 가능
    - 육지 SMP: 같은 규칙 → I 까지
    - 수요 실적: I 오전은 일부만 있으므로 I-1 하루 전체까지만
    - KPX 하루전 수요예측(fcst_jeju): 공표 시각 미확인 → 기본 제외 (`use_fcst=True`로만 사용)
    - 기상 지표(발전량 담당 제공): `available_at <= 발행 시각`인 예보 중 가장 최근 것만 사용

모델: 내일·모레 각각 LightGBM(L1 손실) 회귀 + 0원 이하 확률 분류기.
매달 1일 발행분부터 그 시점에 정답이 확정된 자료로 다시 학습한다 (2024년 검증으로 정한 방식).
비교 기준 B1: 발행일에 알고 있는 마지막 날(I)의 같은 시간 가격.

실행:
    python -m vpp.smp_forecast                       # 2024-06-01 ~ 2025-12-31 대상일
    python -m vpp.smp_forecast --weather data/processed/weather/jeju_renewable_weather_features.csv

입력 (`--processed-dir`, 기본 data/processed/market/):
    market_hourly.csv, smp_forecast_hourly.csv   ← vpp.market 결과
출력 (`--out-dir`, 기본 data/processed/forecast/):
    smp_forecast.csv   발행일·대상시각별 예측 (최적화 입력용)
    smp_metrics.csv    내일·모레별 오차 요약
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

KST = "Asia/Seoul"
ISSUE_HOUR = 11
HORIZONS = (1, 2)  # 1 = 내일(D), 2 = 모레(D+1)
TRAIN_START = "2021-01-01"
WEATHER_TOLERANCE = pd.Timedelta(hours=48)  # 이보다 오래된 예보는 쓰지 않는다

BASE_FEATURES = [
    "hour", "dow", "month", "doy_sin", "doy_cos", "is_holiday", "is_offday",
    "p_last", "p_last2", "p_week", "p_last_mean", "p_last_min", "p_last_max",
    "p_last_low_hours", "p_week_mean", "land_last_mean", "demand_prev",
]  # fmt: skip
REG_PARAMS = dict(
    objective="regression_l1", learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_estimators=300,
    verbose=-1, random_state=0,
)  # fmt: skip
CLF_PARAMS = dict(
    learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, n_estimators=300, verbose=-1, random_state=0,
)  # fmt: skip

WEATHER_META = {"issued_at", "available_at", "target_ts", "lead_hours"}
WEATHER_QC_WORDS = ("coverage", "capacity", "quality", "reliab", "confidence", "신뢰", "커버")


# ---------------------------------------------------------------- 입력


def load_market(processed_dir: Path) -> pd.DataFrame:
    """시장 정리 결과를 시간별 표 하나로. ts는 KST 구간 시작(시간대 정보 없음)."""
    market = pd.read_csv(processed_dir / "market_hourly.csv", parse_dates=["ts"])
    land = pd.read_csv(processed_dir / "smp_forecast_hourly.csv", parse_dates=["ts"])
    cols = ["ts", "smp_da", "demand_mwh", "fcst_jeju"]
    return market[cols].merge(land[["ts", "smp_land"]], on="ts", how="left")


def wide(hourly: pd.DataFrame, col: str) -> pd.DataFrame:
    """시간별 값 → 날짜(행) x 시간 0~23(열)."""
    w = hourly.assign(date=hourly["ts"].dt.normalize(), hour=hourly["ts"].dt.hour)
    return w.pivot_table(index="date", columns="hour", values=col).reindex(columns=range(24))


def to_long(table: pd.DataFrame) -> pd.Series:
    """날짜 x 시간 표 → (issue_date, hour) 인덱스의 Series. 빈 값도 남긴다."""
    t = table.copy()
    t.index.name, t.columns.name = "issue_date", "hour"
    return (
        t.reset_index()
        .melt(id_vars="issue_date", var_name="hour")
        .set_index(["issue_date", "hour"])["value"]
    )


def back(table: pd.DataFrame, days: int) -> pd.DataFrame:
    """행 X 에 X-days 날짜의 값을 놓는다 (days 가 음수면 미래 값)."""
    return table.shift(days, freq="D")


# ---------------------------------------------------------------- 특징


def build_samples(hourly: pd.DataFrame, horizons=HORIZONS, use_fcst: bool = False) -> pd.DataFrame:
    """발행일 I 의 11시 기준으로, 대상일 I+h 각 시간의 특징과 정답을 만든다."""
    price, land, demand = (
        wide(hourly, "smp_da"),
        wide(hourly, "smp_land"),
        wide(hourly, "demand_mwh"),
    )
    fcst = wide(hourly, "fcst_jeju")
    daily = pd.DataFrame(
        {
            "p_last_mean": price.mean(axis=1),
            "p_last_min": price.min(axis=1),
            "p_last_max": price.max(axis=1),
            "p_last_low_hours": (price <= 0).sum(axis=1).where(price.notna().all(axis=1)),
            "land_last_mean": land.mean(axis=1),
        }
    )  # 행 = 발행일 I, 값 = I 자신의 하루 요약 (I 11시에 이미 공표됨)

    frames = []
    for h in horizons:
        parts = {
            "y": back(price, -h),  # 정답: I+h 의 가격
            "p_last": price,  # I 같은 시간
            "p_last2": back(price, 1),  # I-1 같은 시간
            "p_week": back(price, 7 - h),  # 대상일 7일 전 = I+h-7 (h<=7 이면 이미 공표)
            "demand_prev": back(demand, 1),  # I-1 수요 실적
        }
        if use_fcst:
            parts["fcst_jeju"] = back(fcst, -h)  # 대상일 수요예측 (공표 시각 확인 후에만)
        long = pd.concat({k: to_long(v) for k, v in parts.items()}, axis=1).reset_index()
        long["horizon"] = h
        long["p_week_mean"] = long["issue_date"].map(back(price, 7 - h).mean(axis=1))
        frames.append(long)

    out = pd.concat(frames, ignore_index=True).merge(
        daily, left_on="issue_date", right_index=True, how="left"
    )
    out["issue_ts"] = out["issue_date"] + pd.Timedelta(hours=ISSUE_HOUR)
    out["target_date"] = out["issue_date"] + pd.to_timedelta(out["horizon"], unit="D")
    out["target_ts"] = out["target_date"] + pd.to_timedelta(out["hour"], unit="h")

    kr = holidays.KR(
        years=range(out["target_date"].dt.year.min(), out["target_date"].dt.year.max() + 1)
    )
    out["dow"] = out["target_date"].dt.dayofweek
    out["month"] = out["target_date"].dt.month
    doy = out["target_date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["is_holiday"] = out["target_date"].dt.date.map(lambda d: d in kr).astype(int)
    out["is_offday"] = ((out["dow"] >= 5) | (out["is_holiday"] == 1)).astype(int)
    return out.sort_values(["issue_date", "horizon", "hour"]).reset_index(drop=True)


def weather_feature_columns(weather: pd.DataFrame) -> list[str]:
    """시각·품질 정보를 뺀 숫자 열만 모델 입력으로 쓴다."""
    cols = []
    for c in weather.columns:
        if c in WEATHER_META or any(w in c.lower() for w in WEATHER_QC_WORDS):
            continue
        if pd.api.types.is_numeric_dtype(weather[c]):
            cols.append(c)
    return cols


def load_weather(path: Path) -> pd.DataFrame:
    """발전량 담당이 넘기는 시간별 기상 지표. 필수 열: available_at, target_ts (KST)."""
    w = pd.read_csv(path)
    for c in ("issued_at", "available_at", "target_ts"):
        if c in w:
            t = pd.to_datetime(w[c])
            w[c] = t.dt.tz_convert(KST).dt.tz_localize(None) if t.dt.tz is not None else t
    return w


def attach_weather(
    samples: pd.DataFrame, weather: pd.DataFrame, tolerance: pd.Timedelta = WEATHER_TOLERANCE
) -> tuple[pd.DataFrame, list[str]]:
    """각 표본에, 발행 시각 이전에 이용 가능했던 예보 중 가장 최근 것을 붙인다."""
    feats = weather_feature_columns(weather)
    right = weather[["available_at", "target_ts", *feats]].dropna(subset=["available_at"])
    right = right.rename(columns={c: f"wx_{c}" for c in feats}).sort_values("available_at")
    left = samples.sort_values("issue_ts")
    merged = pd.merge_asof(
        left, right, left_on="issue_ts", right_on="available_at", by="target_ts",
        direction="backward", tolerance=tolerance, allow_exact_matches=True,
    )  # fmt: skip
    merged = merged.rename(columns={"available_at": "wx_available_at"})
    return merged.sort_values(["issue_date", "horizon", "hour"]).reset_index(drop=True), [
        f"wx_{c}" for c in feats
    ]


# ---------------------------------------------------------------- 학습·예측


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.to_period("M").to_timestamp(), end, freq="MS"))


def walk_forward(
    samples: pd.DataFrame, start: str, end: str, features: list[str], train_start: str = TRAIN_START
) -> pd.DataFrame:
    """내일 대상일이 [start, end]가 되는 발행일(start-1 ~ end-1)마다 48시간을 매달 재학습으로 예측.

    발행일마다 내일·모레가 모두 들어가도록 발행일 기준으로 고른다 (마지막 발행일의 모레는 end+1).
    한 달 동안 쓰는 모델은 그 달 첫 발행 시각(1일 11시)에 정답이 공표된 표본만으로 학습한다.
    즉 학습 표본의 대상일 <= 그 달 1일 (그날 가격은 전날 18시에 공표).
    """
    first = pd.Timestamp(start) - pd.Timedelta(days=1)
    last = pd.Timestamp(end) - pd.Timedelta(days=1)
    pool = samples[samples["target_date"] >= train_start]
    want = pool[(pool["issue_date"] >= first) & (pool["issue_date"] <= last)]
    first_issue = want["issue_date"].min()
    outputs = []
    for month in months_between(first_issue, want["issue_date"].max()):
        nxt = month + pd.offsets.MonthBegin(1)
        for h in sorted(want["horizon"].unique()):
            test = want[(want["horizon"] == h) & (want["issue_date"] >= month)
                        & (want["issue_date"] < nxt)]  # fmt: skip
            if test.empty:
                continue
            train = pool[(pool["horizon"] == h) & (pool["target_date"] <= month)]
            train = train.dropna(subset=["y", "p_last", "p_week"])
            reg = lgb.LGBMRegressor(**REG_PARAMS).fit(train[features], train["y"])
            clf = lgb.LGBMClassifier(**CLF_PARAMS).fit(
                train[features], (train["y"] <= 0).astype(int)
            )
            outputs.append(
                pd.DataFrame(
                    {
                        "issue_date": test["issue_date"],
                        "issued_at": test["issue_ts"],
                        "horizon": h,
                        "target_ts": test["target_ts"],
                        "smp_pred": reg.predict(test[features]),
                        "p_low": clf.predict_proba(test[features])[:, 1],
                        "smp_b1": test["p_last"],
                        "smp_actual": test["y"],
                        "trained_until": month,
                        "train_rows": len(train),
                    }
                )
            )
    return pd.concat(outputs, ignore_index=True)


# ---------------------------------------------------------------- 평가·출력


def metrics(pred: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """대상 시각이 [start, end] 안인 예측만 연도·내일/모레별로 요약한다."""
    rows = []
    scored = pred.dropna(subset=["smp_actual"]).copy()
    day = scored["target_ts"].dt.normalize()
    if start is not None:
        scored = scored[day >= pd.Timestamp(start)]
    if end is not None:
        scored = scored[day.loc[scored.index] <= pd.Timestamp(end)]
    scored["year"] = scored["target_ts"].dt.year
    for (h, year), g in scored.groupby(["horizon", "year"]):
        y, low = g["smp_actual"], g["smp_actual"] <= 0
        for name, p in (("B1", g["smp_b1"]), ("모델", g["smp_pred"])):
            ok = p.notna()
            err = (p[ok] - y[ok]).abs()
            rows.append({
                "대상": "내일" if h == 1 else "모레", "연도": year, "방법": name,
                "시간수": int(ok.sum()), "MAE": round(err.mean(), 2),
                "RMSE": round(float(np.sqrt(((p[ok] - y[ok]) ** 2).mean())), 2),
                "MAE_0원이하제외": round(err[~low[ok]].mean(), 2),
            })  # fmt: skip
        hit = (g["p_low"] >= 0.5) & low
        false = (g["p_low"] >= 0.5) & ~low
        rows.append({
            "대상": "내일" if h == 1 else "모레", "연도": year, "방법": "저가경보(p>=0.5)",
            "시간수": int(low.sum()), "적중": int(hit.sum()), "헛경보": int(false.sum()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def to_output(pred: pd.DataFrame) -> pd.DataFrame:
    """발전량 예측과 같은 이름(issue_time, target_time, lead_hour)으로 내보낸다.

    시각은 시간대 포함 KST. lead_hour 는 발행 시각부터 대상 시각까지의 시간 수
    (내일 00시 = 13, 모레 23시 = 60). 가격은 원/kWh(smp_hat)와 원/MWh 둘 다.
    """
    out = pred.rename(columns={"issued_at": "issue_time", "target_ts": "target_time",
                               "smp_pred": "smp_hat"})  # fmt: skip
    for c in ("issue_time", "target_time"):
        out[c] = out[c].dt.tz_localize(KST)
    lead = (out["target_time"] - out["issue_time"]) / pd.Timedelta(hours=1)
    out["lead_hour"] = lead.astype(int)
    out["issue_date"] = out["issue_date"].dt.date
    out["trained_until"] = out["trained_until"].dt.date
    out["smp_hat_krw_per_mwh"] = out["smp_hat"] * 1000.0
    cols = [
        "issue_date", "issue_time", "horizon", "lead_hour", "target_time", "smp_hat",
        "smp_hat_krw_per_mwh", "p_low", "smp_b1", "smp_actual", "trained_until", "train_rows",
    ]  # fmt: skip
    return out[cols].round({"smp_hat": 3, "smp_hat_krw_per_mwh": 1, "p_low": 4})


def window_for_issue(forecast: pd.DataFrame, issue_date) -> pd.DataFrame:
    """한 발행일의 48시간 창 → optimization.solve_window 에 넣을 (ts, price_krw_per_mwh)."""
    day = pd.Timestamp(issue_date).date()
    sel = forecast[pd.to_datetime(forecast["issue_date"]).dt.date == day].sort_values("target_time")
    return sel.rename(columns={"target_time": "ts", "smp_hat_krw_per_mwh": "price_krw_per_mwh"})[
        ["ts", "price_krw_per_mwh", "horizon", "lead_hour"]
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/market"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/forecast"))
    parser.add_argument("--start", default="2024-06-01", help="첫 대상일")
    parser.add_argument("--end", default="2025-12-31", help="마지막 대상일")
    parser.add_argument("--weather", type=Path, default=None, help="기상 지표 CSV (선택)")
    parser.add_argument(
        "--use-fcst",
        action="store_true",
        help="KPX 하루전 수요예측 사용 (공표 시각이 11시 전으로 확인된 경우만)",
    )
    args = parser.parse_args()

    samples = build_samples(load_market(args.processed_dir), use_fcst=args.use_fcst)
    features = BASE_FEATURES + (["fcst_jeju"] if args.use_fcst else [])
    if args.weather is not None:
        samples, wx = attach_weather(samples, load_weather(args.weather))
        features += wx
        print(f"기상 지표 {len(wx)}개 사용: {wx}")

    pred = walk_forward(samples, args.start, args.end, features)
    table = metrics(pred, args.start, args.end)
    print(table.to_string(index=False))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    to_output(pred).to_csv(args.out_dir / "smp_forecast.csv", index=False, encoding="utf-8-sig")
    table.to_csv(args.out_dir / "smp_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {args.out_dir / 'smp_forecast.csv'} ({len(pred)}행)")


if __name__ == "__main__":
    main()
