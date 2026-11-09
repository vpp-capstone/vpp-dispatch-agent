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
    python -m vpp.smp_forecast --variants            # + 비교용 가격 입력(전날 가격, 실제 가격)
    python -m vpp.smp_forecast --issue-hour 18       # 18시 재계획용 (내일 = 실제, 모레 = 재예측)
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
# 지난 가격을 더 잘 요약하는 특징 (2024년 선택 구간에서 효과 확인 후 채택)
#   p_med3·p_med7: 최근 3·7일 같은 시간 중앙값 (튀는 날에 덜 흔들림)
#   p_sametype   : 대상일과 같은 요일 유형(평일/토/일·공휴일) 중 가장 최근 날의 같은 시간 가격
#   land_last_h·land_last2_h: 육지 SMP의 최근 이틀 같은 시간 값
EXTRA_FEATURES = ["p_med3", "p_med7", "p_sametype", "land_last_h", "land_last2_h"]
FEATURES = BASE_FEATURES + EXTRA_FEATURES
ISSUE_HOURS = (11, 18)  # 11시 입찰 마감 / 18시 하루전 가격 공표 후 재계획
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


def day_type(d: pd.Timestamp, kr) -> str:
    """평일 / 토요일 / 일요일·공휴일."""
    if d.date() in kr or d.day_of_week == 6:
        return "sun"
    return "sat" if d.day_of_week == 5 else "wd"


def same_type_tables(price: pd.DataFrame, kr) -> dict[str, pd.DataFrame]:
    """유형별 표: 행 L = 'L 까지 공표된 날 중 그 유형의 가장 최근 날'의 24시간 가격."""
    types = pd.Series([day_type(d, kr) for d in price.index], index=price.index)
    out = {}
    for t in ("wd", "sat", "sun"):
        rows = price.where(types.eq(t), axis=0)
        out[t] = rows.ffill()
    return out


def build_samples(
    hourly: pd.DataFrame,
    horizons=HORIZONS,
    use_fcst: bool = False,
    issue_hour: int = ISSUE_HOUR,
) -> pd.DataFrame:
    """발행일 I 의 issue_hour 기준으로, 대상일 I+h 각 시간의 특징과 정답을 만든다.

    11시 발행: 하루전 가격은 I 까지 공표. 18시 발행: I+1 가격이 막 공표되어 I+1 까지 사용.
    수요 실적은 두 경우 모두 I-1 하루 전체까지만 쓴다 (I 는 아직 하루가 끝나지 않음).
    """
    if issue_hour not in ISSUE_HOURS:
        raise ValueError(f"issue_hour 는 {ISSUE_HOURS} 중 하나: {issue_hour}")
    known = 1 if issue_hour >= 18 else 0  # 가격이 공표된 마지막 날 L = I + known
    price, land, demand = (
        wide(hourly, "smp_da"),
        wide(hourly, "smp_land"),
        wide(hourly, "demand_mwh"),
    )
    fcst = wide(hourly, "fcst_jeju")
    years = range(price.index.year.min(), price.index.year.max() + 2)
    kr = holidays.KR(years=years)
    daily = pd.DataFrame(
        {
            "p_last_mean": price.mean(axis=1),
            "p_last_min": price.min(axis=1),
            "p_last_max": price.max(axis=1),
            "p_last_low_hours": (price <= 0).sum(axis=1).where(price.notna().all(axis=1)),
            "land_last_mean": land.mean(axis=1),
        }
    ).shift(-known, freq="D")  # 행 = 발행일 I, 값 = 마지막 공표일 L 의 하루 요약
    med3 = pd.concat([back(price, k) for k in range(3)]).groupby(level=0).median()
    med7 = pd.concat([back(price, k) for k in range(7)]).groupby(level=0).median()
    by_type = same_type_tables(price, kr)

    frames = []
    for h in horizons:
        if h <= known:
            raise ValueError(f"{issue_hour}시 발행에서 {h}일 뒤 가격은 이미 공표됨")
        parts = {
            "y": back(price, -h),  # 정답: I+h 의 가격
            "p_last": back(price, -known),  # L 같은 시간 (B1)
            "p_last2": back(price, 1 - known),  # L-1 같은 시간
            "p_week": back(price, 7 - h),  # 대상일 7일 전 = I+h-7 (이미 공표)
            "demand_prev": back(demand, 1),  # I-1 수요 실적
            "p_med3": back(med3, -known),
            "p_med7": back(med7, -known),
            "land_last_h": back(land, -known),
            "land_last2_h": back(land, 1 - known),
        }
        for t, table in by_type.items():
            parts[f"_type_{t}"] = back(table, -known)
        if use_fcst:
            parts["fcst_jeju"] = back(fcst, -h)  # 대상일 수요예측 (공표 시각 확인 후에만)
        long = pd.concat({k: to_long(v) for k, v in parts.items()}, axis=1).reset_index()
        long["horizon"] = h
        long["p_week_mean"] = long["issue_date"].map(back(price, 7 - h).mean(axis=1))
        target_type = (long["issue_date"] + pd.Timedelta(days=h)).map(lambda d: day_type(d, kr))
        long["p_sametype"] = np.select(
            [target_type.eq(t) for t in ("wd", "sat", "sun")],
            [long[f"_type_{t}"] for t in ("wd", "sat", "sun")],
            default=np.nan,
        )
        frames.append(long.drop(columns=[f"_type_{t}" for t in by_type]))

    out = pd.concat(frames, ignore_index=True).merge(
        daily, left_on="issue_date", right_index=True, how="left"
    )
    out["issue_ts"] = out["issue_date"] + pd.Timedelta(hours=issue_hour)
    out["target_date"] = out["issue_date"] + pd.to_timedelta(out["horizon"], unit="D")
    out["target_ts"] = out["target_date"] + pd.to_timedelta(out["hour"], unit="h")

    out["dow"] = out["target_date"].dt.day_of_week
    out["month"] = out["target_date"].dt.month
    doy = out["target_date"].dt.day_of_year
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
    (11시 발행: 내일 00시 = 13 ~ 모레 23시 = 60, 18시 발행: 6 ~ 53).
    가격은 원/kWh(smp_hat)와 원/MWh 둘 다.
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


PRICE_SOURCES = ("model", "b1", "actual")


def price_variant(forecast: pd.DataFrame, source: str) -> pd.DataFrame:
    """같은 형식에서 smp_hat 만 바꾼 비교용 가격 입력.

    model  = 모델 예측 그대로
    b1     = 발행 시점에 알던 마지막 날의 같은 시간 가격 (단순 기준)
    actual = 실제 하루전 가격 (완전예측 상한, 운영에서는 알 수 없음)
    """
    if source not in PRICE_SOURCES:
        raise ValueError(f"source 는 {PRICE_SOURCES} 중 하나: {source}")
    out = forecast.copy()
    if source != "model":
        out["smp_hat"] = out["smp_b1" if source == "b1" else "smp_actual"]
        out["smp_hat_krw_per_mwh"] = (out["smp_hat"] * 1000.0).round(1)
    out["price_source"] = source
    return out


def window_for_issue(forecast: pd.DataFrame, issue_date) -> pd.DataFrame:
    """한 발행일의 48시간 창 → optimization.solve_window 에 넣을 (ts, price_krw_per_mwh)."""
    day = pd.Timestamp(issue_date).date()
    sel = forecast[pd.to_datetime(forecast["issue_date"]).dt.date == day].sort_values("target_time")
    return sel.rename(columns={"target_time": "ts", "smp_hat_krw_per_mwh": "price_krw_per_mwh"})[
        ["ts", "price_krw_per_mwh", "horizon", "lead_hour"]
    ].reset_index(drop=True)


def published_rows(hourly: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """18시 발행용: 발행일 I 의 다음 날(I+1) 가격은 이미 공표됐으므로 실제 값을 그대로 채운다."""
    price = hourly.set_index("ts")["smp_da"]
    issues = pred[["issue_date", "issued_at", "trained_until", "train_rows"]].drop_duplicates(
        "issue_date"
    )
    rows = []
    for r in issues.itertuples(index=False):
        ts = pd.date_range(r.issue_date + pd.Timedelta(days=1), periods=24, freq="h")
        p = price.reindex(ts).to_numpy()
        rows.append(pd.DataFrame({
            "issue_date": r.issue_date, "issued_at": r.issued_at, "horizon": 1, "target_ts": ts,
            "smp_pred": p, "p_low": (p <= 0).astype(float), "smp_b1": p, "smp_actual": p,
            "trained_until": r.trained_until, "train_rows": r.train_rows,
        }))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def forecast(
    hourly: pd.DataFrame,
    start: str,
    end: str,
    issue_hour: int = ISSUE_HOUR,
    features: list[str] | None = None,
    use_fcst: bool = False,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """발행일마다 48시간 예측. 18시 발행은 내일 = 공표된 실제 가격, 모레 = 재예측."""
    horizons = HORIZONS if issue_hour == ISSUE_HOUR else (2,)
    samples = build_samples(hourly, horizons=horizons, use_fcst=use_fcst, issue_hour=issue_hour)
    feats = list(features or FEATURES) + (["fcst_jeju"] if use_fcst else [])
    if weather is not None:
        samples, wx = attach_weather(samples, weather)
        feats += wx
    pred = walk_forward(samples, start, end, feats)
    if issue_hour != ISSUE_HOUR:
        pred = pd.concat([published_rows(hourly, pred), pred], ignore_index=True)
        pred = pred.sort_values(["issue_date", "horizon", "target_ts"]).reset_index(drop=True)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/market"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/forecast"))
    parser.add_argument("--start", default="2024-06-01", help="첫 대상일")
    parser.add_argument("--end", default="2025-12-31", help="마지막 대상일")
    parser.add_argument(
        "--issue-hour", type=int, default=ISSUE_HOUR, choices=ISSUE_HOURS,
        help="11 = 입찰 마감 예측, 18 = 하루전 가격 공표 후 재계획용 (내일은 실제, 모레만 예측)",
    )  # fmt: skip
    parser.add_argument("--base-features", action="store_true", help="새 특징 없이 기존 특징만")
    parser.add_argument(
        "--variants", action="store_true",
        help="비교용 가격 입력(전날 가격 b1, 실제 가격 actual)도 같은 형식으로 저장",
    )  # fmt: skip
    parser.add_argument("--weather", type=Path, default=None, help="기상 지표 CSV (선택)")
    parser.add_argument(
        "--use-fcst",
        action="store_true",
        help="KPX 하루전 수요예측 사용 (공표 시각이 11시 전으로 확인된 경우만)",
    )
    args = parser.parse_args()

    weather = load_weather(args.weather) if args.weather is not None else None
    features = BASE_FEATURES if args.base_features else FEATURES
    pred = forecast(
        load_market(args.processed_dir), args.start, args.end, issue_hour=args.issue_hour,
        features=features, use_fcst=args.use_fcst, weather=weather,
    )  # fmt: skip
    predicted = pred if args.issue_hour == ISSUE_HOUR else pred[pred["horizon"] == 2]
    table = metrics(predicted, args.start, args.end)  # 18시의 내일은 공표 값이라 평가에서 뺀다
    print(table.to_string(index=False))

    suffix = "" if args.issue_hour == ISSUE_HOUR else f"_{args.issue_hour}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = to_output(pred)
    out.to_csv(args.out_dir / f"smp_forecast{suffix}.csv", index=False, encoding="utf-8-sig")
    table.to_csv(args.out_dir / f"smp_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {args.out_dir / f'smp_forecast{suffix}.csv'} ({len(out)}행)")
    if args.variants:
        for src in ("b1", "actual"):
            path = args.out_dir / f"smp_forecast{suffix}_{src}.csv"
            price_variant(out, src).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"저장: {path}")


if __name__ == "__main__":
    main()
