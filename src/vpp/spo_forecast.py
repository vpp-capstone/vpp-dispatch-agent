"""SMP 예측을 '정확도'가 아니라 'ESS 운영 수익' 기준으로 학습한다 (SPO+).

기존 SMP 예측(`smp_forecast.py`, v6)은 평균 절대오차를 줄이도록 학습한다. 이렇게 학습하면 예측이
무난한 평균값 쪽으로 몰려 하루 안의 가격 차(낮 저가·저녁 고가, 0원 이하 시간)가 지워지고,
최적화가 ESS 차익을 작게 보고 ESS를 덜 쓴다.

여기서는 같은 LightGBM·같은 특징·같은 매달 재학습 규칙을 그대로 두고, 학습 손실만
SPO+(Elmachtoub & Grigas, "Smart Predict, then Optimize", Management Science 68(1), 2022)로 바꾼다.

- 학습 중 대상일마다 24시간 ESS 운영 LP(태양광·낮 판매 상한·SOC·효율·열화비)를 푼다.
- SPO+ 기울기 = 2 * (실제 가격으로 세운 판매 계획 - (2*예측 - 실제) 가격으로 세운 판매 계획).
  두 계획이 같으면 예측이 틀려도 판단은 그대로라 고치지 않고, 다르면 판단을 바꾸는 오차만 고친다.
- 정확도가 무너지지 않게 L2 앵커(anchor * (예측 - 실제))를 약하게 더한다.
- 시작점은 같은 학습 구간의 시간순 블록 OOF L1 예측이고, 그 위에 SPO+ 로 rounds 번 더 학습한다.

출력은 v6 `smp_forecast.csv` 와 같은 형식이다. 점예측(smp_hat, smp_hat_krw_per_mwh)만 바꾸고
분위수·확률·실측 열은 기준 파일 그대로 둔다.
그래서 `distribution_backtest --forecast` 에 바로 넣는다.

    python -m vpp.spo_forecast --base data/processed/forecast/smp_forecast.csv \
        --generation artifacts/2025/evaluation/solar_improved_2025_with_actual.parquet \
        --out data/processed/forecast/smp_forecast_spo_2025.csv

결과와 해석: docs/results/spo_forecast_20261117.md
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

from vpp import smp_forecast as sf

HOURS = 24


@dataclass(frozen=True)
class DayLP:
    """학습 손실 안에서 푸는 하루 24시간 ESS 운영 LP (판매 수익 - 열화비 최대화).

    변수 x = [판매 s, 충전 c, 방전 d, 출력제한 k, SOC e] 각 24시간.
    가격·단가 단위는 원/kWh, 전력량은 MWh.
    태양광 연계형으로 보고 계통 충전은 없다. 하루 끝 SOC 는 시작 SOC 이상으로 둔다.
    기본값은 대표 조건(국내 구축비 예시 2MW/8MWh, 낮 8~18시 판매 상한 2MW)이다.
    """

    power_mw: float = 2.0
    energy_mwh: float = 8.0
    efficiency: float = 0.95
    soc_min: float = 0.1
    soc_max: float = 0.9
    cap_mw: float = 2.0
    cap_start: int = 8
    cap_end: int = 18
    # 원/kWh. project_economics_v0.yaml 국내 구축비 예시 8MWh 의 throughput_cost_krw_per_mwh / 1000
    throughput_cost: float = 20.8349295
    rec: float = 70.0  # 원/kWh, REC 단가 x 가중치 근사 (학습용 LP 에서만 사용)

    def template(self, solar: np.ndarray) -> tuple:
        n = HOURS
        eye = sp.identity(n, format="csr")
        zero = sp.csr_matrix((n, n))
        balance = sp.hstack([eye, eye, -eye, eye, zero])  # s + c - d + k = 태양광
        diff = sp.identity(n, format="csr") - sp.eye(n, k=-1, format="csr")
        soc = sp.hstack([zero, -self.efficiency * eye, (1 / self.efficiency) * eye, zero, diff])
        a_eq = sp.vstack([balance, soc]).tocsr()
        e0 = 0.5 * self.energy_mwh
        b_eq = np.concatenate([np.asarray(solar, float), np.r_[e0, np.zeros(n - 1)]])
        hours = np.arange(n)
        cap = np.where((hours >= self.cap_start) & (hours < self.cap_end), self.cap_mw, None)
        lo, hi = self.soc_min * self.energy_mwh, self.soc_max * self.energy_mwh
        bounds = (
            [(0, u) for u in cap]
            + [(0, self.power_mw)] * n
            + [(0, self.power_mw)] * n
            + [(0, None)] * n
            + [(lo, hi)] * (n - 1)
            + [(e0, hi)]
        )
        return a_eq, b_eq, bounds

    def theta(self, price: np.ndarray) -> np.ndarray:
        """최소화 계수: 판매 -(가격 + REC), 충·방전 +열화비."""
        th = np.zeros(5 * HOURS)
        th[:HOURS] = -(np.asarray(price, float) + self.rec)
        th[HOURS : 2 * HOURS] = self.throughput_cost
        th[2 * HOURS : 3 * HOURS] = self.throughput_cost
        return th

    def sell(self, template: tuple, price: np.ndarray) -> np.ndarray:
        a_eq, b_eq, bounds = template
        r = linprog(self.theta(price), A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs-ds")
        if r.status != 0:
            raise RuntimeError(f"하루 LP 실패: {r.message}")
        return r.x[:HOURS]


DEFAULT_LP = DayLP()


def spo_plus_grad(
    lp: DayLP, template: tuple, price_hat: np.ndarray, price: np.ndarray, sell_true=None
) -> np.ndarray:
    """예측 가격에 대한 SPO+ 기울기 (원/kWh 단위 예측 기준).

    판매 계수는 -(가격 + REC) 이므로 계수 기울기 2(w*(c) - w*(2ĉ - c)) 에 -1 을 곱한다.
    """
    if sell_true is None:
        sell_true = lp.sell(template, price)
    sell_tilde = lp.sell(template, 2 * np.asarray(price_hat, float) - np.asarray(price, float))
    return -2.0 * (sell_true - sell_tilde)


def solar_profiles(solar: pd.Series, days) -> dict:
    """학습 대상일의 태양광 24시간 패턴.

    실측이 있는 해의 같은 월·일·시 값을 빌려 쓴다 (2월 29일은 28일 값).
    """
    lookup = {(t.month, t.day, t.hour): v for t, v in solar.dropna().items()}
    out = {}
    for d in days:
        day = min(d.day, 28) if d.month == 2 else d.day
        row = [lookup.get((d.month, day, h), 0.0) for h in range(HOURS)]
        out[d] = np.nan_to_num(np.asarray(row, float))
    return out


def daily_samples(hourly: pd.DataFrame, weather: pd.DataFrame | None, horizon: int):
    """대상일 x 24시간이 모두 있는 학습 표본만 남긴다 (v6 와 같은 특징)."""
    smp = sf.build_samples(hourly, horizons=(horizon,), use_fcst=False, issue_hour=sf.ISSUE_HOUR)
    feats = list(sf.FEATURES)
    if weather is not None:
        smp, wcols = sf.attach_weather(smp, weather)
        feats += wcols
    smp = smp[smp["target_date"] >= sf.TRAIN_START].dropna(subset=["y", "p_last", "p_week"])
    full = smp.groupby("target_date")["hour"].transform("count") == HOURS
    return smp[full].sort_values(["target_date", "hour"]).reset_index(drop=True), feats


def _oof_l1(train: pd.DataFrame, feats: list[str], folds: int) -> np.ndarray:
    """학습 구간 안 시간순 블록 OOF L1 예측 (SPO+ 시작점, 학습 표본 과적합 방지)."""
    days = np.array(sorted(train["target_date"].unique()))
    oof = np.full(len(train), np.nan)
    for block in np.array_split(days, folds):
        hold = train["target_date"].isin(block).to_numpy()
        m = lgb.LGBMRegressor(**sf.REG_PARAMS).fit(train.loc[~hold, feats], train.loc[~hold, "y"])
        oof[hold] = m.predict(train.loc[hold, feats])
    return oof


def fit_spo(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feats: list[str],
    solar: pd.Series,
    lp: DayLP = DEFAULT_LP,
    anchor: float = 0.1,
    rounds: int = 60,
    learning_rate: float = 0.05,
    folds: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """(L1 예측, SPO+ 예측) — 같은 특징·같은 학습 구간. train·test 는 대상일 x 24행 정렬."""
    l1 = lgb.LGBMRegressor(**sf.REG_PARAMS).fit(train[feats], train["y"])
    pred_l1 = l1.predict(test[feats])
    days = list(pd.to_datetime(train["target_date"].unique()))
    profiles = solar_profiles(solar, days)
    templates = [lp.template(profiles[d]) for d in days]
    y = train["y"].to_numpy(float).reshape(-1, HOURS)
    sell_true = np.stack([lp.sell(tm, y[i]) for i, tm in enumerate(templates)])

    def objective(preds, _data):
        p = preds.reshape(-1, HOURS)
        g = np.stack(
            [spo_plus_grad(lp, tm, p[i], y[i], sell_true[i]) for i, tm in enumerate(templates)]
        )
        grad = g.ravel() + anchor * (preds - y.ravel())
        return grad, np.full_like(preds, anchor + 1e-3)

    params = {k: v for k, v in sf.REG_PARAMS.items() if k not in ("objective", "n_estimators")}
    params.update(objective=objective, learning_rate=learning_rate, verbose=-1)
    data = lgb.Dataset(
        train[feats], label=train["y"], init_score=_oof_l1(train, feats, folds), free_raw_data=False
    )
    booster = lgb.train(params, data, num_boost_round=rounds)
    return pred_l1, pred_l1 + booster.predict(test[feats], raw_score=True)


def walk_forward_spo(
    hourly: pd.DataFrame,
    weather: pd.DataFrame | None,
    solar: pd.Series,
    months: list[str],
    lp: DayLP = DEFAULT_LP,
    anchor: float = 0.1,
    rounds: int = 60,
    learning_rate: float = 0.05,
    log=print,
) -> pd.DataFrame:
    """v6 와 같은 매달 재학습. 그 달(첫 달은 전날 발행분 포함) 발행분을 예측한다."""
    rows = []
    for h in sf.HORIZONS:
        smp, feats = daily_samples(hourly, weather, h)
        for i, m in enumerate(pd.PeriodIndex(months, freq="M")):
            start = m.start_time - (pd.Timedelta(days=1) if i == 0 else pd.Timedelta(0))
            nxt = (m + 1).start_time
            train = smp[smp["target_date"] <= start].reset_index(drop=True)
            test = smp[(smp["issue_date"] >= start) & (smp["issue_date"] < nxt)]
            test = test.reset_index(drop=True)
            if test.empty:
                continue
            t0 = time.time()
            p_l1, p_spo = fit_spo(train, test, feats, solar, lp, anchor, rounds, learning_rate)
            rows.append(
                pd.DataFrame(
                    {
                        "issue_date": test["issue_date"].dt.date.astype(str),
                        "horizon": h,
                        "target_ts": test["target_ts"],
                        "smp_l1": p_l1,
                        "smp_spo": p_spo,
                    }
                )
            )
            log(f"h{h} {m}: {time.time() - t0:.0f}s")
    return pd.concat(rows, ignore_index=True)


def replace_point(base: pd.DataFrame, preds: pd.DataFrame, column: str = "smp_spo") -> pd.DataFrame:
    """v6 형식 파일의 점예측만 바꾼다. 맞는 행이 없으면 기준 값을 그대로 둔다."""
    target = pd.to_datetime(base["target_time"].astype(str).str[:19])
    key = (
        base["issue_date"].astype(str)
        + "|"
        + base["horizon"].astype(str)
        + "|"
        + target.astype(str)
    )
    pkey = (
        preds["issue_date"].astype(str)
        + "|"
        + preds["horizon"].astype(str)
        + "|"
        + pd.to_datetime(preds["target_ts"]).astype(str)
    )
    new = key.map(dict(zip(pkey, preds[column], strict=True)))
    out = base.copy()
    hit = new.notna()
    out.loc[hit, "smp_hat"] = new[hit].round(3)
    out.loc[hit, "smp_hat_krw_per_mwh"] = (new[hit] * 1000).round(1)
    return out


def main(argv=None) -> None:
    from vpp import distribution_backtest as db

    ap = argparse.ArgumentParser(description="SPO+ 로 SMP 점예측을 운영 수익 기준으로 다시 학습")
    ap.add_argument("--base", required=True, help="v6 형식 smp_forecast.csv (분위수·확률 열 유지)")
    ap.add_argument("--generation", required=True, help="태양광 실측이 든 발전량 예측 평가 parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--market-dir", default="data/processed/market")
    ap.add_argument(
        "--weather", default="data/processed/weather/jeju_renewable_weather_features.csv"
    )
    ap.add_argument("--start", default="2025-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--solar-mw", type=float, default=5.7)
    ap.add_argument("--anchor", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    a = ap.parse_args(argv)

    hourly = sf.load_market(Path(a.market_dir))
    weather = sf.load_weather(Path(a.weather)) if a.weather else None
    solar = db.load_generation(a.generation, a.solar_mw)[0]
    months = [str(p) for p in pd.period_range(a.start, a.end, freq="M")]
    preds = walk_forward_spo(
        hourly, weather, solar, months, anchor=a.anchor, rounds=a.rounds,
        learning_rate=a.learning_rate,
    )  # fmt: skip
    base = pd.read_csv(a.base)
    out = replace_point(base, preds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    preds.to_csv(Path(a.out).with_name(Path(a.out).stem + "_raw.csv"), index=False)
    print(f"저장: {a.out} (교체 {int((out['smp_hat'] != base['smp_hat']).sum())}행)")


if __name__ == "__main__":
    main()
