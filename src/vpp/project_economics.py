"""배전망 ESS 사업 참여 판단: 20년 현금흐름과 투자 지표.

운영 백테스트(48시간 롤링 MILP를 1년 돌린 결과)가 1년치 시장수익·태양광 전력량·ESS 충방전량을
주면, 이 모듈이 공고 조건(구축비 지원, 20년 의무 가동)과 비용 가정(O&M, 보험·안전검사,
배터리·PCS 교체)을 붙여 20년 현금흐름을 만들고 다음 지표를 낸다.

- NPV, IRR, 단순·할인 회수기간
- 손익분기 정부지원율, 손익분기 구축비, 필요한 최소 연간 시장수익, 손익분기 PPA 단가,
  NPV를 0으로 만드는 연간 보상 필요액(비증설대안 NWAs 보상 등)
  (VPP가 태양광 소유자에게 kWh당 줄 수 있는 최대 단가. PPA 0원이어도 손해면 nan)
- 배터리 교체 시점

20년치 시간별 운전을 다시 풀지 않는다. 1년 백테스트 결과에 연 변화율(보수·기준·낙관)을 곱한다.

운영 결과 CSV (한 행 = 선로 조건 x 배터리 x 운전 방식, 1년):

| 열 | 뜻 |
| --- | --- |
| case | 선로·사업 조건 이름 (예: cap_1.7MW) |
| battery | 설정 파일 batteries 의 키 |
| operation | 운전 방식 (예: forecast, perfect) |
| market_revenue_krw | 태양광+ESS 1년 시장수익 (SMP+REC 판매, 용량정산금, 편차 반영) |
|  | 열화비는 빼지 않는다 |
| solar_to_vpp_mwh | VPP가 태양광에서 받은 전력량(바로 판매 + ESS 충전). PPA 지급 기준 |
| ess_charge_mwh, ess_discharge_mwh | 1년 충방전량. 배터리 수명(교체 시점) 계산에 쓴다 |
| days | 백테스트 일수 (없으면 365). 365가 아니면 1년으로 환산 |

선택 열 (항목별 수익, 1년 기준 원). 다섯 열이 모두 있으면 공고 표준 수익 배분(``standard``)도
계산한다.

| 열 | 뜻 |
| --- | --- |
| solar_direct_revenue_krw | 태양광을 바로 판 SMP+REC 수익 |
| ess_discharge_revenue_krw | ESS 방전분을 판 SMP+REC 수익 |
| solar_capacity_payment_krw | 태양광 용량정산금 |
| ess_capacity_payment_krw | ESS 용량정산금 = min(방전량, 실효용량) x RCP. |
|  | market_revenue_krw 에 넣지 않는다 |
| imbalance_penalty_krw | 임밸런스 페널티 (양수 = 낸 돈) |

``ess_power_mw``, ``ess_energy_mwh`` 열이 있으면 그 행은 그 ESS 용량으로 구축비·지원금을 계산한다
(용량 비교용, 공고 상한 4MW/20MWh 이내).

시간별 운전 결과(``RollingResult.hourly``)에서 이 행을 만들 때는
``operating_row_from_hourly`` 를 쓴다.

ESS 없이 같은 선로 상한 안에서 태양광만 판 기준선은 ``battery`` 를 ``none`` 으로 둔 행으로 넣는다
(case·operation 이 같은 행끼리 비교). 기준선이 있으면 두 관점을 함께 낸다.

- ``incremental`` (참여 판단 기준): ESS를 넣어서 늘어난 시장수익과 태양광 수전량만 본다.
  VPP 몫 = 시장수익 증가분 - PPA 단가 x 태양광 수전 증가분 - ESS 비용.
  태양광 소유자 몫 = PPA 단가 x 태양광 수전 증가분 (출력제어로 버리던 전력을 팔게 된 몫).
- ``total``: 태양광+ESS 묶음 전체 (PPA 모드면 모든 태양광 수전량에 PPA 지급).
  공고 대상 태양광은 접속대기라 ESS 없이는 접속하지 못한다. 그래서 공고 조건의 참여 판단은
  이 관점(또는 standard)으로 하고, incremental 은 ESS 없이도 접속할 수 있는 경우(유연접속)용이다.
- ``standard``: 2차 공고 표준 수익 배분 예시대로 항목별 수익을 VPP와 발전사업자가 나눈다.
  VPP 몫 = ESS 방전 수익 + 태양광 용량정산금 50% (+ ESS 용량정산금) - 페널티. PPA 없음.

ESS 용량정산금은 인정 여부가 불확실해서 ``ess_capacity_payment_scenarios`` 로 넣은 경우와 뺀 경우를
모두 계산한다 (결과 열 ``ess_capacity_payment``).

열화비는 배터리 구입비를 처리량에 나눠 붙인 값이라, 구입비를 따로 넣는 여기서는 빼지 않는다
(두 번 빼지 않기 위해). 운전 판단에 쓸 열화비는 ``Battery.throughput_cost_krw_per_mwh``로 만든다.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OPERATING_COLUMNS = [
    "case",
    "battery",
    "operation",
    "market_revenue_krw",
    "solar_to_vpp_mwh",
    "ess_charge_mwh",
    "ess_discharge_mwh",
]
BREAKDOWN_COLUMNS = [
    "solar_direct_revenue_krw",
    "ess_discharge_revenue_krw",
    "solar_capacity_payment_krw",
    "ess_capacity_payment_krw",
    "imbalance_penalty_krw",
]
# standard_vpp_shares 의 키 -> 운영 결과 열
STANDARD_ITEMS = {
    "solar_energy": "solar_direct_revenue_krw",
    "ess_energy": "ess_discharge_revenue_krw",
    "solar_capacity_payment": "solar_capacity_payment_krw",
    "ess_capacity_payment": "ess_capacity_payment_krw",
    "imbalance_penalty": "imbalance_penalty_krw",
}


@dataclass(frozen=True)
class Battery:
    key: str
    label: str
    usd_per_kwh: float
    battery_share: float
    soh: float
    efficiency: float
    cycle_life: float
    calendar_life_years: float
    dod: float = 0.8
    source: str = ""

    def capex_krw(self, energy_mwh: float, fx: float) -> float:
        """시스템 구축비 (설치 용량 기준)."""
        return self.usd_per_kwh * fx * energy_mwh * 1000.0

    def usable_mwh(self, energy_mwh: float) -> float:
        return energy_mwh * self.soh

    def lifetime_throughput_mwh(self, energy_mwh: float) -> float:
        """수명 동안 처리량(충전+방전). degradation_stage1 과 같은 정의: 2 x 횟수 x 용량 x DoD."""
        return 2.0 * self.cycle_life * self.usable_mwh(energy_mwh) * self.dod

    def throughput_cost_krw_per_mwh(self, energy_mwh: float, fx: float) -> float:
        """운전 판단(MILP)에 쓸 열화비: 배터리 몫 구입비 ÷ 수명 처리량."""
        return (
            self.capex_krw(energy_mwh, fx)
            * self.battery_share
            / self.lifetime_throughput_mwh(energy_mwh)
        )

    def life_years(self, energy_mwh: float, annual_throughput_mwh: float) -> float:
        """처리량 기준 수명과 달력 수명 중 짧은 쪽 (년)."""
        if annual_throughput_mwh <= 0:
            return float(self.calendar_life_years)
        return min(
            float(self.calendar_life_years),
            self.lifetime_throughput_mwh(energy_mwh) / annual_throughput_mwh,
        )


@dataclass(frozen=True)
class EconomicsConfig:
    ess_power_mw: float
    ess_energy_mwh: float
    solar_mw: float
    project_years: int
    subsidy_rate: float
    subsidy_cap_krw_per_mwh: float | None
    fx: float
    discount_rate: float
    cost_escalation: float
    om_rate: float
    insurance_rate: float
    battery_price_decline: float
    pcs_life_years: float
    pcs_share: float
    revenue_scenarios: dict[str, float]
    split_mode: str
    ppa_prices: tuple[float, ...]
    solar_capex_krw: float | None
    batteries: dict[str, Battery] = field(default_factory=dict)
    standard_vpp_shares: dict[str, float] = field(default_factory=dict)
    ess_cp_scenarios: tuple[bool, ...] = (False,)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EconomicsConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        b, f, s = raw["business"], raw["finance"], raw["revenue_split"]
        ppa = s.get("ppa_krw_per_kwh") or []
        cfg = cls(
            ess_power_mw=float(b["ess_power_mw"]),
            ess_energy_mwh=float(b["ess_energy_mwh"]),
            solar_mw=float(b["solar_mw"]),
            project_years=int(b["project_years"]),
            subsidy_rate=float(b["subsidy_rate"]),
            subsidy_cap_krw_per_mwh=_opt_float(b.get("subsidy_cap_krw_per_mwh")),
            fx=float(f["fx_krw_per_usd"]),
            discount_rate=float(f["discount_rate"]),
            cost_escalation=float(f["cost_escalation"]),
            om_rate=float(f["om_rate_of_capex"]),
            insurance_rate=float(f["insurance_inspection_rate_of_capex"]),
            battery_price_decline=float(f["battery_price_decline"]),
            pcs_life_years=float(f["pcs_life_years"]),
            pcs_share=float(f["pcs_share_of_capex"]),
            revenue_scenarios={k: float(v) for k, v in raw["revenue_scenarios"].items()},
            split_mode=str(s["mode"]),
            ppa_prices=tuple(float(p) for p in (ppa if isinstance(ppa, list) else [ppa])),
            solar_capex_krw=_opt_float(s.get("solar_capex_krw")),
            batteries={k: Battery(key=k, **v) for k, v in raw["batteries"].items()},
            standard_vpp_shares={
                k: float(v) for k, v in (s.get("standard_vpp_shares") or {}).items()
            },
            ess_cp_scenarios=tuple(
                bool(x) for x in raw.get("ess_capacity_payment_scenarios") or [False]
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.split_mode not in ("ppa", "integrated"):
            raise ValueError("revenue_split.mode 는 ppa 또는 integrated 여야 합니다.")
        if self.split_mode == "ppa" and not self.ppa_prices:
            raise ValueError("ppa 모드에는 ppa_krw_per_kwh 가 필요합니다.")
        if self.split_mode == "integrated" and self.solar_capex_krw is None:
            raise ValueError("integrated 모드에는 solar_capex_krw 가 필요합니다.")
        if not 0 <= self.subsidy_rate <= 1:
            raise ValueError("subsidy_rate 는 0~1 이어야 합니다.")
        unknown = set(self.standard_vpp_shares) - set(STANDARD_ITEMS)
        if unknown or not all(0 <= v <= 1 for v in self.standard_vpp_shares.values()):
            raise ValueError(f"standard_vpp_shares 가 올바르지 않습니다: {sorted(unknown)}")
        for bat in self.batteries.values():
            if not (0 < bat.soh <= 1 and 0 < bat.efficiency <= 1 and bat.cycle_life > 0):
                raise ValueError(f"배터리 {bat.key} 값이 올바르지 않습니다.")

    def subsidy_krw(self, capex_krw: float, rate: float | None = None) -> float:
        r = self.subsidy_rate if rate is None else rate
        amount = r * capex_krw
        if self.subsidy_cap_krw_per_mwh is not None:
            amount = min(amount, self.subsidy_cap_krw_per_mwh * self.ess_energy_mwh)
        return amount


def _opt_float(v) -> float | None:
    return None if v is None else float(v)


@dataclass(frozen=True)
class Operating:
    """운영 백테스트 1년 결과 (1년으로 환산된 값)."""

    market_revenue_krw: float
    solar_to_vpp_mwh: float
    ess_throughput_mwh: float


FLOW_COLUMNS = [
    "capex",
    "subsidy",
    "market_revenue",
    "ppa_payment",
    "om",
    "insurance_inspection",
    "battery_replacement",
    "pcs_replacement",
]


def _flows(
    cfg: EconomicsConfig,
    battery: Battery,
    op: Operating,
    *,
    growth: float = 0.0,
    ppa_krw_per_kwh: float | None = None,
    subsidy_rate: float | None = None,
    capex_krw: float | None = None,
    revenue_scale: float = 1.0,
    include_solar_capex: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """현금흐름 항목별 배열(0년~project_years 년).

    손익분기를 반복 계산하므로 numpy 로만 계산한다.
    """
    n = cfg.project_years
    capex = battery.capex_krw(cfg.ess_energy_mwh, cfg.fx) if capex_krw is None else capex_krw
    subsidy = cfg.subsidy_krw(capex, subsidy_rate)
    life = battery.life_years(cfg.ess_energy_mwh, op.ess_throughput_mwh)
    t = np.arange(n + 1, dtype=float)
    run = (t >= 1).astype(float)
    esc = (1 + cfg.cost_escalation) ** np.maximum(t - 1, 0)
    f = {c: np.zeros(n + 1) for c in FLOW_COLUMNS}
    f["capex"][0] = -capex
    if cfg.split_mode == "integrated" and include_solar_capex:
        f["capex"][0] -= float(cfg.solar_capex_krw or 0.0)
    f["subsidy"][0] = subsidy
    f["market_revenue"] = (
        run * op.market_revenue_krw * revenue_scale * (1 + growth) ** np.maximum(t - 1, 0)
    )
    if cfg.split_mode == "ppa":
        price = ppa_krw_per_kwh if ppa_krw_per_kwh is not None else cfg.ppa_prices[0]
        f["ppa_payment"] = run * -price * 1000.0 * op.solar_to_vpp_mwh
    f["om"] = run * -capex * cfg.om_rate * esc
    f["insurance_inspection"] = run * -capex * cfg.insurance_rate * esc
    for y in replacement_years(life, n):
        f["battery_replacement"][y] = (
            -capex * battery.battery_share * (1 - cfg.battery_price_decline) ** y
        )
    for y in replacement_years(cfg.pcs_life_years, n):
        f["pcs_replacement"][y] -= capex * cfg.pcs_share * (1 + cfg.cost_escalation) ** y
    return f, {"capex": capex, "subsidy": subsidy, "battery_life_years": life}


def _npv_arr(net: np.ndarray, rate: float) -> float:
    return float((net / (1 + rate) ** np.arange(len(net))).sum())


def cashflow(cfg: EconomicsConfig, battery: Battery, op: Operating, **kw) -> pd.DataFrame:
    """0년(구축) ~ project_years 년의 현금흐름표 (원). 양수 = 들어오는 돈.

    kw: growth, ppa_krw_per_kwh, subsidy_rate, capex_krw, revenue_scale, include_solar_capex
    """
    f, attrs = _flows(cfg, battery, op, **kw)
    df = pd.DataFrame(f, index=pd.Index(np.arange(cfg.project_years + 1), name="year"))
    df["net"] = df[FLOW_COLUMNS].sum(axis=1)
    df["discounted"] = df["net"] / (1 + cfg.discount_rate) ** df.index.to_numpy()
    df["cumulative"] = df["net"].cumsum()
    df["cumulative_discounted"] = df["discounted"].cumsum()
    df.attrs.update(attrs)
    return df


def replacement_years(life_years: float, project_years: int) -> list[int]:
    """수명이 다하는 해(올림)마다 교체. 사업 마지막 해 이후는 교체하지 않는다."""
    if not math.isfinite(life_years) or life_years <= 0:
        return []
    out, k = [], 1
    while (y := math.ceil(life_years * k - 1e-9)) < project_years:
        out.append(int(y))
        k += 1
    return out


def npv(net: pd.Series, rate: float) -> float:
    return float((net / (1 + rate) ** np.asarray(net.index)).sum())


def irr(net: pd.Series) -> float:
    """NPV=0 이 되는 할인율. 부호가 바뀌지 않으면 nan."""
    lo, hi = -0.99, 1.0
    f_lo, f_hi = npv(net, lo), npv(net, hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(net, mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def payback_year(cumulative: pd.Series) -> float:
    """누적 현금흐름이 처음 0 이상이 되는 시점(년, 선형 보간). 못 넘으면 nan."""
    c = cumulative.to_numpy()
    for y in range(1, len(c)):
        if c[y] >= 0 > c[y - 1]:
            return (y - 1) + (-c[y - 1]) / (c[y] - c[y - 1])
    return 0.0 if c[0] >= 0 else float("nan")


def _solve(fn, lo: float, hi: float, iters: int = 100) -> float:
    """증가 또는 감소하는 fn 의 0점 (구간 안에 없으면 nan)."""
    f_lo, f_hi = fn(lo), fn(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    for _ in range(iters):
        mid = (lo + hi) / 2
        f_mid = fn(mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def evaluate(cfg: EconomicsConfig, battery: Battery, op: Operating, **kw) -> dict:
    """한 조합의 투자 지표."""
    cf = cashflow(cfg, battery, op, **kw)
    r = cfg.discount_rate
    capex = cf.attrs["capex"]

    def npv_with(**over):
        # 손익분기를 구하는 값만 덮어쓰고, 나머지 입력(구축비·지원율 등)은 그대로 둔다
        f, _ = _flows(cfg, battery, op, **{**kw, **over})
        return _npv_arr(sum(f.values()), r)

    be_rate = _solve(lambda s: npv_with(subsidy_rate=s), 0.0, 1.0)
    if math.isnan(be_rate) and npv_with(subsidy_rate=0.0) >= 0:
        be_rate = 0.0
    be_capex = _solve(lambda c: npv_with(capex_krw=c), 1.0, capex * 20)
    be_scale = _solve(lambda s: npv_with(revenue_scale=s), 0.0, 50.0)
    be_ppa = float("nan")
    if cfg.split_mode == "ppa" and op.solar_to_vpp_mwh > 0:
        be_ppa = _solve(lambda p: npv_with(ppa_krw_per_kwh=p), 0.0, 2000.0)
    return {
        "capex_krw": capex,
        "subsidy_krw": cf.attrs["subsidy"],
        "battery_life_years": round(cf.attrs["battery_life_years"], 1),
        "battery_replacement_years": ",".join(
            str(y) for y in cf.index[cf["battery_replacement"] < 0]
        )
        or "-",
        "npv_krw": npv(cf["net"], r),
        "irr": irr(cf["net"]),
        "payback_years": payback_year(cf["cumulative"]),
        "discounted_payback_years": payback_year(cf["cumulative_discounted"]),
        "breakeven_subsidy_rate": be_rate,
        "breakeven_capex_krw": be_capex,
        "min_annual_market_revenue_krw": op.market_revenue_krw * be_scale
        if not math.isnan(be_scale)
        else float("nan"),
        "breakeven_ppa_krw_per_kwh": be_ppa,
        "needed_annual_support_krw": needed_annual_support(npv(cf["net"], r), r, cfg.project_years),
    }


def needed_annual_support(npv_krw: float, rate: float, years: int) -> float:
    """NPV 를 0으로 만드는 1~years 년 매년 같은 보상액 (NWAs 보상 등). 이미 흑자면 0."""
    annuity = sum((1 + rate) ** -t for t in range(1, years + 1))
    return max(0.0, -npv_krw / annuity)


def load_operating(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in OPERATING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"운영 결과 파일에 열이 없습니다: {missing}")
    days = df["days"] if "days" in df.columns else 365
    scale = 365.0 / pd.Series(days, index=df.index).astype(float)
    cols = ["market_revenue_krw", "solar_to_vpp_mwh", "ess_charge_mwh", "ess_discharge_mwh"]
    for c in cols + [c for c in BREAKDOWN_COLUMNS if c in df.columns]:
        df[c] = df[c].astype(float) * scale
    return df


BASELINE_BATTERY = "none"


@dataclass(frozen=True)
class _View:
    row: pd.Series
    basis: str
    op: Operating  # ESS 용량정산금 제외
    ess_cp_vpp_krw: float  # ESS 용량정산금을 넣을 때 VPP 수익에 더할 금액 (nan = 자료 없음)
    d_solar_mwh: float  # PPA 를 주는 태양광 수전량 (standard 는 0)
    owner_fixed_krw: float  # standard: 발전사업자 몫 (ESS 용량정산금 제외)
    owner_ess_cp_krw: float  # standard: ESS 용량정산금 중 발전사업자 몫


def _has_breakdown(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in BREAKDOWN_COLUMNS)


def _standard_view(cfg: EconomicsConfig, r: pd.Series, thr: float) -> _View:
    sh = {k: cfg.standard_vpp_shares.get(k, 0.0) for k in STANDARD_ITEMS}
    vpp = owner = 0.0
    for key, col in STANDARD_ITEMS.items():
        if key == "ess_capacity_payment":
            continue
        sign = -1.0 if key == "imbalance_penalty" else 1.0
        vpp += sign * sh[key] * r[col]
        owner += sign * (1 - sh[key]) * r[col]
    cp = r["ess_capacity_payment_krw"]
    sh_cp = sh["ess_capacity_payment"]
    return _View(r, "standard", Operating(vpp, 0.0, thr), sh_cp * cp, 0.0, owner, (1 - sh_cp) * cp)


def _operating_views(cfg: EconomicsConfig, operating: pd.DataFrame) -> list[_View]:
    """ESS 행마다 관점(total / incremental / standard)별 입력."""
    operating = operating.copy()
    # 용량 비교 행은 case 이름에 용량이 붙는다. 기준선은 line_case(선로 조건)로 짝짓는다.
    if "line_case" in operating.columns:
        operating["line_case"] = operating["line_case"].fillna(operating["case"])
    else:
        operating["line_case"] = operating["case"]
    base = operating[operating["battery"] == BASELINE_BATTERY].set_index(["line_case", "operation"])
    if base.index.duplicated().any():
        raise ValueError("같은 case·operation 의 기준선(battery=none) 행이 두 개 이상입니다.")
    breakdown = _has_breakdown(operating)
    nan = float("nan")
    views = []
    for _, r in operating[operating["battery"] != BASELINE_BATTERY].iterrows():
        thr = r["ess_charge_mwh"] + r["ess_discharge_mwh"]
        cp = r["ess_capacity_payment_krw"] if "ess_capacity_payment_krw" in r.index else nan
        op = Operating(r["market_revenue_krw"], r["solar_to_vpp_mwh"], thr)
        views.append(_View(r, "total", op, cp, r["solar_to_vpp_mwh"], nan, nan))
        key = (r["line_case"], r["operation"])
        if key in base.index:
            b = base.loc[key]
            d_solar = r["solar_to_vpp_mwh"] - b["solar_to_vpp_mwh"]
            op = Operating(r["market_revenue_krw"] - b["market_revenue_krw"], d_solar, thr)
            views.append(_View(r, "incremental", op, cp, d_solar, nan, nan))
        if breakdown and cfg.standard_vpp_shares:
            views.append(_standard_view(cfg, r, thr))
    return views


def _cfg_for_row(cfg: EconomicsConfig, r: pd.Series) -> EconomicsConfig:
    """운영 결과 행에 ess_power_mw·ess_energy_mwh 가 있으면 그 용량으로 계산한다 (용량 비교용)."""
    vals = {}
    for col, attr in (("ess_power_mw", "ess_power_mw"), ("ess_energy_mwh", "ess_energy_mwh")):
        if col in r.index and pd.notna(r[col]):
            vals[attr] = float(r[col])
    return replace(cfg, **vals) if vals else cfg


def summarize(cfg: EconomicsConfig, operating: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """운영 결과 x 관점 x ESS 용량정산금 x 시장수익 시나리오 x PPA 단가 조합의 지표와
    기준 시나리오 현금흐름."""
    unknown = set(operating["battery"]) - set(cfg.batteries) - {BASELINE_BATTERY}
    if unknown:
        raise ValueError(f"설정에 없는 배터리: {sorted(unknown)}")
    rows, flows = [], []
    for v in _operating_views(cfg, operating):
        r = v.row
        bat = cfg.batteries[r["battery"]]
        cfg_r = _cfg_for_row(cfg, r)
        extra = {"include_solar_capex": v.basis == "total"}
        if v.basis == "standard" or cfg.split_mode != "ppa":
            ppas: tuple = (None,)
        else:
            ppas = cfg.ppa_prices
        for with_cp in cfg.ess_cp_scenarios:
            if with_cp and math.isnan(v.ess_cp_vpp_krw):
                continue
            add = v.ess_cp_vpp_krw if with_cp else 0.0
            op = Operating(
                v.op.market_revenue_krw + add, v.op.solar_to_vpp_mwh, v.op.ess_throughput_mwh
            )
            for scen, g in cfg.revenue_scenarios.items():
                for ppa in ppas:
                    kw = {"growth": g, "ppa_krw_per_kwh": 0.0 if ppa is None else ppa, **extra}
                    m = evaluate(cfg_r, bat, op, **kw)
                    if v.basis == "standard":
                        owner = v.owner_fixed_krw + (v.owner_ess_cp_krw if with_cp else 0.0)
                    else:
                        owner = (ppa or 0.0) * 1000.0 * v.d_solar_mwh
                    rows.append(
                        {
                            "case": r["case"],
                            "battery": r["battery"],
                            "battery_label": bat.label,
                            "operation": r["operation"],
                            "basis": v.basis,
                            "ess_capacity_payment": with_cp,
                            "revenue_scenario": scen,
                            "ppa_krw_per_kwh": ppa,
                            "annual_market_revenue_krw": op.market_revenue_krw,
                            "annual_solar_to_vpp_mwh": op.solar_to_vpp_mwh,
                            "solar_owner_gain_krw_per_year": owner,
                            "ess_power_mw": cfg_r.ess_power_mw,
                            "ess_energy_mwh": cfg_r.ess_energy_mwh,
                            "throughput_cost_krw_per_mwh": bat.throughput_cost_krw_per_mwh(
                                cfg_r.ess_energy_mwh, cfg.fx
                            ),
                            **m,
                            "decision": "참여 검토" if m["npv_krw"] > 0 else "비참여",
                        }
                    )
                    if scen == "base":
                        cf = cashflow(cfg_r, bat, op, **kw).reset_index()
                        meta = (
                            ("ppa_krw_per_kwh", float("nan") if ppa is None else ppa),
                            ("ess_capacity_payment", with_cp),
                            ("basis", v.basis),
                            ("operation", r["operation"]),
                            ("battery", r["battery"]),
                            ("case", r["case"]),
                        )
                        for k, val in meta:
                            cf.insert(0, k, val)
                        flows.append(cf)
    return pd.DataFrame(rows), pd.concat(flows, ignore_index=True)


def operating_row_from_hourly(
    hourly: pd.DataFrame,
    *,
    case: str,
    battery: str,
    operation: str,
    solar_capacity_credit_mw: float = 0.0,
    ess_capacity_credit_mw: float = 0.0,
    rcp_krw_per_kwh_h: float = 0.0,
    imbalance_penalty_krw: float = 0.0,
) -> dict:
    """시간별 운전 결과 한 기간을 운영 결과 CSV 한 행으로 바꾼다.

    hourly: ``RollingResult.hourly`` 처럼 ts, price_krw_per_mwh(정산 SMP), sell_mwh, charge_mwh,
    discharge_mwh 열을 가진 표. ``rec_krw_per_mwh`` 가 있으면 판매 단가에 더하고, ``committed``
    가 있으면 확정 구간만 쓴다. 판매량(sell)에는 방전분이 포함돼 있다고 본다.

    용량정산금 = Σ min(판매량, 실효용량) x RCP (태양광은 바로 판 양, ESS 는 방전량 기준).
    실효용량 예: 태양광 5.7MW x 3.84%, ESS 4MW x 65.63%. RCP 예: 제주 22.05원/kW-h.
    열화비는 빼지 않는다 (사업성에서는 배터리 비용을 구축비·교체비로 넣는다).
    """
    h = hourly
    if "committed" in h.columns:
        h = h[h["committed"].astype(bool)]
    need = ["ts", "price_krw_per_mwh", "sell_mwh", "charge_mwh", "discharge_mwh"]
    missing = [c for c in need if c not in h.columns]
    if missing:
        raise ValueError(f"시간별 결과에 열이 없습니다: {missing}")
    price = h["price_krw_per_mwh"].to_numpy(float)
    if "rec_krw_per_mwh" in h.columns:
        price = price + h["rec_krw_per_mwh"].fillna(0.0).to_numpy(float)
    sell = h["sell_mwh"].to_numpy(float)
    ch = h["charge_mwh"].to_numpy(float)
    dis = np.minimum(h["discharge_mwh"].to_numpy(float), sell)
    direct = np.maximum(sell - dis, 0.0)
    to_krw = 1000.0 * rcp_krw_per_kwh_h
    solar_cp = float(np.minimum(direct, solar_capacity_credit_mw).sum() * to_krw)
    ess_cp = float(np.minimum(dis, ess_capacity_credit_mw).sum() * to_krw)
    solar_rev = float(price @ direct)
    ess_rev = float(price @ dis)
    return {
        "case": case,
        "battery": battery,
        "operation": operation,
        "market_revenue_krw": solar_rev + ess_rev + solar_cp - imbalance_penalty_krw,
        "solar_to_vpp_mwh": float(direct.sum() + ch.sum()),
        "ess_charge_mwh": float(ch.sum()),
        "ess_discharge_mwh": float(h["discharge_mwh"].sum()),
        "days": int(pd.to_datetime(h["ts"]).dt.date.nunique()),
        "solar_direct_revenue_krw": solar_rev,
        "ess_discharge_revenue_krw": ess_rev,
        "solar_capacity_payment_krw": solar_cp,
        "ess_capacity_payment_krw": ess_cp,
        "imbalance_penalty_krw": float(imbalance_penalty_krw),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="배전망 ESS 20년 사업성 (참여 판단)")
    p.add_argument("--config", default="config/project_economics_v0.yaml")
    p.add_argument("--operating", required=True, help="운영 백테스트 1년 결과 CSV")
    p.add_argument("--out-dir", default="data/processed/economics")
    args = p.parse_args(argv)
    cfg = EconomicsConfig.from_yaml(args.config)
    summary, flows = summarize(cfg, load_operating(args.operating))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "economics_summary.csv", index=False, encoding="utf-8-sig")
    flows.to_csv(out / "economics_cashflow.csv", index=False, encoding="utf-8-sig")
    view = summary[(summary["revenue_scenario"] == "base") & ~summary["ess_capacity_payment"]]
    for basis in ("standard", "total"):
        if (view["basis"] == basis).any():
            view = view[view["basis"] == basis]
            break
    cols = [
        "case",
        "battery",
        "operation",
        "basis",
        "ppa_krw_per_kwh",
        "npv_krw",
        "irr",
        "payback_years",
        "breakeven_subsidy_rate",
        "decision",
    ]
    shown = view[cols].assign(
        npv_억=(view["npv_krw"] / 1e8).round(1), irr=(view["irr"] * 100).round(1)
    )
    print(shown.drop(columns="npv_krw").to_string(index=False))
    print(f"저장: {out / 'economics_summary.csv'}, {out / 'economics_cashflow.csv'}")


if __name__ == "__main__":
    main()
