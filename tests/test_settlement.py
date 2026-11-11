"""settlement: 실현 수익의 중복 계산·부호·계산 불가 처리 검사."""

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_execution import ESS, plan, run
from vpp import settlement as stl


def executed(actual=(9, 1, 0, 12), rec=None):
    p = plan(sell=[6, 6, 2, 6], charge=[4, 0, 0, 2], discharge=[0, 3, 2, 0],
             gen=[10, 3, 0, 8], rec=rec)
    return run(p, list(actual)).hourly


def test_symmetric_default_equals_supply_times_price_no_double_count():
    h = executed()
    price = np.array([100_000.0, 150_000.0, 200_000.0, 80_000.0])
    s = stl.settle(h, price, ESS)
    smp_part = s["committed_sales_krw"] + s["deviation_krw"]
    assert np.allclose(smp_part, s["supply_mwh"] * price)
    assert np.allclose(s["wear_krw"], 1000.0 * (s["charge_mwh"] + s["discharge_mwh"]))
    assert np.allclose(s["realized_net_krw"],
                       smp_part + s["rec_krw"] - s["wear_krw"])
    totals = stl.settlement_totals(s)
    assert totals["computable"] and totals["basis"] == "assumed_actual_smp"
    assert totals["realized_net_krw"] == pytest.approx(s["realized_net_krw"].sum())


def test_negative_price_flips_sign_of_deviation_settlement():
    # 시간 2: 발전 0, 계획 방전 2 → 판매 2. 실측 0 이면 그대로 2 공급 (부족 없음).
    # 부족을 만들기 위해 잔량이 바닥난 상태를 쓴다.
    p = plan(sell=[4], discharge=[4], gen=[0])
    h = run(p, [0], start=ESS.energy_min_mwh).hourly  # 방전 불가 → 부족 4
    assert h["shortfall_mwh"].iloc[0] == pytest.approx(4)
    pos = stl.settle(h, [50_000.0], ESS).iloc[0]
    neg = stl.settle(h, [-50_000.0], ESS).iloc[0]
    assert pos["deviation_krw"] == pytest.approx(-200_000)  # 양수 가격: 부족분 환수
    assert neg["deviation_krw"] == pytest.approx(+200_000)  # 음수 가격: 못 낸 만큼 비용 회피
    assert pos["committed_sales_krw"] + pos["deviation_krw"] == pytest.approx(0)
    # 초과 공급은 반대 부호
    big = run(plan(sell=[5], gen=[5]), [12]).hourly  # 초과 공급 7
    assert big["surplus_mwh"].iloc[0] == pytest.approx(7)
    assert stl.settle(big, [-10_000.0], ESS)["deviation_krw"].iloc[0] == pytest.approx(-70_000)


def test_multipliers_and_penalty_are_explicit_settings():
    p = plan(sell=[4], discharge=[4], gen=[0])
    h = run(p, [0], start=ESS.energy_min_mwh).hourly
    params = stl.SettlementParams(shortfall_price_multiplier=1.5,
                                  shortfall_penalty_krw_per_mwh=10_000)
    row = stl.settle(h, [100_000.0], ESS, params).iloc[0]
    assert row["deviation_krw"] == pytest.approx(-4 * 100_000 * 1.5 - 4 * 10_000)
    assert params.metadata()["shortfall_penalty_krw_per_mwh"] == 10_000
    with pytest.raises(ValueError, match="day_ahead_price"):
        stl.SettlementParams(day_ahead_price="day_ahead_market")
    with pytest.raises(ValueError, match="0 이상"):
        stl.SettlementParams(surplus_price_multiplier=-1)


def test_rec_applies_once_to_actual_supply_not_to_committed_or_generation():
    h = executed(actual=(9, 1, 0, 12), rec=70_000.0)
    s = stl.settle(h, np.full(4, 100_000.0), ESS)
    assert np.allclose(s["rec_krw"], 70_000.0 * s["supply_mwh"])
    assert s["rec_krw"].sum() != pytest.approx(70_000.0 * s["committed_sell_mwh"].sum())
    assert s["rec_krw"].sum() != pytest.approx(70_000.0 * s["actual_generation_mwh"].sum())
    # 발전 → 충전 → 방전 판매 경로는 한 번만 붙는다: 공급량 기준이므로 충전분엔 없다
    assert (s["rec_krw"] >= 0).all()


def test_missing_price_marks_day_not_computable_instead_of_zero():
    h = executed()
    s = stl.settle(h, [100_000.0, np.nan, 100_000.0, 100_000.0], ESS)
    assert s["settlement_basis"].tolist()[1] == "price_missing"
    totals = stl.settlement_totals(s)
    assert totals["computable"] is False and totals["price_missing_hours"] == 1
    assert totals["realized_net_krw"] is None and "계산 불가" in totals["note"]
    with pytest.raises(ValueError, match="길이"):
        stl.settle(h, [1.0], ESS)


def test_settlement_yaml_round_trip(tmp_path):
    cfg = tmp_path / "execution.yaml"
    cfg.write_text(yaml.safe_dump({"settlement": {"surplus_price_multiplier": 0.5}}),
                   encoding="utf-8")
    params = stl.load_settlement(cfg)
    assert params.surplus_price_multiplier == 0.5 and params.deviation_price == "actual_smp"
    assert stl.load_settlement(None) == stl.SettlementParams()
    repo = stl.load_settlement("config/execution_v0.yaml")
    assert repo.shortfall_penalty_krw_per_mwh == 0 and repo.status == stl.STATUS
    assert isinstance(pd.DataFrame([repo.metadata()]), pd.DataFrame)
