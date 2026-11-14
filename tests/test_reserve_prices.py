"""FCR 원본 날짜 배치, 시간 평균, 결측, 정산 단위와 CLI 연결을 검증한다."""

import json

import numpy as np
import pandas as pd
import pytest

from tests.test_execution import ESS
from tests.test_hourly_settlement import PARAMS, inputs
from vpp.hourly_settlement import hourly_settlement_totals, settle_hourly
from vpp.hourly_settlement import main as settle_main
from vpp.market.reserve_prices import PRICE_COLUMN, SOURCE, main, to_hourly


def raw_prices():
    # 원본처럼 날짜는 각 3행 묶음의 가운데에 있다. 날짜 전방 채움은 하루를 잘못 붙인다.
    headers = [["거래일자", "구분"] + [
        f"{hour:02d}시" if slot == 0 else None for hour in range(1, 25) for slot in range(4)
    ], [None, None] + [1, 2, 3, 4] * 24]
    rows = []
    for day, prices in [("20240601", [10, 20, 30, 40]), ("20240602", [2, 4, 6, 8])]:
        rows.extend([
            [None, "주파수제어예비력"] + prices * 24,
            [day, "1차예비력"] + [999] * 96,
            [None, "3차운전예비력"] + [999] * 96,
        ])
    return pd.DataFrame(headers + rows)


def test_hour_start_dates_and_four_quarter_average_use_only_fcr():
    result = to_hourly(raw_prices())
    assert len(result) == 48
    assert result.ts.iloc[0] == pd.Timestamp("2024-06-01T00:00+09:00")
    assert result.ts.iloc[23] == pd.Timestamp("2024-06-01T23:00+09:00")
    assert result.ts.iloc[24] == pd.Timestamp("2024-06-02T00:00+09:00")
    assert result[PRICE_COLUMN].tolist() == [25] * 24 + [5] * 24
    assert result["fcr_quarter_count"].eq(4).all()


def test_partial_quarters_are_missing_and_real_zero_is_preserved():
    raw = raw_prices()
    raw.iloc[2, 2] = np.nan
    raw.iloc[2, 6:10] = 0
    result = to_hourly(raw)
    assert pd.isna(result.loc[0, PRICE_COLUMN])
    assert result.loc[0, "fcr_quarter_count"] == 3
    assert result.loc[1, PRICE_COLUMN] == 0
    assert result.loc[1, "fcr_quarter_count"] == 4


@pytest.mark.parametrize("issue", ["duplicate_day", "bad_kind", "bad_header", "negative", "text"])
def test_invalid_source_is_not_silently_repaired(issue):
    raw = raw_prices()
    if issue == "duplicate_day":
        raw.iloc[6, 0] = "20240601"
    elif issue == "bad_kind":
        raw.iloc[3, 1] = "주파수제어예비력"
    elif issue == "bad_header":
        raw.iloc[1, 2] = 4
    else:
        raw.iloc[2, 2] = -1 if issue == "negative" else "invalid"
    with pytest.raises(ValueError):
        to_hourly(raw)


def test_fcr_changes_only_penalty_at_one_thousand_times_hourly_price():
    execution, market, dispatch = inputs()
    baseline = settle_hourly(execution, market, ESS, params=PARAMS, dispatch=dispatch).iloc[0]
    result = settle_hourly(
        execution, market, ESS, params=PARAMS,
        dispatch=dispatch.drop(columns=PRICE_COLUMN), fcr=to_hourly(raw_prices()),
    ).iloc[0]
    assert result[PRICE_COLUMN] == 25
    assert result["fcr_price_krw_per_mwh"] == 25_000
    # 허용범위 밖 0.25 MWh × FCR 25,000원/MWh = 6,250원 추가 차감.
    assert result["imbalance_penalty_krw"] == pytest.approx(46_250)
    assert result["realized_net_krw"] == pytest.approx(baseline.realized_net_krw - 6_250)
    for col in ("energy_revenue_krw", "rec_krw", "wear_krw", "net_before_penalty_krw"):
        assert result[col] == baseline[col]
    assert result["fcr_price_source"] == SOURCE


def test_price_connection_does_not_invent_dispatch_or_zero_penalty():
    execution, market, _ = inputs()
    result = settle_hourly(execution, market, ESS, params=PARAMS, fcr=to_hourly(raw_prices()))
    row = result.iloc[0]
    assert row[PRICE_COLUMN] == 25
    assert row["dispatch_source"] == "missing"
    assert pd.isna(row["set_point_mw"])
    assert pd.isna(row["imbalance_penalty_krw"])
    assert pd.isna(row["realized_net_krw"])


def test_fcr_is_aligned_by_time_with_no_fill_from_dispatch():
    execution, market, dispatch = inputs((8.4, 8.4), (150, 150))
    fcr = pd.DataFrame({"ts": execution.ts.iloc[1:], PRICE_COLUMN: 0.0})
    result = settle_hourly(execution, market, ESS, params=PARAMS, dispatch=dispatch, fcr=fcr)
    assert pd.isna(result.loc[0, PRICE_COLUMN])
    assert pd.isna(result.loc[0, "imbalance_penalty_krw"])
    assert result.loc[0, "fcr_price_source"] == "missing"
    assert result.loc[1, "imbalance_penalty_krw"] == pytest.approx(40_000)


def test_conflicting_fcr_sources_raise_instead_of_silently_overwriting():
    execution, market, dispatch = inputs()
    with pytest.raises(ValueError, match="FCR 가격이 다릅니다"):
        settle_hourly(
            execution, market, ESS, params=PARAMS, dispatch=dispatch, fcr=to_hourly(raw_prices()),
        )


def test_conversion_and_settlement_cli_keep_provenance_and_missing_net(tmp_path):
    xlsx = tmp_path / "reserves.xlsx"
    raw_prices().to_excel(xlsx, sheet_name="실시간예비력가격", header=False, index=False)
    fcr = tmp_path / "fcr.csv"
    assert main(["--input", str(xlsx), "--out", str(fcr)]) == 0
    source = json.loads(fcr.with_suffix(".source.json").read_text())
    assert source["hours"] == 48
    assert len(source["source_sha256"]) == 64
    execution, market, _ = inputs()
    execution.to_csv(tmp_path / "execution.csv", index=False)
    market.to_csv(tmp_path / "market.csv", index=False)
    outdir = tmp_path / "settlement"
    assert settle_main([
        "--execution", str(tmp_path / "execution.csv"), "--market", str(tmp_path / "market.csv"),
        "--fcr", str(fcr), "--ess", "config/ess_v0.yaml", "--outdir", str(outdir),
    ]) == 0
    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["inputs"]["fcr"] == str(fcr)
    assert summary["totals"]["fcr_price_missing_hours"] == 0
    assert summary["totals"]["penalty_missing_hours"] == 1
    assert summary["totals"]["realized_net_krw"] is None


@pytest.mark.parametrize("with_dispatch", [False, True])
def test_review_screen_distinguishes_missing_penalty_from_computed_amount(
    tmp_path, monkeypatch, with_dispatch,
):
    from dashboard import settlement_review
    from streamlit.testing.v1 import AppTest

    execution, market, dispatch = inputs()
    result = settle_hourly(
        execution, market, ESS, params=PARAMS, fcr=to_hourly(raw_prices()),
        dispatch=dispatch.drop(columns=PRICE_COLUMN) if with_dispatch else None,
    )
    result.to_csv(tmp_path / "hourly_settlement.csv", index=False)
    (tmp_path / "summary.json").write_text(json.dumps({"inputs": {}, "parameters": {}}))
    monkeypatch.setattr(settlement_review, "DEFAULT_RESULT", tmp_path)
    app = AppTest.from_string("from dashboard.settlement_review import render\nrender()").run()
    assert not app.exception
    assert not app.error
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["FCR 가격 연결"] == "1/1 시간"
    assert metrics["임밸런스 페널티"] == ("46,250 원" if with_dispatch else "미계산")
    assert (metrics["최종 순수익"] == "미계산") is (not with_dispatch)
    assert len(app.warning) == (0 if with_dispatch else 1)
    assert len(app.dataframe) == 1
    assert hourly_settlement_totals(result)["fcr_price_missing_hours"] == 0
    assert metrics["허용오차율"] == ("1.5%" if with_dispatch else "판단 대기")


def test_review_existing_example_explains_each_price_branch(tmp_path, monkeypatch):
    from dashboard import settlement_review
    from streamlit.testing.v1 import AppTest

    settle_main([
        "--execution", "data/sample/settlement_hourly/execution.csv",
        "--market", "data/sample/settlement_hourly/market.csv",
        "--dispatch", "data/sample/settlement_hourly/dispatch.csv",
        "--ess", "config/ess_v0.yaml", "--outdir", str(tmp_path),
    ])
    monkeypatch.setattr(settlement_review, "DEFAULT_RESULT", tmp_path)
    monkeypatch.setattr(settlement_review, "EXAMPLE_RESULT", tmp_path)
    app = AppTest.from_string("from dashboard.settlement_review import render\nrender()").run()
    app.radio[0].set_value("조건별 검산 예제").run()
    for hour, tolerance, explanation in [(1, "1.5%", "최소입찰가격"),
                                         (3, "6%", "FCR 가격"),
                                         (4, "1.5%", "입찰하한가격")]:
        app.selectbox[0].set_value(pd.Timestamp(f"2024-06-01T{hour:02d}:00+09:00")).run()
        assert not app.exception
        metrics = {metric.label: metric.value for metric in app.metric}
        assert metrics["허용오차율"] == tolerance
        assert any(explanation in item.value for item in app.markdown)
        assert any("가상" in item.value for item in app.info)
