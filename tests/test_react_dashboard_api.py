"""The React adapter must preserve source identity, nulls and archive integrity."""

import hashlib
import json
from datetime import date

import pandas as pd
import pytest
from api import dashboard as adapter

from tests.test_overview_data import saved_experiment


@pytest.fixture
def portfolio(tmp_path, monkeypatch):
    folder = tmp_path / "result"
    folder.mkdir()
    saved_experiment(folder, rec=True)
    (folder / "validation.json").write_text(json.dumps({"plan_checks_all_passed": True}))
    release = {"id": "verified", "label": "fixture", "backtest": str(folder)}
    monkeypatch.setattr(adapter, "releases", lambda root: ({"state": "needs_data"}, [release]))
    monkeypatch.setenv("VPP_PROJECT_ROOT", str(tmp_path))
    return tmp_path, folder


def test_saved_forecast_is_not_reported_as_execution(portfolio, client):
    response = client.get(
        "/api/v1/dashboard/snapshot",
        params={"source": "portfolio", "selection": "verified", "day": "2024-06-01"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["id"] == "portfolio:verified:2024-06-01"
    assert data["executionAvailable"] is False
    assert data["actualIncome"] is None and data["cumulativeIncome"] is None
    assert all(h["actualSupply"] is None and h["income"] is None for h in data["hours"])
    assert data["plannedIncome"] == pytest.approx(sum(r["planned"] for r in data["ledger"]))
    assert all(r["actual"] is None for r in data["ledger"])


@pytest.mark.parametrize("prefix", ["", "/Users/former/team project/", "/home/member/repo/",
                                    "C:\\Users\\member\\팀 프로젝트\\"])
def test_shared_portfolio_opens_after_relocation_without_rewriting_data(
    tmp_path, monkeypatch, client, prefix
):
    root = tmp_path / "another checkout"
    relative = "data/processed/operator_pipeline/backtests/verified"
    folder = root / relative
    folder.mkdir(parents=True)
    saved_experiment(folder, rec=True)
    (folder / "validation.json").write_text('{"plan_checks_all_passed": true}')
    saved_path = prefix + (relative.replace("/", "\\") if "\\" in prefix else relative)
    release = {"id": "verified", "label": "shared", "backtest": saved_path}
    status = root / "data/processed/operator_pipeline/status.json"
    status.write_text(json.dumps({"active": release, "releases": {"shared": release}}))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setenv("VPP_PROJECT_ROOT", str(root))
    monkeypatch.chdir(tmp_path)  # Resolving must not depend on the server's working directory.
    response = client.get("/api/v1/dashboard/snapshot", params={
        "source": "portfolio", "selection": "verified", "day": "2024-06-01",
    })
    assert response.status_code == 200, response.text
    assert len(response.json()["hours"]) == 24
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_unknown_selection_bad_date_and_unverified_plan_are_rejected(portfolio, client):
    query = {"source": "portfolio", "selection": "verified", "day": "2024-06-01"}
    assert (
        client.get(
            "/api/v1/dashboard/snapshot", params={**query, "selection": "../secret"}
        ).status_code
        == 404
    )
    assert (
        client.get("/api/v1/dashboard/snapshot", params={**query, "day": "not-date"}).status_code
        == 422
    )
    assert (
        client.get("/api/v1/dashboard/snapshot", params={**query, "day": "2024-06-02"}).status_code
        == 422
    )
    (portfolio[1] / "validation.json").write_text('{"plan_checks_all_passed": false}')
    assert client.get("/api/v1/dashboard/snapshot", params=query).status_code == 422


def test_archive_integrity_scope_and_capacity_are_checked(tmp_path, monkeypatch):
    file = tmp_path / "hourly.parquet"
    file.write_bytes(b"modified")
    meta = {
        "schema": 2,
        "operation": "forecast_exec",
        "basis": "standard",
        "ess_capacity_payment": False,
        "capacity_mwh": 8,
        "file": file.name,
        "sha256": "original",
    }
    row = pd.Series({"id": "test", "ess_energy_mwh": 8})
    monkeypatch.setattr(adapter, "archives", lambda *args: [(tmp_path / "manifest.json", meta)])
    with pytest.raises(ValueError, match="검증값"):
        adapter.load_archive(tmp_path, row)
    meta["capacity_mwh"] = 20
    with pytest.raises(ValueError, match="용량"):
        adapter.load_archive(tmp_path, row)
    meta["capacity_mwh"] = 8
    meta["file"] = "../outside.parquet"
    with pytest.raises(ValueError, match="범위"):
        adapter.load_archive(tmp_path, row)


def test_catalog_demo_readiness_requires_validation_without_removing_candidates(
    tmp_path, monkeypatch
):
    rows = pd.DataFrame([
        {"id": name, "equipment": name, "line_case": "cap_2.50MW",
         "ess_energy_mwh": 8, "ess_power_mw": 2, "npv_krw": 100}
        for name in ("ready", "invalid", "missing")
    ])
    monkeypatch.setenv("VPP_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(adapter, "releases", lambda root: ({}, []))
    monkeypatch.setattr(adapter, "load_display", lambda root: (rows, ""))
    monkeypatch.setattr(adapter, "revenue_breakdown", lambda row: {"total": 200})
    monkeypatch.setattr(
        adapter, "archives",
        lambda root, candidate: (
            [] if candidate == "missing" else [(tmp_path / "manifest.json", {})]
        ),
    )
    checked = []

    def load(root, row):
        checked.append(row.id)
        if row.id == "invalid":
            raise ValueError("선택 설비와 저장된 ESS 용량이 다릅니다.")
        return pd.DataFrame(), {}, "archive"

    monkeypatch.setattr(adapter, "load_archive", load)
    candidates = {c["id"]: c for c in adapter.catalog()["candidates"]}
    assert checked == ["ready", "invalid"]
    assert set(candidates) == {"ready", "invalid", "missing"}
    assert candidates["ready"]["hasArchive"] is True
    assert candidates["invalid"]["hasArchive"] is False
    assert "용량" in candidates["invalid"]["archiveIssue"]
    assert candidates["invalid"]["income"] == 200
    assert candidates["missing"]["hasArchive"] is False
    assert candidates["missing"]["archiveIssue"] is None


def test_distribution_snapshot_keeps_missing_plan_income_and_exact_receipt_sum(
    monkeypatch, tmp_path
):
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2025-05-01", periods=48, freq="h"),
            "forecast_generation_mwh": 2.0,
            "actual_generation_mwh": 1.8,
            "committed_sell_mwh": 1.5,
            "supply_mwh": 1.4,
            "planned_charge_mwh": 0.5,
            "planned_discharge_mwh": 0.0,
            "charge_mwh": 0.4,
            "discharge_mwh": 0.0,
            "planned_energy_end_mwh": 4.0,
            "soc_end": 0.5,
            "shortfall_mwh": 0.1,
            "fixed_plan_supply_mwh": 1.35,
            "fixed_plan_charge_mwh": 0.45,
            "fixed_plan_discharge_mwh": 0.0,
            "fixed_plan_energy_end_mwh": 3.8,
            "fixed_plan_soc_end": 0.475,
            "fixed_plan_shortfall_mwh": 0.15,
            "expected_plan_vpp_ess_krw": 80.0,
            "expected_plan_vpp_solar_cp_krw": 20.0,
            "expected_plan_vpp_penalty_krw": 0.0,
            "expected_plan_vpp_income_krw": 100.0,
            "fixed_plan_vpp_ess_krw": 90.0,
            "fixed_plan_vpp_solar_cp_krw": 20.0,
            "fixed_plan_vpp_penalty_krw": -15.0,
            "fixed_plan_vpp_income_krw": 95.0,
            "vpp_ess_krw": 100.0,
            "vpp_solar_cp_krw": 20.0,
            "vpp_penalty_krw": -10.0,
            "vpp_income_krw": 110.0,
        }
    )
    row = pd.DataFrame(
        [
            {
                "id": "A",
                "equipment": "2/8",
                "ess_energy_mwh": 8.0,
                "ess_power_mw": 2.0,
                "line_case": "cap_2.50MW",
                "source_file": "report.csv",
            }
        ]
    )
    monkeypatch.setattr(adapter, "load_display", lambda root: (row, ""))
    monkeypatch.setattr(
        adapter,
        "load_archive",
        lambda *a: (frame, {"sha256": hashlib.sha256(b"x").hexdigest()}, "archive"),
    )
    result = adapter.distribution_snapshot(tmp_path, "A", date(2025, 5, 1))
    assert len(result["hours"]) == 24
    assert result["actualIncome"] == 2640
    assert result["cumulativeIncome"] == 2640  # no future-day lookahead
    assert sum(r["actual"] for r in result["ledger"]) == result["actualIncome"]
    assert result["plannedIncome"] == 2400
    assert result["fixedPlanIncome"] == 2280
    assert all(h["fixedIncome"] == 95 for h in result["hours"])
    assert result["provenance"]["snapshot"] == "archive"
    assert result["selection"] == "A"


def test_period_sums_saved_days_and_rejects_incomplete_ranges(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2025-05-01", periods=72, freq="h"),
            "committed_sell_mwh": 1.5,
            "supply_mwh": 1.4,
            "planned_charge_mwh": 0.5,
            "planned_discharge_mwh": 0.25,
            "charge_mwh": 0.4,
            "discharge_mwh": 0.2,
            "planned_energy_end_mwh": 4.0,
            "soc_end": 0.5,
            "shortfall_mwh": 0.1,
            "fixed_plan_supply_mwh": 1.35,
            "fixed_plan_charge_mwh": 0.45,
            "fixed_plan_discharge_mwh": 0.2,
            "fixed_plan_soc_end": 0.48,
            "fixed_plan_shortfall_mwh": 0.15,
            "expected_plan_vpp_income_krw": 8.0,
            "fixed_plan_vpp_income_krw": 9.0,
            "vpp_income_krw": 10.0,
        }
    )
    row = pd.DataFrame([{"id": "A", "ess_energy_mwh": 8.0}])
    monkeypatch.setattr(adapter, "load_display", lambda root: (row, ""))
    monkeypatch.setattr(adapter, "load_archive", lambda *a: (frame, {}, "archive"))
    result = adapter.distribution_period(tmp_path, "A", date(2025, 5, 1), date(2025, 5, 2))
    assert [d["date"] for d in result["days"]] == ["2025-05-01", "2025-05-02"]
    first = result["days"][0]
    assert first["income"] == pytest.approx(240)
    assert first["actualSupply"] == pytest.approx(33.6)
    assert first["plannedSoc"] == pytest.approx(50) and first["soc"] == pytest.approx(50)
    assert first["plannedIncome"] == pytest.approx(192)
    assert first["fixedIncome"] == pytest.approx(216)
    with pytest.raises(ValueError, match="완전하지"):
        adapter.distribution_period(tmp_path, "A", date(2025, 4, 30), date(2025, 5, 2))
    with pytest.raises(ValueError, match="시작일"):
        adapter.distribution_period(tmp_path, "A", date(2025, 5, 3), date(2025, 5, 1))
    with pytest.raises(ValueError, match="최대"):
        adapter.period_frame(frame, date(2024, 1, 1), date(2025, 5, 3))


def test_period_portfolio_reports_plan_without_inventing_execution(portfolio, client):
    response = client.get(
        "/api/v1/dashboard/period",
        params={"source": "portfolio", "selection": "verified",
                "start": "2024-06-01", "end": "2024-06-01"},
    )
    assert response.status_code == 200, response.text
    (day,) = response.json()["days"]
    assert day["income"] is None and day["actualSupply"] is None
    snap = client.get(
        "/api/v1/dashboard/snapshot",
        params={"source": "portfolio", "selection": "verified", "day": "2024-06-01"},
    ).json()
    assert day["plannedIncome"] == pytest.approx(snap["plannedIncome"])
