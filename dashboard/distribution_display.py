"""Read-only presentation of published distribution ESS results.

No optimizer, economics calculator, configuration writer, or pipeline is called here.
NPV/support values come from the saved report's tables, at their published precision.
An unreported combination stays missing; it is never interpolated or recomputed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RESULTS = "docs/results"
OPERATING = "distribution_ess_operating_20261115.csv"
SIZING = "distribution_ess_sizing_20261115.csv"
REPORT = "distribution_ess_economics_20261115.md"
ECONOMICS_SUMMARIES = (
    "docs/results/economics_operating/economics_summary.csv",
    "docs/results/economics_sizing/economics_summary.csv",
)
BATTERIES = {
    "kr_notice_example": "국내 구축비 예시",
    "new_2021_pnnl": "신품 · PNNL 2021",
    "new_2025_bnef": "신품 · BNEF 2025",
    "new_2025_ember": "신품 · Ember 2025",
    "second_life_example": "사용후 배터리 예시",
}
REPORT_COLUMNS = (
    "kr_notice_example", "new_2021_pnnl", "new_2025_bnef",
    "new_2025_ember", "second_life_example",
)
IDENTITY = ["line_case", "battery", "ess_power_mw", "ess_energy_mwh"]
ECONOMIC_FIELDS = ("irr", "discounted_payback_years", "needed_annual_support_krw")


def line_label(line: str) -> str:
    kind, cap = line.split("_", 1)
    return f"{'고정' if kind == 'cap' else '동적'} {float(cap[:-2]):.2f} MW"


def equipment_label(row: pd.Series) -> str:
    return (f"{BATTERIES.get(row.battery, row.battery)} · "
            f"{row.ess_power_mw:g} MW / {row.ess_energy_mwh:g} MWh")


def _tables(section: str) -> list[list[list[str]]]:
    tables, current = [], []
    for line in [*section.splitlines(), ""]:
        if line.startswith("|"):
            cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            if not all(re.fullmatch(r"[:\-\s]+", cell) for cell in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    return tables


def report_values(text: str) -> dict[tuple, dict]:
    """Extract only standard/base/forecast_exec/no-ESS-CP tables from the fixed report."""
    sections = {}
    for block in re.split(r"(?m)^## ", text)[1:]:
        match = re.match(r"(\d+)\.", block)
        if match:
            sections[int(match[1])] = block
    values = {}

    def put(line, battery, power, energy, field, value, section):
        key = (line, battery, float(power), float(energy))
        record = values.setdefault(key, {})
        # Report amounts are in 억원. Unit conversion is the only money arithmetic.
        amount = float(value.replace(",", "")) * 1e8
        if field in record and record[field] != amount:
            raise ValueError("결과 보고서에 같은 조합의 값이 서로 다릅니다.")
        record[field] = amount
        if field == "npv_krw":
            record["npv_source"] = f"결과 보고서 §{section} · 소수 1자리 억원"

    for section, prefix in ((1, "cap"), (5, "dyn")):
        tables = _tables(sections.get(section, ""))
        if not tables:
            continue
        table = tables[0]
        if len(table[0]) != 6 or "국내" not in table[0][1]:
            raise ValueError("결과 보고서의 기본 용량 표 형식을 확인하세요.")
        for row in table[1:]:
            line = f"{prefix}_{float(row[0].replace('MW', '')):.2f}MW"
            for battery, value in zip(REPORT_COLUMNS, row[1:], strict=True):
                put(line, battery, 4, 20, "npv_krw", value, section)

    sizing = sections.get(3, "")
    for marker, battery in (("**국내 예시 구축비", "kr_notice_example"),
                            ("**신품 2025 BNEF", "new_2025_bnef")):
        if marker not in sizing:
            continue
        # Skip the heading line, including its closing **, before finding the next heading.
        tail = sizing.split(marker, 1)[1].partition("\n")[2]
        tail = re.split(r"(?m)^\*\*", tail, maxsplit=1)[0]
        tables = _tables(tail)
        if not tables:
            continue
        table = tables[0]
        lines = []
        for label in table[0][1:]:
            match = re.fullmatch(r"(고정|동적)\s+([\d.]+)MW", label)
            if not match:
                raise ValueError("결과 보고서의 용량 비교 선로 표기를 확인하세요.")
            lines.append(f"{'cap' if match[1] == '고정' else 'dyn'}_{float(match[2]):.2f}MW")
        for row in table[1:]:
            match = re.fullmatch(r"([\d.]+)MW/([\d.]+)MWh", row[0])
            if not match:
                raise ValueError("결과 보고서의 설비 용량 표기를 확인하세요.")
            for line, value in zip(lines, row[1:], strict=True):
                put(line, battery, match[1], match[2], "npv_krw", value, 3)

    tables = _tables(sections.get(4, ""))
    if tables:
        for row in tables[0][1:]:
            match = re.fullmatch(r"(고정|동적)\s+([\d.]+)MW", row[0])
            if not match:
                raise ValueError("결과 보고서의 보상 필요액 선로 표기를 확인하세요.")
            line = f"{'cap' if match[1] == '고정' else 'dyn'}_{float(match[2]):.2f}MW"
            for battery, value in zip(REPORT_COLUMNS, row[1:], strict=True):
                put(line, battery, 4, 20, "needed_annual_support_krw", value, 4)
    return values


def economics_values(root: Path) -> dict[tuple, dict]:
    """Read exact saved metrics without calculating or interpolating them in the UI."""
    values: dict[tuple, dict] = {}
    required = {
        "case", "battery", "operation", "basis", "ess_capacity_payment",
        "revenue_scenario", "ess_power_mw", "ess_energy_mwh", *ECONOMIC_FIELDS,
    }
    for relative in ECONOMICS_SUMMARIES:
        path = root / relative
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if missing := required - set(frame):
            raise ValueError(f"{path.name}: 경제성 표시 열이 없습니다: {sorted(missing)}")
        cp = frame["ess_capacity_payment"].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
        selected = frame[
            frame.operation.eq("forecast_exec")
            & frame.basis.eq("standard")
            & frame.revenue_scenario.eq("base")
            & cp.eq(False)
        ].copy()
        selected["line_case"] = selected["case"].str.replace(
            r"_ess[\d.]+MW[\d.]+MWh$", "", regex=True
        )
        source_file = OPERATING if "economics_operating" in relative else SIZING
        for row in selected.itertuples(index=False):
            key = (source_file, row.line_case, row.battery,
                   float(row.ess_power_mw), float(row.ess_energy_mwh))
            record = values.setdefault(key, {})
            for field in ECONOMIC_FIELDS:
                value = getattr(row, field)
                if pd.isna(value):
                    continue
                value = float(value)
                if field in record and record[field] != value:
                    raise ValueError("경제성 요약에 같은 조합의 값이 서로 다릅니다.")
                record[field] = value
            record["economics_source"] = str(path.relative_to(root)).replace("\\", "/")
    return values


def load_display(root: Path) -> tuple[pd.DataFrame, str]:
    """Read the two existing result CSVs; selection has no write-back path."""
    base = root / RESULTS
    parts = []
    required = {*IDENTITY[:2], "operation", "ess_charge_mwh", "ess_discharge_mwh",
                "ess_discharge_revenue_krw", "days"}
    for name in (OPERATING, SIZING):
        frame = pd.read_csv(base / name)
        if missing := required - set(frame):
            raise ValueError(f"{name}: 표시할 열이 없습니다: {sorted(missing)}")
        # The operating report explicitly describes its default ESS as 4 MW / 20 MWh.
        # These are metadata for that saved batch, not the current operating ESS settings.
        if name == OPERATING:
            frame = frame.assign(ess_power_mw=4.0, ess_energy_mwh=20.0)
        elif not set(IDENTITY).issubset(frame):
            raise ValueError("용량 비교 결과에 정격 용량 열이 없습니다.")
        frame = frame[(frame.operation == "forecast_exec") & (frame.battery != "none")].copy()
        numeric = ["ess_power_mw", "ess_energy_mwh", "ess_charge_mwh", "ess_discharge_mwh",
                   "ess_discharge_revenue_krw", "days"]
        if frame[numeric].isna().any().any():
            raise ValueError(f"{name}: 선택한 결과에 누락된 수치가 있습니다.")
        if not frame.days.eq(365).all():
            raise ValueError(f"{name}: 이 화면은 저장된 365일 결과만 표시합니다.")
        frame["source_file"] = name
        parts.append(frame)
    rows = pd.concat(parts, ignore_index=True).drop_duplicates(IDENTITY, keep="last")
    rows = rows.reset_index(drop=True)
    if rows.empty or not rows.line_case.str.fullmatch(r"(?:cap|dyn)_[\d.]+MW").all():
        raise ValueError("표시할 선로 조건이 없거나 선로 표기가 맞지 않습니다.")
    report = (base / REPORT).read_text(encoding="utf-8") if (base / REPORT).exists() else ""
    points = report_values(report)
    metrics = economics_values(root)
    rows["id"] = rows.apply(lambda r: f"{r.line_case}|{r.battery}|{r.ess_power_mw:g}:{r.ess_energy_mwh:g}", axis=1)
    for field in ("npv_krw", "needed_annual_support_krw", "npv_source"):
        rows[field] = [points.get(tuple(row), {}).get(field) for row in rows[IDENTITY].itertuples(index=False, name=None)]
    for field in (*ECONOMIC_FIELDS, "economics_source"):
        calculated = [metrics.get(tuple(row), {}).get(field)
                      for row in rows[["source_file", *IDENTITY]].itertuples(index=False, name=None)]
        if field == "needed_annual_support_krw":
            rows[field] = rows[field].where(rows[field].notna(), calculated)
        else:
            rows[field] = calculated
    rows["equipment"] = rows.apply(equipment_label, axis=1)
    return rows, report


def visible_candidates(rows: pd.DataFrame, line: str, battery="all", size="all") -> pd.DataFrame:
    selected = rows[rows.line_case == line]
    if battery != "all":
        selected = selected[selected.battery == battery]
    if size != "all":
        power, energy = map(float, size.split(":"))
        selected = selected[(selected.ess_power_mw == power) & (selected.ess_energy_mwh == energy)]
    return selected.sort_values("npv_krw", ascending=False, na_position="last")


def selected_id(rows: pd.DataFrame, current: str | None) -> str | None:
    if rows.empty:
        return None
    if current in rows.id.values:
        return current
    preferred = rows[(rows.battery == "kr_notice_example") & (rows.ess_power_mw == 2) & (rows.ess_energy_mwh == 8)]
    return str((preferred if not preferred.empty else rows).iloc[0].id)
