"""Render one saved economics result as a Korean PDF download."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _ttf_font_path() -> Path | None:
    candidates = (
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    )
    return next((value for value in candidates if value.exists()), None)


def _cid_font() -> str:
    """Register ReportLab's built-in Korean font for minimal CI images."""
    fallback = "HYSMyeongJo-Medium"
    if fallback not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(fallback))
    return fallback


def _font() -> str:
    name = "EconomicsKorean"
    if name in pdfmetrics.getRegisteredFontNames():
        return name
    path = _ttf_font_path()
    if path is not None:
        pdfmetrics.registerFont(TTFont(name, str(path)))
        return name

    # GitHub Actions 같은 최소 Ubuntu 환경에는 Nanum 글꼴이 없을 수 있다.
    # ReportLab이 기본 제공하는 한국어 CID 글꼴을 사용하면 시스템 패키지 없이도
    # PDF 생성 자체는 동일하게 검증할 수 있다. Docker에서는 위의 Nanum 글꼴을 쓴다.
    return _cid_font()


def _number(value, digits: int = 1) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value):,.{digits}f}"


def _won(value) -> str:
    if value is None or pd.isna(value):
        return "산출값 없음"
    value = float(value)
    if abs(value) >= 1e8:
        return f"{value / 1e8:,.2f}억 원"
    return f"{value / 1e4:,.1f}만 원"


def _percent(value, missing: str = "산출값 없음") -> str:
    return missing if value is None or pd.isna(value) else f"{float(value) * 100:,.1f}%"


def _years(value, missing: str = "20년 내 회수 불가") -> str:
    return missing if value is None or pd.isna(value) else f"{float(value):,.1f}년"


def build_economics_pdf(
    candidate: pd.Series,
    summary: pd.Series,
    cashflow: pd.DataFrame,
    capacity_summary: pd.Series | None = None,
) -> bytes:
    """Return a PDF containing saved summary metrics and the 20-year cashflow."""
    font = _font()
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=13 * mm,
        title="상세 경제성 분석 결과",
        author="Jeju VPP Dispatch Agent",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "KoreanTitle", parent=styles["Title"], fontName=font, fontSize=19, leading=25
    )
    heading = ParagraphStyle(
        "KoreanHeading", parent=styles["Heading2"], fontName=font, fontSize=12, leading=17
    )
    body = ParagraphStyle(
        "KoreanBody", parent=styles["BodyText"], fontName=font, fontSize=8.5, leading=13
    )
    center = ParagraphStyle("KoreanCenter", parent=body, alignment=TA_CENTER)
    cash_center = ParagraphStyle(
        "KoreanCashflow", parent=center, fontSize=7.2, leading=8.5
    )
    story = [
        Paragraph("상세 경제성 분석 결과", title),
        Paragraph(str(candidate.equipment), heading),
        Paragraph("기준 시나리오 · 표준 배분 · ESS 용량정산금 민감도 비교", body),
        Spacer(1, 5 * mm),
    ]
    reference_year = int(summary.revenue_reference_year)
    assumptions = [
        (
            "수입 기준자료",
            f"{reference_year}년 1년간 예측으로 확정한 판매계획을 실제 발전량·시장가격에 "
            "적용한 ESS 모의 실행·정산 결과를 365일 기준 연간 값으로 사용한다.",
        ),
        (
            "시장수입 장기 적용",
            f"장기 발전량·시장가격 전망이 없어 {reference_year}년 기준수입에 연 변화율 "
            f"{float(summary.revenue_growth_rate) * 100:.1f}%를 적용한다. 기준 시나리오에서는 "
            "1~20년차의 명목 시장수입이 동일하다.",
        ),
        (
            "비용과 현재가치",
            f"운영비·교체비는 매년 {float(summary.cost_escalation) * 100:.1f}% 상승하고, "
            f"현금흐름은 {float(summary.discount_rate) * 100:.1f}% 할인율로 현재가치화한다. "
            "교체 배터리 단가는 매년 "
            f"{float(summary.battery_price_decline) * 100:.1f}% 하락한다고 가정한다.",
        ),
        (
            "정부지원과 수익 배분",
            f"구축비 지원율은 최대 {float(summary.subsidy_rate) * 100:.0f}% 가정을 적용한다. "
            "VPP 수입은 ESS 방전수익 100%, 태양광 용량정산금 50%, 임밸런스 페널티 100%를 "
            "반영한다. 기본 판단에서는 ESS 용량정산금을 제외하고, 인정 가정은 별도 민감도 "
            "시나리오로 비교한다.",
        ),
        (
            "장기 운영 범위",
            f"분석기간은 {int(summary.project_years)}년이다. 20년의 시간별 운전을 매년 다시 "
            "최적화하지 않고 1년 모의 운영 결과를 장기 현금흐름으로 확장한다. 배터리 용량감소에 "
            "따른 연도별 운영량 변화도 별도로 재계산하지 않는다.",
        ),
        (
            "결과 해석",
            "이 결과는 장기 수입을 예측하거나 보장하는 값이 아니라, 동일한 가정 아래 설비 후보와 "
            "참여 타당성을 비교하기 위한 시나리오 분석이다.",
        ),
    ]
    assumption_cells = [
        [Paragraph(label, body), Paragraph(description, body)]
        for label, description in assumptions
    ]
    assumption_table = Table(
        assumption_cells,
        colWidths=[45 * mm, 187 * mm],
        hAlign="LEFT",
        splitByRow=0,
    )
    assumption_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F3F1")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEDC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            Paragraph("분석 기준과 해석 범위", heading),
            assumption_table,
            PageBreak(),
            Paragraph("핵심 경제성 지표", heading),
        ]
    )
    replacement = str(summary.battery_replacement_years)
    summary_rows = [
        (
            f"{reference_year}년 모의 운영 기준 연간 VPP 수입",
            _won(summary.annual_market_revenue_krw),
        ),
        ("구축비", _won(summary.capex_krw)),
        ("정부지원금", _won(summary.subsidy_krw)),
        ("20년 NPV", _won(summary.npv_krw)),
        ("IRR", _percent(summary.irr)),
        ("단순 투자비 회수기간", _years(summary.payback_years)),
        ("할인 투자비 회수기간", _years(summary.discounted_payback_years)),
        ("배터리 예상 수명", _years(summary.battery_life_years, "산출값 없음")),
        ("배터리 교체 연도", "교체 없음" if replacement == "-" else f"{replacement}년차"),
        (
            "손익분기 정부지원율",
            _percent(summary.breakeven_subsidy_rate, "100% 지원으로도 손익분기 불가"),
        ),
        ("손익분기 구축비", _won(summary.breakeven_capex_krw)),
        (
            "손익분기 PPA 단가",
            "해당 없음"
            if pd.isna(summary.breakeven_ppa_krw_per_kwh)
            else f"{_number(summary.breakeven_ppa_krw_per_kwh)}원/kWh",
        ),
        ("필요한 최소 연간 시장수익", _won(summary.min_annual_market_revenue_krw)),
        (
            "필요한 연간 추가 보상액",
            "필요 없음"
            if float(summary.needed_annual_support_krw) == 0
            else _won(summary.needed_annual_support_krw),
        ),
    ]
    cells = [[Paragraph(label, body), Paragraph(value, body)] for label, value in summary_rows]
    summary_table = Table(
        cells,
        colWidths=[55 * mm, 70 * mm],
        hAlign="LEFT",
        splitByRow=0,
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F6F5")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#52676D")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEDC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            summary_table,
            Spacer(1, 2 * mm),
            Paragraph(
                "NPV는 20년 수입과 비용을 할인율로 현재가치화한 결과이며, IRR은 NPV를 0으로 "
                "만드는 수익률이다. 필요한 연간 추가 보상액은 현재 가정에서 NPV를 0으로 만들기 "
                "위해 매년 동일하게 필요한 금액이다.",
                body,
            ),
        ]
    )
    if capacity_summary is not None:
        base_payback = summary.discounted_payback_years
        capacity_payback = capacity_summary.discounted_payback_years
        if pd.isna(base_payback) and not pd.isna(capacity_payback):
            payback_change = "20년 내 회수로 전환"
        elif pd.isna(base_payback) or pd.isna(capacity_payback):
            payback_change = "비교 불가"
        else:
            payback_change = f"{float(base_payback) - float(capacity_payback):,.1f}년 단축"
        irr_change = (
            "비교 불가"
            if pd.isna(summary.irr) or pd.isna(capacity_summary.irr)
            else f"+{(float(capacity_summary.irr) - float(summary.irr)) * 100:,.1f}%p"
        )
        comparison = [
            (
                "연간 ESS 용량정산금",
                "미포함",
                _won(capacity_summary.annual_ess_capacity_payment_krw),
                f"+{_won(capacity_summary.annual_ess_capacity_payment_krw)}",
            ),
            (
                f"{reference_year}년 기준 연간 VPP 수입",
                _won(summary.annual_market_revenue_krw),
                _won(capacity_summary.annual_market_revenue_krw),
                "+"
                + _won(
                    capacity_summary.annual_market_revenue_krw
                    - summary.annual_market_revenue_krw
                ),
            ),
            (
                "20년 NPV",
                _won(summary.npv_krw),
                _won(capacity_summary.npv_krw),
                f"+{_won(capacity_summary.npv_krw - summary.npv_krw)}",
            ),
            (
                "IRR",
                _percent(summary.irr),
                _percent(capacity_summary.irr),
                irr_change,
            ),
            (
                "할인 투자비 회수기간",
                _years(summary.discounted_payback_years),
                _years(capacity_summary.discounted_payback_years),
                payback_change,
            ),
            (
                "필요한 연간 추가 보상액",
                _won(summary.needed_annual_support_krw),
                _won(capacity_summary.needed_annual_support_krw),
                _won(
                    summary.needed_annual_support_krw
                    - capacity_summary.needed_annual_support_krw
                )
                + " 감소",
            ),
        ]
        comparison_rows = [
            [
                Paragraph("비교 항목", center),
                Paragraph("기준: 미포함", center),
                Paragraph("인정 가정: 포함", center),
                Paragraph("개선 효과", center),
            ]
        ]
        comparison_rows.extend(
            [Paragraph(value, center) for value in row] for row in comparison
        )
        comparison_table = Table(
            comparison_rows,
            colWidths=[56 * mm, 52 * mm, 52 * mm, 52 * mm],
            repeatRows=1,
            hAlign="LEFT",
            splitByRow=0,
        )
        comparison_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315F83")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEDC")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend(
            [
                PageBreak(),
                Paragraph("ESS 용량정산금 인정 여부 민감도", heading),
                Paragraph(
                    "인정 가정은 ESS 용량정산금이 VPP에 100% 귀속되는 경우의 잠재적 개선 "
                    "효과를 보여준다. 실제 인정 여부가 확정되지 않은 수익이므로 참여 판단의 "
                    "기준값으로 사용하거나 확정 수익으로 해석해서는 안 된다.",
                    body,
                ),
                Spacer(1, 2 * mm),
                comparison_table,
            ]
        )
    story.extend([PageBreak(), Paragraph("20년 연도별 현금흐름", heading)])
    headers = (
        "연도",
        f"시장수입({reference_year} 기준 가정)",
        "운영비",
        "배터리 교체비",
        "PCS 교체비",
        "순현금흐름",
        "할인 누적 현금흐름",
    )
    flow_rows = [[Paragraph(value, cash_center) for value in headers]]
    for row in cashflow.sort_values("year").itertuples(index=False):
        flow_rows.append(
            [
                Paragraph(f"{int(row.year)}년차", cash_center),
                Paragraph(_won(row.market_revenue), cash_center),
                Paragraph(_won(row.om + row.insurance_inspection), cash_center),
                Paragraph(_won(row.battery_replacement), cash_center),
                Paragraph(_won(row.pcs_replacement), cash_center),
                Paragraph(_won(row.net), cash_center),
                Paragraph(_won(row.cumulative_discounted), cash_center),
            ]
        )
    flow_table = Table(
        flow_rows,
        colWidths=[18 * mm, 35 * mm, 32 * mm, 36 * mm, 32 * mm, 36 * mm, 43 * mm],
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=0,
    )
    flow_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B7F73")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DEDC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.extend(
        [
            Paragraph("0년차는 설비 구축과 정부지원금이 반영된 투자 시점입니다.", body),
            Spacer(1, 2 * mm),
            flow_table,
            Spacer(1, 4 * mm),
            Paragraph(
                "이 문서는 저장된 경제성 결과를 사용하며 PDF 요청 시 재계산하지 않습니다.",
                body,
            ),
        ]
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7)
        canvas.setFillColor(colors.HexColor("#6D7E82"))
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 7 * mm, f"{doc.page}쪽")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
