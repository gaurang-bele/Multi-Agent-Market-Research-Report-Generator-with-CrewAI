"""
report/pdf_generator.py

Compiles the final Markdown report and Pandas statistics into a
professionally styled A4 PDF using ReportLab.

Features:
  - Branded header banner with document metadata
  - Styled section headings (H1 / H2 / H3) and body paragraphs
  - Tabular grid rendering Pandas summary statistics
  - Automatic "Page X of Y" footer on every page
  - Multi-page support via SimpleDocTemplate with Platypus flowables
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable


# ─────────────────────────────────────────────────────────────────────────────
# Colour Palette
# ─────────────────────────────────────────────────────────────────────────────

BRAND_DARK = colors.HexColor("#1A1F36")      # Header banner background
BRAND_ACCENT = colors.HexColor("#4F67FF")    # Accent lines / table header
BRAND_LIGHT = colors.HexColor("#F4F6FF")     # Table row alternate
TEXT_DARK = colors.HexColor("#1E2235")
TEXT_MUTED = colors.HexColor("#6B7280")
WHITE = colors.white


# ─────────────────────────────────────────────────────────────────────────────
# Style Sheet
# ─────────────────────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()

    styles = {
        "h1": ParagraphStyle(
            "CustomH1",
            parent=base["Heading1"],
            fontSize=20,
            textColor=BRAND_DARK,
            spaceAfter=10,
            spaceBefore=18,
            fontName="Helvetica-Bold",
        ),
        "h2": ParagraphStyle(
            "CustomH2",
            parent=base["Heading2"],
            fontSize=15,
            textColor=BRAND_ACCENT,
            spaceAfter=6,
            spaceBefore=14,
            fontName="Helvetica-Bold",
        ),
        "h3": ParagraphStyle(
            "CustomH3",
            parent=base["Heading3"],
            fontSize=12,
            textColor=BRAND_DARK,
            spaceAfter=4,
            spaceBefore=10,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "CustomBody",
            parent=base["Normal"],
            fontSize=10,
            textColor=TEXT_DARK,
            leading=15,
            spaceAfter=6,
            fontName="Helvetica",
        ),
        "bullet": ParagraphStyle(
            "CustomBullet",
            parent=base["Normal"],
            fontSize=10,
            textColor=TEXT_DARK,
            leading=14,
            leftIndent=18,
            spaceAfter=3,
            bulletIndent=6,
            fontName="Helvetica",
        ),
        "meta": ParagraphStyle(
            "Meta",
            fontSize=8,
            textColor=TEXT_MUTED,
            alignment=TA_RIGHT,
            fontName="Helvetica",
        ),
        "footer": ParagraphStyle(
            "Footer",
            fontSize=8,
            textColor=TEXT_MUTED,
            alignment=TA_CENTER,
            fontName="Helvetica",
        ),
    }
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Header Banner — drawn on canvas (not a flowable)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_header_banner(canvas, doc, query: str, generated_at: str) -> None:
    """Draw the top header banner with branding and document metadata."""
    canvas.saveState()
    page_w, page_h = A4

    # Banner background rectangle
    banner_h = 2.8 * cm
    canvas.setFillColor(BRAND_DARK)
    canvas.rect(0, page_h - banner_h, page_w, banner_h, fill=1, stroke=0)

    # Accent stripe
    canvas.setFillColor(BRAND_ACCENT)
    canvas.rect(0, page_h - banner_h, 6 * mm, banner_h, fill=1, stroke=0)

    # Title text
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(1.4 * cm, page_h - 1.6 * cm, "Market Research Report")

    # Subtitle / query
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#A5B4FC"))
    truncated_query = query[:70] + "..." if len(query) > 70 else query
    canvas.drawString(1.4 * cm, page_h - 2.2 * cm, f"Query: {truncated_query}")

    # Metadata (right-aligned)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#9CA3AF"))
    canvas.drawRightString(page_w - 1.2 * cm, page_h - 1.6 * cm, "Powered by CrewAI")
    canvas.drawRightString(page_w - 1.2 * cm, page_h - 2.1 * cm, f"Generated: {generated_at}")

    canvas.restoreState()


def _draw_footer(canvas, doc) -> None:
    """Draw 'Page X of Y' footer on every page."""
    canvas.saveState()
    page_w, _ = A4

    # Thin separator line
    canvas.setStrokeColor(colors.HexColor("#E5E7EB"))
    canvas.setLineWidth(0.5)
    canvas.line(1.5 * cm, 1.6 * cm, page_w - 1.5 * cm, 1.6 * cm)

    # Footer text
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawCentredString(page_w / 2, 1.1 * cm, f"Page {doc.page} of <total>")
    canvas.drawString(1.5 * cm, 1.1 * cm, "Multi-Agent Market Research System")
    canvas.drawRightString(page_w - 1.5 * cm, 1.1 * cm, "CONFIDENTIAL")

    canvas.restoreState()


class _PageNumDoc(BaseDocTemplate):
    """
    Custom BaseDocTemplate that injects total page count into footer
    via a two-pass build (first pass collects page count, second renders).
    """

    def __init__(self, filename: str, query: str, generated_at: str, **kwargs):
        super().__init__(filename, **kwargs)
        self.query = query
        self.generated_at = generated_at
        self._total_pages = 0

    def handle_documentEnd(self):
        self._total_pages = self.page
        super().handle_documentEnd()

    def afterPage(self):
        pass  # handled in onPage callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Markdown → ReportLab Flowables Parser
# ─────────────────────────────────────────────────────────────────────────────

def _inline_md(text: str) -> str:
    """Convert inline Markdown (**bold**, *italic*, `code`) to ReportLab XML tags."""
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Code
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    # Escape bare ampersands that are not already entities
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", text)
    return text


def _markdown_to_flowables(markdown_text: str, styles: dict) -> list:
    """
    Parse a Markdown string and convert it to a list of ReportLab Platypus
    flowables (Paragraphs, Spacers, Tables, HRFlowables).
    Supports: # H1, ## H2, ### H3, bullet lists (- / *), and body paragraphs.
    Markdown table parsing is handled separately.
    """
    flowables = []
    lines = markdown_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # H1
        if stripped.startswith("# ") and not stripped.startswith("## "):
            flowables.append(Spacer(1, 6))
            flowables.append(Paragraph(_inline_md(stripped[2:]), styles["h1"]))
            flowables.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_ACCENT, spaceAfter=4))

        # H2
        elif stripped.startswith("## ") and not stripped.startswith("### "):
            flowables.append(Spacer(1, 4))
            flowables.append(Paragraph(_inline_md(stripped[3:]), styles["h2"]))

        # H3
        elif stripped.startswith("### "):
            flowables.append(Paragraph(_inline_md(stripped[4:]), styles["h3"]))

        # Horizontal rule
        elif stripped in ("---", "***", "___"):
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E5E7EB"), spaceAfter=6))

        # Markdown table — collect all table lines
        elif stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            tbl = _parse_markdown_table(table_lines, styles)
            if tbl:
                flowables.append(Spacer(1, 4))
                flowables.append(tbl)
                flowables.append(Spacer(1, 6))
            continue

        # Bullet list
        elif stripped.startswith(("- ", "* ", "+ ")):
            bullet_text = _inline_md(stripped[2:])
            flowables.append(Paragraph(f"• {bullet_text}", styles["bullet"]))

        # Numbered list
        elif re.match(r"^\d+\.\s", stripped):
            content = re.sub(r"^\d+\.\s", "", stripped)
            flowables.append(Paragraph(f"• {_inline_md(content)}", styles["bullet"]))

        # Empty line → small spacer
        elif stripped == "":
            if flowables and not isinstance(flowables[-1], Spacer):
                flowables.append(Spacer(1, 4))

        # Regular paragraph
        else:
            flowables.append(Paragraph(_inline_md(stripped), styles["body"]))

        i += 1

    return flowables


def _parse_markdown_table(table_lines: list[str], styles: dict) -> Optional[Table]:
    """Parse Markdown pipe-table lines into a ReportLab Table."""
    if len(table_lines) < 2:
        return None

    rows = []
    for line in table_lines:
        # Skip separator rows (---|---)
        if re.match(r"^\|[-| :]+\|$", line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return None

    num_cols = max(len(r) for r in rows)
    # Pad shorter rows
    data = [r + [""] * (num_cols - len(r)) for r in rows]

    # Convert cells to Paragraphs
    header_style = ParagraphStyle(
        "TableHeader",
        fontSize=9,
        textColor=WHITE,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        "TableCell",
        fontSize=8,
        textColor=TEXT_DARK,
        fontName="Helvetica",
        alignment=TA_LEFT,
    )

    table_data = []
    for row_idx, row in enumerate(data):
        s = header_style if row_idx == 0 else cell_style
        table_data.append([Paragraph(_inline_md(cell), s) for cell in row])

    # Column widths — distribute evenly within content area
    content_width = A4[0] - 3 * cm  # 1.5cm margins each side
    col_width = content_width / num_cols

    tbl = Table(table_data, colWidths=[col_width] * num_cols, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                # Header row
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                # Alternating body rows
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BRAND_LIGHT]),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                # Grid
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Pandas Stats Table
# ─────────────────────────────────────────────────────────────────────────────

def _build_stats_table(stats_df: pd.DataFrame) -> Optional[Table]:
    """Render a Pandas DataFrame (describe() output) as a styled ReportLab Table."""
    if stats_df is None or stats_df.empty:
        return None

    rounded = stats_df.round(2)
    headers = ["Metric"] + list(rounded.columns)
    rows = [[str(idx)] + [str(v) for v in row] for idx, row in rounded.iterrows()]
    data = [headers] + rows

    header_style = ParagraphStyle(
        "StatsHeader", fontSize=8, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        "StatsCell", fontSize=8, textColor=TEXT_DARK,
        fontName="Helvetica", alignment=TA_RIGHT,
    )
    index_style = ParagraphStyle(
        "StatsIndex", fontSize=8, textColor=BRAND_DARK,
        fontName="Helvetica-Bold", alignment=TA_LEFT,
    )

    table_data = []
    for r_idx, row in enumerate(data):
        if r_idx == 0:
            table_data.append([Paragraph(cell, header_style) for cell in row])
        else:
            row_cells = [Paragraph(row[0], index_style)]
            row_cells += [Paragraph(cell, cell_style) for cell in row[1:]]
            table_data.append(row_cells)

    num_cols = len(headers)
    content_width = A4[0] - 3 * cm
    index_col_w = 2.5 * cm
    data_col_w = (content_width - index_col_w) / max(num_cols - 1, 1)
    col_widths = [index_col_w] + [data_col_w] * (num_cols - 1)

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("BACKGROUND", (0, 1), (0, -1), BRAND_LIGHT),
                ("ROWBACKGROUNDS", (1, 1), (-1, -1), [WHITE, colors.HexColor("#F9FAFB")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compile_report(
    markdown_text: str,
    query: str = "Market Research Report",
    stats_df: Optional[pd.DataFrame] = None,
    output_path: str = "output/market_report.pdf",
) -> str:
    """
    Compile the final PDF report from Markdown content and optional stats DataFrame.

    Args:
        markdown_text : Full Markdown string from the Content Writer agent.
        query         : The original research query (shown in header banner).
        stats_df      : Optional Pandas DataFrame (from df.describe()) for the stats table.
        output_path   : Path where the PDF will be written.

    Returns:
        Absolute path to the written PDF file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    styles = _build_styles()

    # ── Document setup ──────────────────────────────────────────────────────
    doc = _PageNumDoc(
        output_path,
        query=query,
        generated_at=generated_at,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=3.5 * cm,   # space for header banner
        bottomMargin=2.0 * cm,
    )

    # Page template with header + footer callbacks
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="main",
    )

    def _on_page(canvas, doc):
        _draw_header_banner(canvas, doc, query, generated_at)
        _draw_footer(canvas, doc)

    page_template = PageTemplate(id="main", frames=[frame], onPage=_on_page)
    doc.addPageTemplates([page_template])

    # ── Build story (flowables) ──────────────────────────────────────────────
    story = []

    # Cover intro block
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            f"<b>Research Query:</b> {_inline_md(query)}",
            ParagraphStyle("QueryBlock", fontSize=11, textColor=BRAND_DARK,
                           fontName="Helvetica-Bold", spaceAfter=4),
        )
    )
    story.append(
        Paragraph(
            f"<i>Generated on {generated_at} by the Multi-Agent Market Research System</i>",
            ParagraphStyle("DateLine", fontSize=9, textColor=TEXT_MUTED,
                           fontName="Helvetica", spaceAfter=12),
        )
    )
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_ACCENT, spaceAfter=10))

    # Main report body from Markdown
    story.extend(_markdown_to_flowables(markdown_text, styles))

    # ── Stats table section ──────────────────────────────────────────────────
    if stats_df is not None and not stats_df.empty:
        story.append(Spacer(1, 0.4 * cm))
        story.append(HRFlowable(width="100%", thickness=1, color=BRAND_ACCENT, spaceAfter=6))
        story.append(Paragraph("Appendix: Statistical Summary (Pandas describe())", styles["h2"]))
        story.append(Spacer(1, 0.2 * cm))
        tbl = _build_stats_table(stats_df)
        if tbl:
            story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))

    # ── Build the document (two-pass for page numbering) ────────────────────
    doc.multiBuild(story)

    # Second pass: replace <total> placeholder with actual page count
    _fix_page_total(output_path, doc._total_pages)

    abs_path = os.path.abspath(output_path)
    print(f"\n✅ PDF report compiled successfully → {abs_path}")
    return abs_path


def _fix_page_total(pdf_path: str, total_pages: int) -> None:
    """
    ReportLab does not natively support 'Page X of Y' in a single pass.
    We perform a simple byte-string replacement on the rendered PDF to
    substitute the '<total>' placeholder with the actual page count.
    This works because the footer text is embedded as a literal string.
    """
    try:
        with open(pdf_path, "rb") as f:
            content = f.read()
        fixed = content.replace(b"<total>", str(total_pages).encode())
        with open(pdf_path, "wb") as f:
            f.write(fixed)
    except Exception:
        pass  # Non-critical — footer still shows "Page X of <total>" if replacement fails
