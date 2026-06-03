"""Export saved dashboard charts to a multi-page PDF (2x2 grid per page)."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import plotly.io as pio
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from athena.ui.charts import build_chart_from_spec

CHARTS_PER_PAGE = 4
COLS_PER_ROW = 2
PNG_WIDTH = 520
PNG_HEIGHT = 360
PNG_SCALE = 2


def _plotly_figure_to_png(fig) -> bytes | None:
    try:
        return pio.to_image(
            fig,
            format="png",
            width=PNG_WIDTH,
            height=PNG_HEIGHT,
            scale=PNG_SCALE,
        )
    except Exception:
        return None


def build_dashboard_pdf(df: pd.DataFrame, specs: list[dict[str, Any]]) -> bytes:
    """Render saved chart specs into a PDF with four charts per page (2x2)."""
    page_w, page_h = letter
    margin = 0.55 * inch
    gutter = 0.2 * inch
    title_h = 0.22 * inch

    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - title_h
    cell_w = (usable_w - gutter) / COLS_PER_ROW
    cell_h = (usable_h - gutter) / COLS_PER_ROW
    img_h = cell_h - title_h

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    slot = 0
    for spec in specs:
        title = (spec.get("title") or "Chart")[:80]
        fig = build_chart_from_spec(df, spec)
        png = _plotly_figure_to_png(fig) if fig is not None else None

        row = (slot % CHARTS_PER_PAGE) // COLS_PER_ROW
        col = slot % COLS_PER_ROW

        x0 = margin + col * (cell_w + gutter)
        y_top = page_h - margin - row * (cell_h + gutter)
        y_img = y_top - img_h
        y_title = y_img - title_h

        if png:
            pdf.drawImage(
                ImageReader(io.BytesIO(png)),
                x0,
                y_img,
                width=cell_w,
                height=img_h,
                preserveAspectRatio=True,
                anchor="sw",
            )
        else:
            pdf.setFont("Helvetica", 9)
            pdf.drawString(x0, y_img + img_h / 2, "Could not render chart")

        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(x0, y_title, title)

        slot += 1
        if slot % CHARTS_PER_PAGE == 0 and slot < len(specs):
            pdf.showPage()

    if slot == 0:
        pdf.setFont("Helvetica", 12)
        pdf.drawString(margin, page_h / 2, "No charts to export.")

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
