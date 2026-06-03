import io
from unittest.mock import patch

import pandas as pd
from PIL import Image

from athena.ui.pdf_export import build_dashboard_pdf


def _tiny_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), color=(30, 30, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_build_dashboard_pdf_non_empty():
    df = pd.DataFrame({
        "Country": ["USA", "Germany", "Poland"],
        "Salary": [120000, 85000, 45000],
    })
    specs = [
        {
            "chart_type": "bar",
            "x": "Country",
            "y": "Salary",
            "title": "Pay by country",
            "rationale": "Compare salaries",
        },
    ]
    with patch("athena.ui.pdf_export._plotly_figure_to_png", return_value=_tiny_png()):
        pdf_bytes = build_dashboard_pdf(df, specs)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 200


def test_build_dashboard_pdf_empty_specs():
    df = pd.DataFrame({"a": [1, 2]})
    pdf_bytes = build_dashboard_pdf(df, [])
    assert pdf_bytes.startswith(b"%PDF")
