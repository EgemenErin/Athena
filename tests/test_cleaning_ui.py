import pandas as pd

from athena.ui.cleaning import _resolution_choices


def test_drop_column_offers_fill_options():
    df = pd.DataFrame({
        "RemoteWork": ["Yes", None, "No", "Yes", None],
        "pay": [1.0, 2.0, None, 4.0, 5.0],
    })
    action = {
        "id": "1",
        "type": "drop_column",
        "column": "RemoteWork",
        "reason": "16% nulls",
    }
    labels = [c[0] for c in _resolution_choices(action, df)]
    assert "Drop column" in labels
    assert "Fill nulls (mode)" in labels
    assert "Fill nulls (median)" not in labels


def test_drop_numeric_column_offers_median():
    df = pd.DataFrame({"pay": [1.0, None, 3.0]})
    action = {"id": "1", "type": "drop_column", "column": "pay", "reason": "nulls"}
    labels = [c[0] for c in _resolution_choices(action, df)]
    assert "Fill nulls (median)" in labels


def test_fill_null_offers_drop():
    df = pd.DataFrame({"pay": [1.0, None, 3.0]})
    action = {"id": "1", "type": "fill_null", "column": "pay", "strategy": "median", "reason": "x"}
    labels = [c[0] for c in _resolution_choices(action, df)]
    assert "Drop column" in labels
