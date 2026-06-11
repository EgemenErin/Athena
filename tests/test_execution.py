import pandas as pd

from athena.llm.execution import (
    extract_code,
    friendly_execution_error,
    run_code,
    validate_code_safety,
)


def _df():
    return pd.DataFrame({
        "Country": ["USA", "Germany", "Poland"],
        "Salary": [120000, 85000, 45000],
    })


# ---------- extract_code ----------

def test_extract_code_from_json():
    raw = '{"code": "result = len(df)"}'
    assert extract_code(raw) == "result = len(df)"


def test_extract_code_from_fenced_json():
    raw = 'Here you go:\n```json\n{"code": "result = df[\\"Salary\\"].mean()"}\n```'
    assert extract_code(raw) == 'result = df["Salary"].mean()'


def test_extract_code_from_python_fence():
    raw = "```python\nresult = df.shape[0]\n```"
    assert extract_code(raw) == "result = df.shape[0]"


def test_extract_code_from_generic_fence():
    raw = "```\nresult = 42\n```"
    assert extract_code(raw) == "result = 42"


def test_extract_code_bare_result():
    assert extract_code("result = 1 + 1") == "result = 1 + 1"


def test_extract_code_none_for_prose():
    assert extract_code("I cannot answer that question.") is None


def test_extract_code_json_with_multiline_code():
    raw = '{"code": "x = df[\\"Salary\\"]\\nresult = x.sum()"}'
    assert extract_code(raw) == 'x = df["Salary"]\nresult = x.sum()'


# ---------- validate_code_safety ----------

def test_safety_blocks_import():
    assert validate_code_safety("import os\nresult = 1") is not None


def test_safety_blocks_from_import():
    assert validate_code_safety("from os import path\nresult = 1") is not None


def test_safety_blocks_open():
    assert validate_code_safety("result = open('x.txt').read()") is not None


def test_safety_blocks_dunder():
    assert validate_code_safety("result = ().__class__") is not None


def test_safety_blocks_os_attribute():
    assert validate_code_safety("result = os.listdir('.')") is not None


def test_safety_blocks_eval():
    assert validate_code_safety("result = eval('1+1')") is not None


def test_safety_allows_normal_pandas():
    code = 'result = df.groupby("Country")["Salary"].mean().reset_index()'
    assert validate_code_safety(code) is None


# ---------- run_code ----------

def test_run_code_success():
    result, error = run_code("result = len(df)", _df())
    assert error is None
    assert result == 3


def test_run_code_missing_result():
    result, error = run_code("x = 5", _df())
    assert result is None
    assert "did not assign" in error


def test_run_code_blocked_code_friendly_error():
    result, error = run_code("import os\nresult = 1", _df())
    assert result is None
    assert "blocked for safety" in error


def test_run_code_timeout():
    slow = "x = 0\nfor i in range(10**8):\n    x += 1\nresult = x"
    result, error = run_code(slow, _df(), timeout=0.2)
    assert result is None
    assert "longer than" in error


def test_run_code_no_builtin_import():
    # Restricted builtins: __import__ unavailable even without the keyword.
    result, error = run_code("result = getattr(pd, 'io')", _df())
    assert result is None
    assert error is not None


# ---------- friendly_execution_error ----------

def test_friendly_error_missing_column():
    err = "KeyError: 'Salry'"
    friendly = friendly_execution_error(err, _df())
    assert "Salry" in friendly
    assert "not found" in friendly
