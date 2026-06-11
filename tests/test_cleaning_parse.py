"""Phase 3 cleaning: parse hardening, retries, heuristics, validation."""

import json

import pandas as pd

import athena.llm.cleaning_columns as cc
from athena.llm.cleaning import validate_action
from athena.llm.cleaning_columns import (
    _ai_analyze_column_batches,
    _parse_batch_actions,
    build_per_column_proposal,
    duplicate_columns,
    is_junk_index_column,
)


# ---------- _parse_batch_actions ----------

def test_parse_malformed_json_returns_empty():
    assert _parse_batch_actions("not json at all", ["a"]) == []
    assert _parse_batch_actions("{broken: json", ["a"]) == []
    assert _parse_batch_actions("", ["a"]) == []


def test_parse_fenced_json():
    raw = '```json\n{"actions": [{"column": "a", "type": "skip", "reason": "ok"}]}\n```'
    parsed = _parse_batch_actions(raw, ["a"])
    assert len(parsed) == 1
    assert parsed[0]["type"] == "skip"


def test_parse_drops_invalid_strategy():
    raw = json.dumps({
        "actions": [
            {"column": "a", "type": "fill_null", "strategy": "banana"},
            {"column": "b", "type": "fill_null", "strategy": "median"},
        ]
    })
    parsed = _parse_batch_actions(raw, ["a", "b"])
    assert [a["column"] for a in parsed] == ["b"]


def test_parse_drops_constant_without_value():
    raw = json.dumps({
        "actions": [{"column": "a", "type": "fill_null", "strategy": "constant"}]
    })
    assert _parse_batch_actions(raw, ["a"]) == []


def test_parse_drops_invalid_outlier_method():
    raw = json.dumps({
        "actions": [
            {"column": "a", "type": "drop_outlier_rows", "method": "magic"},
            {"column": "b", "type": "drop_outlier_rows", "method": "zscore"},
        ]
    })
    parsed = _parse_batch_actions(raw, ["a", "b"])
    assert [a["column"] for a in parsed] == ["b"]


def test_parse_drops_invalid_percentiles():
    raw = json.dumps({
        "actions": [
            {"column": "a", "type": "cap_outliers", "lower_percentile": 99, "upper_percentile": 1},
            {"column": "b", "type": "cap_outliers", "lower_percentile": "x", "upper_percentile": 99},
            {"column": "c", "type": "cap_outliers", "lower_percentile": 1, "upper_percentile": 99},
        ]
    })
    parsed = _parse_batch_actions(raw, ["a", "b", "c"])
    assert [a["column"] for a in parsed] == ["c"]


def test_parse_drops_unknown_and_duplicate_columns():
    raw = json.dumps({
        "actions": [
            {"column": "ghost", "type": "skip"},
            {"column": "a", "type": "skip"},
            {"column": "a", "type": "drop_column"},
        ]
    })
    parsed = _parse_batch_actions(raw, ["a"])
    assert len(parsed) == 1
    assert parsed[0]["type"] == "skip"


# ---------- batch retries ----------

def test_batch_retry_after_failed_call(monkeypatch):
    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    profiles = {c: cc.column_profile(df, c) for c in df.columns}
    calls = {"n": 0}

    def fake_run(df_, batch_cols, profiles_, idx, total):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # first call fails
        return [{"id": c, "column": c, "type": "skip", "reason": "ok"} for c in batch_cols]

    monkeypatch.setattr(cc, "_run_ai_batch", fake_run)
    outcome = _ai_analyze_column_batches(df, profiles)
    assert calls["n"] == 2
    assert outcome["failed_batches"] == 0
    assert {a["column"] for a in outcome["actions"]} == {"a", "b"}


def test_batch_partial_response_retried_with_missing_columns(monkeypatch):
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    profiles = {c: cc.column_profile(df, c) for c in df.columns}
    seen_batches = []

    def fake_run(df_, batch_cols, profiles_, idx, total):
        seen_batches.append(list(batch_cols))
        if len(seen_batches) == 1:
            return [{"id": "a", "column": "a", "type": "skip", "reason": "ok"}]
        return [{"id": c, "column": c, "type": "skip", "reason": "ok"} for c in batch_cols]

    monkeypatch.setattr(cc, "_run_ai_batch", fake_run)
    outcome = _ai_analyze_column_batches(df, profiles)
    assert seen_batches[1] == ["b", "c"]  # smaller follow-up batch
    assert {a["column"] for a in outcome["actions"]} == {"a", "b", "c"}


def test_failed_batches_counted(monkeypatch):
    df = pd.DataFrame({"a": [1], "b": [2]})
    profiles = {c: cc.column_profile(df, c) for c in df.columns}
    monkeypatch.setattr(cc, "_run_ai_batch", lambda *args: None)
    outcome = _ai_analyze_column_batches(df, profiles)
    assert outcome["failed_batches"] == outcome["total_batches"] == 1
    assert outcome["actions"] == []


# ---------- proposal merge + source badges ----------

def test_heuristic_proposal_tags_source():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    proposal = build_per_column_proposal(df, use_ai=False)
    column_actions = [a for a in proposal["actions"] if a.get("column")]
    assert column_actions
    assert all(a.get("source") == "heuristic" for a in column_actions)


def test_proposal_merges_ai_and_heuristic(monkeypatch):
    df = pd.DataFrame({"a": [1, 2, None], "b": [None, "y", "z"]})

    def fake_batches(df_, profiles, columns=None):
        return {
            "actions": [{"id": "a", "column": "a", "type": "skip", "reason": "ai said ok"}],
            "failed_batches": 1,
            "total_batches": 2,
        }

    monkeypatch.setattr(cc, "_ai_analyze_column_batches", fake_batches)
    proposal = build_per_column_proposal(df, use_ai=True)
    by_col = {a["column"]: a for a in proposal["actions"] if a.get("column")}
    assert by_col["a"]["source"] == "ai"
    assert by_col["b"]["source"] == "heuristic"
    assert proposal["ai_failed_batches"] == 1
    assert proposal["ai_total_batches"] == 2


# ---------- structural heuristics ----------

def test_junk_index_column_detected():
    df = pd.DataFrame({"Unnamed: 0": range(5), "value": [1, 2, 3, 4, 5]})
    assert is_junk_index_column(df, "Unnamed: 0") is True
    assert is_junk_index_column(df, "value") is False
    proposal = build_per_column_proposal(df, use_ai=False)
    by_col = {a["column"]: a for a in proposal["actions"] if a.get("column")}
    assert by_col["Unnamed: 0"]["type"] == "drop_column"


def test_duplicate_columns_detected():
    df = pd.DataFrame({
        "name": ["a", "b", "c"],
        "name_copy": ["a", "b", "c"],
        "other": [1, 2, 3],
    })
    dupes = duplicate_columns(df)
    assert dupes == {"name_copy": "name"}
    proposal = build_per_column_proposal(df, use_ai=False)
    by_col = {a["column"]: a for a in proposal["actions"] if a.get("column")}
    assert by_col["name_copy"]["type"] == "drop_column"
    assert by_col["name"]["type"] != "drop_column"


def test_numeric_like_text_coerced_for_profiles():
    # "1 000 000" style outliers hidden in text columns get detected after coercion.
    df = pd.DataFrame({
        "pay_text": ["100", "110", "105", "2000000"],
    })
    proposal = build_per_column_proposal(df, use_ai=False)
    by_col = {a["column"]: a for a in proposal["actions"] if a.get("column")}
    assert by_col["pay_text"]["type"] in ("drop_outlier_rows", "cap_outliers")


# ---------- validate_action hardening ----------

def test_validate_blocks_mean_on_object_column():
    df = pd.DataFrame({"name": ["a", None, "c"]})
    err = validate_action(df, {"type": "fill_null", "column": "name", "strategy": "mean"})
    assert err is not None
    assert "not numeric" in err


def test_validate_allows_mode_on_object_column():
    df = pd.DataFrame({"name": ["a", None, "c"]})
    err = validate_action(df, {"type": "fill_null", "column": "name", "strategy": "mode"})
    assert err is None
