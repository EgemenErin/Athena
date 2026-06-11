"""Offline smoke test for the manual CSV scenarios (run directly, no Ollama)."""

import numpy as np
import pandas as pd

from athena.llm.chart_suggestions import _fallback_chart_suggestions, chart_description
from athena.llm.cleaning import apply_cleaning_actions, preview_stats
from athena.llm.cleaning_columns import build_per_column_proposal
from athena.llm.schema import build_schema_string
from athena.ui.charts import build_chart_from_spec, chart_truncation_note

rng = np.random.default_rng(7)
n = 300

df = pd.DataFrame({
    "Unnamed: 0": range(n),
    "Country": rng.choice([f"Country{i}" for i in range(60)], n),
    "JobTitle": rng.choice([f"Title{i}" for i in range(40)], n),
    "LanguageHaveWorkedWith": rng.choice(
        ["Python;SQL", "Python;Java;SQL", "JavaScript", "Go;Rust", None], n
    ),
    "Remote": rng.choice([0, 1], n),
    "Verified": pd.array(rng.choice([True, False, None], n), dtype="boolean"),
    "Salary": np.where(rng.random(n) < 0.02, 5_000_000, rng.normal(90_000, 20_000, n)),
    "Salary_copy": np.nan,
    "Month": rng.choice(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"], n
    ),
})
df["Salary_copy"] = df["Salary"]
df.loc[rng.choice(n, 30, replace=False), "Salary"] = np.nan
df = pd.concat([df, df.head(10)], ignore_index=True)  # duplicates

print("=== schema ===")
print(build_schema_string(df)[:400], "…\n")

print("=== cleaning proposal (heuristic only) ===")
proposal = build_per_column_proposal(df, use_ai=False)
print(proposal["summary"])
for a in proposal["actions"]:
    print(f"  [{a.get('source', 'rule'):9s}] {a.get('type'):18s} {a.get('column') or '-':24s} {a.get('reason', '')[:60]}")

selected = [a for a in proposal["actions"] if a.get("type") != "skip"]
cleaned = apply_cleaning_actions(df, selected)
print("before:", preview_stats(df), "-> after:", preview_stats(cleaned))

print("\n=== chart suggestions (fallback) ===")
specs = _fallback_chart_suggestions(df, n=6)
for spec in specs:
    fig = build_chart_from_spec(df, spec)
    note = chart_truncation_note(fig) if fig else None
    status = "OK " if fig else "FAIL"
    print(f"  [{status}] {spec['chart_type']:9s} x={spec.get('x')!r:30} y={spec.get('y')!r:12} agg={spec.get('aggregation')}")
    if note:
        print(f"         note: {note}")
    assert fig is not None, f"chart failed to render: {spec}"

flag_specs = [s for s in specs if s.get("aggregation") == "pct_true"]
mean_of_flag = [
    s for s in specs
    if s.get("chart_type") in ("bar", "scatter")
    and s.get("y") in ("Remote", "Verified")
    and s.get("aggregation") not in ("pct_true", "count")
]
assert not mean_of_flag, "boolean flags must never be averaged"
print(f"\npct_true suggestions: {len(flag_specs)}; no mean-of-boolean charts. All good.")
