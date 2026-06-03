from athena.llm.personas import (
    BUSINESS_ANALYST_RULES,
    SENIOR_ANALYST_RULES,
    SUMMARY_ANALYST_RULES,
)
from athena.llm.prompts import build_system_prompt


def test_senior_analyst_rules_present():
    assert "EXACT question" in SENIOR_ANALYST_RULES or "exact" in SENIOR_ANALYST_RULES.lower()
    assert "corr()" in SENIOR_ANALYST_RULES or "correlation" in SENIOR_ANALYST_RULES.lower()
    assert "semicolon" in SENIOR_ANALYST_RULES.lower()


def test_build_system_prompt_includes_schema():
    schema = "Column: 'Country' dtype: object"
    prompt = build_system_prompt(schema)
    assert schema in prompt
    assert "senior data analyst" in prompt.lower()


def test_summary_grounding_rules():
    assert "invent" in SUMMARY_ANALYST_RULES.lower() or "only describe" in SUMMARY_ANALYST_RULES.lower()


def test_business_analyst_rules():
    assert "business analyst" in BUSINESS_ANALYST_RULES.lower()
    assert "backticks" in BUSINESS_ANALYST_RULES.lower()
