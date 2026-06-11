import ollama
import pandas as pd

from athena.config import (
    ENABLE_AGENT_PIPELINE,
    MAX_RETRIES,
    MAX_REVIEW_RETRIES,
    MODEL,
)
from athena.llm.agents import (
    build_analyst_user_message,
    deterministic_review,
    plan_analysis,
    review_result,
)
from athena.llm.execution import extract_code, friendly_execution_error, run_code
from athena.llm.prompts import build_system_prompt
from athena.llm.schema import categorical_columns, numeric_columns
from athena.llm.summary import summarise_result


def _chat_codegen(
    messages: list[dict],
    df: pd.DataFrame,
) -> tuple[str | None, str | None, object, str | None]:
    """Run model, extract code, execute. Returns (code, result, error, raw)."""
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": 0.1},
    )
    raw = response["message"]["content"]
    code = extract_code(raw)

    if not code:
        return None, None, "Model did not return a recognisable code block.", raw

    result, error = run_code(code, df)
    return code, result, error, raw


def generate_and_run(
    question: str,
    df: pd.DataFrame,
    schema: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Main entry point called by the Streamlit app.

    Optional agent pipeline: Planner → Analyst (code) → Reviewer (one retry).
    """
    system_prompt = build_system_prompt(schema, df)
    history = chat_history or []
    numeric_hint = ", ".join(numeric_columns(df)[:12]) or "none"
    categorical_hint = ", ".join(categorical_columns(df)[:12]) or "none"

    plan = None
    if ENABLE_AGENT_PIPELINE:
        plan = plan_analysis(question, schema, df)

    user_content = build_analyst_user_message(question, plan)
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_content}]
    )

    code, result, error, raw = _chat_codegen(messages, df)

    if error and MAX_RETRIES > 0:
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"That code raised an error:\n\n{error}\n\n"
                    'Fix it and return only corrected code as {"code": "..."} JSON.\n\n'
                    "Important constraints:\n"
                    f"- Numeric columns likely to aggregate: {numeric_hint}\n"
                    f"- Non-numeric/grouping columns: {categorical_hint}\n"
                    "- Never run mean/median/sum/min/max on string columns.\n"
                    "- For numeric-like text columns, use pd.to_numeric(df['col'], errors='coerce') before comparisons.\n"
                    "- Never use DataFrame.append(); use pd.concat([a, b], ignore_index=True).\n"
                    "- If a requested column doesn't exist, pick the closest valid column from schema."
                ),
            },
        ]
        code, result, error, raw = _chat_codegen(retry_messages, df)

    review_feedback = None
    if (
        ENABLE_AGENT_PIPELINE
        and not error
        and code
        and MAX_REVIEW_RETRIES > 0
    ):
        review = review_result(question, plan, result, code)
        if not review["ok"] and review.get("feedback"):
            review_feedback = review["feedback"]
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Reviewer: the result does not answer the question.\n\n"
                        f"{review['feedback']}\n\n"
                        'Return only corrected code as {"code": "..."} JSON that fixes this.'
                    ),
                },
            ]
            code2, result2, error2, raw2 = _chat_codegen(retry_messages, df)
            if code2:
                code, result, error, raw = code2, result2, error2, raw2

    if error:
        error = friendly_execution_error(error, df)

    # Final rule check: if the answer still looks wrong after retries,
    # tell the UI so the user is not misled by a NaN/empty "result".
    result_warning = None
    if not error and code:
        final_verdict = deterministic_review(question, plan, result)
        if not final_verdict["ok"]:
            result_warning = final_verdict["feedback"]

    plan_summary = None
    if plan and plan.get("columns"):
        plan_summary = ", ".join(plan["columns"][:5])

    return {
        "code": code,
        "result": result,
        "error": error,
        "raw_response": raw,
        "plan": plan,
        "plan_summary": plan_summary,
        "review_feedback": review_feedback,
        "result_warning": result_warning,
    }


if __name__ == "__main__":
    from athena.llm.schema import build_schema_string

    print("Running smoke test against Ollama...\n")

    sample_df = pd.DataFrame({
        "Country": ["USA", "Germany", "Poland", "USA", "Germany"],
        "LanguageHaveWorkedWith": ["Python;SQL", "Python;Java", "Python", "JavaScript", "Python;SQL"],
        "ConvertedCompYearly": [120000, 85000, 45000, 95000, 90000],
        "YearsCodePro": [5, 8, 2, 3, 10],
    })

    schema = build_schema_string(sample_df)
    print("Schema:\n", schema, "\n")

    question = "What is the average yearly compensation by country?"
    print(f"Question: {question}\n")

    output = generate_and_run(question, sample_df, schema)

    if output["error"]:
        print("ERROR:", output["error"])
    else:
        print("Generated code:\n", output["code"])
        print("\nResult:\n", output["result"])
        print("\nNarrative:", summarise_result(question, output["result"]))
