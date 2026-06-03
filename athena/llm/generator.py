import ollama
import pandas as pd

from athena.config import MAX_RETRIES, MODEL
from athena.llm.execution import extract_code, friendly_execution_error, run_code
from athena.llm.prompts import build_system_prompt
from athena.llm.schema import categorical_columns, numeric_columns
from athena.llm.summary import summarise_result


def generate_and_run(
    question: str,
    df: pd.DataFrame,
    schema: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Main entry point called by the Streamlit app.

    Flow:
      1. Send user question to Ollama with schema in the system prompt
      2. Extract the returned Python code block
      3. Execute it in a sandbox against the real DataFrame
      4. On failure, send the error back for one automatic fix attempt
      5. Return a result dict
    """
    system_prompt = build_system_prompt(schema)
    history = chat_history or []
    numeric_hint = ", ".join(numeric_columns(df)[:12]) or "none"
    categorical_hint = ", ".join(categorical_columns(df)[:12]) or "none"

    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )

    response = ollama.chat(model=MODEL, messages=messages)
    raw = response["message"]["content"]
    code = extract_code(raw)

    if not code:
        return {
            "code": None,
            "result": None,
            "error": "Model did not return a recognisable code block.",
            "raw_response": raw,
        }

    result, error = run_code(code, df)

    if error and MAX_RETRIES > 0:
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"That code raised an error:\n\n{error}\n\n"
                    "Fix it and return only the corrected ```python block.\n\n"
                    "Important constraints:\n"
                    f"- Numeric columns likely to aggregate: {numeric_hint}\n"
                    f"- Non-numeric/grouping columns: {categorical_hint}\n"
                    "- Never run mean/median/sum/min/max on string columns.\n"
                    "- If a requested column doesn't exist, pick the closest valid column from schema."
                ),
            },
        ]
        retry_response = ollama.chat(model=MODEL, messages=retry_messages)
        raw = retry_response["message"]["content"]
        code = extract_code(raw)
        if code:
            result, error = run_code(code, df)

    if error:
        error = friendly_execution_error(error, df)

    return {
        "code": code,
        "result": result,
        "error": error,
        "raw_response": raw,
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
