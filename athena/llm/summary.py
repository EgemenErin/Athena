import ollama
import pandas as pd

from athena.config import MODEL
from athena.llm.personas import SUMMARY_ANALYST_RULES


def _describe_result_shape(result) -> str:
    if isinstance(result, pd.DataFrame):
        rows, cols = result.shape
        numeric_cols = result.select_dtypes(include="number").columns.tolist()
        if rows == cols and rows <= 6 and len(numeric_cols) == cols:
            return f"square numeric matrix ({rows}×{cols}), likely correlation or pivot — not a ranked list"
        if rows == 1 and cols <= 3:
            return f"single-row table ({rows}×{cols})"
        return f"table ({rows} rows × {cols} columns)"
    if isinstance(result, pd.Series):
        return f"series ({len(result)} values)"
    return f"scalar ({type(result).__name__})"


def summarise_result(question: str, result) -> str:
    """
    Second LLM call: grounded insight tied to the actual result preview.
    """
    if isinstance(result, pd.DataFrame):
        result_preview = result.head(10).to_string(index=False)
    elif isinstance(result, pd.Series):
        result_preview = result.head(10).to_string()
    else:
        result_preview = str(result)

    shape_note = _describe_result_shape(result)

    prompt = (
        f"{SUMMARY_ANALYST_RULES}\n\n"
        f'User question: "{question}"\n'
        f"Result type: {shape_note}\n\n"
        f"Query result preview:\n{result_preview}\n\n"
        "Write the insight now."
    )

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},
    )
    return response["message"]["content"].strip()
