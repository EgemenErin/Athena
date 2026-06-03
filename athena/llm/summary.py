import ollama
import pandas as pd

from athena.config import MODEL


def summarise_result(question: str, result) -> str:
    """
    Second LLM call: takes the question + query result and returns
    a 1–2 sentence plain-English insight with specific numbers.
    """
    if isinstance(result, pd.DataFrame):
        result_preview = result.head(10).to_string(index=False)
    elif isinstance(result, pd.Series):
        result_preview = result.head(10).to_string()
    else:
        result_preview = str(result)

    prompt = (
        f'A user asked: "{question}"\n\n'
        f"The data query returned:\n{result_preview}\n\n"
        "Write 1–2 sentences summarising the key insight. "
        "Be specific — mention actual numbers or names from the data. "
        "Do not say 'the data shows' or 'based on the results'."
    )

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()
