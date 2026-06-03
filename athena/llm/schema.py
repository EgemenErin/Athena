import pandas as pd


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c].dtype)]


def build_schema_string(df: pd.DataFrame, max_values: int = 10) -> str:
    lines = [f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"]

    for col in df.columns:
        dtype = df[col].dtype
        null_count = int(df[col].isna().sum())

        if pd.api.types.is_numeric_dtype(dtype):
            col_min = df[col].min()
            col_max = df[col].max()
            lines.append(
                f"Column: '{col}'  dtype: {dtype}\n"
                f"  range: {col_min} – {col_max}  |  nulls: {null_count}"
            )
        else:
            samples = df[col].dropna().unique()[:max_values]
            sample_str = ", ".join(str(v) for v in samples)
            n_unique = df[col].nunique()
            if n_unique > max_values:
                sample_str += f", ... ({n_unique} unique)"
            lines.append(
                f"Column: '{col}'  dtype: {dtype}\n"
                f"  sample values: {sample_str}  |  nulls: {null_count}"
            )

    return "\n".join(lines)
