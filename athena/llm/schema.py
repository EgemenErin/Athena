import pandas as pd


def is_quantitative_dtype(dtype) -> bool:
    """Numeric dtypes safe for mean, diff, outliers — excludes boolean."""
    return (
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
    )


def is_boolean_like(series: pd.Series) -> bool:
    """True for bool / nullable boolean dtypes and 0/1 integer flag columns."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return True
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return False
    values = series.dropna().unique()
    if len(values) == 0 or len(values) > 2:
        return False
    try:
        return set(float(v) for v in values) <= {0.0, 1.0}
    except (TypeError, ValueError):
        return False


def boolean_flag_columns(df: pd.DataFrame, max_cols: int = 8) -> list[str]:
    """Boolean and 0/1 flag columns suitable for % True charts."""
    out: list[str] = []
    for col in df.columns:
        if is_boolean_like(df[col]) and df[col].dropna().nunique() == 2:
            out.append(col)
        if len(out) >= max_cols:
            break
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_quantitative_dtype(df[c].dtype)]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in numeric_columns(df)]


def looks_numeric_string(series: pd.Series, sample_size: int = 100) -> bool:
    """True when a text column's values are mostly parseable as numbers."""
    if pd.api.types.is_numeric_dtype(series.dtype):
        return False
    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return False
    converted = pd.to_numeric(sample.astype(str), errors="coerce")
    return converted.notna().mean() >= 0.8


def coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse numeric-like text columns so comparisons and aggregates work."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col].dtype):
            continue
        if looks_numeric_string(out[col]):
            out[col] = pd.to_numeric(out[col].astype(str), errors="coerce")
    return out


def comparable_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Columns safe for >, <, mean, etc. (native numeric or numeric-like text)."""
    out: list[str] = []
    for col in df.columns:
        if is_quantitative_dtype(df[col].dtype):
            out.append(col)
        elif looks_numeric_string(df[col]):
            out.append(col)
    return out


def build_schema_string(df: pd.DataFrame, max_values: int = 10) -> str:
    lines = [f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"]

    for col in df.columns:
        dtype = df[col].dtype
        null_count = int(df[col].isna().sum())

        if pd.api.types.is_bool_dtype(dtype):
            samples = df[col].dropna().unique()[:max_values]
            sample_str = ", ".join(str(v) for v in samples)
            lines.append(
                f"Column: '{col}'  dtype: {dtype} (boolean)\n"
                f"  sample values: {sample_str}  |  nulls: {null_count}"
            )
        elif is_quantitative_dtype(dtype):
            col_min = df[col].min()
            col_max = df[col].max()
            lines.append(
                f"Column: '{col}'  dtype: {dtype} (numeric)\n"
                f"  range: {col_min} – {col_max}  |  nulls: {null_count}"
            )
        elif looks_numeric_string(df[col]):
            converted = pd.to_numeric(df[col].astype(str), errors="coerce")
            col_min = converted.min()
            col_max = converted.max()
            lines.append(
                f"Column: '{col}'  dtype: {dtype} (numeric-like text — use pd.to_numeric before math)\n"
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


def build_column_index(df: pd.DataFrame, max_per_line: int = 8) -> str:
    """Compact list of all column names for suggestion prompts."""
    cols = list(df.columns)
    if not cols:
        return "Available columns: (none)"
    lines = ["Available columns — every question must use at least one exact name in backticks:"]
    for i in range(0, len(cols), max_per_line):
        chunk = cols[i : i + max_per_line]
        lines.append(", ".join(f"`{c}`" for c in chunk))
    return "\n".join(lines)
