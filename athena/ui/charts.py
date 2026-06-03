from typing import Any

import pandas as pd
import plotly.express as px

from athena.llm.chart_transforms import apply_transforms

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#14161c",
    plot_bgcolor="#14161c",
    font_color="#e8eaef",
    font_family="Source Sans 3, sans-serif",
    colorway=["#e8a838", "#3dd6c6", "#f07178", "#a8b4ff", "#7ec97e"],
)

MAX_BAR_CATEGORIES = 25
MAX_PIE_SLICES = 10
MAX_LINE_POINTS = 500


def _apply_plotly_theme(fig, *, tick_angle: int = -35, show_legend: bool = False) -> None:
    fig.update_layout(
        **PLOTLY_THEME,
        xaxis_tickangle=tick_angle,
        showlegend=show_legend,
        margin=dict(t=30, b=60, l=40, r=20),
        coloraxis_showscale=False,
    )
    fig.update_traces(marker_line_width=0)


def prepare_chart_data(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame | None:
    """Build a plotting-ready frame from a chart suggestion spec."""
    chart_type = spec.get("chart_type")
    x = spec.get("x")
    y = spec.get("y")

    if chart_type in ("histogram", "scatter"):
        plot_df, plot_x, plot_y = apply_transforms(df, spec)
        return plot_df

    work, plot_x, plot_y = apply_transforms(df, spec)
    if work is None or not plot_x or plot_x not in work.columns:
        return None

    if chart_type == "line":
        agg = spec.get("aggregation", "mean")
        if agg == "count" or y == "count":
            plot_df = (
                work.groupby(plot_x, dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values(plot_x)
            )
            return plot_df.head(MAX_LINE_POINTS)
        if not plot_y or plot_y not in work.columns:
            return None
        plot_df = work[[plot_x, plot_y]].dropna().copy()
        if pd.api.types.is_datetime64_any_dtype(plot_df[plot_x].dtype):
            plot_df = plot_df.sort_values(plot_x)
        else:
            try:
                numeric_x = pd.to_numeric(plot_df[plot_x], errors="coerce")
                if numeric_x.notna().mean() >= 0.9:
                    plot_df = plot_df.assign(**{plot_x: numeric_x}).dropna(subset=[plot_x])
                    plot_df = plot_df.sort_values(plot_x)
                else:
                    plot_df[plot_x] = pd.to_datetime(plot_df[plot_x], errors="coerce")
                    plot_df = plot_df.dropna(subset=[plot_x]).sort_values(plot_x)
            except Exception:
                plot_df = plot_df.sort_values(plot_x)
        return plot_df.head(MAX_LINE_POINTS)

    if chart_type == "pie":
        counts = work[plot_x].value_counts(dropna=True).head(MAX_PIE_SLICES)
        if len(counts) < 2:
            return None
        return counts.reset_index(name="count")

    if chart_type == "bar":
        agg = spec.get("aggregation", "mean")
        if agg == "count" or y == "count":
            grouped = (
                work.groupby(plot_x, dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            return grouped.head(MAX_BAR_CATEGORIES)
        if not plot_y or plot_y not in work.columns:
            return None
        grouped = (
            work.groupby(plot_x, dropna=False)[plot_y]
            .mean()
            .reset_index(name=plot_y)
            .sort_values(plot_y, ascending=False)
        )
        return grouped.head(MAX_BAR_CATEGORIES)

    return None


def build_chart_from_spec(df: pd.DataFrame, spec: dict[str, Any]):
    """Render a Plotly figure from a validated chart suggestion spec."""
    plot_df = prepare_chart_data(df, spec)
    if plot_df is None or plot_df.empty:
        return None

    chart_type = spec.get("chart_type")
    x = spec.get("x")
    y = spec.get("y")
    title = spec.get("title") or ""

    if chart_type == "histogram" and len(plot_df.columns) >= 1:
        x_col = plot_df.columns[0]
        fig = px.histogram(plot_df, x=x_col, title=title)
        _apply_plotly_theme(fig, tick_angle=0)
        return fig

    if chart_type == "scatter" and x in plot_df.columns and y in plot_df.columns:
        fig = px.scatter(plot_df, x=x, y=y, title=title)
        _apply_plotly_theme(fig, tick_angle=0)
        return fig

    if chart_type == "line" and len(plot_df.columns) >= 2:
        x_col, y_plot = plot_df.columns[0], plot_df.columns[1]
        fig = px.line(plot_df, x=x_col, y=y_plot, title=title)
        _apply_plotly_theme(fig, tick_angle=-25)
        return fig

    if chart_type == "pie":
        name_col = plot_df.columns[0]
        fig = px.pie(plot_df, names=name_col, values="count", title=title)
        _apply_plotly_theme(fig, tick_angle=0, show_legend=True)
        return fig

    if chart_type == "bar" and len(plot_df.columns) >= 2:
        x_col, y_col = plot_df.columns[0], plot_df.columns[1]
        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            color=y_col,
            title=title,
            color_continuous_scale=["#1a1d26", "#e8a838"],
        )
        _apply_plotly_theme(fig)
        return fig

    return None


def _is_square_numeric_matrix(df: pd.DataFrame) -> bool:
    rows, cols = df.shape
    if rows != cols or rows == 0 or cols > 6:
        return False
    numeric = df.select_dtypes(include="number")
    return len(numeric.columns) == cols


def _pick_xy_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    if df.shape[1] != 2:
        return None
    c0, c1 = df.columns[0], df.columns[1]
    if pd.api.types.is_numeric_dtype(df[c1]) and not pd.api.types.is_numeric_dtype(df[c0]):
        return c0, c1
    if pd.api.types.is_numeric_dtype(df[c0]) and not pd.api.types.is_numeric_dtype(df[c1]):
        return c1, c0
    if pd.api.types.is_numeric_dtype(df[c0]) and pd.api.types.is_numeric_dtype(df[c1]):
        return None
    return None


def should_chart(result) -> bool:
    """Return True only when a bar chart would help interpret the result."""
    if isinstance(result, pd.Series):
        if len(result) < 2 or len(result) > MAX_BAR_CATEGORIES:
            return False
        return pd.api.types.is_numeric_dtype(result.dtype)

    if not isinstance(result, pd.DataFrame):
        return False

    rows, cols = result.shape
    if cols != 2 or rows < 2:
        return False
    if rows > MAX_BAR_CATEGORIES:
        return False
    if _is_square_numeric_matrix(result):
        return False
    if pd.api.types.is_numeric_dtype(result.iloc[:, 0]) and pd.api.types.is_numeric_dtype(
        result.iloc[:, 1]
    ):
        if rows <= 5:
            return False
    return _pick_xy_columns(result) is not None


def try_make_chart(result):
    if not should_chart(result):
        return None

    if isinstance(result, pd.Series):
        result = result.reset_index()
        if len(result.columns) == 2:
            result.columns = ["category", "value"]
        else:
            return None

    xy = _pick_xy_columns(result)
    if not xy:
        return None

    x_col, y_col = xy
    plot_df = result.head(MAX_BAR_CATEGORIES)
    fig = px.bar(
        plot_df,
        x=x_col,
        y=y_col,
        color=y_col,
        color_continuous_scale=["#1a1d26", "#e8a838"],
    )
    _apply_plotly_theme(fig)
    return fig
