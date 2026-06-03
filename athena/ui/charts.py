import pandas as pd
import plotly.express as px

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#14161c",
    plot_bgcolor="#14161c",
    font_color="#e8eaef",
    font_family="Source Sans 3, sans-serif",
    colorway=["#e8a838", "#3dd6c6", "#f07178", "#a8b4ff", "#7ec97e"],
)


def try_make_chart(result):
    if not isinstance(result, (pd.DataFrame, pd.Series)):
        return None

    if isinstance(result, pd.Series):
        result = result.reset_index()
        result.columns = ["category", "value"]

    if result.shape[1] == 2:
        x_col, y_col = result.columns[0], result.columns[1]
        if pd.api.types.is_numeric_dtype(result[y_col]):
            fig = px.bar(
                result.head(20),
                x=x_col,
                y=y_col,
                color=y_col,
                color_continuous_scale=["#1a1d26", "#e8a838"],
            )
            fig.update_layout(
                **PLOTLY_THEME,
                xaxis_tickangle=-35,
                showlegend=False,
                margin=dict(t=30, b=60, l=40, r=20),
                coloraxis_showscale=False,
            )
            fig.update_traces(marker_line_width=0)
            return fig

    return None
