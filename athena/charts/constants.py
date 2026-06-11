"""Shared chart limits and options (single source of truth)."""

MAX_BAR_CATEGORIES = 25
MAX_PIE_SLICES = 10
MAX_LINE_POINTS = 500

MIN_HISTOGRAM_UNIQUE = 5
MAX_HISTOGRAM_UNIQUE_RATIO = 0.5
MAX_CODE_CARDINALITY = 80

# Aggregations supported by chart specs and prepare_chart_data.
# "pct_true" is the share of True/1 values in a boolean flag column.
SUPPORTED_AGGREGATIONS = ("count", "mean", "median", "sum", "min", "max", "pct_true")

# Bar aggregations the user can pick in the chart builder (count handled separately).
BUILDER_BAR_AGGREGATIONS = ("mean", "median", "sum", "min", "max")

AGGREGATION_LABELS = {
    "mean": "Average",
    "median": "Median",
    "sum": "Total",
    "min": "Minimum",
    "max": "Maximum",
    "count": "Count",
    "pct_true": "% True",
}
