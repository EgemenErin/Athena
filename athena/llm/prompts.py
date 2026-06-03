from athena.llm.personas import SENIOR_ANALYST_RULES


def build_system_prompt(schema: str) -> str:
    return f"""{SENIOR_ANALYST_RULES}

Here is the schema of `df`:
{schema}

Example (grouped aggregation):
```python
result = (
    df[df["ConvertedCompYearly"] > 0]
    .groupby("Country")["ConvertedCompYearly"]
    .median()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
```

Example (count distinct after filter):
```python
subset = df[df["LearnCode"].str.contains("Books", case=False, na=False)]
activities = subset["CodingActivities"].dropna().str.split(";").explode().str.strip()
result = activities.nunique()
```

Example (compare groups — use concat, never append):
```python
masters = df[df["EdLevel"].str.contains("master", case=False, na=False)]
bachelors = df[df["EdLevel"].str.contains("bachelor", case=False, na=False)]
m_act = masters["CodingActivities"].dropna().str.split(";").explode().str.strip().value_counts()
b_act = bachelors["CodingActivities"].dropna().str.split(";").explode().str.strip().value_counts()
result = pd.concat(
    [m_act.head(10).rename("masters"), b_act.head(10).rename("bachelors")],
    axis=1,
).fillna(0)
```
"""
