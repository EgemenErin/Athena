<img width="1366" height="768" alt="athena" src="https://github.com/user-attachments/assets/85d5d5c7-6605-4b33-b45a-5676366eff23" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-local-black)](https://ollama.com/)

> **Talk to your data** — upload any CSV, ask questions in plain English, and get tables, charts, and narrative insights. Everything runs on your machine via [Ollama](https://ollama.com/); your data never leaves your device.

Repository: [github.com/EgemenErin/Athena](https://github.com/EgemenErin/Athena)

---

## Description

**Athena** is a universal CSV data assistant built for analysts, researchers, and anyone who works with tabular exports but does not want to write pandas by hand. Upload a spreadsheet, describe what you want to know in natural language, and Athena generates safe pandas code, executes it in a sandbox, visualizes the result when appropriate, and summarizes the finding in plain English.

Athena is designed to run **locally on your machine**. It talks to Ollama on the same computer — no commercial LLM API keys, no cloud hosting, and no sending sensitive CSVs to third-party services.


## ✨ Features

- **Natural-language queries** — Ask questions like “What is the average salary by country?” and get executable pandas code plus results.
- **Local-first privacy** — Powered by Ollama (`qwen2.5-coder:14b` by default); no external LLM API required.
- **Sandboxed execution** — Generated code runs in an isolated scope with only `df` and `pd` available; imports, file I/O, and system modules are blocked.
- **Smart recovery** — Automatic retry with schema-aware hints when generated code fails.
- **Senior analyst prompts** — Code generation follows strict column mapping, question-type rules, and survey-style data patterns (semicolon lists, filters).
- **Agent pipeline** — Optional Planner → Analyst → Reviewer flow for better answers; toggle in `athena/config.py`.
- **AI-suggested questions** — Business-analyst-style sidebar prompts tied to real columns.
- **Dashboard** — AI-suggested graphs (bar, histogram, scatter, line, pie) with live Plotly previews; **Make a chart** builder with column pickers and heuristic chart suggestions; save charts to **My charts**; export saved charts to PDF (2×2 per page). Handles numeric text (coerce), high-cardinality metrics (bins), and list-like columns (explode `;`-separated values).
- **Rich responses** — DataFrames, scalar metrics, Plotly bar charts (only when appropriate), and grounded narrative insights.
- **Multi-turn chat** — Follow-up questions retain conversation context for chained analysis.
- **AI-assisted cleaning** — Per-column analysis (no action cap): AI reviews every column in batches and recommends drop, fill (median/mean/mode), outlier handling, or skip; apply what you approve, with undo.

---

## 🛠️ Built With

| Layer | Technology |
|--------|------------|
| **UI** | [Streamlit](https://streamlit.io/) |
| **Data** | [pandas](https://pandas.pydata.org/) |
| **Charts** | [Plotly Express](https://plotly.com/python/plotly-express/) |
| **LLM runtime** | [Ollama](https://ollama.com/) + `qwen2.5-coder:14b` (default) |
| **Language** | Python 3.10+ |

**Project layout**

```
Athena/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── requirements-dev.txt
├── .streamlit/
│   └── config.toml
├── static/                     # Logos, favicon, OG image
├── demo/
│   └── demo.csv
├── athena/
│   ├── config.py               # Model (env), retry limits, agent toggle
│   ├── charts/                 # Chart constants and aggregations
│   ├── llm/                    # Ollama prompts, sandbox, suggestions
│   │   ├── schema.py
│   │   ├── execution.py
│   │   ├── generator.py
│   │   ├── suggestions.py
│   │   ├── chart_suggestions.py
│   │   ├── cleaning.py
│   │   ├── cleaning_columns.py
│   │   ├── agents.py
│   │   ├── personas.py
│   │   └── summary.py
│   └── ui/                     # Styles, sidebar, chat, dashboard, cleaning
│       ├── styles.py
│       ├── sidebar.py
│       ├── main.py
│       ├── cleaning.py
│       ├── dashboard.py
│       ├── charts.py
│       ├── pdf_export.py
│       ├── branding.py
│       └── helpers.py
├── tests/
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Before you begin, install:

1. **[Python](https://www.python.org/downloads/)** 3.10 or newer (3.11+ recommended)
2. **[Ollama](https://ollama.com/download)** — local model server
3. **Git** (optional, for cloning)

Pull the default model (required on first run):

```bash
ollama pull qwen2.5-coder:14b
```

Verify Ollama is running:

```bash
ollama list
```

You should see `qwen2.5-coder:14b` (or a variant whose name contains `qwen2.5-coder`) in the list.

### Installation

```bash
# Clone the repository
git clone https://github.com/EgemenErin/Athena.git
cd Athena

# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

| Variable / setting | Where | Description |
|--------------------|-------|-------------|
| `OLLAMA_MODEL` | Environment variable | Ollama model name (default: `qwen2.5-coder:14b`) |
| `OLLAMA_HOST` | Environment variable | Ollama API URL if not the default `http://127.0.0.1:11434` |
| `MAX_RETRIES` | `athena/config.py` | Automatic fix attempts on execution errors (default: `1`) |
| `ENABLE_AGENT_PIPELINE` | `athena/config.py` | Planner + Reviewer before/after codegen (default: `True`) |
| `MAX_REVIEW_RETRIES` | `athena/config.py` | Reviewer-driven codegen retries when result mismatches question (default: `1`) |
| `CLEANING_BATCH_SIZE` | `athena/config.py` | Columns per cleaning LLM batch (default: `30`) |

**Changing the model** — set `OLLAMA_MODEL` (recommended) or edit the default in [`athena/config.py`](athena/config.py). Pull the model locally (`ollama pull <name>`). The sidebar status pill checks that a model whose name contains the configured base tag (e.g. `qwen2.5-coder`) is available.

Ollama also respects the standard `OLLAMA_HOST` environment variable used by the Ollama CLI.

### Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### Smoke test (optional)

Verify the LLM pipeline without the UI:

```bash
python -m athena.llm.generator
```

This runs a small in-memory DataFrame through code generation and execution against Ollama.

Cleaning executor smoke test (no Ollama):

```bash
python -m athena.llm.cleaning
pytest tests/
```

Install dev dependencies first: `pip install -r requirements-dev.txt`.

---

## 💡 Usage

### Web UI workflow

1. **Upload** — In the sidebar, drop a `.csv` file (surveys, sales, logs, exports, etc.).
2. **Clean (optional)** — **Clean data** tab → **Analyze with AI** walks **every column** (120 columns = 120 recommendations, batched LLM calls). Review per-column actions (drop, fill strategy, outliers, or skip), apply selected, or use **Fill all missing → 0%** for a one-shot null fix. **Download cleaned CSV** when ready.
3. **Dashboard** — Open **Dashboard** for AI-suggested charts (bar, histogram, scatter, line, pie) rendered as live Plotly previews. Use **↻ Refresh** for new ideas (Ollama when available, rule-based fallback otherwise).
4. **Explore** — Review row/column stats and column types, or click a **Suggested question** in the sidebar.
5. **Ask** — Type a question in the chat, e.g.:
   - *What are the top 10 countries by median yearly compensation?*
   - *How many rows have missing values in the `email` column?*
   - *Show average `score` grouped by `region`.*
6. **Inspect** — Read the narrative insight, table or chart, and expand **View generated code** to see the pandas logic.

Follow-up questions work in context (e.g. *“Now filter that to Europe only”*).

### Agent pipeline

When `ENABLE_AGENT_PIPELINE` is `True` (default), each chat question runs:

1. **Planner** — picks columns, filters, and expected result shape (count vs table vs correlation).
2. **Senior Analyst** — writes pandas code using the plan and analyst rules.
3. **Reviewer** — checks if the result answers the question; triggers one retry if not.

Set `ENABLE_AGENT_PIPELINE = False` in [`athena/config.py`](athena/config.py) for faster responses with a single model call.

Charts are shown only for suitable results (e.g. category + metric tables), not for correlation matrices or scalars.

### Example questions

| Goal | Example prompt |
|------|----------------|
| Top-N ranking | *What are the top 10 most common values in `Category`?* |
| Aggregation | *What is the average, min, and max of `Revenue`?* |
| Group-by | *Show average `Price` grouped by `Country` (top 10).* |
| Row filter | *Which rows have the highest `Score`?* |
| Data quality | *Which columns have the most missing values?* |

### Programmatic use

The core logic lives in the `athena.llm` package:

```python
import pandas as pd
from athena.llm import build_schema_string, generate_and_run, summarise_result

df = pd.read_csv("your_data.csv")
schema = build_schema_string(df)

output = generate_and_run(
    question="How many unique customers are there?",
    df=df,
    schema=schema,
    chat_history=[],
)

if output["error"]:
    print(output["error"])
else:
    print(output["result"])
    print(summarise_result("How many unique customers are there?", output["result"]))
```

---

## 🤝 Contributing

Contributions are welcome. To keep changes easy to review:

1. **Fork** the repository and create a branch from `main`.
2. **Describe** your change in the PR (bug fix, feature, docs).
3. **Test locally** — run `streamlit run app.py` and, when touching the engine, `python -m athena.llm.generator` and `pytest tests/`.
4. **Keep scope focused** — one logical change per PR when possible.
5. **Match style** — follow existing patterns under `athena/`; avoid unrelated refactors.

Please open an issue first for large features (new chart types, alternate LLM backends, export formats) so approach can be discussed.

---

## 📄 License

This project is licensed under the **MIT License**. A `LICENSE` file may be added to the repository separately; until then, the MIT terms at [opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) apply.

---

## 📬 Contact

**Egemen Erin**

- GitHub: [@EgemenErin](https://github.com/EgemenErin)
- Repository: [github.com/EgemenErin/Athena](https://github.com/EgemenErin/Athena)
- Email: egemeneriin@protonmail.com

Questions, bugs, and feature requests: please use [GitHub Issues](https://github.com/EgemenErin/Athena/issues).

---

<p align="center">
  <sub>Athena · Runs on your machine · Ollama · Your data stays local</sub>
</p>
