import streamlit as st

STYLES = """

@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Spline+Sans+Mono:wght@400;500&display=swap');

:root {
    --bg: #FAF9F5;
    --surface: #FFFFFF;
    --surface-2: #F1EFE9;
    --border: #E5E2D8;
    --border-strong: #D4D0C2;
    --ink: #1C1B17;
    --muted: #76715F;
    --accent: #2D4FDE;
    --accent-soft: #E9EDFC;
    --accent-border: #C5D0F6;
    --green: #0E8A6D;
    --danger: #C5483D;
    --serif: 'Instrument Serif', Georgia, serif;
    --sans: 'Instrument Sans', sans-serif;
    --mono: 'Spline Sans Mono', ui-monospace, monospace;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, #F2EFE6 0%, transparent 60%),
        var(--bg);
}

/* faint grid texture, like graph paper */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(to right, rgba(28, 27, 23, 0.025) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(28, 27, 23, 0.025) 1px, transparent 1px);
    background-size: 56px 56px;
    z-index: 0;
}

h1, h2, h3 {
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    color: var(--ink) !important;
}

/* ---------- sidebar ---------- */

[data-testid="stSidebar"] {
    background: var(--bg);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
}

.brand-header {
    margin-bottom: 1rem;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin: 0;
}

.brand-text-img {
    height: 1.65rem;
    width: auto;
    display: block;
}

.brand-logo-img {
    height: 1.9rem;
    width: auto;
    display: block;
    flex-shrink: 0;
}

.brand-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin: 0;
    padding-top: 0.65rem;
    line-height: 1.45;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--green);
    border-radius: 999px;
    padding: 4px 12px;
    font-family: var(--mono);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.status-pill::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 3px rgba(14, 138, 109, 0.15);
}

/* ---------- stat cards ---------- */

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 0.75rem 0;
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 1px 2px rgba(28, 27, 23, 0.04);
}

.stat-card.wide {
    grid-column: 1 / -1;
}

.stat-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 5px;
}

.stat-value {
    font-family: var(--mono);
    font-size: 1.3rem;
    font-weight: 500;
    color: var(--ink);
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}

.section-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    margin: 1.4rem 0 0.6rem 0;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.suggest-hint {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
    line-height: 1.4;
}

.chart-desc {
    font-size: 0.84rem;
    color: var(--muted);
    line-height: 1.45;
    margin: 0.15rem 0 0.65rem 0;
}

/* ---------- top section nav (segmented control) ---------- */

[data-testid="stRadio"] [role="radiogroup"][aria-label="Section"] {
    display: inline-flex;
    gap: 2px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 3px;
}

[data-testid="stRadio"] [role="radiogroup"][aria-label="Section"] label {
    margin: 0 !important;
    padding: 5px 18px;
    border-radius: 7px;
    cursor: pointer;
    transition: background 0.15s, box-shadow 0.15s;
}

[data-testid="stRadio"] [role="radiogroup"][aria-label="Section"] label > div:first-child {
    display: none;
}

[data-testid="stRadio"] [role="radiogroup"][aria-label="Section"] label p {
    font-size: 0.86rem !important;
    font-weight: 500;
    color: var(--muted);
}

[data-testid="stRadio"] [role="radiogroup"][aria-label="Section"] label:hover p {
    color: var(--ink);
}

[data-testid="stRadio"] [role="radiogroup"][aria-label="Section"] label:has(input:checked) {
    background: var(--surface);
    box-shadow: 0 1px 3px rgba(28, 27, 23, 0.1), inset 0 0 0 1px var(--border);
}

[data-testid="stRadio"] [role="radiogroup"][aria-label="Section"] label:has(input:checked) p {
    color: var(--ink);
}

/* ---------- chat ---------- */

[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 0.92rem;
    box-shadow: 0 1px 2px rgba(28, 27, 23, 0.04);
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--surface-2);
    border-left: 3px solid var(--accent);
}

[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 12px !important;
    color: var(--ink) !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

code, pre {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
}

.insight-box {
    background: var(--accent-soft);
    border: 1px solid var(--accent-border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 8px 0 12px;
    font-size: 0.92rem;
    line-height: 1.55;
    color: #1F2A56;
}

/* ---------- landing ---------- */

.landing-hero {
    text-align: center;
    padding: 3.5rem 1rem 2.25rem;
    max-width: 660px;
    margin: 0 auto;
    animation: rise 0.5s ease both;
}

.landing-hero .hero-kicker {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-bottom: 1rem;
}

.landing-hero h1 {
    font-family: var(--sans) !important;
    font-size: 3.4rem !important;
    font-weight: 600 !important;
    font-style: normal;
    color: var(--ink) !important;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin-bottom: 0.9rem !important;
}

.landing-hero h1 em {
    font-family: inherit;
    font-style: normal;
    font-weight: 600;
    color: var(--accent);
}

.landing-hero p {
    color: var(--muted);
    font-size: 1.05rem;
    line-height: 1.6;
}

.step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.35rem;
    height: 100%;
    box-shadow: 0 1px 2px rgba(28, 27, 23, 0.04);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    animation: rise 0.5s ease both;
}

.step-card:nth-child(1) { animation-delay: 0.05s; }

.step-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent-border);
    box-shadow: 0 8px 20px rgba(45, 79, 222, 0.08);
}

.step-num {
    font-family: var(--mono);
    color: var(--accent);
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    background: var(--accent-soft);
    border: 1px solid var(--accent-border);
    border-radius: 6px;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.step-card h3 {
    font-size: 1rem !important;
    color: var(--ink) !important;
    margin-bottom: 0.4rem !important;
}

.step-card p {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 0;
}

@keyframes rise {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes athena-spin {
    to { transform: rotate(360deg); }
}

/* ---------- loading ---------- */

[data-testid="stSpinner"] > div > :first-child {
    display: none !important;
}

[data-testid="stSpinner"] > div {
    gap: 0.55rem !important;
}

[data-testid="stSpinner"] > div::before {
    content: '';
    width: 14px;
    height: 14px;
    border: 2px solid var(--accent-border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: athena-spin 0.7s linear infinite;
    flex-shrink: 0;
}

[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span {
    font-family: var(--sans) !important;
    color: var(--muted) !important;
    font-size: 0.86rem !important;
    opacity: 1 !important;
}

/* ---------- page headers ---------- */

.stMain .chat-header, .chat-header {
    font-family: var(--sans) !important;
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0;
}

.chat-meta {
    color: var(--muted);
    font-family: var(--sans) !important;
    font-size: 0.95rem;
    line-height: 1.55;
    letter-spacing: normal;
    margin-bottom: 1rem;
}

/* ---------- widgets ---------- */

[data-testid="stFileUploader"] {
    border: 1px dashed var(--border-strong) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

hr {
    border-color: var(--border) !important;
    margin: 1.25rem 0 !important;
}

.stButton > button,
.stDownloadButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--ink) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    box-shadow: 0 1px 2px rgba(28, 27, 23, 0.05) !important;
    transition: border-color 0.15s, background 0.15s, box-shadow 0.15s !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--surface) !important;
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: #2240C4 !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 10px rgba(45, 79, 222, 0.25) !important;
}

/* sidebar suggestion buttons: quieter, left-aligned */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    padding: 0.6rem 0.85rem !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    line-height: 1.35 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 2.5rem !important;
    color: var(--ink) !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent) !important;
    background: var(--accent-soft) !important;
    color: var(--ink) !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--surface) !important;
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
}

[data-testid="stMetricValue"] {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

"""


def inject_styles() -> None:
    st.markdown("<style>" + STYLES + "</style>", unsafe_allow_html=True)
