import streamlit as st

STYLES = """

@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
    --bg: #0c0d10;
    --surface: #14161c;
    --surface-2: #1a1d26;
    --border: #2a2f3d;
    --text: #e8eaef;
    --muted: #8b93a7;
    --accent: #e8a838;
    --accent-dim: #c4892a;
    --teal: #3dd6c6;
    --danger: #f07178;
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    color: var(--text);
}

.stApp {
    background: radial-gradient(ellipse 120% 80% at 50% -20%, #1a1520 0%, var(--bg) 55%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111318 0%, #0e1014 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
}

.brand-title {
    font-family: 'Fraunces', serif;
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem 0;
    line-height: 1.1;
}

.brand-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 1rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(61, 214, 198, 0.08);
    border: 1px solid rgba(61, 214, 198, 0.25);
    color: var(--teal);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.status-pill::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 8px var(--teal);
}

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
}

.stat-card.wide {
    grid-column: 1 / -1;
}

.stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 4px;
}

.stat-value {
    font-family: 'Fraunces', serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.1;
}

.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin: 1.25rem 0 0.6rem 0;
    font-weight: 600;
}

.suggest-hint {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
    line-height: 1.4;
}

[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 0.92rem;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid var(--accent);
}

[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-dim) !important;
    box-shadow: 0 0 0 1px var(--accent-dim) !important;
}

code, pre {
    font-family: ui-monospace, 'Cascadia Code', monospace !important;
    font-size: 0.8rem !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.insight-box {
    background: linear-gradient(135deg, rgba(232, 168, 56, 0.08) 0%, var(--surface) 100%);
    border: 1px solid rgba(232, 168, 56, 0.22);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 8px 0 12px;
    font-size: 0.92rem;
    line-height: 1.55;
    color: #f0e6d4;
}

.landing-hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    max-width: 640px;
    margin: 0 auto;
}

.landing-hero h1 {
    font-family: 'Fraunces', serif !important;
    font-size: 2.6rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.04em;
    margin-bottom: 0.75rem !important;
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
}

.step-num {
    font-family: 'Fraunces', serif;
    color: var(--accent);
    font-size: 1.5rem;
    margin-bottom: 0.35rem;
}

.step-card h3 {
    font-size: 1rem !important;
    color: var(--text) !important;
    margin-bottom: 0.4rem !important;
}

.step-card p {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 0;
}

.chat-header {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.25rem 0;
}

.chat-meta {
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-dim) !important;
}

hr {
    border-color: var(--border) !important;
    margin: 1.25rem 0 !important;
}

/* Suggestion buttons */
[data-testid="stSidebar"] .stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    padding: 0.65rem 0.85rem !important;
    font-size: 0.82rem !important;
    line-height: 1.35 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: border-color 0.15s, background 0.15s !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 2.5rem !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent-dim) !important;
    background: var(--surface-2) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(232, 168, 56, 0.12) !important;
    border-color: var(--accent-dim) !important;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

"""


def inject_styles() -> None:
    st.markdown("<style>" + STYLES + "</style>", unsafe_allow_html=True)
