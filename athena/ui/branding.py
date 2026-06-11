from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


def _img_data_uri(filename: str) -> str:
    data = base64.b64encode((STATIC_DIR / filename).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def render_brand_header() -> None:
    text_uri = _img_data_uri("athena-text.png")
    logo_uri = _img_data_uri("athena-logo-sm.png")
    st.markdown(
        f"""
        <div class="brand-header">
            <div class="brand-row">
                <img class="brand-text-img" src="{text_uri}" alt="athena" />
                <img class="brand-logo-img" src="{logo_uri}" alt="" />
            </div>
            <p class="brand-sub">Upload once — clean, dashboard, and chat.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_meta_tags() -> None:
    components.html(
        """
        <script>
        (function () {
            const doc = window.parent.document;
            if (doc.querySelector('meta[property="og:image"]')) return;

            const image = new URL('app/static/og-image.png', window.parent.location).href;
            const meta = [
                ['property', 'og:title', 'athena — ai data analysis tool'],
                ['property', 'og:description', 'Upload a CSV, clean your data, build dashboards, and chat with your dataset.'],
                ['property', 'og:image', image],
                ['property', 'og:type', 'website'],
                ['name', 'twitter:card', 'summary_large_image'],
                ['name', 'twitter:image', image],
            ];

            meta.forEach(([attr, key, value]) => {
                const el = doc.createElement('meta');
                el.setAttribute(attr, key);
                el.setAttribute('content', value);
                doc.head.appendChild(el);
            });
        })();
        </script>
        """,
        height=0,
    )
