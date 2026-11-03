"""Cards, badges, callouts, empty states and the provenance panel.

Every dynamic string is HTML-escaped, and every status carries an icon AND a word, so no state is shown by colour alone.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Iterable, Optional, Sequence

import streamlit as st

ICONS = {"ok": "✔", "warn": "▲", "error": "✖", "info": "ℹ"}
DEFAULT_WORDS = {"ok": "OK", "warn": "Caution", "error": "Problem", "info": "Note"}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def badge(kind: str, text: Optional[str] = None) -> str:
    """HTML of a status pill: icon + word (kind is one of ok, warn, error, info)."""
    kind = kind if kind in ICONS else "info"
    return f'<span class="rd-badge {kind}"><span aria-hidden="true">{ICONS[kind]}</span>{esc(text or DEFAULT_WORDS[kind])}</span>'


def page_header(title: str, lede: str) -> None:
    st.markdown(f'<h1 class="rd-page-title">{esc(title)}</h1><p class="rd-lede">{esc(lede)}</p>', unsafe_allow_html=True)


def kpi_card_html(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="n">{esc(note)}</div>' if note else ""
    size = " sm" if len(str(value)) > 12 else ""
    return f'<div class="rd-card"><div class="k">{esc(label)}</div><div class="v{size}">{esc(value)}</div>{note_html}</div>'


def render_kpis(items: Sequence[Dict[str, str]]) -> None:
    """A responsive row of KPI cards. Each item: label, value, optional note."""
    cards = "".join(kpi_card_html(i["label"], i["value"], i.get("note", "")) for i in items)
    st.markdown(f'<div class="rd-kpis">{cards}</div>', unsafe_allow_html=True)


def callout(kind: str, title: str, body: str = "", *, html_body: bool = False) -> None:
    """A bordered message. `body` is escaped unless `html_body` is True (then the caller guarantees it is safe)."""
    kind = kind if kind in ICONS else "info"
    text = body if html_body else esc(body)
    body_html = f"<div>{text}</div>" if text else ""
    st.markdown(
        f'<div class="rd-callout {kind}" role="note"><div class="t"><span aria-hidden="true">{ICONS[kind]}</span> {esc(title)}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def code_block(command: str) -> None:
    st.markdown(f'<div class="rd-code">{esc(command)}</div>', unsafe_allow_html=True)


def empty_state(title: str, detail: str, commands: Iterable[str] = (), missing: Iterable[str] = ()) -> None:
    """The standard message for a missing artifact: what is absent, why it matters, and the exact commands that create it."""
    parts = [f"<div>{esc(detail)}</div>"] if detail else []
    missing = list(missing)
    if missing:
        parts.append("<div style='margin-top:0.4rem'>Missing: " + ", ".join(f"<code>{esc(m)}</code>" for m in missing) + "</div>")
    commands = [c for c in commands if c]
    if commands:
        parts.append("<div style='margin-top:0.4rem'>Create it from the project root (PowerShell):</div>")
        parts += [f'<div class="rd-code">{esc(c)}</div>' for c in commands]
    st.markdown(
        f'<div class="rd-callout warn" role="alert"><div class="t"><span aria-hidden="true">{ICONS["warn"]}</span> {esc(title)}</div>{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def loaded_error(result, title: str = "This result is not available yet") -> None:
    """Show the error of a failed `Loaded` with its creating command."""
    empty_state(title, result.error or "", [result.command] if result.command else [])


def provenance_panel(records: Sequence[Dict[str, Any]], note: str = "") -> None:
    """A collapsed 'Data and model provenance' area: which saved artifact produced what is shown."""
    with st.expander("Data and model provenance", expanded=False):
        if note:
            st.markdown(f'<p class="rd-prov">{esc(note)}</p>', unsafe_allow_html=True)
        from components.tables import html_table

        rows = []
        for r in records:
            rows.append({
                "Shown": r.get("role", ""),
                "Artifact": r.get("file", ""),
                "Status": "found" if r.get("exists", True) else "NOT FOUND",
                "Written (UTC)": r.get("modified_utc", "-"),
                "SHA-256 (first 12)": r.get("sha256", "-"),
            })
        st.markdown(html_table(__import__("pandas").DataFrame(rows), caption=None), unsafe_allow_html=True)
        st.markdown('<p class="rd-prov">Hashes identify the exact saved file. Metrics are recomputed from the saved forecasts, '
                    "not retrained, and the final holdout is not read anywhere in this app.</p>", unsafe_allow_html=True)
