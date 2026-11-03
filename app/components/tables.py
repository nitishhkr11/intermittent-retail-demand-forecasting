"""Theme-aware tables.

`st.dataframe` draws on a canvas that follows Streamlit's own theme and cannot be restyled from the app, so tables here are
plain HTML (escaped) styled by the same tokens as everything else. Missing numbers are shown as "n/a", never left blank.
"""

from __future__ import annotations

import html
import math
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st

Formatter = Union[str, Callable[[Any], str]]


def _format(value: Any, fmt: Optional[Formatter]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT or (not isinstance(value, (str, bool)) and pd.isna(value)):
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if callable(fmt):
        return fmt(value)
    if isinstance(fmt, str):
        return fmt.format(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.3f}" if abs(value) < 1000 else f"{float(value):,.1f}"
    return str(value)


def html_table(
    df: pd.DataFrame,
    *,
    caption: Optional[str] = None,
    formats: Optional[Mapping[str, Formatter]] = None,
    best: Optional[Mapping[str, str]] = None,
    max_height: Optional[int] = 460,
) -> str:
    """HTML of a table. `best={'column': 'min'|'max'}` marks the best row with a text label (not colour alone)."""
    formats = dict(formats or {})
    best_row = None
    marker_col = None
    if best and len(df):
        column, how = next(iter(best.items()))
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            best_row = values.idxmin() if how == "min" else values.idxmax()
            marker_col = df.columns[0]
    numeric = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])}
    head = "".join(f'<th scope="col" class="{"num" if c in numeric else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    body = []
    for idx, row in df.iterrows():
        cells = []
        for c in df.columns:
            text = html.escape(_format(row[c], formats.get(c)))
            if idx == best_row and c == marker_col:
                text += f' <span class="rd-hint">(lowest {html.escape(next(iter(best)))})</span>' if best and next(iter(best.values())) == "min" else f' <span class="rd-hint">(highest {html.escape(next(iter(best)))})</span>'
            cells.append(f'<td class="{"num" if c in numeric else ""}">{text}</td>')
        body.append(f'<tr class="{"best" if idx == best_row else ""}">{"".join(cells)}</tr>')
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    style = f' style="max-height:{max_height}px"' if max_height else ""
    return f'<div class="rd-table-wrap"{style}><table class="rd-table">{cap}<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def render_table(
    df: pd.DataFrame,
    *,
    caption: Optional[str] = None,
    formats: Optional[Mapping[str, Formatter]] = None,
    best: Optional[Mapping[str, str]] = None,
    max_height: Optional[int] = 460,
    download_name: Optional[str] = None,
    key: Optional[str] = None,
) -> None:
    """Show a themed table; with `download_name`, also offer the same rows as a CSV file."""
    st.markdown(html_table(df, caption=caption, formats=formats, best=best, max_height=max_height), unsafe_allow_html=True)
    if download_name:
        st.download_button("Download this table (CSV)", df.to_csv(index=False).encode("utf-8"), file_name=download_name,
                           mime="text/csv", key=key or f"dl_{download_name}")
