"""Plotly charts for the dashboard, all styled from `app.theme` so they follow the light/dark toggle.

Encoding rules: series differ by colour AND dash/marker, bars carry value labels, intervals are drawn as a labelled band,
and every chart has a text or table equivalent on its page. Aggregate and item-level errors are different quantities and
are drawn in separate charts only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import theme
from src.evaluation.model_comparison import model_family

DASHES = theme.LINE_DASHES
SYMBOLS = theme.MARKER_SYMBOLS


def show(fig: go.Figure) -> None:
    """Draw a themed Plotly figure at the full column width (the width argument changed name in Streamlit 1.50)."""
    import streamlit as st

    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except ValueError:  # pragma: no cover - unusual version string
        major, minor = 99, 0
    width = {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    st.plotly_chart(fig, theme=None, config={"displaylogo": False}, **width)


def _fig(height: int = 380, title: Optional[str] = None, hovermode: str = "x unified") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**theme.chart_layout(height=height, title=title))
    fig.update_layout(hovermode=hovermode)
    return fig


def _family_slots(models: Sequence[str], scope: str) -> Dict[str, int]:
    """Stable colour slot per family (first seen order within the fixed family list)."""
    order: List[str] = []
    for m in models:
        fam = model_family(m, scope)
        if fam not in order:
            order.append(fam)
    return {fam: i for i, fam in enumerate(order)}


def history_trend_chart(series: pd.Series, *, window: int = 28) -> go.Figure:
    t = theme.tokens()
    colors = theme.series_colors()
    fig = _fig(380, "Daily units sold, CA_1 FOODS_3 (development window)")
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Daily sales (all items)",
                             line=dict(color=t["history"], width=1), opacity=0.55,
                             hovertemplate="%{y:,.0f} units<extra>daily</extra>"))
    fig.add_trace(go.Scatter(x=series.index, y=series.rolling(window, min_periods=window).mean(), mode="lines",
                             name=f"{window}-day average", line=dict(color=colors[0], width=2.6, dash="solid"),
                             hovertemplate="%{y:,.0f} units<extra>" + f"{window}-day average" + "</extra>"))
    fig.update_yaxes(title_text="Units per day", rangemode="tozero")
    fig.update_xaxes(title_text=None, rangeslider=dict(visible=False))
    return fig


def class_bar_chart(table: pd.DataFrame, *, title: str) -> go.Figure:
    """Horizontal bars of item counts per demand class, labelled with count and share (class is on the axis, not a colour)."""
    t = theme.tokens()
    order = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    data = table.set_index("demand_class").reindex([c for c in order if c in set(table["demand_class"])] + [c for c in table["demand_class"] if c not in order])
    total = float(data["items"].sum())
    labels = [f"{int(v):,} items ({v / total:.1%})" for v in data["items"]]
    fig = _fig(280, title, hovermode="closest")
    fig.add_trace(go.Bar(y=data.index, x=data["items"], orientation="h", text=labels, textposition="outside", cliponaxis=False,
                         marker=dict(color=theme.series_colors()[0], line=dict(color=t["surface"], width=2)),
                         textfont=dict(color=t["text"], size=13), hovertemplate="%{y}: %{x:,} items<extra></extra>", showlegend=False))
    fig.update_yaxes(autorange="reversed", title_text=None)
    fig.update_xaxes(title_text="Number of items", range=[0, float(data["items"].max()) * 1.35])
    return fig


def forecast_chart(
    history: pd.Series,
    window: pd.DataFrame,
    *,
    model: str,
    cutoff: pd.Timestamp,
    unit: str,
    history_days: int = 90,
) -> go.Figure:
    """Past sales up to the training cutoff, then the saved forecast beside what actually happened."""
    t = theme.tokens()
    colors = theme.series_colors()
    shown = history.loc[history.index <= cutoff].iloc[-history_days:]
    fig = _fig(430, f"{model}: saved backtest forecast vs actual sales ({unit})")
    fig.add_trace(go.Scatter(x=shown.index, y=shown.values, mode="lines", name="Actual sales before the cutoff (training data)",
                             line=dict(color=t["history"], width=1.8), hovertemplate="%{y:,.1f}<extra>actual (training)</extra>"))
    if {"lower", "upper"} <= set(window.columns) and window["lower"].notna().all():
        fig.add_trace(go.Scatter(x=list(window["date"]) + list(window["date"])[::-1],
                                 y=list(window["upper"]) + list(window["lower"])[::-1], fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor="rgba(57,135,229,0.20)" if theme.current_theme() == "dark" else "rgba(42,120,214,0.16)",
                                 name="80% prediction interval (saved)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=window["date"], y=window["actual"], mode="lines+markers", name="Actual sales in the forecast window",
                             line=dict(color=t["history"], width=1.8, dash="dot"), marker=dict(symbol="circle", size=6, color=t["history"]),
                             hovertemplate="%{y:,.1f}<extra>actual</extra>"))
    fig.add_trace(go.Scatter(x=window["date"], y=window["predicted"], mode="lines+markers", name=f"{model} forecast",
                             line=dict(color=colors[0], width=2.6, dash="dash"), marker=dict(symbol="diamond", size=7, color=colors[0]),
                             hovertemplate="%{y:,.1f}<extra>forecast</extra>"))
    fig.add_shape(type="line", x0=cutoff, x1=cutoff, y0=0, y1=1, yref="paper", line=dict(color=t["muted"], width=1.5, dash="dot"))
    fig.add_annotation(x=cutoff, y=1, yref="paper", text="training cutoff", showarrow=False, xanchor="right", yanchor="top",
                       font=dict(color=t["text_secondary"], size=12), bgcolor=t["surface"])
    fig.update_yaxes(title_text=f"Units per day ({unit})", rangemode="tozero")
    return fig


def comparison_bar_chart(summary: pd.DataFrame, metric: str, *, scope: str, title: str, unit: str) -> go.Figure:
    """Mean metric across folds per model with the fold-to-fold standard deviation as an error bar; labelled values."""
    t = theme.tokens()
    colors = theme.series_colors()
    data = summary.dropna(subset=[f"{metric}_mean"]).sort_values(f"{metric}_mean")
    slots = _family_slots(list(data["model"]), scope)
    patterns = ["", "/", "x", "."]
    fig = _fig(max(300, 52 * len(data) + 120), title, hovermode="closest")
    for family, slot in slots.items():
        part = data[data["family"] == family]
        fig.add_trace(go.Bar(
            y=part["model"], x=part[f"{metric}_mean"], orientation="h", name=family,
            error_x=dict(type="data", array=part[f"{metric}_std"].fillna(0), color=t["text_secondary"], thickness=1.6, width=4),
            marker=dict(color=colors[slot % len(colors)],
                        pattern=dict(shape=patterns[slot % len(patterns)], bgcolor=colors[slot % len(colors)], fgcolor=t["surface"], size=7, solidity=0.35),
                        line=dict(color=colors[slot % len(colors)], width=1)),
            customdata=np.stack([part[f"{metric}_std"].fillna(np.nan), part["folds_scored"]], axis=-1),
            hovertemplate="%{y}<br>mean " + metric + " %{x:,.3f} (std %{customdata[0]:,.3f} over %{customdata[1]} folds)<extra></extra>",
        ))
    # value labels sit just beyond the error bar so the whisker never runs through the number
    fig.add_trace(go.Scatter(
        x=data[f"{metric}_mean"] + data[f"{metric}_std"].fillna(0), y=data["model"], mode="text", showlegend=False, hoverinfo="skip",
        text=[f"  {v:,.2f}" for v in data[f"{metric}_mean"]], textposition="middle right", cliponaxis=False,
        textfont=dict(color=t["text"], size=12)))
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=list(data["model"]), title_text=None)
    top = float((data[f"{metric}_mean"] + data[f"{metric}_std"].fillna(0)).max()) if len(data) else 1.0
    fig.update_xaxes(title_text=f"Mean {metric} over folds ({unit}); bars show one standard deviation across folds", range=[0, top * 1.28])
    fig.update_layout(barmode="overlay")
    return fig


def fold_lines_chart(fold_table: pd.DataFrame, metric: str, *, scope: str, title: str, unit: str) -> go.Figure:
    """One line per model across the five folds: colour = family, dash and marker = model."""
    colors = theme.series_colors()
    ok = fold_table[fold_table["success"].astype(bool)]
    models = list(dict.fromkeys(ok["model"]))
    slots = _family_slots(models, scope)
    within: Dict[str, int] = {}
    fig = _fig(430, title)
    for model in models:
        fam = model_family(model, scope)
        k = within.get(fam, 0)
        within[fam] = k + 1
        part = ok[ok["model"] == model].sort_values("fold")
        color = colors[slots[fam] % len(colors)]
        fig.add_trace(go.Scatter(x=part["fold"], y=part[metric], mode="lines+markers", name=model,
                                 line=dict(color=color, width=2, dash=DASHES[k % len(DASHES)]),
                                 marker=dict(symbol=SYMBOLS[k % len(SYMBOLS)], size=7, color=color),
                                 hovertemplate="%{y:,.3f}<extra>" + model + "</extra>"))
    fig.update_xaxes(title_text="Fold (1 = earliest forecast window, 5 = latest)", dtick=1)
    fig.update_yaxes(title_text=f"{metric} ({unit})", rangemode="tozero")
    fig.update_layout(legend=dict(orientation="v", x=1.01, y=1, yanchor="top"), margin=dict(r=210))
    return fig


def acf_chart(info: Dict[str, Any], *, title: str) -> go.Figure:
    t = theme.tokens()
    colors = theme.series_colors()
    fig = _fig(320, title, hovermode="closest")
    fig.add_trace(go.Bar(x=info["lags"], y=info["acf"], name="ACF of forecast errors", marker=dict(color=colors[0], line=dict(color=t["surface"], width=1)),
                         hovertemplate="lag %{x}: %{y:.2f}<extra></extra>"))
    band = info["band"]
    for sign in (1, -1):
        fig.add_hline(y=sign * band, line=dict(color=t["muted"], width=1.5, dash="dash"),
                      annotation_text="approx. 95% band (no autocorrelation)" if sign == 1 else None,
                      annotation_font=dict(color=t["text_secondary"], size=11), annotation_position="top right")
    fig.update_xaxes(title_text="Lag (days)", dtick=1)
    fig.update_yaxes(title_text="Autocorrelation", range=[-1, 1])
    return fig


def inventory_chart(forecast: Sequence[float], realised: Optional[Sequence[float]], *, protection_days: int, expected: float,
                    reorder_point: float, safety_stock: float, start: str) -> go.Figure:
    """Cumulative forecast demand over the protection interval, with the reorder point as the stocking level."""
    t = theme.tokens()
    colors = theme.series_colors()
    days = list(range(1, protection_days + 1))
    cum = np.cumsum(np.asarray(forecast, dtype=float)[:protection_days])
    fig = _fig(380, f"Demand over the protection interval (forecast origin {start})")
    fig.add_trace(go.Scatter(x=days, y=cum, mode="lines+markers", name="Cumulative forecast demand",
                             line=dict(color=colors[0], width=2.6, dash="dash"), marker=dict(symbol="diamond", size=7, color=colors[0]),
                             hovertemplate="day %{x}: %{y:,.1f} units<extra>forecast</extra>"))
    if realised is not None:
        real = np.cumsum(np.asarray(realised, dtype=float)[:protection_days])
        fig.add_trace(go.Scatter(x=days, y=real, mode="lines+markers", name="Demand that actually occurred (known only afterwards)",
                                 line=dict(color=t["history"], width=1.8, dash="dot"), marker=dict(symbol="circle", size=6, color=t["history"]),
                                 hovertemplate="day %{x}: %{y:,.1f} units<extra>actual</extra>"))
    fig.add_hline(y=reorder_point, line=dict(color=colors[1], width=2.4), annotation_text=f"Reorder point {reorder_point:,.1f}",
                  annotation_font=dict(color=t["text"], size=12), annotation_position="top left")
    fig.add_hline(y=expected, line=dict(color=t["muted"], width=1.5, dash="dash"),
                  annotation_text=f"Expected demand {expected:,.1f} (reorder point minus safety stock {safety_stock:,.1f})",
                  annotation_font=dict(color=t["text_secondary"], size=11), annotation_position="bottom left")
    fig.update_xaxes(title_text="Days after the forecast origin", dtick=max(1, protection_days // 14))
    fig.update_yaxes(title_text="Units", rangemode="tozero")
    return fig
