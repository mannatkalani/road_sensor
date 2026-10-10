"""Plotly figures sharing ONE custom dark template (transparent background, theme colours)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from core import config as C

TEMPLATE_NAME = "roadpulse"
_AXIS = dict(gridcolor=C.COLORS["border"], zerolinecolor=C.COLORS["border"], linecolor=C.COLORS["border"],
             tickfont=dict(family=C.FONT_MONO, size=11, color=C.COLORS["muted"]),
             title=dict(font=dict(color=C.COLORS["muted"], size=12)))


def register_template() -> str:
    """Register the RoadPulse Plotly template (idempotent) and return its name."""
    pio.templates[TEMPLATE_NAME] = go.layout.Template(layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=f"{C.FONT_BODY}, sans-serif", color=C.COLORS["text"], size=13),
        title=dict(font=dict(family=f"{C.FONT_HEADING}, sans-serif", size=18, color=C.COLORS["text"])),
        colorway=[C.COLORS["accent"], C.COLORS["smooth"], C.COLORS["pothole"], "#7C9CFF", "#C084FC", C.COLORS["muted"]],
        xaxis=_AXIS, yaxis=_AXIS,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C.COLORS["muted"])),
        margin=dict(l=48, r=24, t=56, b=44),
        hoverlabel=dict(bgcolor=C.COLORS["panel"], bordercolor=C.COLORS["border"],
                        font=dict(family=C.FONT_MONO, color=C.COLORS["text"])),
    ))
    return TEMPLATE_NAME


def _fig(title: str = "", height: int = 360, **layout) -> go.Figure:
    register_template()
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE_NAME, title=title, height=height, **layout)
    return fig


def waveform_fig(ride_df: pd.DataFrame, events: pd.DataFrame | None = None, ride_id: str | None = None,
                 title: str = "Vertical vibration with detected events") -> go.Figure:
    """acc_z (gravity removed) over time, with shaded spans for each detected event."""
    g = ride_df if ride_id is None else ride_df[ride_df["ride_id"] == ride_id]
    g = g.sort_values("time_s")
    step = max(1, len(g) // 6000)
    g = g.iloc[::step]
    fig = _fig(title, 360, xaxis_title="Time (s)", yaxis_title="acc_z, gravity removed (m/s²)")
    fig.add_trace(go.Scatter(x=g["time_s"], y=g["acc_z_dyn"], mode="lines", name="acc_z",
                             line=dict(color=C.COLORS["muted"], width=1)))
    if events is not None and len(events):
        ev = events if ride_id is None else events[events["ride_id"] == ride_id]
        for _, r in ev.iterrows():
            fig.add_vrect(x0=r["peak_time"] - 0.6, x1=r["peak_time"] + 0.6, line_width=0,
                          fillcolor=C.CLASS_COLORS[int(r["class_id"])], opacity=0.35)
        for cid in C.EVENT_CLASSES:
            sub = ev[ev["class_id"] == cid]
            if len(sub):
                fig.add_trace(go.Scatter(x=sub["peak_time"], y=[g["acc_z_dyn"].max() * 1.05] * len(sub), mode="markers",
                                         name=C.CLASS_NAMES[cid], marker=dict(color=C.CLASS_COLORS[cid], size=9, symbol="triangle-down"),
                                         text=sub["event_id"] if "event_id" in sub else None,
                                         hovertemplate="%{text}<br>%{x:.1f} s<extra></extra>"))
    return fig


def probability_fig(pred_windows: pd.DataFrame, ride_id: str | None = None) -> go.Figure:
    """Event probabilities per window over time."""
    g = pred_windows if ride_id is None else pred_windows[pred_windows["ride_id"] == ride_id]
    g = g.sort_values("start_time")
    t = g["start_time"] + C.WINDOW_SECONDS / 2
    fig = _fig("Model probability per window", 280, xaxis_title="Time (s)", yaxis_title="Probability", yaxis_range=[0, 1.02])
    fig.add_trace(go.Scatter(x=t, y=g["p_speed_breaker"], name="Speed breaker", mode="lines", line=dict(color=C.COLORS["speed_breaker"])))
    fig.add_trace(go.Scatter(x=t, y=g["p_pothole"], name="Pothole", mode="lines", line=dict(color=C.COLORS["pothole"])))
    return fig


def threshold_sweep_fig(sweep: pd.DataFrame) -> go.Figure:
    """Detection quality against the threshold.

    With labels: recall (left axis) and false alarms per km (right axis). Without labels: events detected
    (left axis) and estimated false alarms per km (right axis).
    """
    has_recall = "recall" in sweep and sweep["recall"].notna().any()
    left_col, left_name = ("recall", "Recall") if has_recall else ("events", "Events detected")
    fig = _fig("Operating trade-off", 340, xaxis_title="Threshold", yaxis_title=left_name,
               yaxis2=dict(title="False alarms / km", overlaying="y", side="right", **{k: v for k, v in _AXIS.items() if k != "title"}))
    if has_recall:
        fig.update_layout(yaxis_range=[0, 1.02])
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[left_col], name=left_name, line=dict(color=C.COLORS["smooth"], width=3)))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["false_alarms_per_km"], name="False alarms / km",
                             yaxis="y2", line=dict(color=C.COLORS["pothole"], width=3)))
    return fig


def gauge_fig(score: float, label: str) -> go.Figure:
    """Gauge for the Trust Meter (0-100)."""
    color = {"High": C.COLORS["smooth"], "Medium": C.COLORS["speed_breaker"], "Low": C.COLORS["pothole"]}[label]
    fig = _fig("", 280)
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=float(score), number=dict(font=dict(family=C.FONT_MONO, size=44, color=color)),
        title=dict(text=f"Trust: {label}", font=dict(family=C.FONT_HEADING, size=18, color=C.COLORS["text"])),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=C.COLORS["muted"]), bar=dict(color=color, thickness=0.28),
                   bgcolor=C.COLORS["panel"], borderwidth=1, bordercolor=C.COLORS["border"],
                   steps=[dict(range=[0, C.TRUST_BANDS[1]], color="#2A1A1C"),
                          dict(range=[C.TRUST_BANDS[1], C.TRUST_BANDS[0]], color="#2A2417"),
                          dict(range=[C.TRUST_BANDS[0], 100], color="#12302D")])))
    return fig


def drift_fig(per_feature: pd.DataFrame) -> go.Figure:
    """How far each feature's ride median sits from the training average, in training standard deviations."""
    d = per_feature.sort_values("z", key=lambda s: s.abs())
    colors = [C.COLORS["pothole"] if abs(z) > 1.5 else C.COLORS["speed_breaker"] if abs(z) > 1 else C.COLORS["smooth"] for z in d["z"]]
    fig = _fig("Feature drift vs training data", max(360, 22 * len(d) + 120), xaxis_title="Standard deviations from training average")
    fig.add_trace(go.Bar(x=d["z"], y=d["label"], orientation="h", marker_color=colors,
                         hovertemplate="%{y}: %{x:.2f} std<extra></extra>"))
    fig.add_vrect(x0=-1, x1=1, fillcolor=C.COLORS["smooth"], opacity=0.07, line_width=0)
    return fig


def signature_fig(expl) -> go.Figure:  # expl: core.explain.EventExplanation
    """The event's waveform over the average signature of its class."""
    col = C.CLASS_COLORS[int(expl.event["class_id"])]
    fig = _fig("Event vs average class signature", 340, xaxis_title="Seconds from estimated event start", yaxis_title="acc_z (m/s²)")
    fig.add_trace(go.Scatter(x=np.r_[expl.sig_t, expl.sig_t[::-1]], y=np.r_[expl.sig_mean + expl.sig_std, (expl.sig_mean - expl.sig_std)[::-1]],
                             fill="toself", fillcolor="rgba(139,152,165,0.15)", line=dict(width=0), name="±1 std", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=expl.sig_t, y=expl.sig_mean, name="Average class signature", line=dict(color=C.COLORS["muted"], dash="dash")))
    fig.add_trace(go.Scatter(x=expl.t_rel, y=expl.z, name="This event", line=dict(color=col, width=3)))
    return fig


def model_comparison_fig(comp: pd.DataFrame) -> go.Figure:
    """Grouped bars of the headline metrics per model."""
    d = comp.copy()
    if "Model" in d.columns:
        d = d.set_index("Model")
    fig = _fig("Model comparison (held-out route D)", 380, yaxis_title="Score", yaxis_range=[0, 1.05], barmode="group")
    for col, color in zip(["Macro F1", "CV Macro F1", "Pothole recall", "Speed breaker recall"],
                          [C.COLORS["accent"], C.COLORS["smooth"], C.COLORS["pothole"], "#7C9CFF"]):
        if col in d.columns:
            fig.add_trace(go.Bar(x=d.index, y=d[col], name=col, marker_color=color, text=d[col].round(2), textposition="outside"))
    return fig


def cv_fig(cv: pd.DataFrame) -> go.Figure:
    """Leave-one-route-out macro-F1 per model and held-out route."""
    fig = _fig("Leave-one-route-out cross-validation", 360, yaxis_title="Macro F1", yaxis_range=[0, 1.05], barmode="group")
    for model, g in cv.groupby("Model"):
        fig.add_trace(go.Bar(x=g["Held-out route"], y=g["Macro F1"], name=model, text=g["Macro F1"].round(2), textposition="outside"))
    fig.update_xaxes(title_text="Held-out route")
    return fig


def per_class_fig(pc: pd.DataFrame, metric: str = "Recall") -> go.Figure:
    """One metric (Precision / Recall / F1) per class and model."""
    fig = _fig(f"{metric} by class", 340, yaxis_title=metric, yaxis_range=[0, 1.05], barmode="group")
    for model, g in pc.groupby("Model"):
        fig.add_trace(go.Bar(x=g["Class"], y=g[metric], name=model, text=g[metric].round(2), textposition="outside"))
    return fig


def priority_fig(ranked: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Top repair priorities as horizontal bars."""
    d = ranked.head(top_n).iloc[::-1]
    fig = _fig("Repair priority", max(300, 28 * len(d) + 120), xaxis_title="Priority (0-100)", xaxis_range=[0, 100])
    fig.add_trace(go.Bar(x=d["priority"], y=[f"#{r} {c} {s}" for r, c, s in zip(d["rank"], d["class_name"], d["spot_id"])],
                         orientation="h", marker_color=[C.CLASS_COLORS[C.POTHOLE if n == "Pothole" else C.SPEED_BREAKER] for n in d["class_name"]],
                         text=d["priority"].round(0), textposition="outside"))
    return fig
