"""Event receipts: why did the model flag this spot? Waveform, class signature and top features."""
from __future__ import annotations

import html
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from core import config as C
from core.drift import load_reference
from core.inference import ModelBundle


@dataclass
class EventExplanation:
    """Everything needed to draw one event receipt."""

    event: dict[str, Any]
    t_rel: np.ndarray            # seconds relative to the estimated event start
    z: np.ndarray                # acc_z_dyn around the event
    sig_t: np.ndarray
    sig_mean: np.ndarray
    sig_std: np.ndarray
    features: pd.DataFrame       # top contributing features with plain-English text
    method: str                  # "SHAP" or "importance x deviation"
    summary: str


# ------------------------------------------------------------------ pieces
def class_signature(reference: dict[str, Any], class_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t_rel, mean, std) of acc_z_dyn around labelled events of ``class_id`` on the training routes."""
    sig = reference["signatures"]
    return (np.asarray(sig["t_rel"], dtype=float),
            np.asarray(sig[str(int(class_id))]["mean"], dtype=float),
            np.asarray(sig[str(int(class_id))]["std"], dtype=float))


def event_waveform(
    ride_df: pd.DataFrame, event: pd.Series | dict[str, Any], pre_s: float = 1.0, post_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """acc_z_dyn around an event, aligned so t=0 is the estimated event start (peak minus half its typical length)."""
    start = float(event["peak_time"]) - C.CLASS_MEAN_DURATION_S[int(event["class_id"])] / 2
    g = ride_df[ride_df["ride_id"] == event["ride_id"]].sort_values("time_s")
    t = g["time_s"].to_numpy(dtype=float)
    mask = (t >= start - pre_s) & (t <= start + post_s)
    return t[mask] - start, g["acc_z_dyn"].to_numpy(dtype=float)[mask]


def _importances(model: Any, n: int) -> np.ndarray:
    """Global feature importance for tree models, |coef| for linear ones, uniform otherwise."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_, dtype=float)
    if hasattr(est, "coef_"):
        return np.abs(np.asarray(est.coef_, dtype=float)).mean(axis=0)
    return np.full(n, 1.0 / n)


def _shap_values(model: Any, x_row: pd.DataFrame, class_id: int) -> np.ndarray | None:
    """SHAP values for one row and class, or None when shap is not installed / not applicable."""
    try:
        import shap  # optional dependency
    except ImportError:
        return None
    try:
        values = shap.TreeExplainer(model).shap_values(x_row)
        if isinstance(values, list):
            return np.asarray(values[class_id])[0]
        arr = np.asarray(values)
        return arr[0, :, class_id] if arr.ndim == 3 else arr[0]
    except Exception:  # noqa: BLE001 - fall back to the built-in explanation
        return None


def top_features(
    bundle: ModelBundle, window_row: pd.Series, class_id: int, reference: dict[str, Any], k: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Top-``k`` features behind the prediction, with plain-English text and the method used."""
    cols = list(bundle.feature_columns)
    values = window_row[cols].astype(float)
    ref = reference["features"]
    z = np.array([(values[c] - ref[c]["mean"]) / (ref[c]["std"] or 1.0) for c in cols])
    shap_vals = _shap_values(bundle.model, pd.DataFrame([values.to_numpy()], columns=cols), class_id)
    if shap_vals is not None and len(shap_vals) == len(cols):
        contrib, method = np.asarray(shap_vals, dtype=float), "SHAP"
    else:
        contrib, method = _importances(bundle.model, len(cols)) * z, "importance x deviation"
    order = np.argsort(-np.abs(contrib))[:k]
    name = C.CLASS_NAMES[int(class_id)].lower()
    rows = []
    for i in order:
        c = cols[i]
        side = "above" if z[i] > 0 else "below"
        base = (f"{C.FEATURE_LABELS[c].capitalize()} is {values[c]:.2f}, "
                f"{abs(z[i]):.1f} std {side} the training average ({ref[c]['mean']:.2f})")
        tail = (f", which pushed the model toward '{name}'." if contrib[i] > 0
                else f", which pushed the model away from '{name}'.") if method == "SHAP" else \
            ", and the model leans heavily on this signal."
        rows.append({"feature": c, "label": C.FEATURE_LABELS[c], "value": float(values[c]),
                     "ref_mean": float(ref[c]["mean"]), "z": float(z[i]), "contribution": float(contrib[i]),
                     "text": base + tail})
    return pd.DataFrame(rows), method


def explain_event(
    bundle: ModelBundle,
    windows: pd.DataFrame,
    event: pd.Series | dict[str, Any],
    ride_df: pd.DataFrame,
    reference: dict[str, Any] | None = None,
    k: int = 5,
) -> EventExplanation:
    """Build the explanation for one detected event."""
    ref = reference or load_reference()
    ev = dict(event)
    cid = int(ev["class_id"])
    t_rel, z = event_waveform(ride_df, ev)
    sig_t, sig_mean, sig_std = class_signature(ref, cid)
    m = windows[(windows["ride_id"] == ev["ride_id"])
                & np.isclose(windows["start_time"], ev["best_window_start"])]
    row = m.iloc[0] if len(m) else windows[windows["ride_id"] == ev["ride_id"]].iloc[
        int(np.argmin(np.abs(windows.loc[windows["ride_id"] == ev["ride_id"], "start_time"].to_numpy()
                              - ev["best_window_start"])))]
    feats, method = top_features(bundle, row, cid, ref, k=k)
    summary = (f"{ev['class_name']} at {ev['peak_time']:.1f} s in ride {ev['ride_id']}: "
               f"{ev['confidence']:.0%} model confidence, {ev['severity_label'].lower()} severity "
               f"({ev['severity']:.0f}/100) at about {ev['speed_kmh']:.0f} km/h.")
    return EventExplanation(event=ev, t_rel=t_rel, z=z, sig_t=sig_t, sig_mean=sig_mean, sig_std=sig_std,
                            features=feats, method=method, summary=summary)


# ------------------------------------------------------------------ receipt card (HTML / PNG)
def _svg_polyline(x: np.ndarray, y: np.ndarray, x_rng: tuple[float, float], y_rng: tuple[float, float],
                  box: tuple[float, float, float, float], color: str, dash: str = "", width: float = 1.6) -> str:
    left, top, w, h = box
    if len(x) < 2:
        return ""
    px = left + (x - x_rng[0]) / (x_rng[1] - x_rng[0]) * w
    py = top + h - (y - y_rng[0]) / (y_rng[1] - y_rng[0]) * h
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{d} points="{pts}"/>'


def receipt_html(expl: EventExplanation) -> str:
    """A self-contained HTML card (inline SVG, no external assets) for download."""
    ev, col = expl.event, C.CLASS_COLORS[int(expl.event["class_id"])]
    lo = float(min(expl.z.min() if len(expl.z) else 0, expl.sig_mean.min()))
    hi = float(max(expl.z.max() if len(expl.z) else 0, expl.sig_mean.max()))
    pad = 0.1 * (hi - lo or 1.0)
    y_rng, x_rng, box = (lo - pad, hi + pad), (-1.0, 2.0), (40.0, 10.0, 500.0, 150.0)
    zero_y = box[1] + box[3] - (0 - y_rng[0]) / (y_rng[1] - y_rng[0]) * box[3]
    svg = (
        f'<svg viewBox="0 0 560 190" width="100%" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="40" y="10" width="500" height="150" fill="none" stroke="{C.COLORS["border"]}"/>'
        f'<line x1="40" x2="540" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="{C.COLORS["border"]}" stroke-dasharray="3 3"/>'
        + _svg_polyline(expl.sig_t, expl.sig_mean, x_rng, y_rng, box, C.COLORS["muted"], "5 4", 1.4)
        + _svg_polyline(expl.t_rel, expl.z, x_rng, y_rng, box, col, "", 1.8)
        + f'<text x="40" y="180" fill="{C.COLORS["muted"]}" font-size="11">-1 s</text>'
          f'<text x="520" y="180" fill="{C.COLORS["muted"]}" font-size="11">+2 s</text>'
          f'<text x="150" y="180" fill="{C.COLORS["muted"]}" font-size="11">'
          f'solid = this event, dashed = average {html.escape(str(ev["class_name"]).lower())}</text></svg>'
    )
    items = "".join(f"<li>{html.escape(t)}</li>" for t in expl.features["text"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>RoadPulse receipt {html.escape(str(ev['event_id']))}</title></head>
<body style="margin:0;background:{C.COLORS['bg']};font-family:Inter,Arial,sans-serif;color:{C.COLORS['text']};padding:24px">
<div style="max-width:600px;margin:auto;background:{C.COLORS['panel']};border:1px solid {C.COLORS['border']};border-top:3px solid {col};border-radius:12px;padding:20px">
<div style="font-family:'Space Grotesk',Arial,sans-serif;font-size:22px;font-weight:700;color:{col}">{html.escape(str(ev['class_name']))}</div>
<div style="color:{C.COLORS['muted']};font-size:13px;margin:4px 0 12px">{html.escape(expl.summary)}</div>
<div style="display:flex;gap:12px;margin-bottom:12px;font-family:'JetBrains Mono',monospace">
<div><div style="color:{C.COLORS['muted']};font-size:11px">Confidence</div><div style="font-size:20px">{ev['confidence']:.0%}</div></div>
<div><div style="color:{C.COLORS['muted']};font-size:11px">Severity</div><div style="font-size:20px">{ev['severity']:.0f}/100</div></div>
<div><div style="color:{C.COLORS['muted']};font-size:11px">Speed</div><div style="font-size:20px">{ev['speed_kmh']:.0f} km/h</div></div></div>
{svg}
<div style="font-weight:600;margin:12px 0 6px">Why the model flagged it <span style="color:{C.COLORS['muted']};font-weight:400;font-size:12px">({html.escape(expl.method)})</span></div>
<ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.5">{items}</ul>
<div style="margin-top:14px;padding-top:10px;border-top:1px solid {C.COLORS['border']};color:{C.COLORS['muted']};font-size:11px">{html.escape(C.SIMULATED_NOTICE)}</div>
</div></body></html>"""


def receipt_png(expl: EventExplanation) -> bytes:
    """Render the receipt as a PNG (matplotlib Agg, no pyplot state)."""
    import textwrap

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ev, col = expl.event, C.CLASS_COLORS[int(expl.event["class_id"])]
    fig = Figure(figsize=(8.5, 8.4), facecolor=C.COLORS["panel"])
    FigureCanvasAgg(fig)
    fig.text(0.05, 0.95, str(ev["class_name"]), color=col, fontsize=20, fontweight="bold", va="top")
    fig.text(0.05, 0.905, "\n".join(textwrap.wrap(expl.summary, 90)), color=C.COLORS["muted"], fontsize=9, va="top")
    ax = fig.add_axes([0.08, 0.55, 0.87, 0.27], facecolor=C.COLORS["bg"])
    ax.plot(expl.sig_t, expl.sig_mean, color=C.COLORS["muted"], linestyle="--", linewidth=1.3, label="average " + str(ev["class_name"]).lower())
    ax.fill_between(expl.sig_t, expl.sig_mean - expl.sig_std, expl.sig_mean + expl.sig_std, color=C.COLORS["muted"], alpha=0.12)
    ax.plot(expl.t_rel, expl.z, color=col, linewidth=1.8, label="this event")
    ax.axhline(0, color=C.COLORS["border"], linewidth=0.8)
    ax.set_xlim(-1, 2)
    ax.set_xlabel("seconds from estimated event start", color=C.COLORS["muted"], fontsize=8)
    ax.set_ylabel("acc_z (gravity removed)", color=C.COLORS["muted"], fontsize=8)
    ax.tick_params(colors=C.COLORS["muted"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(C.COLORS["border"])
    ax.legend(facecolor=C.COLORS["panel"], edgecolor=C.COLORS["border"], labelcolor=C.COLORS["text"], fontsize=8)
    stats = f"confidence {ev['confidence']:.0%}   severity {ev['severity']:.0f}/100   speed {ev['speed_kmh']:.0f} km/h"
    fig.text(0.05, 0.47, stats, color=C.COLORS["text"], fontsize=10, family="monospace")
    fig.text(0.05, 0.435, f"Why the model flagged it ({expl.method})", color=C.COLORS["text"], fontsize=10, fontweight="bold")
    y = 0.405
    for t in expl.features["text"]:
        wrapped = textwrap.wrap(t, 95)
        fig.text(0.06, y, "\n".join(wrapped), color=C.COLORS["text"], fontsize=8, va="top")
        y -= 0.024 * len(wrapped) + 0.014
    fig.text(0.05, 0.03, "\n".join(textwrap.wrap(C.SIMULATED_NOTICE, 105)), color=C.COLORS["muted"], fontsize=7, va="bottom")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor=C.COLORS["panel"])
    return buf.getvalue()
