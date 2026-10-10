"""Merge flagged windows into events, score severity, and evaluate events against labels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core import config as C

EVENT_COLUMNS = [
    "event_id", "ride_id", "route_id", "class_id", "class_name", "start_time", "end_time", "peak_time",
    "lat", "lon", "n_windows", "confidence", "max_prob", "peak_abs_z", "speed_kmh",
    "severity", "severity_label", "best_window_start",
]


def severity_score(peak_abs_z: float, class_id: int) -> float:
    """0-100 severity from the peak vertical jolt relative to a typical event of that class."""
    ref = C.SEVERITY_REF_PEAK[int(class_id)]
    return float(np.clip(100.0 * peak_abs_z / ref, 0.0, 100.0))


def severity_label(score: float) -> str:
    low, high = C.SEVERITY_BANDS
    return "Low" if score < low else "Medium" if score < high else "High"


def _interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    ok = ~np.isnan(fp)
    return float(np.interp(x, xp[ok], fp[ok])) if ok.sum() >= 2 else float("nan")


def merge_windows_to_events(
    pred_windows: pd.DataFrame,
    ride_df: pd.DataFrame | None = None,
    *,
    window_s: float = C.WINDOW_SECONDS,
    gap_s: float = C.EVENT_MERGE_GAP_S,
) -> pd.DataFrame:
    """Merge runs of consecutive flagged windows (per ride) into single events.

    The event class is the one with the larger summed probability over the run. The event time is
    the probability-weighted centre of its windows. When the preprocessed ``ride_df`` is supplied,
    location, speed and peak jolt are read from the raw signal at that time.
    """
    has_route = "route_id" in pred_windows.columns
    rows: list[dict[str, Any]] = []
    rides = {rid: g.sort_values("time_s") for rid, g in ride_df.groupby("ride_id")} if ride_df is not None else {}
    for rid, g in pred_windows.groupby("ride_id", sort=False):
        flagged = g.sort_values("start_time")
        flagged = flagged[flagged["pred"] > 0]
        if flagged.empty:
            continue
        starts = flagged["start_time"].to_numpy(dtype=float)
        breaks = np.where(np.diff(starts) > gap_s + 1e-6)[0] + 1
        ride = rides.get(rid)
        for n, idx in enumerate(np.split(np.arange(len(flagged)), breaks), start=1):
            run = flagged.iloc[idx]
            p1 = run["p_speed_breaker"].to_numpy(dtype=float)
            p2 = run["p_pothole"].to_numpy(dtype=float)
            class_id = C.SPEED_BREAKER if p1.sum() > p2.sum() else C.POTHOLE
            pc = p1 if class_id == C.SPEED_BREAKER else p2
            centers = run["start_time"].to_numpy(dtype=float) + window_s / 2
            peak_time = float(np.average(centers, weights=pc + 1e-9))
            best = int(np.argmax(pc))

            lat = lon = float("nan")
            peak_abs_z = float(run["z_absmax"].max()) if "z_absmax" in run else float("nan")
            speed = float(np.average(run["speed_mean"], weights=pc + 1e-9)) if "speed_mean" in run else float("nan")
            if ride is not None and len(ride):
                t = ride["time_s"].to_numpy(dtype=float)
                lat = _interp(peak_time, t, ride["lat"].to_numpy(dtype=float))
                lon = _interp(peak_time, t, ride["lon"].to_numpy(dtype=float))
                speed = _interp(peak_time, t, ride["speed_kmh"].to_numpy(dtype=float))
                near = np.abs(t - peak_time) <= C.EVENT_PEAK_HALF_WINDOW_S
                if near.any():
                    peak_abs_z = float(np.abs(ride["acc_z_dyn"].to_numpy(dtype=float)[near]).max())
            if np.isnan(lat) and "lat" in run:
                lat = float(np.nanmean(run["lat"])) if run["lat"].notna().any() else float("nan")
                lon = float(np.nanmean(run["lon"])) if run["lon"].notna().any() else float("nan")

            sev = severity_score(peak_abs_z, class_id) if not np.isnan(peak_abs_z) else 0.0
            rows.append({
                "event_id": f"{rid}-E{n:03d}", "ride_id": rid,
                "route_id": run["route_id"].iloc[0] if has_route else None,
                "class_id": class_id, "class_name": C.CLASS_NAMES[class_id],
                "start_time": float(starts[idx[0]]), "end_time": float(starts[idx[-1]] + window_s),
                "peak_time": peak_time, "lat": lat, "lon": lon, "n_windows": int(len(run)),
                "confidence": float(pc.mean()), "max_prob": float(pc.max()),
                "peak_abs_z": peak_abs_z, "speed_kmh": speed,
                "severity": sev, "severity_label": severity_label(sev),
                "best_window_start": float(starts[idx[best]]),
            })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


# ------------------------------------------------------------------ distance / duration
def distance_km(df: pd.DataFrame) -> float:
    """Distance travelled, integrated from speed (robust to missing GPS). Gaps over 1 s are ignored."""
    total = 0.0
    for _, g in df.groupby("ride_id", sort=False):
        g = g.sort_values("time_s")
        t = g["time_s"].to_numpy(dtype=float)
        v = g["speed_kmh"].to_numpy(dtype=float)
        if len(t) < 2:
            continue
        dt = np.clip(np.diff(t), 0.0, 1.0)
        total += float(np.nansum(v[:-1] * dt) / 3600.0)
    return total


def duration_s(df: pd.DataFrame) -> float:
    """Total recorded duration over all rides, in seconds."""
    return float(sum(g["time_s"].max() - g["time_s"].min() for _, g in df.groupby("ride_id", sort=False)))


def summarize_events(events: pd.DataFrame) -> dict[str, int]:
    """Event counts per class name (always includes both event classes)."""
    counts = events["class_name"].value_counts().to_dict() if len(events) else {}
    return {name: int(counts.get(name, 0)) for name in (C.CLASS_NAMES[1], C.CLASS_NAMES[2])}


# ------------------------------------------------------------------ evaluation against labels
def ground_truth_events(df: pd.DataFrame) -> pd.DataFrame:
    """Labelled events (contiguous label segments) from a table with a ``label`` column."""
    cols = ["ride_id", "class_id", "start_time", "end_time", "lat", "lon"]
    if "label" not in df.columns:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for rid, g in df.groupby("ride_id", sort=False):
        g = g.sort_values("time_s").reset_index(drop=True)
        lab = g["label"].to_numpy()
        prev = np.concatenate([[0], lab[:-1]])
        seg = np.cumsum(lab != prev)
        for s in np.unique(seg[lab > 0]):
            m = (seg == s) & (lab > 0)
            sub = g[m]
            rows.append({
                "ride_id": rid, "class_id": int(lab[m][0]),
                "start_time": float(sub["time_s"].iloc[0]), "end_time": float(sub["time_s"].iloc[-1]),
                "lat": float(sub["lat"].mean()) if sub["lat"].notna().any() else float("nan"),
                "lon": float(sub["lon"].mean()) if sub["lon"].notna().any() else float("nan"),
            })
    return pd.DataFrame(rows, columns=cols)


@dataclass
class EventMetrics:
    """Event-level precision / recall plus per-km false-alarm and miss rates."""

    per_class: pd.DataFrame
    summary: dict[str, float]


def _gap(peak: float, start: float, end: float) -> float:
    return 0.0 if start <= peak <= end else min(abs(peak - start), abs(peak - end))


def evaluate_events(
    pred_events: pd.DataFrame,
    gt_events: pd.DataFrame,
    km: float,
    tol_s: float = C.MATCH_TOLERANCE_S,
) -> EventMetrics:
    """Match predicted events to labelled ones (same ride, same class, peak within ``tol_s`` of the label).

    Each labelled event can be matched once. Unmatched predictions are false alarms; unmatched
    labelled events are misses.
    """
    rows = []
    for cid in C.EVENT_CLASSES:
        p = pred_events[pred_events["class_id"] == cid] if len(pred_events) else pred_events
        g = gt_events[gt_events["class_id"] == cid]
        tp = 0
        for rid in set(g["ride_id"]) | (set(p["ride_id"]) if len(p) else set()):
            pr = p[p["ride_id"] == rid] if len(p) else p
            gr = g[g["ride_id"] == rid]
            pairs = sorted(
                (_gap(pt, gs, ge), i, j)
                for i, pt in enumerate(pr["peak_time"].to_numpy(dtype=float))
                for j, (gs, ge) in enumerate(zip(gr["start_time"], gr["end_time"]))
            )
            used_p: set[int] = set()
            used_g: set[int] = set()
            for d, i, j in pairs:
                if d <= tol_s and i not in used_p and j not in used_g:
                    used_p.add(i)
                    used_g.add(j)
                    tp += 1
        n_pred, n_gt = len(p), len(g)
        fp, fn = n_pred - tp, n_gt - tp
        prec = tp / n_pred if n_pred else float("nan")
        rec = tp / n_gt if n_gt else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec and not np.isnan(prec) and not np.isnan(rec)) else (
            0.0 if n_pred and n_gt else float("nan"))
        rows.append({"class_id": cid, "class_name": C.CLASS_NAMES[cid], "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1})
    per_class = pd.DataFrame(rows)
    tp, fp, fn = int(per_class["tp"].sum()), int(per_class["fp"].sum()), int(per_class["fn"].sum())
    summary = {
        "tp": tp, "fp": fp, "fn": fn, "distance_km": km,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "false_alarms_per_km": fp / km if km > 0 else float("nan"),
        "missed_per_km": fn / km if km > 0 else float("nan"),
        "n_predicted": int(len(pred_events)), "n_actual": int(len(gt_events)),
    }
    return EventMetrics(per_class=per_class, summary=summary)


def estimate_false_alarms(events: pd.DataFrame) -> float:
    """Label-free estimate of false alarms: the sum of (1 - confidence) over detected events.

    Valid only if the model's probabilities are well calibrated; treat it as a rough guide.
    """
    return float((1.0 - events["confidence"]).sum()) if len(events) else 0.0
