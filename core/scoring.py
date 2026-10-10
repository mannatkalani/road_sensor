"""Repair Priority = severity x confirmations x low-speed exposure."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

LOW_SPEED_KMH = 30.0


def exposure_factor(speed_kmh: float) -> float:
    """1.0 at or above 30 km/h, rising to 2.0 at a standstill (slow traffic means more people exposed)."""
    if speed_kmh is None or np.isnan(speed_kmh):
        return 1.0
    return float(1.0 + np.clip((LOW_SPEED_KMH - speed_kmh) / LOW_SPEED_KMH, 0.0, 1.0))


def priority_score(severity: float, confirmations: int, n_rides_total: int, speed_kmh: float) -> float:
    """0-100 repair priority: severity (0-100) x confirmations x exposure, scaled by the maximum possible."""
    sev = float(np.clip(severity / 100.0, 0.0, 1.0))
    conf = float(min(max(confirmations, 0), max(n_rides_total, 1)))
    raw = sev * conf * exposure_factor(speed_kmh)
    return float(100.0 * raw / (max(n_rides_total, 1) * 2.0))


def rank_priority(
    spots: pd.DataFrame,
    n_rides_total: int,
    classes: Iterable[int] | None = None,
    min_confirmations: int = 1,
) -> pd.DataFrame:
    """Rank spots by repair priority (highest first) with a plain-English reason for each."""
    cols = ["rank", "spot_id", "class_name", "priority", "severity", "confirmations", "speed_kmh",
            "exposure", "lat", "lon", "why"]
    if spots.empty:
        return pd.DataFrame(columns=cols)
    df = spots.copy()
    if classes is not None:
        df = df[df["class_id"].isin(list(classes))]
    df = df[df["n_rides"] >= min_confirmations]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["severity"] = df["severity_mean"]
    df["confirmations"] = df["n_rides"]
    df["exposure"] = df["speed_kmh"].map(exposure_factor)
    df["priority"] = [
        priority_score(s, c, n_rides_total, v)
        for s, c, v in zip(df["severity"], df["confirmations"], df["speed_kmh"])
    ]
    df = df.sort_values("priority", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    def _why(r: pd.Series) -> str:
        sev = "severe" if r["severity"] >= 70 else "moderate" if r["severity"] >= 35 else "mild"
        speed = f" at about {r['speed_kmh']:.0f} km/h" if not np.isnan(r["speed_kmh"]) else ""
        conf = f"confirmed by {int(r['confirmations'])} of {n_rides_total} ride(s)"
        return f"{sev.capitalize()} {r['class_name'].lower()}, {conf}{speed}."

    df["why"] = df.apply(_why, axis=1)
    return df[cols]
