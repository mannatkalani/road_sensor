"""Cross-ride consensus: cluster events by location so repeated detections confirm each other."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from core import config as C

SPOT_COLUMNS = [
    "spot_id", "class_id", "class_name", "lat", "lon", "n_events", "n_rides", "confidence",
    "confirmed", "severity_mean", "severity_max", "speed_kmh", "model_confidence", "peak_abs_z", "rides",
]


def cluster_events(
    events: pd.DataFrame,
    n_rides_total: int | None = None,
    radius_m: float = C.CONSENSUS_RADIUS_M,
    min_confirmations: int = C.MIN_CONFIRMATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group same-class events within ``radius_m`` of each other (haversine DBSCAN, one cluster per spot).

    Returns ``(events_with_spot_id, spots)``. A spot's consensus ``confidence`` is the share of all
    rides that detected it (``n_rides / n_rides_total``). Events without GPS cannot be clustered
    and are left out of both tables.
    """
    ev = events.dropna(subset=["lat", "lon"]).copy() if len(events) else events.copy()
    if ev.empty:
        ev["spot_id"] = pd.Series(dtype=object)
        return ev, pd.DataFrame(columns=SPOT_COLUMNS)
    total = int(n_rides_total or ev["ride_id"].nunique())
    ev["spot_id"] = ""
    eps = radius_m / C.EARTH_RADIUS_M
    for cid in C.EVENT_CLASSES:
        mask = (ev["class_id"] == cid).to_numpy()
        if not mask.any():
            continue
        coords = np.radians(ev.loc[mask, ["lat", "lon"]].to_numpy(dtype=float))
        labels = DBSCAN(eps=eps, min_samples=1, metric="haversine").fit_predict(coords)
        prefix = "P" if cid == C.POTHOLE else "B"
        ev.loc[mask, "spot_id"] = [f"{prefix}{int(lab) + 1:03d}" for lab in labels]

    spots = (
        ev.groupby("spot_id")
        .agg(
            class_id=("class_id", "first"), lat=("lat", "mean"), lon=("lon", "mean"),
            n_events=("event_id", "count"), n_rides=("ride_id", "nunique"),
            severity_mean=("severity", "mean"), severity_max=("severity", "max"),
            speed_kmh=("speed_kmh", "mean"), model_confidence=("confidence", "mean"),
            peak_abs_z=("peak_abs_z", "mean"),
            rides=("ride_id", lambda s: ", ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    spots["class_name"] = spots["class_id"].map(lambda i: C.CLASS_NAMES[int(i)])
    spots["confidence"] = (spots["n_rides"] / max(total, 1)).clip(upper=1.0)
    spots["confirmed"] = spots["n_rides"] >= min_confirmations
    spots = spots.sort_values(["n_rides", "severity_max"], ascending=False).reset_index(drop=True)
    return ev, spots[SPOT_COLUMNS]


def simulate_repeat_passes(ride_df: pd.DataFrame, n_passes: int = 4, seed: int = 7,
                           gps_jitter_m: float = 3.0) -> pd.DataFrame:
    """DEMO ONLY: fabricate repeat passes over the same road from ONE recorded ride.

    Each pass keeps the original event timing and location but adds fresh sensor noise, a per-pass
    speed scale and per-second GPS jitter. The passes are NOT independent evidence; they exist so the
    consensus workflow can be demonstrated when only a single ride is available.
    """
    rng = np.random.default_rng(seed)
    base = ride_df.sort_values("time_s").reset_index(drop=True)
    rid = str(base["ride_id"].iloc[0])
    sec = np.floor(base["time_s"].to_numpy(dtype=float) - float(base["time_s"].iloc[0])).astype(int)
    cos_lat = np.cos(np.radians(base["lat"].mean())) if base["lat"].notna().any() else 1.0
    parts = []
    for k in range(1, n_passes + 1):
        p = base.copy()
        n = len(p)
        for c in ("acc_x", "acc_y", "acc_z"):
            p[c] = p[c] + rng.normal(0, 0.30, n)
        for c in ("gyro_x", "gyro_y", "gyro_z"):
            p[c] = p[c] + rng.normal(0, 0.02, n)
        p["speed_kmh"] = np.clip(p["speed_kmh"] * (1 + rng.normal(0, 0.06)) + rng.normal(0, 0.3, n), 0, None)
        p["lat"] = p["lat"] + rng.normal(0, gps_jitter_m / 111_320, sec.max() + 1)[sec]
        p["lon"] = p["lon"] + rng.normal(0, gps_jitter_m / (111_320 * max(cos_lat, 1e-6)), sec.max() + 1)[sec]
        p["ride_id"] = f"{rid}-pass{k}"
        parts.append(p)
    return pd.concat(parts, ignore_index=True)
