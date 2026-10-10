"""Exports: GeoJSON, CSV, complaint-letter text and Google Maps deep links (no API keys, no integrations)."""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd

MAX_WAYPOINTS = 9   # Google Maps URLs accept up to 9 waypoints


def _py(v: Any) -> Any:
    """Convert numpy / pandas scalars to JSON-safe Python values (NaN -> None)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA or v is None:
        return None
    return v if isinstance(v, (str, int, bool)) else str(v)


def dataframe_to_geojson(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon",
                         properties: Sequence[str] | None = None) -> str:
    """Point FeatureCollection (GeoJSON, [lon, lat] order). Rows without coordinates are skipped."""
    props = list(properties) if properties is not None else [c for c in df.columns if c not in (lat_col, lon_col)]
    features = []
    for _, r in df.dropna(subset=[lat_col, lon_col]).iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r[lon_col]), float(r[lat_col])]},
            "properties": {p: _py(r[p]) for p in props if p in r.index},
        })
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=1)


def events_to_geojson(events: pd.DataFrame) -> str:
    return dataframe_to_geojson(events, properties=[
        "event_id", "ride_id", "class_name", "confidence", "severity", "severity_label", "speed_kmh", "peak_time"])


def spots_to_geojson(spots: pd.DataFrame) -> str:
    cols = [c for c in ("rank", "spot_id", "class_name", "priority", "severity", "severity_mean", "n_rides",
                        "confirmations", "confidence", "confirmed", "speed_kmh", "why") if c in spots.columns]
    return dataframe_to_geojson(spots, properties=cols)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 CSV for download buttons."""
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------------ Google Maps deep links
def google_maps_pin_link(lat: float, lon: float) -> str:
    """Link that opens Google Maps with a pin at the location."""
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


def nearest_neighbour_order(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Greedy visiting order starting from the first point (keeps the highest-priority spot first)."""
    remaining = list(points)
    if not remaining:
        return []
    route = [remaining.pop(0)]
    while remaining:
        last = route[-1]
        i = min(range(len(remaining)),
                key=lambda k: (remaining[k][0] - last[0]) ** 2 + (remaining[k][1] - last[1]) ** 2)
        route.append(remaining.pop(i))
    return route


def google_maps_route_link(points: Iterable[tuple[float, float]], mode: str = "driving") -> str:
    """Directions deep link through the points (origin, up to 9 waypoints, destination)."""
    pts = [(float(a), float(b)) for a, b in points if not (np.isnan(a) or np.isnan(b))]
    if not pts:
        raise ValueError("No coordinates to build a route from.")
    if len(pts) == 1:
        return google_maps_pin_link(*pts[0])
    pts = pts[: MAX_WAYPOINTS + 2]
    fmt = lambda p: f"{p[0]:.6f},{p[1]:.6f}"  # noqa: E731
    url = (f"https://www.google.com/maps/dir/?api=1&origin={fmt(pts[0])}&destination={fmt(pts[-1])}"
           f"&travelmode={mode}")
    if len(pts) > 2:
        url += "&waypoints=" + quote("|".join(fmt(p) for p in pts[1:-1]), safe=",")
    return url


# ------------------------------------------------------------------ complaint letter
def complaint_letter(
    ranked: pd.DataFrame,
    *,
    authority: str = "The Municipal Commissioner",
    area: str = "the surveyed route",
    sender: str = "[Your name]",
    contact: str = "[Your contact details]",
    data_source: str = "smartphone sensor rides analysed with RoadPulse",
    n_rides: int | None = None,
    top_n: int = 10,
    today: date | None = None,
) -> str:
    """Plain-text complaint letter listing the highest-priority spots with map links."""
    day = (today or date.today()).strftime("%d %B %Y")
    top = ranked.head(top_n)
    lines = [
        f"Date: {day}", "", f"To,\n{authority}", "",
        f"Subject: Request to inspect and repair {len(top)} road defect(s) near {area}", "",
        f"Respectfully,\n\nI am writing to report road defects detected from {data_source}"
        + (f" ({n_rides} ride(s))." if n_rides else "."),
        "The locations below are ranked by repair priority (severity, number of confirming rides and "
        "slow-speed exposure).", "",
    ]
    for _, r in top.iterrows():
        lines.append(f"{int(r['rank'])}. {r['class_name']} (priority {r['priority']:.0f}/100) - {r['why']}")
        lines.append(f"   Location: {r['lat']:.6f}, {r['lon']:.6f}")
        lines.append(f"   Map: {google_maps_pin_link(r['lat'], r['lon'])}")
    lines += [
        "", "These detections come from an automated model and may include errors. Please verify each "
        "location on site before work is scheduled.",
    ]
    if "simulated" in data_source.lower() or "sample" in data_source.lower():
        lines += ["", "NOTE: this letter was generated from SIMULATED demo data and does not describe real roads."]
    lines += ["", "Thank you for your attention.", "", f"Sincerely,\n{sender}\n{contact}"]
    return "\n".join(lines)
