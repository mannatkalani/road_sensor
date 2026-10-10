"""Folium map builders using dark OpenStreetMap-based CARTO tiles (no API keys)."""
from __future__ import annotations

import html
from typing import Iterable

import folium
import numpy as np
import pandas as pd

from core import config as C

TILE_URL = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
TILE_ATTR = "&copy; OpenStreetMap contributors &copy; CARTO"
DEFAULT_CENTER = (18.5204, 73.8567)


def _base_map(center: tuple[float, float], zoom: int = 14) -> folium.Map:
    m = folium.Map(location=list(center), zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(TILE_URL, attr=TILE_ATTR, name="CartoDB dark_matter", subdomains="abcd", max_zoom=20).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    return m


def _center(df: pd.DataFrame) -> tuple[float, float]:
    ok = df.dropna(subset=["lat", "lon"]) if len(df) else df
    return (float(ok["lat"].mean()), float(ok["lon"].mean())) if len(ok) else DEFAULT_CENTER


def _fit(m: folium.Map, df: pd.DataFrame) -> None:
    ok = df.dropna(subset=["lat", "lon"])
    if len(ok) >= 2:
        m.fit_bounds([[ok["lat"].min(), ok["lon"].min()], [ok["lat"].max(), ok["lon"].max()]], padding=(30, 30))


def _legend(items: Iterable[tuple[str, str]]) -> folium.Element:
    rows = "".join(
        f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};'
        f'margin-right:6px"></span>{html.escape(t)}</div>' for t, c in items)
    return folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:{C.COLORS["panel"]};'
        f'color:{C.COLORS["text"]};border:1px solid {C.COLORS["border"]};border-radius:8px;padding:8px 12px;'
        f'font:12px Inter,Arial,sans-serif">{rows}</div>')


def event_popup(r: pd.Series) -> str:
    """HTML popup for one event row."""
    return (f"<b>{html.escape(str(r['class_name']))}</b><br>severity {r['severity']:.0f}/100 "
            f"({html.escape(str(r['severity_label']))})<br>confidence {r['confidence']:.0%}<br>"
            f"speed {r['speed_kmh']:.0f} km/h<br>ride {html.escape(str(r['ride_id']))} at {r['peak_time']:.1f} s")


def map_to_html(m: folium.Map) -> str:
    """Full HTML document for embedding with ``st.iframe``."""
    return m.get_root().render()


def ride_map(ride_df: pd.DataFrame, events: pd.DataFrame | None = None, max_points: int = 3000) -> folium.Map:
    """Route polyline plus event markers coloured by class (size grows with severity)."""
    track = ride_df.dropna(subset=["lat", "lon"])
    m = _base_map(_center(track))
    for _, g in track.groupby("ride_id", sort=False):
        g = g.sort_values("time_s")
        g = g.iloc[:: max(1, len(g) // max_points)]
        folium.PolyLine(g[["lat", "lon"]].to_numpy().tolist(), color=C.COLORS["smooth"], weight=3, opacity=0.7).add_to(m)
    if events is not None and len(events):
        for _, r in events.dropna(subset=["lat", "lon"]).iterrows():
            color = C.CLASS_COLORS[int(r["class_id"])]
            folium.CircleMarker([r["lat"], r["lon"]], radius=5 + r["severity"] / 20, color=color, weight=2,
                                fill=True, fill_color=color, fill_opacity=0.55,
                                popup=folium.Popup(event_popup(r), max_width=240)).add_to(m)
    _fit(m, track)
    m.get_root().add_child(_legend([("Route", C.COLORS["smooth"]), ("Speed breaker", C.COLORS["speed_breaker"]),
                                    ("Pothole", C.COLORS["pothole"])]))
    return m


def confidence_color(confidence: float, confirmed: bool) -> str:
    """Muted for unconfirmed spots, amber for partly confirmed, red for strongly confirmed."""
    if not confirmed:
        return C.COLORS["muted"]
    return C.COLORS["pothole"] if confidence >= 0.67 else C.COLORS["speed_breaker"]


def consensus_map(spots: pd.DataFrame) -> folium.Map:
    """Spots coloured by cross-ride confidence; radius grows with the number of confirming rides."""
    m = _base_map(_center(spots))
    for _, r in spots.dropna(subset=["lat", "lon"]).iterrows():
        color = confidence_color(float(r["confidence"]), bool(r["confirmed"]))
        pop = (f"<b>{html.escape(str(r['class_name']))}</b> {html.escape(str(r['spot_id']))}<br>"
               f"{int(r['n_rides'])} ride(s): {html.escape(str(r['rides']))}<br>"
               f"confidence {r['confidence']:.0%}<br>severity {r['severity_mean']:.0f}/100")
        folium.CircleMarker([r["lat"], r["lon"]], radius=5 + 3 * int(r["n_rides"]), color=color, weight=2,
                            fill=True, fill_color=color, fill_opacity=0.5,
                            popup=folium.Popup(pop, max_width=240)).add_to(m)
    _fit(m, spots)
    m.get_root().add_child(_legend([("Unconfirmed (1 ride)", C.COLORS["muted"]), ("Confirmed", C.COLORS["speed_breaker"]),
                                    ("Strongly confirmed (>=67% of rides)", C.COLORS["pothole"])]))
    return m


def priority_map(ranked: pd.DataFrame, top_n: int = 25) -> folium.Map:
    """Numbered markers for the top-ranked repair spots."""
    top = ranked.head(top_n).dropna(subset=["lat", "lon"])
    m = _base_map(_center(top))
    for _, r in top.iterrows():
        color = C.COLORS["pothole"] if r["priority"] >= 50 else C.COLORS["speed_breaker"] if r["priority"] >= 20 else C.COLORS["muted"]
        icon = folium.DivIcon(
            icon_size=(26, 26), icon_anchor=(13, 13),
            html=(f'<div style="width:26px;height:26px;border-radius:50%;background:{color};color:#0B0F14;'
                  f'font:700 12px JetBrains Mono,monospace;display:flex;align-items:center;justify-content:center;'
                  f'border:2px solid #0B0F14">{int(r["rank"])}</div>'))
        pop = (f"<b>#{int(r['rank'])} {html.escape(str(r['class_name']))}</b><br>priority {r['priority']:.0f}/100<br>"
               f"{html.escape(str(r['why']))}")
        folium.Marker([r["lat"], r["lon"]], icon=icon, popup=folium.Popup(pop, max_width=260)).add_to(m)
    _fit(m, top)
    return m
