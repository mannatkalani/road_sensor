"""Page 1: load rides, validate them, detect events, and show them on a map and waveform."""
from __future__ import annotations

import streamlit as st

from core import app_state as S
from core import charts, export, maps, ui
from core import config as C

ui.hero("Upload and analyze rides",
        "Load phone sensor files, flag speed breakers and potholes, and see exactly where they are.")
ui.source_tabs("p1")

ds = S.get_dataset()
if ds is None:
    ui.empty_state("No rides loaded yet", "Pick a sample ride or upload a file above to start.")
    st.stop()

ui.dataset_strip(ds)
rides = ds.rides
ride = st.selectbox("Ride to inspect", rides, key="p1_ride") if len(rides) > 1 else rides[0]
all_a = S.analyze(ds.df)
a = all_a.for_ride(ride)
trust = S.trust_for(all_a, ride, ds.warnings)
ui.training_note(ds, ride)

counts = {n: int((a.events["class_name"] == n).sum()) for n in C.CLASS_NAMES[1:]}
ui.tiles([
    {"label": "Distance", "value": f"{a.km:.2f} km", "sub": "Integrated from the speed signal"},
    {"label": "Duration", "value": ui.fmt_duration(a.duration_s)},
    {"label": "Events found", "value": len(a.events),
     "sub": f"{counts['Speed breaker']} speed breakers, {counts['Pothole']} potholes", "tone": "amber"},
    {"label": "Trust level", "value": trust.label, "sub": f"Score {trust.score:.0f} of 100", "tone": ui.tone_for_trust(trust.label)},
])

tab_map, tab_signal, tab_events = st.tabs(["Map", "Signal", "Events"])
with tab_map:
    if a.pre["lat"].notna().any():
        ui.show_map(maps.ride_map(a.pre, a.events))
    else:
        st.info("This ride has no GPS fixes, so there is no map. The Signal and Events tabs still work.")
with tab_signal:
    ui.plot(charts.waveform_fig(a.pre, a.events), key="p1_wave")
    with st.expander("Model probability over time"):
        ui.plot(charts.probability_fig(a.pred_windows), key="p1_prob")
with tab_events:
    ui.events_table(a.events, key="p1_events")
    if not a.events.empty:
        c1, c2 = st.columns(2)
        c1.download_button("Download events (CSV)", export.to_csv_bytes(a.events), f"{ride}_events.csv", "text/csv", key="p1_csv")
        c2.download_button("Download events (GeoJSON)", export.events_to_geojson(a.events), f"{ride}_events.geojson",
                           "application/geo+json", key="p1_geojson")
