"""Page 4: confirm events across several rides so repeated detections reinforce each other."""
from __future__ import annotations

import streamlit as st

from core import app_state as S
from core import consensus, maps, ui

ui.hero("Consensus map", "When several rides flag the same spot, confidence goes up. Load multiple rides over the same roads to see it.")
ui.source_tabs("p4", allow_demo=True)

ds = S.get_dataset()
if ds is None:
    ui.empty_state("No rides loaded yet", "Upload several rides, load all sample rides, or create demo passes above.")
    st.stop()

ui.dataset_strip(ds)
if ds.source == "demo":
    st.warning("These are simulated repeat passes of a single ride, created to demonstrate consensus. They are not independent evidence.")

n = len(ds.rides)
a = S.analyze(ds.df)
c1, c2 = st.columns(2)
radius = c1.slider("Match radius (metres)", 5, 50, 15, 5, key="p4_radius")
min_conf = c2.slider("Rides needed to confirm", 1, max(n, 2), min(2, max(n, 2)), key="p4_min")
_, spots = consensus.cluster_events(a.events, n_rides_total=n, radius_m=float(radius), min_confirmations=int(min_conf))

confirmed = int(spots["confirmed"].sum()) if len(spots) else 0
ui.tiles([
    {"label": "Rides", "value": n}, {"label": "Events detected", "value": len(a.events)},
    {"label": "Distinct spots", "value": len(spots)},
    {"label": "Confirmed spots", "value": confirmed, "sub": f"Seen by at least {min_conf} ride(s)", "tone": "teal" if confirmed else ""},
])
if spots.empty:
    st.info("No events with GPS positions were found, so there is nothing to cluster.")
    st.stop()
if n > 1 and confirmed == 0:
    st.info("No spot was seen by enough rides. Different sample rides cover different roads, so they rarely overlap. "
            "Upload repeat rides over the same route, or use Demo repeat passes to see how confirmation works.")

only_confirmed = st.checkbox("Show confirmed spots only", value=False, key="p4_only")
shown = spots[spots["confirmed"]] if only_confirmed else spots
if shown.empty:
    st.info("Nothing to show with the current filter.")
else:
    ui.show_map(maps.consensus_map(shown))
    st.dataframe(shown[["spot_id", "class_name", "n_rides", "confidence", "severity_mean", "speed_kmh", "rides", "lat", "lon"]],
                 width="stretch", hide_index=True, column_config={
        "spot_id": "Spot", "class_name": "Type", "n_rides": "Rides",
        "confidence": st.column_config.ProgressColumn("Consensus", min_value=0.0, max_value=1.0, format="percent"),
        "severity_mean": st.column_config.NumberColumn("Severity", format="%.0f"),
        "speed_kmh": st.column_config.NumberColumn("Speed (km/h)", format="%.0f"), "rides": "Seen in",
        "lat": st.column_config.NumberColumn("Latitude", format="%.5f"), "lon": st.column_config.NumberColumn("Longitude", format="%.5f")})
