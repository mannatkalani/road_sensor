"""Page 6: a ranked repair list with exports and a complaint-letter draft."""
from __future__ import annotations

import streamlit as st

from core import app_state as S
from core import charts, consensus, export, maps, scoring, ui
from core import config as C

ui.hero("Priority queue", "Spots ranked for repair by severity, how many rides confirmed them, and how slowly traffic passes them.")
ds = ui.require_dataset()
n = len(ds.rides)
a = S.analyze(ds.df)
if a.events.empty:
    st.info("No events were flagged at the current operating mode, so there is nothing to rank.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
radius = c1.slider("Match radius (m)", 5, 50, 15, 5, key="p6_radius")
min_conf = c2.slider("Rides needed", 1, max(n, 1), 1, key="p6_min") if n > 1 else 1
top_n = c3.slider("Spots to show", 3, 25, 10, key="p6_top")
potholes_only = c4.checkbox("Potholes only", value=True, key="p6_pot")

_, spots = consensus.cluster_events(a.events, n_rides_total=n, radius_m=float(radius))
ranked = scoring.rank_priority(spots, n, classes=[C.POTHOLE] if potholes_only else None, min_confirmations=int(min_conf))
if ranked.empty:
    st.info("No spots match these filters. Lower the number of rides needed or include speed breakers.")
    st.stop()

top = ranked.head(top_n)
ui.tiles([{"label": "Spots ranked", "value": len(ranked)},
          {"label": "Highest priority", "value": f"{ranked['priority'].iloc[0]:.0f} / 100", "tone": "red", "sub": ranked["why"].iloc[0]},
          {"label": "Rides analysed", "value": n}])
if ds.source != "upload":
    st.caption("This ranking uses simulated data and is for demonstration only.")

ui.show_map(maps.priority_map(top, top_n))
st.dataframe(top[["rank", "spot_id", "class_name", "priority", "severity", "confirmations", "speed_kmh", "why"]], width="stretch", hide_index=True,
             column_config={"rank": "Rank", "spot_id": "Spot", "class_name": "Type",
                            "priority": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%d"),
                            "severity": st.column_config.NumberColumn("Severity", format="%.0f"),
                            "confirmations": "Rides", "speed_kmh": st.column_config.NumberColumn("Speed (km/h)", format="%.0f"), "why": "Why"})
with st.expander("Priority chart"):
    ui.plot(charts.priority_fig(top, top_n), key="p6_chart")

ui.section("Export")
d1, d2, link_col = st.columns(3)
d1.download_button("Download ranking (CSV)", export.to_csv_bytes(top), "roadpulse_priority.csv", "text/csv", key="p6_csv")
d2.download_button("Download ranking (GeoJSON)", export.spots_to_geojson(top), "roadpulse_priority.geojson", "application/geo+json", key="p6_geo")
pts = export.nearest_neighbour_order([(r.lat, r.lon) for r in top.head(export.MAX_WAYPOINTS + 2).itertuples()])
link_col.link_button("Open route in Google Maps", export.google_maps_route_link(pts), key="p6_gmaps")
st.caption("The link opens Google Maps with your top spots as stops. RoadPulse does not connect to any Google service.")

ui.section("Complaint letter", "Edit the details, then copy or download the letter. Detections come from an automated model, so the letter asks for on-site verification.")
f1, f2 = st.columns(2)
authority = f1.text_input("Addressed to", "The Municipal Commissioner", key="p6_auth")
area = f2.text_input("Area or route", "the surveyed route", key="p6_area")
sender = f1.text_input("Your name", "[Your name]", key="p6_sender")
contact = f2.text_input("Your contact details", "[Your contact details]", key="p6_contact")
letter = export.complaint_letter(top, authority=authority, area=area, sender=sender, contact=contact,
                                 data_source=ds.data_source_text, n_rides=n, top_n=top_n)
edited = st.text_area("Letter draft", letter, height=380, key=f"p6_letter_{hash(letter) & 0xffff}")
st.download_button("Download letter (TXT)", edited.encode("utf-8"), "roadpulse_complaint_letter.txt", "text/plain", key="p6_letter_dl")
