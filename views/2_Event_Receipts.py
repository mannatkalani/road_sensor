"""Page 2: a receipt for any detected event showing why the model flagged it."""
from __future__ import annotations

import streamlit as st

from core import app_state as S
from core import charts, explain, ui

ui.hero("Event receipts", "Pick any detected event to see the waveform, how it compares with a typical event of its type, "
                          "and the signals that drove the prediction.")
ds = ui.require_dataset()
all_a = S.analyze(ds.df)
rides = [r for r in ds.rides if (all_a.events["ride_id"] == r).any()]
if not rides:
    st.info("No events were flagged at the current operating mode. Try Municipal Audit on the Operating Modes page.")
    st.stop()

ride = st.selectbox("Ride", rides, key="p2_ride") if len(rides) > 1 else rides[0]
a = all_a.for_ride(ride)
ev = a.events.reset_index(drop=True)
idx = st.selectbox(
    "Event", list(ev.index), key=f"p2_event_{ride}",
    format_func=lambda i: f"{ev.loc[i, 'event_id']}: {ev.loc[i, 'class_name']} at {ev.loc[i, 'peak_time']:.1f} s, "
                          f"{ev.loc[i, 'severity_label'].lower()} severity")
event = ev.loc[idx]
expl = S.explain_event(a, event)

ui.tiles([
    {"label": "Type", "value": event["class_name"], "tone": "red" if event["class_id"] == 2 else "amber"},
    {"label": "Model confidence", "value": f"{event['confidence']:.0%}", "sub": f"Peak probability {event['max_prob']:.0%}"},
    {"label": "Severity", "value": f"{event['severity']:.0f} / 100", "sub": f"{event['severity_label']} level"},
    {"label": "Speed at the spot", "value": f"{event['speed_kmh']:.0f} km/h", "sub": f"Time {event['peak_time']:.1f} s into the ride"},
])

left, right = st.columns([3, 2])
with left:
    ui.section("Receipt")
    ui.show_html(explain.receipt_html(expl), height=720)
with right:
    ui.section("Compared with a typical event", "Dashed line is the average signature of this event type on the training routes.")
    ui.plot(charts.signature_fig(expl), key="p2_sig")
    ui.section("Why the model flagged it", f"Method: {expl.method}")
    ui.bullets(list(expl.features["text"]))

d1, d2 = st.columns(2)
d1.download_button("Download receipt (PNG)", explain.receipt_png(expl), f"{event['event_id']}_receipt.png", "image/png", key="p2_png")
d2.download_button("Download receipt (HTML)", explain.receipt_html(expl), f"{event['event_id']}_receipt.html", "text/html", key="p2_html")
