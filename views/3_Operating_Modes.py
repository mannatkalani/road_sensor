"""Page 3: choose how cautious the detector is and see the trade-off live."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import app_state as S
from core import charts, ui
from core import config as C

ui.hero("Operating modes", "Pick how cautious the detector should be. The numbers below update for the rides you loaded.")
ds = ui.require_dataset()

modes = [*C.OPERATING_MODES, S.CUSTOM_MODE]
current = S.get_mode()
choice = st.radio("Operating mode", modes, index=modes.index(current), horizontal=True)
if choice != current:
    if choice == S.CUSTOM_MODE:
        th = S.get_thresholds()
        S.set_custom_thresholds(th[C.SPEED_BREAKER], th[C.POTHOLE])
    else:
        S.set_mode(choice)
    st.rerun()

if choice == S.CUSTOM_MODE:
    th = S.get_thresholds()
    c1, c2 = st.columns(2)
    sb = c1.slider("Speed breaker threshold", 0.05, 0.95, float(th[C.SPEED_BREAKER]), 0.05, key="p3_sb")
    ph = c2.slider("Pothole threshold", 0.05, 0.95, float(th[C.POTHOLE]), 0.05, key="p3_ph")
    if (sb, ph) != (th[C.SPEED_BREAKER], th[C.POTHOLE]):
        S.set_custom_thresholds(sb, ph)
        st.rerun()
else:
    cfg = C.OPERATING_MODES[choice]
    st.caption(f"{cfg['tagline']}. {cfg['description']}")

a = S.analyze(ds.df)
counts = {n: int((a.events["class_name"] == n).sum()) for n in C.CLASS_NAMES[1:]}
tiles = [{"label": "Events detected", "value": len(a.events), "sub": f"{counts['Speed breaker']} speed breakers, {counts['Pothole']} potholes",
          "tone": "amber"}]
if a.has_labels:
    s = a.metrics.summary
    tiles += [{"label": "False alarms per km", "value": f"{s['false_alarms_per_km']:.2f}", "sub": f"Measured: {s['fp']} false alarms over {s['distance_km']:.1f} km", "tone": "red"},
              {"label": "Recall", "value": f"{s['recall']:.0%}", "sub": f"{s['tp']} of {s['tp'] + s['fn']} labelled events found", "tone": "teal"},
              {"label": "Precision", "value": f"{s['precision']:.0%}", "sub": f"{s['tp']} of {s['tp'] + s['fp']} flags were real"}]
else:
    tiles += [{"label": "False alarms per km (estimated)", "value": f"{a.est_false_alarms / max(a.km, 1e-9):.2f}",
               "sub": "From the model's own confidence", "tone": "red"}]
ui.tiles(tiles)
if not a.has_labels:
    st.info("These rides have no labels, so false alarms are estimated from the model's own confidence and recall and precision are unavailable.")

if a.has_labels:
    ui.section("Per class")
    pc = a.metrics.per_class[["class_name", "tp", "fp", "fn", "precision", "recall", "f1"]]
    st.dataframe(pc, width="stretch", hide_index=True, column_config={
        "class_name": "Type", "tp": "Found", "fp": "False alarms", "fn": "Missed",
        "precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="percent"),
        "recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="percent"),
        "f1": st.column_config.NumberColumn("F1", format="%.2f")})

ui.section("How each mode behaves on these rides")
cmp = S.mode_comparison(a)
st.dataframe(cmp, width="stretch", hide_index=True, column_config={
    "False alarms per km": st.column_config.NumberColumn(format="%.2f"),
    "Recall": st.column_config.NumberColumn(format="percent"), "Precision": st.column_config.NumberColumn(format="percent")})

ui.section("The trade-off",
           "A higher threshold makes the detector wait until it is very sure, so it raises fewer false alarms but misses more real events. "
           "A lower threshold catches more real events at the price of more false alarms to review.")
ui.plot(charts.threshold_sweep_fig(S.sweep_for(ds.df)), key="p3_sweep")
if ds.source == "sample":
    st.caption("Sample rides R01, R05 and R10 are on training routes, so the model has seen those roads. Load a route D ride for an honest test.")
