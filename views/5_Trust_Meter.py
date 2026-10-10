"""Page 5: how closely does a ride resemble the data the model was trained on?"""
from __future__ import annotations

import streamlit as st

from core import app_state as S
from core import charts, ui

ui.hero("Trust meter", "Checks whether a ride looks like the data the model learned from, and says what to expect if it does not.")
ds = ui.require_dataset()
all_a = S.analyze(ds.df)

options = ["All rides", *ds.rides]
choice = st.selectbox("Ride", options, key="p5_ride") if len(ds.rides) > 1 else ds.rides[0]
ride_id = None if choice == "All rides" else choice
rep = S.trust_for(all_a, ride_id, ds.warnings)
if ride_id:
    ui.training_note(ds, ride_id)

left, right = st.columns([2, 3])
with left:
    ui.plot(charts.gauge_fig(rep.score, rep.label), key="p5_gauge")
    ui.tiles([{"label": "Roughness vs training smooth road", "value": f"{rep.roughness_percentile:.0f}th percentile",
               "sub": f"Based on {rep.n_windows} two-second windows"}])
with right:
    ui.section("What this means")
    ui.bullets(rep.reasons)
    ui.section("Recommended operating mode")
    st.write(f"**{rep.recommended_mode}**. " + {
        "Balanced": "The ride looks familiar, so the default trade-off is reasonable.",
        "Driver Alert": "Expect extra false alarms, so only alert when the model is very sure.",
        "Municipal Audit": "Subtle events may be missed, so flag anything plausible and review it."}[rep.recommended_mode])
    if S.get_mode() == rep.recommended_mode:
        st.caption(f"You are already using {rep.recommended_mode}.")
    elif st.button(f"Switch to {rep.recommended_mode}", type="primary", key="p5_apply"):
        S.set_mode(rep.recommended_mode)
        st.rerun()

ui.section("Which signals differ from the training data", "Bars show how far this ride's typical value sits from the training average, in standard deviations.")
ui.plot(charts.drift_fig(rep.per_feature), key="p5_drift")
with st.expander("Feature table"):
    st.dataframe(rep.per_feature[["label", "ride_median", "ref_mean", "z", "out_of_range_pct"]], width="stretch", hide_index=True,
                 column_config={"label": "Signal", "ride_median": st.column_config.NumberColumn("This ride (median)", format="%.2f"),
                                "ref_mean": st.column_config.NumberColumn("Training average", format="%.2f"),
                                "z": st.column_config.NumberColumn("Std from average", format="%.2f"),
                                "out_of_range_pct": st.column_config.NumberColumn("Windows outside training range", format="%.0f%%")})
st.caption("The reference statistics come from the simulated training routes, so a High score means the ride resembles simulated data, not that the model is accurate on real roads.")
