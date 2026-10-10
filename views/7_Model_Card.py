"""Page 7: what the model is, how it scored, and where it should not be trusted."""
from __future__ import annotations

import streamlit as st

from core import app_state as S
from core import charts, ui

ui.hero("Model card", "What this model is, how it scored, and where it should not be trusted. Plain language, no spin.")
st.warning("The training data is simulated. Every score on this page shows how well the model recovers patterns from a simulator, "
           "not how well it finds potholes on real roads.")

try:
    mc = S.model_card_data()
except Exception as exc:  # noqa: BLE001
    st.error(f"Model assets could not be loaded: {exc}")
    st.stop()

bundle, windows = mc["bundle"], mc["windows"]
routes = ", ".join(mc["reference"]["meta"]["training_routes"])
ui.tiles([
    {"label": "Model", "value": bundle.model_name, "sub": "Chosen by macro-F1 on the held-out route"},
    {"label": "Inputs", "value": f"{len(bundle.feature_columns)} features", "sub": "Vibration, rotation, speed and frequency-band power"},
    {"label": "Window", "value": "2 s", "sub": f"Step 1 s at {bundle.sampling_hz} Hz"},
    {"label": "Trained on", "value": f"Routes {routes}", "sub": f"{mc['reference']['meta']['n_windows']:,} windows, simulated", "tone": "amber"},
])

if mc["comparison"] is not None:
    ui.section("How the models scored", "All scores are on route D, which was held out of training. Higher is better.")
    comp = mc["comparison"].set_index("Model")
    st.dataframe(comp, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in comp.columns})
    ui.plot(charts.model_comparison_fig(comp), key="p7_comp")
    st.caption("XGBoost and Random Forest are statistically tied on this small test set. Accuracy is not shown as a headline because "
               "about 95% of windows are smooth road, so a model that never flags anything already scores about 95%.")
if mc["cv"] is not None:
    ui.section("Does it hold up on unseen routes?", "Each bar trains on two training routes and tests on the third.")
    ui.plot(charts.cv_fig(mc["cv"]), key="p7_cv")
if mc["per_class"] is not None:
    ui.section("Per-class results")
    metric = st.radio("Metric", ["Recall", "Precision", "F1"], horizontal=True, key="p7_metric")
    ui.plot(charts.per_class_fig(mc["per_class"], metric), key="p7_pc")
    st.caption("Recall is the share of real events found. Precision is the share of flags that were real.")

test_counts = ""
if windows is not None and "route_id" in windows:
    t = windows[windows["route_id"] == "D"]["label"].value_counts()
    test_counts = f" The test route has only {int(t.get(1, 0))} speed-breaker windows and {int(t.get(2, 0))} pothole windows."

ui.section("What it can and cannot do")
ui.bullets([
    "It can flag windows of phone data that look like speed breakers or potholes in the simulator it learned from.",
    "It can merge those windows into events, place them on a map, and rank them for repair.",
    "It cannot promise real-world accuracy. It has never seen a real road.",
    "It cannot tell a pothole from a similar jolt, such as a rail crossing or a manhole cover, unless those were in training.",
])
ui.section("Limitations")
ui.bullets([
    "Simulated training data: real roads, phones and drivers will differ.",
    "Small test set." + test_counts + " Every score carries wide error bars.",
    "One phone position: the model assumes the phone lies flat with the z-axis pointing up.",
    "Sample rides R01, R05 and R10 are on training routes, so results there are optimistic. R11 and R12 are on the held-out route.",
    "The Trust Meter compares rides with statistics from the simulated routes.",
    "Demo repeat passes on the Consensus Map are copies of one ride with added noise and are not independent evidence.",
    "Thresholds were not tuned for any real deployment.",
])
ui.section("From simulation to real roads")
st.markdown(
    "1. Record real rides with a fixed phone mount, and note or film each pothole and speed breaker.\n"
    "2. Label the events, then add the rides as new routes and keep some routes out of training.\n"
    "3. Retrain, and check calibration, so that 80% confidence really means about 80% correct.\n"
    "4. Test across drivers, vehicles, phones and speeds, not only across routes.\n"
    "5. Pilot with a small municipal audit and compare the ranking with what inspectors find on site.")
