"""RoadPulse: detect speed breakers and potholes from smartphone sensor data.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging

import streamlit as st

st.set_page_config(page_title="RoadPulse", page_icon=":material/route:", layout="wide", initial_sidebar_state="expanded")

from core import app_state as S  # noqa: E402  (after set_page_config)
from core import ui  # noqa: E402
from core.inference import ModelLoadError  # noqa: E402

log = logging.getLogger("roadpulse")

pages = [
    st.Page("views/1_Upload_Analyze.py", title="Upload & Analyze", icon=":material/upload_file:", default=True),
    st.Page("views/2_Event_Receipts.py", title="Event Receipts", icon=":material/receipt_long:"),
    st.Page("views/3_Operating_Modes.py", title="Operating Modes", icon=":material/tune:"),
    st.Page("views/4_Consensus_Map.py", title="Consensus Map", icon=":material/hub:"),
    st.Page("views/5_Trust_Meter.py", title="Trust Meter", icon=":material/verified:"),
    st.Page("views/6_Priority_Queue.py", title="Priority Queue", icon=":material/format_list_numbered:"),
    st.Page("views/7_Model_Card.py", title="Model Card", icon=":material/description:"),
]
current = st.navigation(pages)

ui.inject_css()
ui.header(S.get_dataset())
ui.simulated_banner(pages[-1])
with st.sidebar:
    ui.sidebar_status()

try:
    current.run()
except ModelLoadError as exc:
    st.error(f"The model could not be loaded. {exc}")
except FileNotFoundError as exc:
    st.error(f"A required file is missing: {exc}")
except Exception:  # noqa: BLE001 - never show a raw traceback to users
    log.exception("Unhandled error while rendering %s", getattr(current, "title", "page"))
    st.error("Something went wrong while analyzing these rides. Check that the file has the expected columns and units, "
             "then try again. If it keeps happening, load a sample ride to confirm the app itself is working.")
