"""Streamlit rendering helpers shared by the pages (theme, tiles, panels). No analysis logic lives here."""
from __future__ import annotations

import html
from typing import Any, Sequence

import pandas as pd
import streamlit as st

from core import app_state as S
from core import config as C
from core import io, maps


# ------------------------------------------------------------------ theme and chrome
def inject_css() -> None:
    """Inject assets/style.css (fonts, cards, tiles, tabs, buttons)."""
    css_path = C.ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def header(ds: S.Dataset | None) -> None:
    """Slim sticky header with the RoadPulse wordmark."""
    status = (f"{len(ds.rides)} ride(s) loaded, {ds.source} data" if ds else "No rides loaded")
    st.markdown(
        f'<div class="rp-header"><span class="rp-glyph"></span>'
        f'<span class="rp-wordmark">Road<b>Pulse</b></span>'
        f'<span class="rp-tag">Road quality from phone sensors</span>'
        f'<span class="rp-status">{html.escape(status)}</span></div>',
        unsafe_allow_html=True)


def simulated_banner(model_card_page: Any | None = None) -> None:
    """Persistent, tasteful reminder that the training data is simulated."""
    left, right = st.columns([9, 2], vertical_alignment="center")
    left.markdown(f'<div class="rp-banner">{html.escape(C.SIMULATED_NOTICE)}</div>', unsafe_allow_html=True)
    if model_card_page is not None:
        right.page_link(model_card_page, label="Read the model card", icon=":material/description:")


def sidebar_status() -> None:
    """Loaded data, current operating mode and a clear button."""
    ds = S.get_dataset()
    st.markdown("**Session**")
    if ds is None:
        st.caption("No rides loaded yet.")
    else:
        st.caption(f"{len(ds.rides)} ride(s) from {ds.source} data")
    st.caption(f"Operating mode: {S.get_mode()}")
    if ds is not None and st.button("Clear loaded rides", key="sb_clear"):
        S.set_dataset(None)
        st.rerun()


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="rp-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>'
        f'<div class="rp-centerline"></div></div>', unsafe_allow_html=True)


def section(title: str, caption: str | None = None) -> None:
    cap = f'<div class="rp-caption">{html.escape(caption)}</div>' if caption else ""
    st.markdown(f'<h3 class="rp-section">{html.escape(title)}</h3>{cap}', unsafe_allow_html=True)


def tiles(items: Sequence[dict[str, str]]) -> None:
    """Metric tiles. Each item: label, value, optional sub and tone (amber, red, teal)."""
    cells = "".join(
        f'<div class="rp-tile {html.escape(i.get("tone", ""))}"><div class="l">{html.escape(i["label"])}</div>'
        f'<div class="v">{html.escape(str(i["value"]))}</div>'
        + (f'<div class="s">{html.escape(i["sub"])}</div>' if i.get("sub") else "") + "</div>" for i in items)
    st.markdown(f'<div class="rp-tiles">{cells}</div>', unsafe_allow_html=True)


def chips(items: Sequence[tuple[str, str]]) -> None:
    """Small status pills: (text, tone) with tone in '', 'amber', 'red', 'teal'."""
    st.markdown("".join(f'<span class="rp-chip {t}">{html.escape(x)}</span>' for x, t in items), unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="rp-empty"><b>{html.escape(title)}</b><br>{html.escape(body)}</div>', unsafe_allow_html=True)


def bullets(lines: Sequence[str]) -> None:
    st.markdown('<ul class="rp-list">' + "".join(f"<li>{html.escape(x)}</li>" for x in lines) + "</ul>", unsafe_allow_html=True)


def plot(fig: Any, key: str | None = None) -> None:
    """Show a Plotly figure with the RoadPulse template intact (Streamlit's own theme override disabled)."""
    st.plotly_chart(fig, width="stretch", theme=None, key=key)


def show_map(m: Any, height: int = 520) -> None:
    """Render a folium map in an iframe (st.iframe treats a string starting with '<' as raw HTML)."""
    st.iframe(maps.map_to_html(m), width="stretch", height=height)


def show_html(html_doc: str, height: int = 600) -> None:
    """Render a self-contained HTML document in an iframe."""
    st.iframe(html_doc, width="stretch", height=height)


# ------------------------------------------------------------------ formatting helpers
def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d} min"


def tone_for_trust(label: str) -> str:
    return {"High": "teal", "Medium": "amber", "Low": "red"}[label]


def training_note(ds: S.Dataset, ride_id: str) -> None:
    """Warn when a sample ride comes from a route the model was trained on."""
    if ds.source != "sample" or "route_id" not in ds.df:
        return
    route = ds.df.loc[ds.df["ride_id"] == ride_id, "route_id"].iloc[0]
    if route != "D":
        st.caption(f"Ride {ride_id} is on training route {route}, so the model has seen this road. Scores here are optimistic. "
                   "Route D rides are the held-out test.")


# ------------------------------------------------------------------ tables
def events_table(events: pd.DataFrame, key: str | None = None) -> None:
    """Event table with progress bars for confidence and severity."""
    if events.empty:
        st.info("No events were flagged in this ride at the current operating mode.")
        return
    view = events[["event_id", "class_name", "peak_time", "confidence", "severity", "severity_label", "speed_kmh", "lat", "lon"]]
    st.dataframe(
        view, width="stretch", hide_index=True, key=key,
        column_config={
            "event_id": "Event", "class_name": "Type",
            "peak_time": st.column_config.NumberColumn("Time (s)", format="%.1f"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="percent"),
            "severity": st.column_config.ProgressColumn("Severity", min_value=0, max_value=100, format="%d"),
            "severity_label": "Level",
            "speed_kmh": st.column_config.NumberColumn("Speed (km/h)", format="%.0f"),
            "lat": st.column_config.NumberColumn("Latitude", format="%.5f"),
            "lon": st.column_config.NumberColumn("Longitude", format="%.5f"),
        })


# ------------------------------------------------------------------ validation report
def render_report(res: S.FileResult) -> None:
    """Friendly validation feedback for one file."""
    if res.read_error:
        st.error(res.read_error)
        return
    rep = res.report
    for e in rep.errors:
        st.error(e)
    for w in rep.warnings:
        st.warning(w)
    if res.fixes_applied:
        st.success("Applied unit fixes: " + ", ".join(io.UNIT_FIXES[f] for f in res.fixes_applied))
    if res.resampled:
        st.success("Resampled to 20 Hz.")
    if rep.ok:
        i = rep.info
        chips([(f"{i['rows']:,} rows", ""), (f"{i['n_rides']} ride(s)", ""), (fmt_duration(i["duration_s"]), ""),
               (f"{i['sampling_hz']:.0f} Hz", "teal"), ("labels found" if i["has_labels"] else "no labels", "")])


# ------------------------------------------------------------------ data sources
def _first_held_out(cat: list[dict[str, str]]) -> str | None:
    held = [c["label"] for c in cat if c["route"] == "D"]
    return held[0] if held else (cat[0]["label"] if cat else None)


def sample_panel(key: str) -> None:
    """Choose bundled sample rides."""
    cat = S.sample_catalog()
    if not cat:
        st.info("No sample rides found in data/sample_rides. Run tools/make_assets.py or upload your own files.")
        return
    labels = [c["label"] for c in cat]
    by_label = {c["label"]: c["name"] for c in cat}
    everything = st.checkbox("Select all sample rides", key=f"{key}_all")
    default = labels if everything else [_first_held_out(cat)]
    chosen = st.multiselect("Sample rides", labels, default=default, key=f"{key}_ms_{int(everything)}")
    st.caption("Sample rides are simulated. Rides on route D were never seen during training.")
    if st.button("Load sample rides", type="primary", key=f"{key}_load", disabled=not chosen):
        S.set_dataset(S.load_sample_dataset([by_label[x] for x in chosen]))
        st.rerun()


def upload_panel(key: str) -> None:
    """Upload CSV/Excel files, show validation, offer repairs, and load the valid ones."""
    files = st.file_uploader("Ride files (CSV or Excel)", type=["csv", "xlsx"], accept_multiple_files=True, key=f"{key}_files")
    st.download_button("Download a CSV template", io.make_template_csv(), "roadpulse_template.csv", "text/csv", key=f"{key}_tpl")
    if not files:
        st.caption("Expected columns: time_s, lat, lon, speed_kmh, acc_x/y/z (m/s² including gravity), gyro_x/y/z (rad/s). "
                   "Optional: ride_id, route_id, label.")
        return
    results: list[S.FileResult] = []
    for f in files:
        data = f.getvalue()
        with st.expander(f.name, expanded=True):
            fixes = st.checkbox("Apply suggested unit fixes", value=True, key=f"{key}_fix_{f.name}")
            resample = st.checkbox("Resample to 20 Hz if needed", value=True, key=f"{key}_rs_{f.name}")
            res = S.process_upload(data, f.name, apply_fixes=fixes, resample=resample)
            if res.needs_mapping:
                st.info("Some required columns were not found. Match them to your columns below.")
                mapping = io.render_column_mapping_ui(res.raw.columns, key_prefix=f"{key}_map_{f.name}")
                res = S.process_upload(data, f.name, mapping=mapping, apply_fixes=fixes, resample=resample)
            render_report(res)
            results.append(res)
    n_ok = sum(r.ok for r in results)
    if st.button(f"Analyze {n_ok} valid file(s)", type="primary", key=f"{key}_go", disabled=n_ok == 0):
        S.set_dataset(S.combine_results(results, "upload"))
        st.rerun()


def demo_panel(key: str) -> None:
    """Create clearly-labelled simulated repeat passes so consensus can be demonstrated."""
    cat = S.sample_catalog()
    if not cat:
        st.info("No sample rides found.")
        return
    labels = [c["label"] for c in cat]
    by_label = {c["label"]: c["name"] for c in cat}
    base = st.selectbox("Base ride", labels, index=labels.index(_first_held_out(cat)), key=f"{key}_demo_base")
    n = st.slider("Number of passes", 3, 6, 4, key=f"{key}_demo_n")
    st.warning("Demo only: the passes are copies of one ride with fresh noise and GPS jitter, so they are not independent evidence. "
               "Real consensus needs real repeat rides over the same road.")
    if st.button("Create demo passes", type="primary", key=f"{key}_demo_go"):
        S.set_dataset(S.make_demo_dataset(by_label[base], n))
        st.rerun()


def source_tabs(key: str, allow_demo: bool = False) -> None:
    """Tabs for choosing where rides come from."""
    names = ["Sample rides", "Upload files"] + (["Demo repeat passes"] if allow_demo else [])
    tabs = st.tabs(names)
    with tabs[0]:
        sample_panel(key)
    with tabs[1]:
        upload_panel(key)
    if allow_demo:
        with tabs[2]:
            demo_panel(key)


def require_dataset() -> S.Dataset:
    """Return the loaded dataset, or show a friendly prompt and stop the page."""
    ds = S.get_dataset()
    if ds is not None:
        return ds
    empty_state("No rides loaded yet", "Load a sample ride or upload your own file to see results on this page.")
    cat = S.sample_catalog()
    if cat:
        held = next((c for c in cat if c["route"] == "D"), cat[0])
        if st.button(f"Load sample ride {held['name']}", type="primary", key="req_load"):
            S.set_dataset(S.load_sample_dataset([held["name"]]))
            st.rerun()
    st.caption("You can also use Upload & Analyze to bring your own files.")
    st.stop()


def dataset_strip(ds: S.Dataset) -> None:
    """Chips describing what is loaded and the current operating mode."""
    chips([(f"{len(ds.rides)} ride(s)", ""), (f"{ds.source} data", "amber" if ds.source != "upload" else "teal"),
           (f"Mode: {S.get_mode()}", "")])
    if ds.warnings:
        with st.expander(f"Input checks ({len(ds.warnings)})"):
            bullets(ds.warnings)
