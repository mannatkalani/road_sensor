"""Smoke tests: every page renders headlessly (Streamlit AppTest) with and without data, plus key interactions.

Extra to the four required test files: these guard the UI layer against regressions.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from core import app_state as S
from core import consensus

MODEL_READY = (ROOT / "models" / "road_quality_model.joblib").exists() and (ROOT / "models" / "reference_stats.json").exists()
pytestmark = pytest.mark.skipif(not MODEL_READY, reason="model assets not built (run tools/make_assets.py)")

PAGES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "views").glob("*.py"))


def run_page(page: str, dataset=None) -> AppTest:
    at = AppTest.from_file(str(ROOT / page), default_timeout=180)
    if dataset is not None:
        at.session_state["rp_dataset"] = dataset
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_data(page):
    at = run_page(page, S.load_sample_dataset(["R11", "R12"]))
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_data(page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]


def test_loading_a_sample_ride_stores_the_dataset():
    at = run_page("views/1_Upload_Analyze.py")
    at.button(key="p1_load").click().run()
    assert not at.exception
    assert at.session_state["rp_dataset"] is not None
    assert any("Events found" in m.value for m in at.markdown)


def test_switching_operating_mode_updates_thresholds():
    at = run_page("views/3_Operating_Modes.py", S.load_sample_dataset(["R11"]))
    at.radio[0].set_value("Municipal Audit").run()
    assert not at.exception
    assert at.session_state["rp_mode"] == "Municipal Audit"
    assert at.session_state["rp_thresholds"] == {1: 0.25, 2: 0.25}


def test_custom_thresholds_are_accepted():
    at = run_page("views/3_Operating_Modes.py", S.load_sample_dataset(["R11"]))
    at.radio[0].set_value("Custom").run()
    at.slider(key="p3_ph").set_value(0.3).run()
    assert not at.exception and at.session_state["rp_mode"] == "Custom"
    assert at.session_state["rp_thresholds"][2] == pytest.approx(0.3)


def test_demo_repeat_passes_are_confirmed_by_consensus():
    ds = S.make_demo_dataset("R11", 4)
    a = S.analyze(ds.df)
    _, spots = consensus.cluster_events(a.events, n_rides_total=4)
    assert spots["confirmed"].sum() >= 5 and spots["n_rides"].max() == 4
    at = run_page("views/4_Consensus_Map.py", ds)
    assert not at.exception and any("Confirmed spots" in m.value for m in at.markdown)


def test_full_app_navigation_runs():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("SIMULATED" in m.value for m in at.markdown)          # honesty banner is always present
    at.switch_page("views/7_Model_Card.py").run()
    assert not at.exception, [e.value for e in at.exception]


def test_upload_pipeline_reports_friendly_errors_and_repairs():
    good = S.process_upload((ROOT / "data" / "sample_rides" / "ride_R11.csv").read_bytes(), "ride_R11.csv")
    assert good.ok and good.report.info["has_labels"]
    bad = S.process_upload(b"time_s,speed_kmh\n1,2\n", "bad.csv")
    assert not bad.ok and bad.needs_mapping and "acc_x" in bad.report.errors[0]
    assert S.process_upload(b"", "empty.csv").read_error
    merged = S.combine_results([good, S.process_upload((ROOT / "data" / "sample_rides" / "ride_R11.csv").read_bytes(), "again.csv")])
    assert len(merged.rides) == 2 and len(set(merged.rides)) == 2     # duplicate ride ids are made unique
