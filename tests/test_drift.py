"""Tests: drift (Trust Meter), explain, maps, charts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.ensemble import RandomForestClassifier

from core import charts, drift, explain, inference, maps
from core import config as C


def make_reference() -> dict:
    feats = {c: {"mean": 0.0, "std": 1.0, "p05": -1.645, "p95": 1.645} for c in C.FEATURE_ORDER}
    feats["z_std"] = {"mean": 1.0, "std": 0.577, "p05": 0.1, "p95": 1.9}
    feats["speed_mean"] = {"mean": 25.0, "std": 5.0, "p05": 17.0, "p95": 33.0}
    t = np.arange(-20, 40) / 20
    wave = lambda a: (a * np.exp(-((t - 0.2) ** 2) / 0.05)).tolist()  # noqa: E731
    return {"features": feats, "smooth_z_std": {"quantiles": list(np.linspace(0, 2, 101))},
            "signatures": {"t_rel": t.tolist(), "1": {"mean": wave(2), "std": [0.5] * 60, "n": 10},
                           "2": {"mean": wave(-5), "std": [0.8] * 60, "n": 10}}}


def make_windows(n: int = 400, shift: float = 0.0, z_scale: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = pd.DataFrame(rng.normal(shift, 1.0, (n, 21)), columns=C.FEATURE_ORDER)
    w["z_std"] = rng.uniform(0, 2, n) * z_scale + (shift if shift else 0.0)
    w["speed_mean"] = rng.normal(25 + shift * 5, 5, n)
    w["ride_id"], w["start_time"] = "r1", np.arange(n, dtype=float)
    return w


# ------------------------------------------------------------------ drift
def test_in_distribution_ride_is_high_trust_and_bounded():
    r = drift.assess_drift(make_windows(), make_reference())
    assert 0.0 <= r.score <= 100.0 and r.label == "High" and r.score >= C.TRUST_BANDS[0]
    assert r.reasons == ["This ride looks similar to the data the model was trained on."]
    assert r.recommended_mode == "Balanced" and len(r.per_feature) == 21


def test_shifted_ride_is_low_trust_with_large_z():
    r = drift.assess_drift(make_windows(shift=3.0), make_reference())
    assert r.label == "Low" and r.score < C.TRUST_BANDS[1]
    assert r.per_feature["z"].abs().max() > 2.0 and r.recommended_mode == "Driver Alert"


def test_rough_ride_explains_false_alarms():
    base = drift.assess_drift(make_windows(), make_reference())
    rough = drift.assess_drift(make_windows(z_scale=3.0), make_reference())
    assert rough.roughness_percentile >= 90 and rough.score < base.score
    assert any("rougher than" in s and "false alarms" in s for s in rough.reasons)


def test_smooth_ride_warns_about_missed_events():
    r = drift.assess_drift(make_windows(z_scale=0.1), make_reference())
    assert r.roughness_percentile <= 10 and any("smoother than" in s and "missed" in s for s in r.reasons)


def test_score_always_within_bounds():
    ref = make_reference()
    for seed, shift in enumerate([-8, -3, -1, 0, 0.5, 2, 6, 20]):
        r = drift.assess_drift(make_windows(60, shift=shift, seed=seed), ref, ["Sampling rate is not 20 Hz"] * 5)
        assert 0.0 <= r.score <= 100.0 and r.label in {"High", "Medium", "Low"}


def test_short_ride_is_never_high():
    r = drift.assess_drift(make_windows(5), make_reference())
    assert r.label != "High" and any("Only 5 window" in s for s in r.reasons)


def test_only_quality_warnings_lower_trust():
    ref, w = make_reference(), make_windows()
    base = drift.assess_drift(w, ref).score
    assert drift.assess_drift(w, ref, ["2.0% of GPS fixes are missing and will be interpolated."]).score == base
    assert drift.assess_drift(w, ref, ["Sampling rate is not 20 Hz."]).score <= base - 5
    assert any(s.startswith("Input check:") for s in drift.assess_drift(w, ref, ["Sampling rate is not 20 Hz."]).reasons)


def test_roughness_percentile_is_monotonic_and_bounded():
    ref = make_reference()
    vals = [drift.roughness_percentile(v, ref) for v in (-1, 0, 0.5, 1, 1.5, 2, 5)]
    assert vals == sorted(vals) and vals[0] == 0.0 and vals[-1] == 100.0


def test_load_reference_from_disk(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(make_reference()))
    assert drift.load_reference(p)["features"]["z_std"]["mean"] == 1.0
    with pytest.raises(FileNotFoundError):
        drift.load_reference(tmp_path / "nope.json")


# ------------------------------------------------------------------ explain
@pytest.fixture(scope="module")
def tiny_bundle() -> inference.ModelBundle:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 21)), columns=C.FEATURE_ORDER)
    y = np.where(X["z_std"] > 0.8, 2, np.where(X["speed_min"] < -0.8, 1, 0))
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    return inference.ModelBundle(model, "tiny-rf", tuple(C.FEATURE_ORDER), tuple(C.CLASS_NAMES), 40, 20, 20)


def make_event() -> dict:
    return {"event_id": "r1-E001", "ride_id": "r1", "class_id": 2, "class_name": "Pothole", "peak_time": 30.0,
            "best_window_start": 29.0, "confidence": 0.9, "severity": 55.0, "severity_label": "Medium", "speed_kmh": 20.0}


def make_ride() -> pd.DataFrame:
    t = np.arange(0, 60, 0.05)
    return pd.DataFrame({"ride_id": "r1", "time_s": t, "acc_z_dyn": np.sin(t)})


def test_top_features_returns_ranked_plain_english(tiny_bundle):
    row = make_windows(3, seed=1).iloc[0]
    feats, method = explain.top_features(tiny_bundle, row, 2, make_reference(), k=5)
    assert len(feats) == 5 and method in {"SHAP", "importance x deviation"}
    assert feats["contribution"].abs().is_monotonic_decreasing
    assert all(isinstance(t, str) and len(t) > 20 for t in feats["text"])


def test_event_waveform_alignment():
    t_rel, z = explain.event_waveform(make_ride(), make_event())
    assert len(t_rel) == len(z) > 50 and t_rel.min() >= -1.0 - 1e-9 and t_rel.max() <= 2.0 + 1e-9
    start = 30.0 - C.CLASS_MEAN_DURATION_S[2] / 2
    assert abs(t_rel[np.argmin(np.abs(t_rel))]) < 0.06 and start > 0


def test_explain_event_and_receipts(tiny_bundle):
    w = make_windows(60)
    ex = explain.explain_event(tiny_bundle, w, make_event(), make_ride(), make_reference())
    assert ex.summary.startswith("Pothole") and len(ex.features) == 5 and len(ex.sig_mean) == 60
    page = explain.receipt_html(ex)
    assert "SIMULATED" in page and "Pothole" in page and "<svg" in page and "Confidence" in page
    png = explain.receipt_png(ex)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000


# ------------------------------------------------------------------ maps
def ride_and_events():
    n = 200
    ride = pd.DataFrame({"ride_id": "r1", "time_s": np.arange(n) / 20, "lat": 18.5 + np.arange(n) * 1e-5, "lon": 73.8})
    ev = pd.DataFrame([{"class_id": 2, "class_name": "Pothole", "severity": 60.0, "severity_label": "Medium",
                        "confidence": 0.9, "speed_kmh": 20.0, "ride_id": "r1", "peak_time": 5.0, "lat": 18.5005, "lon": 73.8}])
    return ride, ev


def test_maps_use_dark_tiles_and_handle_empty():
    ride, ev = ride_and_events()
    m = maps.ride_map(ride, ev)
    assert isinstance(m, folium.Map)
    html = maps.map_to_html(m)
    assert "dark_all" in html and "OpenStreetMap" in html and "Pothole" in html
    assert isinstance(maps.ride_map(ride, None), folium.Map)
    empty_spots = pd.DataFrame(columns=["lat", "lon"])
    assert isinstance(maps.consensus_map(empty_spots.assign(confidence=[], confirmed=[], n_rides=[])), folium.Map)
    assert isinstance(maps.priority_map(pd.DataFrame(columns=["lat", "lon", "priority", "rank"])), folium.Map)


def test_consensus_and_priority_maps_render_markers():
    spots = pd.DataFrame([{"spot_id": "P001", "class_name": "Pothole", "lat": 18.5, "lon": 73.8, "n_rides": 3, "rides": "a, b, c",
                           "confidence": 1.0, "confirmed": True, "severity_mean": 70.0}])
    assert "P001" in maps.map_to_html(maps.consensus_map(spots))
    ranked = pd.DataFrame([{"rank": 1, "class_name": "Pothole", "lat": 18.5, "lon": 73.8, "priority": 60.0, "why": "Severe pothole."}])
    assert "Severe pothole." in maps.map_to_html(maps.priority_map(ranked))
    assert maps.confidence_color(0.9, True) == C.COLORS["pothole"] and maps.confidence_color(0.2, False) == C.COLORS["muted"]


# ------------------------------------------------------------------ charts
def test_template_and_figures_build(tiny_bundle):
    assert charts.register_template() == "roadpulse"
    ride, ev = ride_and_events()
    ride["acc_z_dyn"] = np.sin(ride["time_s"])
    ex = explain.explain_event(tiny_bundle, make_windows(60), make_event(), make_ride(), make_reference())
    rep = drift.assess_drift(make_windows(), make_reference())
    pw = make_windows(20).assign(p_speed_breaker=0.1, p_pothole=0.2)
    figs = [
        charts.waveform_fig(ride, ev), charts.probability_fig(pw), charts.gauge_fig(rep.score, rep.label),
        charts.drift_fig(rep.per_feature), charts.signature_fig(ex),
        charts.threshold_sweep_fig(pd.DataFrame({"threshold": [0.2, 0.5], "recall": [1, 0.8], "false_alarms_per_km": [0.5, 0.1]})),
        charts.model_comparison_fig(pd.DataFrame({"Macro F1": [0.9], "CV Macro F1": [0.88]}, index=["XGBoost"])),
        charts.cv_fig(pd.DataFrame({"Model": ["A", "A"], "Held-out route": ["A", "B"], "Macro F1": [0.9, 0.8]})),
        charts.per_class_fig(pd.DataFrame({"Model": ["A"], "Class": ["Pothole"], "Recall": [0.8], "Precision": [0.7], "F1": [0.75]})),
        charts.priority_fig(pd.DataFrame({"rank": [1], "class_name": ["Pothole"], "spot_id": ["P001"], "priority": [50.0]})),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)
    assert all(f.layout.paper_bgcolor == "rgba(0,0,0,0)" or f.layout.template.layout.paper_bgcolor == "rgba(0,0,0,0)" for f in figs)
