"""Tests: events, consensus, scoring."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from core import config as C
from core import consensus, events, scoring


def pred_windows(starts, classes, probs=0.9, ride="r1") -> pd.DataFrame:
    rows = []
    for s, c in zip(starts, classes):
        rows.append({"ride_id": ride, "route_id": "A", "start_time": float(s), "pred": c,
                     "p_speed_breaker": probs if c == 1 else 0.05, "p_pothole": probs if c == 2 else 0.05,
                     "speed_mean": 20.0, "z_absmax": 4.0, "lat": 18.5, "lon": 73.8})
    return pd.DataFrame(rows)


def ride_signal(duration=60) -> pd.DataFrame:
    n = duration * 20
    t = np.arange(n) / 20
    z = np.zeros(n)
    z[int(30.0 * 20)] = -6.0                              # one sharp dip at t = 30 s
    return pd.DataFrame({"ride_id": "r1", "time_s": t, "lat": 18.5 + t * 1e-5, "lon": 73.8, "speed_kmh": 30.0, "acc_z_dyn": z})


# ------------------------------------------------------------------ events
def test_consecutive_windows_merge_into_one_event():
    pw = pred_windows([10, 11, 12, 30], [2, 2, 2, 1])
    ev = events.merge_windows_to_events(pw)
    assert len(ev) == 2
    first = ev.iloc[0]
    assert first["class_id"] == 2 and first["n_windows"] == 3
    assert np.isclose(first["peak_time"], 12.0)            # centre of windows 10, 11, 12 (+1 s)
    assert np.isclose(first["start_time"], 10.0) and np.isclose(first["end_time"], 14.0)
    assert ev.iloc[1]["class_id"] == 1 and ev.iloc[1]["event_id"] == "r1-E002"


def test_gap_larger_than_one_step_splits_events():
    ev = events.merge_windows_to_events(pred_windows([10, 12], [2, 2]))
    assert len(ev) == 2


def test_class_chosen_by_summed_probability_and_no_flags_gives_empty():
    pw = pred_windows([5, 6, 7], [2, 1, 1], probs=0.9)
    pw.loc[0, "p_pothole"] = 0.99
    pw.loc[[1, 2], "p_speed_breaker"] = 0.9
    assert events.merge_windows_to_events(pw).iloc[0]["class_id"] == 1
    assert events.merge_windows_to_events(pred_windows([5, 6], [0, 0])).empty


def test_event_uses_raw_signal_for_peak_location_and_speed():
    ev = events.merge_windows_to_events(pred_windows([29], [2]), ride_signal())
    r = ev.iloc[0]
    assert np.isclose(r["peak_abs_z"], 6.0) and np.isclose(r["speed_kmh"], 30.0)
    assert np.isclose(r["lat"], 18.5 + 30.0 * 1e-5, atol=1e-6)


def test_severity_bounds_and_labels():
    for c in C.EVENT_CLASSES:
        assert events.severity_score(0.0, c) == 0.0 and events.severity_score(1e6, c) == 100.0
    assert events.severity_label(10) == "Low" and events.severity_label(50) == "Medium" and events.severity_label(90) == "High"
    assert events.severity_score(4.0, 2) < events.severity_score(6.0, 2)


def test_distance_and_duration():
    t = np.arange(0, 101) * 1.0
    df = pd.DataFrame({"ride_id": "r", "time_s": t, "speed_kmh": 36.0})
    assert np.isclose(events.distance_km(df), 1.0)         # 36 km/h for 100 s = 1 km
    assert np.isclose(events.duration_s(df), 100.0)


def test_ground_truth_and_evaluation():
    n = 400
    df = pd.DataFrame({"ride_id": "r1", "time_s": np.arange(n) / 20, "lat": 18.5, "lon": 73.8, "label": 0})
    df.loc[100:107, "label"] = 2                            # pothole at 5.0-5.35 s
    df.loc[200:220, "label"] = 1                            # breaker at 10.0-11.0 s
    gt = events.ground_truth_events(df)
    assert list(gt["class_id"]) == [2, 1] and len(gt) == 2

    def ev(cid, peak, rid="r1"):
        return {"event_id": f"{rid}-{peak}", "ride_id": rid, "class_id": cid, "peak_time": peak}

    pred = pd.DataFrame([ev(2, 5.2), ev(1, 10.4), ev(2, 15.0)])      # 2 hits + 1 false alarm
    m = events.evaluate_events(pred, gt, km=2.0)
    assert (m.summary["tp"], m.summary["fp"], m.summary["fn"]) == (2, 1, 0)
    assert np.isclose(m.summary["false_alarms_per_km"], 0.5) and np.isclose(m.summary["recall"], 1.0)
    wrong = events.evaluate_events(pd.DataFrame([ev(1, 5.2)]), gt, km=1.0)   # wrong class: FP + FN
    assert wrong.summary["tp"] == 0 and wrong.summary["fp"] == 1 and wrong.summary["fn"] == 2
    miss = events.evaluate_events(pred.iloc[0:0], gt, km=1.0)
    assert miss.summary["fn"] == 2 and np.isclose(miss.summary["missed_per_km"], 2.0)


def test_each_label_matches_at_most_once():
    gt = pd.DataFrame([{"ride_id": "r1", "class_id": 2, "start_time": 5.0, "end_time": 5.3}])
    pred = pd.DataFrame([{"event_id": "a", "ride_id": "r1", "class_id": 2, "peak_time": 5.1},
                         {"event_id": "b", "ride_id": "r1", "class_id": 2, "peak_time": 5.2}])
    m = events.evaluate_events(pred, gt, km=1.0)
    assert (m.summary["tp"], m.summary["fp"]) == (1, 1)


# ------------------------------------------------------------------ consensus
def make_events() -> pd.DataFrame:
    rows = []
    for i, rid in enumerate(["a", "b", "c"]):
        rows.append({"event_id": f"{rid}-1", "ride_id": rid, "class_id": 2, "lat": 18.5000 + i * 2e-5,
                     "lon": 73.8000, "severity": 60.0 + i, "speed_kmh": 20.0, "confidence": 0.9, "peak_abs_z": 5.0})
    rows.append({"event_id": "a-2", "ride_id": "a", "class_id": 2, "lat": 18.6, "lon": 73.9, "severity": 40.0,
                 "speed_kmh": 30.0, "confidence": 0.8, "peak_abs_z": 3.0})
    rows.append({"event_id": "a-3", "ride_id": "a", "class_id": 1, "lat": 18.5000, "lon": 73.8000, "severity": 30.0,
                 "speed_kmh": 12.0, "confidence": 0.8, "peak_abs_z": 2.0})
    return pd.DataFrame(rows)


def test_nearby_events_from_different_rides_confirm_each_other():
    ev, spots = consensus.cluster_events(make_events(), n_rides_total=3)
    pot = spots[spots["class_id"] == 2].sort_values("n_rides", ascending=False)
    assert len(pot) == 2
    top = pot.iloc[0]
    assert top["n_rides"] == 3 and np.isclose(top["confidence"], 1.0) and bool(top["confirmed"])
    far = pot.iloc[1]
    assert far["n_rides"] == 1 and np.isclose(far["confidence"], 1 / 3) and not bool(far["confirmed"])
    assert ev["spot_id"].ne("").all()


def test_classes_are_clustered_separately():
    _, spots = consensus.cluster_events(make_events(), n_rides_total=3)
    assert set(spots["class_id"]) == {1, 2}
    assert spots.loc[spots["class_id"] == 1, "n_rides"].iloc[0] == 1


def test_consensus_handles_empty_and_missing_gps():
    ev, spots = consensus.cluster_events(make_events().iloc[0:0])
    assert spots.empty and list(spots.columns) == consensus.SPOT_COLUMNS
    gpsless = make_events()
    gpsless["lat"] = np.nan
    assert consensus.cluster_events(gpsless)[1].empty


# ------------------------------------------------------------------ scoring
def test_exposure_factor():
    assert scoring.exposure_factor(0) == 2.0 and scoring.exposure_factor(30) == 1.0 and scoring.exposure_factor(80) == 1.0
    assert scoring.exposure_factor(float("nan")) == 1.0
    assert scoring.exposure_factor(10) > scoring.exposure_factor(20)


def test_priority_is_monotonic_and_bounded():
    base = scoring.priority_score(50, 2, 4, 20)
    assert scoring.priority_score(80, 2, 4, 20) > base
    assert scoring.priority_score(50, 3, 4, 20) > base
    assert scoring.priority_score(50, 2, 4, 10) > base
    for sev in (0, 50, 100):
        for conf in (0, 1, 4):
            for spd in (0, 15, 60):
                assert 0.0 <= scoring.priority_score(sev, conf, 4, spd) <= 100.0
    assert np.isclose(scoring.priority_score(100, 4, 4, 0), 100.0)


def test_rank_priority_orders_and_filters():
    _, spots = consensus.cluster_events(make_events(), n_rides_total=3)
    ranked = scoring.rank_priority(spots, 3)
    assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))
    assert ranked["priority"].is_monotonic_decreasing
    assert ranked.iloc[0]["confirmations"] == 3 and "confirmed by 3 of 3" in ranked.iloc[0]["why"]
    assert set(scoring.rank_priority(spots, 3, classes=[2])["class_name"]) == {"Pothole"}
    assert scoring.rank_priority(spots, 3, min_confirmations=3).shape[0] == 1
    assert scoring.rank_priority(spots.iloc[0:0], 3).empty


# ------------------------------------------------------------------ added helpers
def test_estimate_false_alarms_sums_uncertainty():
    ev = pd.DataFrame({"confidence": [0.9, 0.6, 1.0]})
    assert np.isclose(events.estimate_false_alarms(ev), 0.5)
    assert events.estimate_false_alarms(ev.iloc[0:0]) == 0.0


def test_simulated_repeat_passes_are_deterministic_and_close():
    n = 200
    base = pd.DataFrame({"ride_id": "R", "time_s": np.arange(n) / 20, "lat": 18.5 + np.arange(n) * 1e-6, "lon": 73.8,
                         "speed_kmh": 25.0, "acc_x": 0.0, "acc_y": 0.0, "acc_z": 9.81,
                         "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0, "label": 0})
    a = consensus.simulate_repeat_passes(base, n_passes=3, seed=1)
    b = consensus.simulate_repeat_passes(base, n_passes=3, seed=1)
    assert a.equals(b) and a["ride_id"].nunique() == 3 and set(a["ride_id"]) == {"R-pass1", "R-pass2", "R-pass3"}
    assert (a["label"] == 0).all() and (a["speed_kmh"] >= 0).all()
    shift_m = np.abs(a["lat"].to_numpy() - np.tile(base["lat"].to_numpy(), 3)) * 111_320
    assert shift_m.max() < C.CONSENSUS_RADIUS_M                       # jitter stays well inside the cluster radius
    assert not np.allclose(a.loc[a["ride_id"] == "R-pass1", "acc_z"], a.loc[a["ride_id"] == "R-pass2", "acc_z"])
