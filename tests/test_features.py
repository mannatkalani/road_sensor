"""Tests: config, preprocess, features, inference (deterministic, no app assets required)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from core import config as C
from core import features, inference, preprocess


def synthetic_ride(n: int = 400, seed: int = 0, ride_id: str = "r1", route: str = "A") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / C.SAMPLING_HZ
    return pd.DataFrame({
        "ride_id": ride_id, "route_id": route, "time_s": t,
        "lat": 18.5 + np.cumsum(np.full(n, 1e-6)), "lon": 73.8 + np.cumsum(np.full(n, 1e-6)),
        "speed_kmh": 25 + rng.normal(0, 1, n),
        "acc_x": rng.normal(0, 0.2, n), "acc_y": rng.normal(0, 0.2, n), "acc_z": 9.81 + rng.normal(0, 0.5, n),
        "gyro_x": rng.normal(0, 0.03, n), "gyro_y": rng.normal(0, 0.03, n), "gyro_z": rng.normal(0, 0.03, n),
        "label": 0,
    })


# ------------------------------------------------------------------ config
def test_config_constants():
    assert len(C.FEATURE_ORDER) == 21 and len(set(C.FEATURE_ORDER)) == 21
    assert (C.WINDOW_SIZE, C.STEP, C.SAMPLING_HZ) == (40, 20, 20)
    assert set(C.FEATURE_LABELS) == set(C.FEATURE_ORDER)
    assert all(c.startswith("#") and len(c) == 7 for c in C.COLORS.values())
    assert C.CLASS_NAMES == ["Smooth", "Speed breaker", "Pothole"]


# ------------------------------------------------------------------ preprocess
def test_preprocess_sorts_and_removes_gravity():
    df = synthetic_ride().sample(frac=1.0, random_state=1)
    out = preprocess.preprocess(df)
    assert out["time_s"].is_monotonic_increasing
    assert np.allclose(out["acc_z_dyn"], out["acc_z"] - C.GRAVITY)
    assert abs(out["acc_z_dyn"].mean()) < 0.2


def test_gps_interpolation_never_crosses_rides():
    a, b = synthetic_ride(100, 1, "a"), synthetic_ride(100, 2, "b")
    b["lat"] = b["lat"] + 1.0
    a.loc[40:60, "lat"] = np.nan
    b.loc[0:10, "lat"] = np.nan                      # leading gap in ride b
    out = preprocess.preprocess(pd.concat([a, b], ignore_index=True))
    assert out["lat"].notna().all()
    lat_b = out.loc[out["ride_id"] == "b", "lat"]
    assert lat_b.iloc[0] > 19.0                      # filled from ride b, not from ride a (~18.5)


def test_resample_uniform_to_20hz():
    n = 500
    t = np.arange(n) / 50.0                          # 50 Hz input
    df = synthetic_ride(n)
    df["time_s"] = t
    out = preprocess.resample_uniform(df)
    assert np.allclose(np.diff(out["time_s"]), 1 / 20, atol=1e-9)
    assert abs(len(out) - (t[-1] - t[0]) * 20) <= 1
    assert out["label"].isin([0, 1, 2]).all()


# ------------------------------------------------------------------ features
def test_window_count_columns_and_times():
    pre = preprocess.preprocess(synthetic_ride(400))
    w = features.build_windows(pre)
    assert len(w) == (400 - C.WINDOW_SIZE) // C.STEP + 1 == 19
    assert list(w[C.FEATURE_ORDER].columns) == C.FEATURE_ORDER and w[C.FEATURE_ORDER].shape[1] == 21
    assert np.isclose(w["start_time"].iloc[0], 0.0) and np.isclose(w["start_time"].iloc[1], 1.0)
    assert {"ride_id", "route_id", "start_time", "lat", "lon", "label"} <= set(w.columns)


def test_feature_values_match_manual_computation():
    pre = preprocess.preprocess(synthetic_ride(400, seed=3))
    w = features.build_windows(pre)
    seg = pre.iloc[20:60]                            # second window (start index 20)
    z = seg["acc_z_dyn"].to_numpy()
    row = w.iloc[1]
    assert np.isclose(row["z_std"], z.std()) and np.isclose(row["z_range"], z.max() - z.min())
    assert np.isclose(row["z_absmax"], np.abs(z).max())
    assert np.isclose(row["y_mean"], seg["acc_y"].mean()) and np.isclose(row["speed_min"], seg["speed_kmh"].min())
    assert np.isclose(row["gx_absmax"], seg["gyro_x"].abs().max())
    p = np.abs(np.fft.rfft(z)) ** 2
    f = np.fft.rfftfreq(40, 1 / 20)
    assert np.isclose(row["pow_2_5hz"], p[(f >= 2) & (f < 5)].sum() / (p.sum() + 1e-9))


def test_power_shares_sum_to_one():
    w = features.build_windows(preprocess.preprocess(synthetic_ride(400, seed=5)))
    total = w[["pow_0_2hz", "pow_2_5hz", "pow_5_10hz"]].sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-6)


def test_label_is_window_maximum():
    df = synthetic_ride(400)
    df.loc[100:105, "label"] = 2
    w = features.build_windows(preprocess.preprocess(df))
    covering = w[(w["start_time"] * 20 <= 105) & (w["start_time"] * 20 + 39 >= 100)]
    assert (covering["label"] == 2).all() and (w.loc[~w.index.isin(covering.index), "label"] == 0).all()


def test_short_ride_skipped_and_nan_windows_dropped():
    short = preprocess.preprocess(synthetic_ride(30))
    assert features.build_windows(short).empty
    df = synthetic_ride(400)
    df.loc[150, "acc_x"] = np.nan                    # poisons the windows covering sample 150
    w = features.build_windows(preprocess.preprocess(df))
    assert len(w) < 19 and w[C.FEATURE_ORDER].notna().all().all()


def test_build_windows_requires_preprocessed_input():
    with pytest.raises(ValueError):
        features.build_windows(synthetic_ride())


def test_select_model_features_uses_given_order():
    w = features.build_windows(preprocess.preprocess(synthetic_ride()))
    cols = list(reversed(C.FEATURE_ORDER))
    assert list(features.select_model_features(w, cols).columns) == cols
    with pytest.raises(ValueError):
        features.select_model_features(w, [*cols, "nope"])


# ------------------------------------------------------------------ inference
@pytest.fixture(scope="module")
def tiny_bundle() -> inference.ModelBundle:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 21)), columns=C.FEATURE_ORDER)
    y = np.where(X["z_std"] > 0.8, 2, np.where(X["speed_min"] < -0.8, 1, 0))
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    return inference.ModelBundle(model, "tiny-rf", tuple(C.FEATURE_ORDER), tuple(C.CLASS_NAMES), 40, 20, 20)


def test_predict_proba_shape_and_rows_sum_to_one(tiny_bundle):
    w = features.build_windows(preprocess.preprocess(synthetic_ride(400)))
    p = inference.predict_proba(tiny_bundle, w)
    assert p.shape == (len(w), 3) and np.allclose(p.sum(axis=1), 1.0)
    assert inference.predict_proba(tiny_bundle, w.iloc[0:0]).shape == (0, 3)


def test_apply_thresholds_rules():
    proba = np.array([[0.9, 0.05, 0.05], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.1, 0.45, 0.45], [0.1, 0.3, 0.6]])
    assert list(inference.apply_thresholds(proba, {1: 0.5, 2: 0.5})) == [0, 1, 2, 0, 2]
    assert list(inference.apply_thresholds(proba, {1: 0.4, 2: 0.4}))[3] == 2          # tie goes to pothole
    assert list(inference.apply_thresholds(proba, {"1": 0.95, "2": 0.95})) == [0, 0, 0, 0, 0]  # string keys accepted
    assert inference.apply_thresholds(np.zeros((0, 3))).shape == (0,)


def test_predict_windows_adds_expected_columns(tiny_bundle):
    w = features.build_windows(preprocess.preprocess(synthetic_ride(400)))
    out = inference.predict_windows(tiny_bundle, w)
    assert {"p_smooth", "p_speed_breaker", "p_pothole", "pred", "pred_name", "confidence"} <= set(out.columns)
    assert len(out) == len(w) and out["confidence"].between(0, 1).all()


def test_bundle_roundtrip_and_errors(tmp_path, tiny_bundle):
    path = tmp_path / "m.joblib"
    joblib.dump({"model": tiny_bundle.model, "model_name": "tiny-rf", "feature_columns": list(C.FEATURE_ORDER),
                 "class_names": list(C.CLASS_NAMES), "window_size": 40, "step": 20, "sampling_hz": 20}, path)
    b = inference.load_bundle(path)
    assert b.model_name == "tiny-rf" and b.feature_columns == tuple(C.FEATURE_ORDER)
    with pytest.raises(inference.ModelLoadError):
        inference.load_bundle(tmp_path / "missing.joblib")
    joblib.dump({"model": tiny_bundle.model}, tmp_path / "bad.joblib")
    with pytest.raises(inference.ModelLoadError):
        inference.load_bundle(tmp_path / "bad.joblib")
