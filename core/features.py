"""Window feature builder. The model depends on this exact recipe: do not change it."""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis, skew

from core import config as C

FS = C.SAMPLING_HZ
WIN = C.WINDOW_SIZE
STEP = C.STEP


def _win(a: np.ndarray) -> np.ndarray:
    """Overlapping windows of ``WIN`` samples taken every ``STEP`` samples."""
    return sliding_window_view(a, WIN)[::STEP]


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Turn preprocessed rides into one row per 2-second window (1-second step).

    ``df`` must come from :func:`core.preprocess.preprocess` (needs ``acc_z_dyn``). Output columns:
    ``ride_id``, ``route_id`` (if present), the 21 features in :data:`core.config.FEATURE_ORDER`,
    ``start_time``, ``lat``, ``lon`` and ``label`` (if present). Windows whose features contain NaN
    are dropped (a window with no GPS is kept; its lat/lon are NaN).
    """
    if "acc_z_dyn" not in df.columns:
        raise ValueError("build_windows expects preprocessed data with an 'acc_z_dyn' column.")
    frames: list[pd.DataFrame] = []
    has_label = "label" in df.columns
    has_route = "route_id" in df.columns
    for rid, g in df.groupby("ride_id", sort=False):
        if len(g) < WIN:
            continue
        z = _win(g["acc_z_dyn"].to_numpy(dtype=float))
        x = _win(g["acc_x"].to_numpy(dtype=float))
        y = _win(g["acc_y"].to_numpy(dtype=float))
        gx = _win(g["gyro_x"].to_numpy(dtype=float))
        gy = _win(g["gyro_y"].to_numpy(dtype=float))
        gz = _win(g["gyro_z"].to_numpy(dtype=float))
        sp = _win(g["speed_kmh"].to_numpy(dtype=float))
        t0 = _win(g["time_s"].to_numpy(dtype=float))
        la = _win(g["lat"].to_numpy(dtype=float))
        lo = _win(g["lon"].to_numpy(dtype=float))

        power = np.abs(np.fft.rfft(z, axis=1)) ** 2
        freqs = np.fft.rfftfreq(WIN, 1 / FS)
        tot = power.sum(axis=1) + 1e-9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feat = pd.DataFrame({
                "z_std": z.std(1), "z_mean": z.mean(1), "z_max": z.max(1), "z_min": z.min(1),
                "z_range": z.max(1) - z.min(1), "z_absmax": np.abs(z).max(1),
                "z_skew": skew(z, axis=1), "z_kurt": kurtosis(z, axis=1),
                "x_std": x.std(1), "y_mean": y.mean(1), "y_std": y.std(1),
                "gx_std": gx.std(1), "gy_std": gy.std(1), "gz_std": gz.std(1),
                "gx_absmax": np.abs(gx).max(1),
                "speed_mean": sp.mean(1), "speed_min": sp.min(1),
                "speed_range": sp.max(1) - sp.min(1),
                "pow_0_2hz": power[:, (freqs >= 0) & (freqs < 2)].sum(1) / tot,
                "pow_2_5hz": power[:, (freqs >= 2) & (freqs < 5)].sum(1) / tot,
                "pow_5_10hz": power[:, freqs >= 5].sum(1) / tot,
            })[C.FEATURE_ORDER]
            feat["start_time"] = t0[:, 0]
            feat["lat"] = np.nanmean(la, axis=1)
            feat["lon"] = np.nanmean(lo, axis=1)
        if has_label:
            feat["label"] = _win(g["label"].to_numpy()).max(1).astype(int)
        feat.insert(0, "ride_id", rid)
        if has_route:
            feat.insert(1, "route_id", g["route_id"].iloc[0])
        frames.append(feat)
    if not frames:
        cols = ["ride_id", *C.FEATURE_ORDER, "start_time", "lat", "lon"]
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=[*C.FEATURE_ORDER, "start_time"]).reset_index(drop=True)


def select_model_features(windows: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Pick model inputs using the order stored in the model file (never a hard-coded order)."""
    missing = [c for c in feature_columns if c not in windows.columns]
    if missing:
        raise ValueError(f"Windows are missing model feature(s): {missing}")
    return windows.loc[:, list(feature_columns)]
