"""Per-ride preprocessing: sorting, gravity removal, GPS interpolation, optional resampling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import config as C


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Sort each ride by time, remove gravity from ``acc_z`` and interpolate GPS within the ride.

    Rides are processed independently, so GPS gaps are never filled across rides.
    """
    work = df.copy()
    if "ride_id" not in work.columns:
        work["ride_id"] = "ride_1"
    parts: list[pd.DataFrame] = []
    for _, g in work.groupby("ride_id", sort=False):
        g = g.sort_values("time_s", kind="stable").drop_duplicates("time_s").reset_index(drop=True)
        g["acc_z_dyn"] = g["acc_z"] - C.GRAVITY
        for col in C.GPS_COLUMNS:
            g[col] = g[col].interpolate(limit_direction="both")
        parts.append(g)
    if not parts:
        return work.assign(acc_z_dyn=pd.Series(dtype=float))
    return pd.concat(parts, ignore_index=True)


def resample_uniform(df: pd.DataFrame, fs: int = C.SAMPLING_HZ) -> pd.DataFrame:
    """Resample every ride onto a uniform ``fs`` Hz grid by linear interpolation.

    Labels use the nearest sample; text columns (ride/route ids) are carried over. Linear
    interpolation does not low-pass filter, so down-sampling very high rates can slightly
    soften sharp peaks.
    """
    numeric = [c for c in [*C.REQUIRED_COLUMNS, *C.GPS_COLUMNS] if c != "time_s"]
    parts: list[pd.DataFrame] = []
    for rid, g in df.groupby("ride_id", sort=False):
        g = g.sort_values("time_s", kind="stable").drop_duplicates("time_s")
        t = g["time_s"].to_numpy(dtype=float)
        if len(t) < 2:
            parts.append(g)
            continue
        grid = np.arange(t[0], t[-1], 1.0 / fs)
        out = pd.DataFrame({"time_s": grid})
        for c in numeric:
            col = g[c].to_numpy(dtype=float)
            ok = ~np.isnan(col)
            out[c] = np.interp(grid, t[ok], col[ok]) if ok.sum() >= 2 else np.nan
        if "label" in g.columns:
            idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
            nearest = np.where(np.abs(grid - t[idx - 1]) <= np.abs(grid - t[idx]), idx - 1, idx)
            out["label"] = g["label"].to_numpy()[nearest].astype(int)
        out["ride_id"] = rid
        if "route_id" in g.columns:
            out["route_id"] = g["route_id"].iloc[0]
        parts.append(out)
    res = pd.concat(parts, ignore_index=True)
    ordered = [c for c in C.TEMPLATE_COLUMNS if c in res.columns]
    return res[ordered + [c for c in res.columns if c not in ordered]]
