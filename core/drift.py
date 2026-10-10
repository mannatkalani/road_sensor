"""Trust Meter: how similar is this ride to the data the model was trained on?"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from core import config as C

KEY_SPEED_FEATURE = "speed_mean"
# Only input problems that make the model's inputs unreliable reduce trust (not e.g. small GPS gaps).
_QUALITY_KEYWORDS = ("sampling rate", "units of g", "gravity", "degrees/s", "m/s instead", "z-axis")


@dataclass
class DriftReport:
    """Trust score (0-100), its label, and plain-English reasons."""

    score: float
    label: str
    roughness_percentile: float
    n_windows: int
    per_feature: pd.DataFrame
    reasons: list[str]
    recommended_mode: str


@lru_cache(maxsize=4)
def _load_reference(path_str: str) -> dict[str, Any]:
    with open(path_str, encoding="utf-8") as fh:
        return json.load(fh)


def load_reference(path: str | Path | None = None) -> dict[str, Any]:
    """Load (and cache) ``reference_stats.json``."""
    p = Path(path or C.REFERENCE_STATS_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"Reference statistics not found at '{p}'. Run: python tools/make_assets.py --dataset <your csv>"
        )
    return _load_reference(str(p))


def roughness_percentile(value: float, reference: dict[str, Any]) -> float:
    """Percentile (0-100) of ``value`` within the training smooth-road ``z_std`` distribution."""
    q = np.maximum.accumulate(np.asarray(reference["smooth_z_std"]["quantiles"], dtype=float))
    return float(np.interp(value, q, np.linspace(0, 100, len(q))))


def _label_for(score: float) -> str:
    high, medium = C.TRUST_BANDS
    return "High" if score >= high else "Medium" if score >= medium else "Low"


def assess_drift(
    windows: pd.DataFrame,
    reference: dict[str, Any] | None = None,
    io_warnings: Iterable[str] | None = None,
) -> DriftReport:
    """Compare a ride's window features with the training statistics.

    Score = 100 minus penalties for (1) features outside the training 5th-95th percentile band beyond the
    10% expected by chance, (2) a shifted feature median (|z|), (3) roughness above the 95th percentile
    of training smooth road, and (4) input-quality warnings. The result is always within 0-100.
    """
    ref = reference or load_reference()
    feats = ref["features"]
    rows = []
    for name in C.FEATURE_ORDER:
        s = feats[name]
        x = windows[name].to_numpy(dtype=float) if name in windows and len(windows) else np.array([])
        if len(x) == 0:
            rows.append({"feature": name, "label": C.FEATURE_LABELS[name], "ride_median": np.nan,
                         "ride_mean": np.nan, "ref_mean": s["mean"], "ref_std": s["std"], "z": 0.0,
                         "out_of_range_pct": 0.0, "p05": s["p05"], "p95": s["p95"]})
            continue
        med = float(np.median(x))
        z = (med - s["mean"]) / (s["std"] if s["std"] > 1e-12 else 1.0)
        oor = float(np.mean((x < s["p05"]) | (x > s["p95"])) * 100)
        rows.append({"feature": name, "label": C.FEATURE_LABELS[name], "ride_median": med,
                     "ride_mean": float(np.mean(x)), "ref_mean": s["mean"], "ref_std": s["std"],
                     "z": float(z), "out_of_range_pct": oor, "p05": s["p05"], "p95": s["p95"]})
    per = pd.DataFrame(rows)

    n = int(len(windows))
    rough_pct = roughness_percentile(float(np.median(windows["z_std"])), ref) if n else 50.0
    oor_share = float(per["out_of_range_pct"].mean() / 100)
    shift = float(np.mean(np.minimum(per["z"].abs(), 3.0)))

    pen_oor = float(np.clip((oor_share - 0.10) / 0.30, 0, 1) * 45)
    pen_shift = float(np.clip((shift - 0.15) / 1.35, 0, 1) * 35)
    pen_rough = float(min(20.0, max(0.0, rough_pct - 95.0) * 4))
    warns = [w for w in (io_warnings or []) if any(k in w.lower() for k in _QUALITY_KEYWORDS)]
    pen_io = float(min(15, 5 * len(warns)))
    score = float(np.clip(100.0 - pen_oor - pen_shift - pen_rough - pen_io, 0.0, 100.0))
    if n < 10:
        score = min(score, C.TRUST_BANDS[0] - 1.0)   # a very short ride can never be "High"
    label = _label_for(score)

    reasons: list[str] = []
    if n < 10:
        reasons.append(f"Only {n} window(s) of data, so this score is uncertain.")
    if rough_pct >= 90:
        reasons.append(f"This ride is rougher than {rough_pct:.0f}% of training smooth road, "
                       "so expect more false alarms (smooth road flagged as events).")
    elif rough_pct <= 10:
        reasons.append(f"This ride is smoother than {100 - rough_pct:.0f}% of training smooth road, "
                       "so subtle events may be missed.")
    sp = per.set_index("feature").loc[KEY_SPEED_FEATURE]
    if abs(sp["z"]) > 1.0 and not np.isnan(sp["ride_median"]):
        direction = "slower" if sp["z"] < 0 else "faster"
        reasons.append(f"Typical speed ({sp['ride_median']:.0f} km/h) is {direction} than most training windows "
                       f"(about {sp['p05']:.0f} to {sp['p95']:.0f} km/h), so speed-based cues may mislead the model.")
    others = per[(per["feature"] != KEY_SPEED_FEATURE) & ((per["z"].abs() > 1.5) | (per["out_of_range_pct"] > 50))]
    for _, r in others.reindex(others["z"].abs().sort_values(ascending=False).index).head(3).iterrows():
        reasons.append(f"The {r['label']} is unusual: {r['out_of_range_pct']:.0f}% of windows fall outside the training range.")
    reasons.extend(f"Input check: {w}" for w in warns[:3])
    if not reasons:
        reasons.append("This ride looks similar to the data the model was trained on.")

    if label == "High":
        mode = "Balanced"
    elif rough_pct <= 10:
        mode = "Municipal Audit"
    else:
        mode = "Driver Alert"
    return DriftReport(score=score, label=label, roughness_percentile=float(rough_pct), n_windows=n,
                       per_feature=per, reasons=reasons, recommended_mode=mode)
