"""Model loading, probability prediction and per-class thresholding."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from core import config as C
from core.features import select_model_features


class ModelLoadError(RuntimeError):
    """The model file is missing or does not have the expected structure."""


@dataclass(frozen=True)
class ModelBundle:
    """The trained classifier plus the metadata needed to use it correctly."""

    model: Any
    model_name: str
    feature_columns: tuple[str, ...]
    class_names: tuple[str, ...]
    window_size: int
    step: int
    sampling_hz: int


@lru_cache(maxsize=4)
def _load(path_str: str) -> ModelBundle:
    path = Path(path_str)
    if not path.exists():
        raise ModelLoadError(
            f"Model file not found at '{path}'. Place road_quality_model.joblib in models/ "
            "or run: python tools/make_assets.py --dataset <your csv>"
        )
    try:
        blob = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Could not read the model file '{path.name}': {exc}") from exc
    required = {"model", "model_name", "feature_columns", "class_names", "window_size", "step", "sampling_hz"}
    missing = required - set(blob)
    if missing:
        raise ModelLoadError(f"Model file is missing key(s): {sorted(missing)}")
    if (int(blob["window_size"]), int(blob["step"]), int(blob["sampling_hz"])) != (
        C.WINDOW_SIZE, C.STEP, C.SAMPLING_HZ,
    ):
        raise ModelLoadError("Model windowing (window/step/Hz) does not match the app's feature pipeline.")
    return ModelBundle(
        model=blob["model"],
        model_name=str(blob["model_name"]),
        feature_columns=tuple(blob["feature_columns"]),
        class_names=tuple(blob["class_names"]),
        window_size=int(blob["window_size"]),
        step=int(blob["step"]),
        sampling_hz=int(blob["sampling_hz"]),
    )


def load_bundle(path: str | Path | None = None) -> ModelBundle:
    """Load (and cache) the model bundle. The cache is keyed by path."""
    return _load(str(path or C.MODEL_PATH))


def predict_proba(bundle: ModelBundle, windows: pd.DataFrame) -> np.ndarray:
    """Class probabilities, shape (n_windows, 3), columns ordered Smooth, Speed breaker, Pothole."""
    if windows.empty:
        return np.zeros((0, 3))
    X = select_model_features(windows, bundle.feature_columns)
    raw = np.asarray(bundle.model.predict_proba(X))
    classes = [int(c) for c in getattr(bundle.model, "classes_", [0, 1, 2])]
    out = np.zeros((len(X), 3))
    for j, c in enumerate(classes):
        out[:, c] = raw[:, j]
    return out


def _normalize_thresholds(thresholds: Mapping[Any, float] | None) -> dict[int, float]:
    merged = dict(C.DEFAULT_THRESHOLDS)
    for k, v in (thresholds or {}).items():
        merged[int(k)] = float(v)
    return merged


def apply_thresholds(proba: np.ndarray, thresholds: Mapping[Any, float] | None = None) -> np.ndarray:
    """Flag a window as class c when p_c >= threshold[c]; if both event classes qualify, the likelier wins."""
    t = _normalize_thresholds(thresholds)
    if len(proba) == 0:
        return np.zeros(0, dtype=int)
    p1, p2 = proba[:, C.SPEED_BREAKER], proba[:, C.POTHOLE]
    ok1, ok2 = p1 >= t[C.SPEED_BREAKER], p2 >= t[C.POTHOLE]
    pred = np.zeros(len(proba), dtype=int)
    pred[ok1 & ~ok2] = C.SPEED_BREAKER
    pred[ok2 & ~ok1] = C.POTHOLE
    both = ok1 & ok2
    pred[both] = np.where(p2[both] >= p1[both], C.POTHOLE, C.SPEED_BREAKER)
    return pred


def predict_windows(
    bundle: ModelBundle,
    windows: pd.DataFrame,
    thresholds: Mapping[Any, float] | None = None,
) -> pd.DataFrame:
    """Windows plus ``p_smooth``, ``p_speed_breaker``, ``p_pothole``, ``pred``, ``pred_name``, ``confidence``."""
    proba = predict_proba(bundle, windows)
    out = windows.copy().reset_index(drop=True)
    out["p_smooth"], out["p_speed_breaker"], out["p_pothole"] = proba[:, 0], proba[:, 1], proba[:, 2]
    out["pred"] = apply_thresholds(proba, thresholds)
    out["pred_name"] = [C.CLASS_NAMES[i] for i in out["pred"]]
    out["confidence"] = proba[np.arange(len(out)), out["pred"].to_numpy()] if len(out) else np.array([])
    return out
