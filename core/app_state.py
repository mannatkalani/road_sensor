"""Streamlit-aware glue: cached pipeline wrappers, session state and file ingestion.

Pages never run analysis themselves; they call the functions in this module (which call the
framework-free modules in ``core/``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st

from core import config as C
from core import consensus, drift, events, explain, features, inference, io, preprocess

DATASET_KEY, MODE_KEY, THRESHOLDS_KEY = "rp_dataset", "rp_mode", "rp_thresholds"
CUSTOM_MODE = "Custom"
DEFAULT_MODE = "Balanced"


# ------------------------------------------------------------------ data classes
@dataclass
class FileResult:
    """Outcome of reading and validating one uploaded file."""

    name: str
    raw: pd.DataFrame | None = None
    df: pd.DataFrame | None = None
    report: io.ValidationReport | None = None
    read_error: str | None = None
    fixes_applied: list[str] = field(default_factory=list)
    resampled: bool = False

    @property
    def ok(self) -> bool:
        return self.df is not None and self.report is not None and self.report.ok

    @property
    def needs_mapping(self) -> bool:
        return self.raw is not None and self.report is not None and any(
            "Missing required column" in e for e in self.report.errors)


@dataclass
class Dataset:
    """Rides currently loaded in the session."""

    df: pd.DataFrame
    source: str                       # "sample" | "upload" | "demo"
    names: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def rides(self) -> list[str]:
        return list(self.df["ride_id"].drop_duplicates())

    @property
    def has_labels(self) -> bool:
        return "label" in self.df.columns

    @property
    def data_source_text(self) -> str:
        return {"sample": "simulated sample rides analysed with RoadPulse",
                "demo": "simulated repeat passes of a sample ride analysed with RoadPulse"}.get(
            self.source, "smartphone sensor rides analysed with RoadPulse")


@dataclass
class Analysis:
    """Everything produced by running the pipeline on a set of rides at given thresholds."""

    pre: pd.DataFrame
    windows: pd.DataFrame
    proba: np.ndarray
    thresholds: dict[int, float]
    pred_windows: pd.DataFrame
    events: pd.DataFrame
    km: float
    duration_s: float
    ground_truth: pd.DataFrame
    metrics: events.EventMetrics | None
    est_false_alarms: float

    @property
    def has_labels(self) -> bool:
        return self.metrics is not None

    def for_ride(self, ride_id: str) -> "Analysis":
        """The same analysis restricted to a single ride."""
        pre = self.pre[self.pre["ride_id"] == ride_id]
        mask = (self.windows["ride_id"] == ride_id).to_numpy()
        return _assemble(pre, self.windows[mask].reset_index(drop=True), self.proba[mask], self.thresholds)


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner=False)
def get_bundle() -> inference.ModelBundle:
    return inference.load_bundle()


@st.cache_resource(show_spinner=False)
def get_reference() -> dict[str, Any]:
    return drift.load_reference()


@st.cache_data(show_spinner=False)
def load_csv_asset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Extracting window features")
def _prep_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = preprocess.preprocess(df)
    return pre, features.build_windows(pre)


@st.cache_data(show_spinner="Scoring windows")
def _score(windows: pd.DataFrame) -> np.ndarray:
    return inference.predict_proba(get_bundle(), windows)


# ------------------------------------------------------------------ analysis
def _assemble(pre: pd.DataFrame, windows: pd.DataFrame, proba: np.ndarray, thresholds: Mapping[Any, float]) -> Analysis:
    th = {int(k): float(v) for k, v in {**C.DEFAULT_THRESHOLDS, **dict(thresholds)}.items()}
    pw = windows.copy().reset_index(drop=True)
    pw["p_smooth"], pw["p_speed_breaker"], pw["p_pothole"] = proba[:, 0], proba[:, 1], proba[:, 2]
    pw["pred"] = inference.apply_thresholds(proba, th)
    pw["pred_name"] = [C.CLASS_NAMES[i] for i in pw["pred"]]
    pw["confidence"] = proba[np.arange(len(pw)), pw["pred"].to_numpy()] if len(pw) else np.array([])
    ev = events.merge_windows_to_events(pw, pre)
    km = events.distance_km(pre) if len(pre) else 0.0
    has_labels = "label" in pre.columns
    gt = events.ground_truth_events(pre) if has_labels else events.ground_truth_events(pre.assign(label=0))
    metrics = events.evaluate_events(ev, gt, km) if has_labels else None
    return Analysis(pre=pre, windows=windows.reset_index(drop=True), proba=proba, thresholds=th, pred_windows=pw,
                    events=ev, km=km, duration_s=events.duration_s(pre) if len(pre) else 0.0, ground_truth=gt,
                    metrics=metrics, est_false_alarms=events.estimate_false_alarms(ev))


def analyze(df: pd.DataFrame, thresholds: Mapping[Any, float] | None = None) -> Analysis:
    """Run the full pipeline. Feature building and scoring are cached; thresholds apply instantly."""
    pre, windows = _prep_windows(df)
    return _assemble(pre, windows, _score(windows), thresholds or get_thresholds())


def threshold_sweep(base: Analysis, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Events, false alarms per km and (with labels) recall/precision across a symmetric threshold grid."""
    rows = []
    for t in (grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)):
        a = _assemble(base.pre, base.windows, base.proba, {1: t, 2: t})
        km = max(a.km, 1e-9)
        row = {"threshold": float(t), "events": len(a.events), "est_false_alarms_per_km": a.est_false_alarms / km,
               "recall": np.nan, "precision": np.nan, "false_alarms_per_km": a.est_false_alarms / km}
        if a.metrics is not None:
            row.update(recall=a.metrics.summary["recall"], precision=a.metrics.summary["precision"],
                       false_alarms_per_km=a.metrics.summary["false_alarms_per_km"])
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Sweeping thresholds")
def sweep_for(df: pd.DataFrame) -> pd.DataFrame:
    """Cached threshold sweep for a set of rides."""
    pre, windows = _prep_windows(df)
    return threshold_sweep(_assemble(pre, windows, _score(windows), C.DEFAULT_THRESHOLDS))


def mode_comparison(a: Analysis) -> pd.DataFrame:
    """Events, false alarms per km and (with labels) recall/precision for every preset mode."""
    rows = []
    for name, cfg in C.OPERATING_MODES.items():
        r = _assemble(a.pre, a.windows, a.proba, cfg["thresholds"])
        km = max(r.km, 1e-9)
        row = {"Mode": name, "Events detected": len(r.events),
               "False alarms per km": (r.metrics.summary["false_alarms_per_km"] if r.metrics else r.est_false_alarms / km),
               "Recall": r.metrics.summary["recall"] if r.metrics else np.nan,
               "Precision": r.metrics.summary["precision"] if r.metrics else np.nan}
        rows.append(row)
    return pd.DataFrame(rows)


def trust_for(a: Analysis, ride_id: str | None = None, io_warnings: list[str] | None = None) -> drift.DriftReport:
    """Trust Meter report for one ride (or all rides when ``ride_id`` is None)."""
    w = a.windows if ride_id is None else a.windows[a.windows["ride_id"] == ride_id]
    return drift.assess_drift(w, get_reference(), io_warnings)


def explain_event(a: Analysis, event: pd.Series | dict[str, Any]) -> explain.EventExplanation:
    """Explanation for one detected event."""
    return explain.explain_event(get_bundle(), a.windows, event, a.pre, get_reference())


# ------------------------------------------------------------------ session state
def get_dataset() -> Dataset | None:
    return st.session_state.get(DATASET_KEY)


def set_dataset(ds: Dataset | None) -> None:
    st.session_state[DATASET_KEY] = ds


def get_mode() -> str:
    return st.session_state.get(MODE_KEY, DEFAULT_MODE)


def get_thresholds() -> dict[int, float]:
    return dict(st.session_state.get(THRESHOLDS_KEY, C.OPERATING_MODES[DEFAULT_MODE]["thresholds"]))


def set_mode(name: str) -> None:
    """Switch to a preset operating mode."""
    st.session_state[MODE_KEY] = name
    st.session_state[THRESHOLDS_KEY] = dict(C.OPERATING_MODES[name]["thresholds"])


def set_custom_thresholds(speed_breaker: float, pothole: float) -> None:
    st.session_state[MODE_KEY] = CUSTOM_MODE
    st.session_state[THRESHOLDS_KEY] = {C.SPEED_BREAKER: float(speed_breaker), C.POTHOLE: float(pothole)}


# ------------------------------------------------------------------ samples and uploads
def sample_catalog() -> list[dict[str, str]]:
    """Sample rides on disk with a label saying whether the model saw their route in training."""
    out = []
    for p in sorted(C.SAMPLE_DIR.glob("ride_*.csv")):
        head = pd.read_csv(p, nrows=1)
        rid = str(head["ride_id"].iloc[0]) if "ride_id" in head else p.stem.replace("ride_", "")
        route = str(head["route_id"].iloc[0]) if "route_id" in head else "?"
        note = "held-out test route D" if route == "D" else f"training route {route}"
        out.append({"name": rid, "path": str(p), "route": route, "label": f"{rid} ({note})"})
    return out


def load_sample_dataset(names: list[str]) -> Dataset:
    """Load the chosen sample rides as a dataset."""
    cat = {c["name"]: c for c in sample_catalog()}
    parts, warns = [], []
    for n in names:
        df, rep = io.prepare(io.read_table(cat[n]["path"]), filename=Path(cat[n]["path"]).name)
        parts.append(df)
        warns += rep.warnings
    return Dataset(df=pd.concat(parts, ignore_index=True), source="sample", names=list(names), warnings=warns)


def make_demo_dataset(sample_name: str, n_passes: int = 4) -> Dataset:
    """Demo: simulated repeat passes of one sample ride (clearly labelled in the UI)."""
    base = load_sample_dataset([sample_name]).df
    passes = consensus.simulate_repeat_passes(base, n_passes=n_passes)
    return Dataset(df=passes, source="demo", names=[f"{sample_name} x{n_passes} simulated passes"])


def process_upload(data: bytes, name: str, mapping: Mapping[str, str | None] | None = None,
                   apply_fixes: bool = False, resample: bool = True) -> FileResult:
    """Read, validate and (optionally) repair one uploaded file."""
    res = FileResult(name=name)
    try:
        raw = io.read_table(data, filename=name)
    except io.InputError as exc:
        res.read_error = str(exc)
        return res
    res.raw = raw
    df, rep = io.prepare(raw, filename=name, mapping=mapping)
    if rep.ok and apply_fixes and rep.suggested_fixes:
        res.fixes_applied = list(rep.suggested_fixes)
        df, rep = io.prepare(io.apply_unit_fixes(df, res.fixes_applied), filename=name)
    if rep.ok and rep.info.get("needs_resample") and resample:
        df, rep = io.prepare(preprocess.resample_uniform(df), filename=name)
        res.resampled = True
    res.df, res.report = df, rep
    return res


def combine_results(results: list[FileResult], source: str = "upload") -> Dataset | None:
    """Merge valid files into one dataset, making ride ids unique across files."""
    ok = [r for r in results if r.ok]
    if not ok:
        return None
    seen: set[str] = set()
    parts, warns = [], []
    for r in ok:
        d = r.df.copy()
        ids = list(d["ride_id"].drop_duplicates())
        if any(i in seen for i in ids):
            d["ride_id"] = f"{Path(r.name).stem}:" + d["ride_id"].astype(str)
        seen.update(d["ride_id"])
        parts.append(d)
        warns += r.report.warnings
    return Dataset(df=pd.concat(parts, ignore_index=True), source=source, names=[r.name for r in ok], warnings=warns)


# ------------------------------------------------------------------ model card data
def model_card_data() -> dict[str, Any]:
    """Tables and counts shown on the Model Card page."""
    out: dict[str, Any] = {"bundle": get_bundle(), "reference": get_reference()}
    for key, path in (("comparison", C.MODEL_COMPARISON_PATH), ("cv", C.CV_RESULTS_PATH),
                      ("per_class", C.PER_CLASS_METRICS_PATH), ("windows", C.TRAINING_WINDOWS_PATH)):
        out[key] = load_csv_asset(str(path)) if Path(path).exists() else None
    return out
