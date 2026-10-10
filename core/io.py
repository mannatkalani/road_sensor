"""Reading, validating and repairing uploaded ride files.

Pure logic (no UI). ``render_column_mapping_ui`` imports Streamlit lazily so that the rest of
this module stays importable and testable without a Streamlit runtime.
"""
from __future__ import annotations

import difflib
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from core import config as C


class InputError(ValueError):
    """Raised for problems the user can fix (bad file type, empty file, unreadable content)."""


# ------------------------------------------------------------------ column aliases
COLUMN_ALIASES: dict[str, list[str]] = {
    "time_s": ["time", "timestamp", "seconds", "sec", "elapsed", "elapsed_s", "t", "time_sec"],
    "lat": ["latitude", "gps_lat", "gpslatitude", "locationlatitude"],
    "lon": ["lng", "long", "longitude", "gps_lon", "gpslongitude", "locationlongitude"],
    "speed_kmh": ["speed", "speedkmh", "kmh", "speed_kph", "kph", "gps_speed", "gpsspeed"],
    "acc_x": ["accx", "ax", "accel_x", "accelerometer_x", "accelerationx", "acceleration_x"],
    "acc_y": ["accy", "ay", "accel_y", "accelerometer_y", "accelerationy", "acceleration_y"],
    "acc_z": ["accz", "az", "accel_z", "accelerometer_z", "accelerationz", "acceleration_z"],
    "gyro_x": ["gyrox", "gx", "gyroscope_x", "gyroscopex", "rotation_rate_x", "rotationratex"],
    "gyro_y": ["gyroy", "gy", "gyroscope_y", "gyroscopey", "rotation_rate_y", "rotationratey"],
    "gyro_z": ["gyroz", "gz", "gyroscope_z", "gyroscopez", "rotation_rate_z", "rotationratez"],
    "ride_id": ["ride", "rideid", "trip", "trip_id", "session"],
    "route_id": ["route", "routeid"],
    "label": ["labels", "class", "target", "event"],
}
MAPPABLE_COLUMNS: list[str] = [*C.REQUIRED_COLUMNS, *C.GPS_COLUMNS, *C.OPTIONAL_COLUMNS]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# ------------------------------------------------------------------ report
@dataclass
class ValidationReport:
    """Result of validating one uploaded file."""

    filename: str = "upload"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def suggested_fixes(self) -> list[str]:
        return list(self.info.get("suggested_fixes", []))


UNIT_FIXES: dict[str, str] = {
    "acc_g_to_ms2": "Multiply accelerometer values by 9.81 (they look like units of g).",
    "gyro_deg_to_rad": "Convert gyroscope values from degrees/s to radians/s.",
    "speed_ms_to_kmh": "Multiply speed by 3.6 (it looks like metres per second).",
}


def apply_unit_fixes(df: pd.DataFrame, fixes: Iterable[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with the requested unit conversions applied."""
    out = df.copy()
    for fix in fixes:
        if fix == "acc_g_to_ms2":
            for c in ("acc_x", "acc_y", "acc_z"):
                out[c] = out[c] * C.GRAVITY
        elif fix == "gyro_deg_to_rad":
            for c in ("gyro_x", "gyro_y", "gyro_z"):
                out[c] = np.deg2rad(out[c])
        elif fix == "speed_ms_to_kmh":
            out["speed_kmh"] = out["speed_kmh"] * 3.6
        else:
            raise InputError(f"Unknown unit fix '{fix}'.")
    return out


# ------------------------------------------------------------------ reading
def read_table(source: Any, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV or .xlsx upload (path, bytes or file-like) into a DataFrame."""
    name = filename or getattr(source, "name", None) or (
        str(source) if isinstance(source, (str, Path)) else "upload.csv"
    )
    ext = Path(str(name)).suffix.lower()
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(bytes(source))

    def _rewind() -> None:
        if hasattr(source, "seek"):
            source.seek(0)

    try:
        if ext in {".xlsx", ".xlsm"}:
            df = pd.read_excel(source, engine="openpyxl")
        elif ext == ".xls":
            raise InputError("Old .xls files are not supported. Save the file as .xlsx or .csv.")
        elif ext in {".csv", ".txt", ".tsv", ""}:
            try:
                df = pd.read_csv(source)
            except pd.errors.ParserError:
                _rewind()
                df = pd.read_csv(source, sep=None, engine="python")
            if df.shape[1] == 1:  # wrong delimiter (semicolon / tab)
                _rewind()
                df = pd.read_csv(source, sep=None, engine="python")
        else:
            raise InputError(f"Unsupported file type '{ext}'. Upload a .csv or .xlsx file.")
    except InputError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise InputError("The file is empty.") from exc
    except Exception as exc:  # noqa: BLE001 - convert to a friendly message
        raise InputError(f"Could not read '{name}'. Check that it is a valid CSV or Excel file.") from exc
    if df.empty:
        raise InputError("The file has headers but no data rows.")
    return df


# ------------------------------------------------------------------ column mapping
def suggest_column_mapping(columns: Iterable[str]) -> dict[str, str | None]:
    """Best-guess mapping from RoadPulse column names to the columns of an uploaded file."""
    cols = [str(c) for c in columns]
    normed = {c: _norm(c) for c in cols}
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    for target in MAPPABLE_COLUMNS:
        match: str | None = None
        wanted = {_norm(target), *(_norm(a) for a in COLUMN_ALIASES.get(target, []))}
        for c in cols:  # exact / alias match
            if c not in used and normed[c] in wanted:
                match = c
                break
        if match is None:  # fuzzy match
            pool = [normed[c] for c in cols if c not in used]
            close = difflib.get_close_matches(_norm(target), pool, n=1, cutoff=0.8)
            if close:
                match = next(c for c in cols if c not in used and normed[c] == close[0])
        if match is not None:
            used.add(match)
        mapping[target] = match
    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: Mapping[str, str | None]) -> pd.DataFrame:
    """Rename columns according to ``mapping`` (target -> source). Unmapped targets are skipped."""
    rename: dict[str, str] = {}
    for target, source in mapping.items():
        if source and source in df.columns and source != target:
            rename[source] = target
    out = df.drop(columns=[t for t in rename.values() if t in df.columns], errors="ignore")
    return out.rename(columns=rename)


def render_column_mapping_ui(columns: Iterable[str], key_prefix: str = "map") -> dict[str, str | None]:
    """Render selectboxes (Streamlit) that let the user map their columns. Returns the mapping."""
    import streamlit as st  # lazy: keep core importable without Streamlit

    cols = list(columns)
    suggestion = suggest_column_mapping(cols)
    options = ["- not in my file -", *cols]
    mapping: dict[str, str | None] = {}
    for target in MAPPABLE_COLUMNS:
        default = suggestion.get(target)
        index = options.index(default) if default in options else 0
        choice = st.selectbox(target, options, index=index, key=f"{key_prefix}_{target}")
        mapping[target] = None if choice == options[0] else choice
    return mapping


# ------------------------------------------------------------------ validation
def _hint_for_missing(target: str, columns: list[str]) -> str:
    close = difflib.get_close_matches(target, columns, n=1, cutoff=0.5)
    return f" (did you mean '{close[0]}'?)" if close else ""


def prepare(
    df: pd.DataFrame,
    filename: str = "upload.csv",
    mapping: Mapping[str, str | None] | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Normalize and validate a raw ride table.

    Returns the cleaned frame (numeric columns coerced, ``ride_id`` filled, labels as int) and a
    :class:`ValidationReport`. When ``report.ok`` is False the frame must not be analysed.
    """
    rep = ValidationReport(filename=filename)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if mapping:
        df = apply_column_mapping(df, mapping)

    # auto-map confident matches for columns that are still missing
    suggestion = suggest_column_mapping(list(df.columns))
    renames = {src: tgt for tgt, src in suggestion.items()
               if src and src != tgt and tgt not in df.columns}
    if renames:
        df = df.rename(columns=renames)
        rep.warnings.append(
            "Interpreted columns: " + ", ".join(f"'{s}' as {t}" for s, t in renames.items()) + "."
        )

    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        cols = list(df.columns)
        rep.errors.append(
            "Missing required column(s): "
            + ", ".join(m + _hint_for_missing(m, cols) for m in missing)
            + ". Rename them or use the column-mapping tool."
        )
        return df, rep

    # GPS is optional: keep the columns but allow NaN
    for c in C.GPS_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
            rep.warnings.append(f"No '{c}' column found. Maps and cross-ride matching are disabled.")

    # identifiers
    stem = Path(filename).stem or "ride"
    if "ride_id" not in df.columns:
        df["ride_id"] = stem
    df["ride_id"] = df["ride_id"].fillna(stem).astype(str)
    if "route_id" in df.columns:
        df["route_id"] = df["route_id"].astype(str)

    # numeric coercion
    numeric = [*C.REQUIRED_COLUMNS, *C.GPS_COLUMNS]
    bad_cells = 0
    for c in numeric:
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad_cells += int(df[c].isna().sum() - before)
    if bad_cells:
        rep.warnings.append(f"{bad_cells} non-numeric cell(s) were treated as missing.")

    n_before = len(df)
    df = df.dropna(subset=["time_s"]).reset_index(drop=True)
    if len(df) < n_before:
        rep.warnings.append(f"Dropped {n_before - len(df)} row(s) with no timestamp.")
    if df.empty:
        rep.errors.append("No usable rows after reading the file.")
        return df, rep

    sensor_nan = df[C.SENSOR_COLUMNS + ["speed_kmh"]].isna().mean().max()
    if sensor_nan > 0.2:
        rep.errors.append(
            f"A sensor column is missing in {sensor_nan:.0%} of rows. At most 20% is allowed."
        )
    elif sensor_nan > 0:
        rep.warnings.append(
            f"Some sensor readings are missing ({sensor_nan:.1%}). Affected windows will be skipped."
        )

    # labels
    has_labels = False
    if "label" in df.columns:
        lab = pd.to_numeric(df["label"], errors="coerce")
        if lab.notna().any():
            if lab.dropna().isin([0, 1, 2]).all():
                if lab.isna().any():
                    rep.warnings.append("Some labels were empty and were treated as 0 (smooth).")
                df["label"] = lab.fillna(0).astype(int)
                has_labels = True
            else:
                rep.errors.append("The 'label' column may only contain 0 (smooth), 1 (speed breaker) or 2 (pothole).")
        else:
            df = df.drop(columns=["label"])

    # per-ride checks
    rides: list[dict[str, Any]] = []
    needs_resample = False
    for rid, g in df.groupby("ride_id", sort=False):
        t = np.sort(g["time_s"].to_numpy(dtype=float))
        dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        dt = np.diff(t)
        dt = dt[dt > 0]
        hz = float(1.0 / np.median(dt)) if len(dt) else float("nan")
        rides.append({"ride_id": rid, "rows": int(len(g)), "duration_s": dur, "sampling_hz": hz})
        if len(g) < C.WINDOW_SIZE:
            rep.errors.append(
                f"Ride '{rid}' has only {len(g)} rows. At least {C.WINDOW_SIZE} readings (2 s at 20 Hz) are needed."
            )
        if len(t) > 1 and (np.diff(np.sort(g['time_s'].to_numpy())) == 0).any():
            rep.warnings.append(f"Ride '{rid}' has duplicate timestamps; duplicates will be removed.")
        if math.isfinite(hz):
            if hz < 10:
                rep.errors.append(f"Ride '{rid}' is sampled at about {hz:.1f} Hz. At least 10 Hz is required.")
            elif abs(hz - C.SAMPLING_HZ) > 1.0:
                needs_resample = True
    if needs_resample:
        rep.warnings.append(
            "Sampling rate is not 20 Hz. The model expects 20 Hz, so the data should be resampled "
            "(Upload & Analyze can do this for you)."
        )

    # unit checks
    fixes: list[str] = []
    acc_z_mean = float(df["acc_z"].mean())
    if 0.8 <= abs(acc_z_mean) <= 1.2 and df["acc_z"].abs().quantile(0.99) < 4:
        rep.warnings.append("Accelerometer looks like units of g (mean of acc_z is about 1).")
        fixes.append("acc_g_to_ms2")
    elif abs(acc_z_mean) < 2.0:
        rep.warnings.append(
            f"acc_z averages {acc_z_mean:.1f} m/s², so gravity may already be removed. "
            "RoadPulse expects raw accelerometer values including gravity (about 9.8)."
        )
    elif not 8.5 <= acc_z_mean <= 11.0:
        rep.warnings.append(
            f"acc_z averages {acc_z_mean:.1f} m/s² instead of about 9.8. "
            "The model assumes the phone lies flat with the z-axis pointing up."
        )
    gyro_p99 = float(df[["gyro_x", "gyro_y", "gyro_z"]].abs().stack().quantile(0.99))
    if gyro_p99 > 35:
        rep.warnings.append("Gyroscope values are very large. They may be in degrees/s instead of rad/s.")
        fixes.append("gyro_deg_to_rad")
    if df["speed_kmh"].quantile(0.99) < 8 and df["speed_kmh"].max() > 0:
        rep.warnings.append("Speed values are small. They may be in m/s instead of km/h.")
        fixes.append("speed_ms_to_kmh")

    # GPS checks
    gps_missing = float(df[C.GPS_COLUMNS].isna().any(axis=1).mean())
    if gps_missing >= 0.999:
        rep.warnings.append("The file contains no GPS fixes. Maps and cross-ride matching are disabled.")
    else:
        lat_bad = (df["lat"].abs() > 90).any()
        lon_bad = (df["lon"].abs() > 180).any()
        if lat_bad or lon_bad:
            rep.errors.append("GPS values are out of range (latitude must be -90..90, longitude -180..180).")
        elif gps_missing > 0:
            rep.warnings.append(f"{gps_missing:.1%} of GPS fixes are missing and will be interpolated.")
        if ((df["lat"] == 0) & (df["lon"] == 0)).mean() > 0.05:
            rep.warnings.append("Many GPS points are exactly (0, 0), which usually means no fix.")

    rep.info.update(
        rows=int(len(df)),
        rides=rides,
        n_rides=len(rides),
        duration_s=float(sum(r["duration_s"] for r in rides)),
        sampling_hz=float(np.nanmedian([r["sampling_hz"] for r in rides])) if rides else float("nan"),
        acc_z_mean=acc_z_mean,
        gps_missing_pct=gps_missing * 100,
        has_labels=has_labels,
        needs_resample=needs_resample,
        suggested_fixes=fixes,
    )
    return df, rep


def load_and_validate(
    source: Any,
    filename: str | None = None,
    mapping: Mapping[str, str | None] | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Convenience: :func:`read_table` followed by :func:`prepare`."""
    name = filename or getattr(source, "name", None) or (
        str(source) if isinstance(source, (str, Path)) else "upload.csv"
    )
    df = read_table(source, filename=name)
    return prepare(df, filename=Path(str(name)).name, mapping=mapping)


# ------------------------------------------------------------------ template
def make_template_csv(seconds: int = 12, seed: int = 0) -> bytes:
    """A small, valid example file (20 Hz) users can download and fill with their own data."""
    rng = np.random.default_rng(seed)
    n = int(seconds * C.SAMPLING_HZ)
    t = np.arange(n) / C.SAMPLING_HZ
    speed = 24 + rng.normal(0, 0.6, n)
    az = C.GRAVITY + rng.normal(0, 0.45, n)
    label = np.zeros(n, dtype=int)
    start = int(5 * C.SAMPLING_HZ)
    k = np.arange(8)
    az[start:start + 8] += -6.0 * np.sin(2 * np.pi * k / 8)   # pothole-like dip then rebound
    label[start:start + 8] = 2
    df = pd.DataFrame({
        "ride_id": "TEMPLATE", "route_id": "T", "time_s": t.round(2),
        "lat": (18.5204 + np.cumsum(speed / 3.6 / C.SAMPLING_HZ) / 111_320).round(6),
        "lon": 73.8567,
        "speed_kmh": speed.round(1),
        "acc_x": rng.normal(0, 0.2, n).round(3), "acc_y": rng.normal(0, 0.15, n).round(3),
        "acc_z": az.round(3),
        "gyro_x": rng.normal(0, 0.03, n).round(4), "gyro_y": rng.normal(0, 0.03, n).round(4),
        "gyro_z": rng.normal(0, 0.03, n).round(4),
        "label": label,
    })[C.TEMPLATE_COLUMNS]
    return df.to_csv(index=False).encode("utf-8")
