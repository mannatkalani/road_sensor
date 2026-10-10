"""Tests: io (reading, validation, mapping, template) and export."""
from __future__ import annotations

import io
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from core import config as C
from core import export
from core import io as rio


def valid_df(n: int = 200, hz: float = 20.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ride_id": "r1", "time_s": np.arange(n) / hz,
        "lat": 18.5 + np.arange(n) * 1e-6, "lon": 73.8 + np.arange(n) * 1e-6,
        "speed_kmh": 25 + rng.normal(0, 1, n),
        "acc_x": rng.normal(0, 0.2, n), "acc_y": rng.normal(0, 0.2, n), "acc_z": 9.81 + rng.normal(0, 0.5, n),
        "gyro_x": rng.normal(0, 0.03, n), "gyro_y": rng.normal(0, 0.03, n), "gyro_z": rng.normal(0, 0.03, n),
    })


# ------------------------------------------------------------------ reading
def test_read_table_csv_bytes_path_and_semicolon(tmp_path):
    df = valid_df()
    raw = df.to_csv(index=False).encode()
    assert rio.read_table(raw, "x.csv").shape == df.shape
    (tmp_path / "r.csv").write_bytes(raw)
    assert rio.read_table(tmp_path / "r.csv").shape == df.shape
    semi = df.to_csv(index=False, sep=";").encode()
    assert rio.read_table(semi, "semi.csv").shape == df.shape


def test_read_table_xlsx_roundtrip():
    df = valid_df(50)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    assert rio.read_table(buf, "ride.xlsx").shape == df.shape


def test_read_table_friendly_errors():
    with pytest.raises(rio.InputError, match="Unsupported"):
        rio.read_table(b"abc", "x.pdf")
    with pytest.raises(rio.InputError, match=r"\.xlsx"):
        rio.read_table(b"abc", "x.xls")
    with pytest.raises(rio.InputError):
        rio.read_table(b"", "empty.csv")
    with pytest.raises(rio.InputError, match="no data rows"):
        rio.read_table(b"a,b,c\n", "header_only.csv")


# ------------------------------------------------------------------ validation
def test_valid_file_passes_with_no_errors():
    df, rep = rio.prepare(valid_df(), "ride.csv")
    assert rep.ok and not rep.errors
    assert rep.info["n_rides"] == 1 and abs(rep.info["sampling_hz"] - 20) < 0.5
    assert df["ride_id"].iloc[0] == "r1"


def test_missing_column_error_names_the_column():
    df = valid_df().drop(columns=["gyro_z"])
    _, rep = rio.prepare(df, "ride.csv")
    assert not rep.ok and "gyro_z" in rep.errors[0]


def test_alias_columns_are_auto_mapped():
    df = valid_df().rename(columns={"acc_x": "AccelerometerX", "lat": "Latitude", "time_s": "timestamp", "gyro_y": "gy"})
    out, rep = rio.prepare(df, "ride.csv")
    assert rep.ok and {"acc_x", "lat", "time_s", "gyro_y"} <= set(out.columns)
    assert any("Interpreted columns" in w for w in rep.warnings)


def test_unit_problems_are_detected_and_fixable():
    d = valid_df()
    d["acc_x"], d["acc_y"], d["acc_z"] = d["acc_x"] / 9.81, d["acc_y"] / 9.81, d["acc_z"] / 9.81   # in g
    out, rep = rio.prepare(d, "g.csv")
    assert "acc_g_to_ms2" in rep.suggested_fixes and any("units of g" in w for w in rep.warnings)
    fixed = rio.apply_unit_fixes(out, ["acc_g_to_ms2"])
    assert 9.0 < fixed["acc_z"].mean() < 10.5

    d = valid_df()
    d[["gyro_x", "gyro_y", "gyro_z"]] = np.random.default_rng(1).normal(0, 1.0, (len(d), 3)) * 57.3
    assert "gyro_deg_to_rad" in rio.prepare(d, "deg.csv")[1].suggested_fixes

    d = valid_df()
    d["speed_kmh"] = 5 + np.random.default_rng(2).normal(0, 0.3, len(d))
    assert "speed_ms_to_kmh" in rio.prepare(d, "ms.csv")[1].suggested_fixes
    with pytest.raises(rio.InputError):
        rio.apply_unit_fixes(valid_df(), ["bogus"])


def test_gravity_removed_and_orientation_warnings():
    d = valid_df()
    d["acc_z"] = d["acc_z"] - 9.81
    assert any("gravity" in w for w in rio.prepare(d, "lin.csv")[1].warnings)
    d = valid_df()
    d["acc_z"] = -9.81 + d["acc_z"] * 0
    assert any("z-axis" in w for w in rio.prepare(d, "flip.csv")[1].warnings)


def test_sampling_rate_checks():
    _, rep = rio.prepare(valid_df(1000, hz=50), "fast.csv")
    assert rep.ok and rep.info["needs_resample"] and any("Sampling rate" in w for w in rep.warnings)
    _, rep = rio.prepare(valid_df(300, hz=5), "slow.csv")
    assert not rep.ok and any("At least 10 Hz" in e for e in rep.errors)


def test_short_ride_and_bad_labels_are_errors():
    _, rep = rio.prepare(valid_df(30), "short.csv")
    assert not rep.ok and any("only 30 rows" in e for e in rep.errors)
    d = valid_df()
    d["label"] = 3
    assert not rio.prepare(d, "bad.csv")[1].ok
    d["label"] = np.where(np.arange(len(d)) % 50 == 0, np.nan, 0)
    out, rep = rio.prepare(d, "partial.csv")
    assert rep.ok and rep.info["has_labels"] and out["label"].dtype.kind == "i"


def test_gps_is_optional_but_validated():
    out, rep = rio.prepare(valid_df().drop(columns=["lat", "lon"]), "nogps.csv")
    assert rep.ok and out["lat"].isna().all() and any("No 'lat' column" in w for w in rep.warnings)
    d = valid_df()
    d["lat"] = np.nan
    assert any("no GPS fixes" in w for w in rio.prepare(d, "x.csv")[1].warnings)
    d = valid_df()
    d.loc[5, "lat"] = 123.0
    assert not rio.prepare(d, "x.csv")[1].ok


def test_ride_id_defaults_to_file_name():
    out, rep = rio.prepare(valid_df().drop(columns=["ride_id"]), "morning_commute.csv")
    assert rep.ok and (out["ride_id"] == "morning_commute").all()


def test_template_csv_is_valid_and_labelled():
    df = rio.read_table(rio.make_template_csv(), "template.csv")
    out, rep = rio.prepare(df, "template.csv")
    assert rep.ok and rep.info["has_labels"] and list(df.columns) == C.TEMPLATE_COLUMNS


def test_column_mapping_suggestions_and_apply():
    cols = ["Time", "Latitude", "Longitude", "speed", "ax", "ay", "az", "gx", "gy", "gz"]
    m = rio.suggest_column_mapping(cols)
    assert m["acc_x"] == "ax" and m["gyro_z"] == "gz" and m["time_s"] == "Time" and m["speed_kmh"] == "speed"
    assert len({v for v in m.values() if v}) == len([v for v in m.values() if v])      # no column used twice
    df = pd.DataFrame(np.zeros((3, len(cols))), columns=cols)
    assert {"acc_x", "lat", "lon"} <= set(rio.apply_column_mapping(df, m).columns)


# ------------------------------------------------------------------ export
def sample_ranked() -> pd.DataFrame:
    return pd.DataFrame([
        {"rank": 1, "spot_id": "P001", "class_name": "Pothole", "priority": 72.0, "lat": 18.5, "lon": 73.8,
         "why": "Severe pothole, confirmed by 3 of 3 ride(s) at about 15 km/h."},
        {"rank": 2, "spot_id": "P002", "class_name": "Pothole", "priority": 40.0, "lat": 18.51, "lon": 73.81,
         "why": "Moderate pothole, confirmed by 2 of 3 ride(s) at about 22 km/h."},
        {"rank": 3, "spot_id": "B001", "class_name": "Speed breaker", "priority": 10.0, "lat": np.nan, "lon": np.nan,
         "why": "Mild speed breaker."},
    ])


def test_geojson_structure_and_coordinate_order():
    gj = json.loads(export.spots_to_geojson(sample_ranked()))
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 2          # row without GPS skipped
    f = gj["features"][0]
    assert f["geometry"]["coordinates"] == [73.8, 18.5]                               # [lon, lat]
    assert f["properties"]["spot_id"] == "P001"
    json.dumps(gj)                                                                    # fully serialisable


def test_csv_and_nan_handling():
    assert export.to_csv_bytes(sample_ranked()).decode().startswith("rank,spot_id")
    ev = pd.DataFrame([{"event_id": "e", "ride_id": "r", "class_name": "Pothole", "confidence": 0.9, "severity": 50.0,
                        "severity_label": "Medium", "speed_kmh": np.nan, "peak_time": 1.0, "lat": 1.0, "lon": 2.0}])
    assert json.loads(export.events_to_geojson(ev))["features"][0]["properties"]["speed_kmh"] is None


def test_google_maps_links():
    pin = export.google_maps_pin_link(18.5, 73.8)
    assert pin == "https://www.google.com/maps/search/?api=1&query=18.500000,73.800000"
    pts = [(18.5, 73.8), (18.51, 73.81), (18.52, 73.82), (18.53, 73.83)]
    url = export.google_maps_route_link(pts)
    assert "origin=18.500000,73.800000" in url and "destination=18.530000,73.830000" in url
    assert "waypoints=18.510000,73.810000%7C18.520000,73.820000" in url and url.endswith("&waypoints=18.510000,73.810000%7C18.520000,73.820000")
    assert export.google_maps_route_link(pts[:1]) == export.google_maps_pin_link(*pts[0])
    many = [(18.5 + i * 0.001, 73.8) for i in range(30)]
    assert export.google_maps_route_link(many).count("%7C") == export.MAX_WAYPOINTS - 1
    with pytest.raises(ValueError):
        export.google_maps_route_link([])


def test_nearest_neighbour_order_keeps_first_point():
    pts = [(0.0, 0.0), (10.0, 0.0), (1.0, 0.0), (5.0, 0.0)]
    assert export.nearest_neighbour_order(pts) == [(0.0, 0.0), (1.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    assert export.nearest_neighbour_order([]) == []


def test_complaint_letter_content():
    ranked = sample_ranked().dropna(subset=["lat"])
    text = export.complaint_letter(ranked, area="Baner Road", sender="A. Citizen", n_rides=3, today=date(2026, 1, 5))
    assert "05 January 2026" in text and "Baner Road" in text and "A. Citizen" in text
    assert "1. Pothole (priority 72/100)" in text and "https://www.google.com/maps/search/?api=1&query=18.500000,73.800000" in text
    assert "verify each location on site" in text and "SIMULATED" not in text
    sim = export.complaint_letter(ranked, data_source="simulated sample rides", today=date(2026, 1, 5))
    assert "SIMULATED demo data" in sim
