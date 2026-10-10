"""One-time asset builder for RoadPulse.

Reads the simulated sensor dataset and writes everything the app expects:

    models/road_quality_model.joblib   best model + metadata (dict)
    models/reference_stats.json        training-route feature statistics + class signatures
    data/training_windows.csv          window features for all rides
    data/cv_results.csv                leave-one-route-out macro-F1 per model
    data/per_class_metrics.csv         precision / recall / F1 per class (test route D)
    data/model_comparison.csv          final comparison table
    data/sample_rides/ride_R*.csv      raw sample rides for the demo

Usage:  python tools/make_assets.py --dataset path/to/pune_road_sensor_dataset.csv

The pipeline mirrors the notebook: routes A, B, C train, route D is the held-out test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import config as C  # noqa: E402
from core.features import build_windows  # noqa: E402
from core.io import prepare  # noqa: E402
from core.preprocess import preprocess  # noqa: E402

TEST_ROUTE = "D"
SAMPLE_RIDES = ["R01", "R05", "R10", "R11", "R12"]
SIG_PRE, SIG_POST = 20, 40   # samples: 1 s before, 2 s after the event start


def make_models() -> dict[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, random_state=42, eval_metric="mlogloss"),
    }


def fit_model(name: str, model, X: pd.DataFrame, y: pd.Series):
    if name == "XGBoost":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def class_signatures(pre: pd.DataFrame, routes: list[str]) -> dict[str, dict]:
    """Average acc_z_dyn around labelled event starts (training routes only)."""
    waves: dict[int, list[np.ndarray]] = {1: [], 2: []}
    for rid, g in pre[pre["route_id"].isin(routes)].groupby("ride_id"):
        lab = g["label"].to_numpy()
        z = g["acc_z_dyn"].to_numpy()
        prev = np.concatenate([[0], lab[:-1]])
        for s in np.where((lab > 0) & (lab != prev))[0]:
            if s - SIG_PRE >= 0 and s + SIG_POST <= len(z):
                waves[int(lab[s])].append(z[s - SIG_PRE:s + SIG_POST])
    out: dict[str, dict] = {}
    for k, ws in waves.items():
        arr = np.array(ws)
        out[str(k)] = {"mean": arr.mean(0).round(4).tolist(), "std": arr.std(0).round(4).tolist(), "n": int(len(arr))}
    out["t_rel"] = (np.arange(-SIG_PRE, SIG_POST) / C.SAMPLING_HZ).round(3).tolist()
    return out


def main(dataset: Path, out_root: Path) -> None:
    models_dir, data_dir = out_root / "models", out_root / "data"
    (data_dir / "sample_rides").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(dataset)
    clean, report = prepare(raw, filename=Path(dataset).name)
    if not report.ok:
        raise SystemExit(f"Dataset failed validation: {report.errors}")
    pre = preprocess(clean)
    windows = build_windows(pre)
    windows.to_csv(data_dir / "training_windows.csv", index=False)

    train = windows[windows["route_id"] != TEST_ROUTE]
    test = windows[windows["route_id"] == TEST_ROUTE]
    X_train, y_train = train[C.FEATURE_ORDER], train["label"]
    X_test, y_test = test[C.FEATURE_ORDER], test["label"]
    train_routes = sorted(train["route_id"].unique().tolist())

    # ---- fit, cross-validate, score
    fitted, rows, cv_rows, pc_rows = {}, [], [], []
    for name, model in make_models().items():
        for tr, va in LeaveOneGroupOut().split(X_train, y_train, train["route_id"]):
            m = fit_model(name, clone(model), X_train.iloc[tr], y_train.iloc[tr])
            cv_rows.append({"Model": name, "Held-out route": train["route_id"].iloc[va].iloc[0],
                            "Macro F1": f1_score(y_train.iloc[va], m.predict(X_train.iloc[va]), average="macro")})
        m = fit_model(name, clone(model), X_train, y_train)
        fitted[name] = m
        pred = m.predict(X_test)
        proba = m.predict_proba(X_test)
        pr, rc, f1, _ = precision_recall_fscore_support(y_test, pred, labels=[0, 1, 2], zero_division=0)
        for k, cls in enumerate(C.CLASS_NAMES):
            pc_rows.append({"Model": name, "Class": cls, "Precision": pr[k], "Recall": rc[k], "F1": f1[k]})
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro"),
            "MCC": matthews_corrcoef(y_test, pred),
            "Macro ROC-AUC": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
            "Pothole recall": recall_score(y_test, pred, labels=[2], average=None)[0],
            "Speed breaker recall": recall_score(y_test, pred, labels=[1], average=None)[0],
        })
    cv_df = pd.DataFrame(cv_rows)
    comp = pd.DataFrame(rows).set_index("Model")
    comp["CV Macro F1"] = cv_df.groupby("Model")["Macro F1"].mean()
    comp = comp[["Accuracy", "Macro F1", "MCC", "CV Macro F1", "Macro ROC-AUC",
                 "Pothole recall", "Speed breaker recall"]].round(4).sort_values("Macro F1", ascending=False)

    cv_df.round(4).to_csv(data_dir / "cv_results.csv", index=False)
    pd.DataFrame(pc_rows).round(4).to_csv(data_dir / "per_class_metrics.csv", index=False)
    comp.to_csv(data_dir / "model_comparison.csv")

    best_name = comp.index[0]
    joblib.dump({
        "model": fitted[best_name], "model_name": best_name,
        "feature_columns": list(C.FEATURE_ORDER), "class_names": list(C.CLASS_NAMES),
        "window_size": C.WINDOW_SIZE, "step": C.STEP, "sampling_hz": C.SAMPLING_HZ,
    }, models_dir / "road_quality_model.joblib")

    # ---- reference statistics (training routes only)
    feats = {}
    for c in C.FEATURE_ORDER:
        v = train[c].to_numpy(dtype=float)
        feats[c] = {"mean": float(v.mean()), "std": float(v.std() or 1e-9),
                    "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
    smooth = train.loc[train["label"] == 0, "z_std"].to_numpy(dtype=float)
    ref = {
        "meta": {"training_routes": train_routes, "n_windows": int(len(train)),
                 "note": "Statistics computed on SIMULATED training data."},
        "features": feats,
        "smooth_z_std": {
            "mean": float(smooth.mean()), "std": float(smooth.std()),
            "p05": float(np.percentile(smooth, 5)), "p25": float(np.percentile(smooth, 25)),
            "p50": float(np.percentile(smooth, 50)), "p75": float(np.percentile(smooth, 75)),
            "p95": float(np.percentile(smooth, 95)),
            "quantiles": [float(q) for q in np.percentile(smooth, np.arange(0, 101))],
        },
        "signatures": class_signatures(pre, train_routes),
    }
    (models_dir / "reference_stats.json").write_text(json.dumps(ref, indent=1))

    # ---- sample rides (raw columns, including labels, for the demo)
    for rid in SAMPLE_RIDES:
        raw[raw["ride_id"] == rid].to_csv(data_dir / "sample_rides" / f"ride_{rid}.csv", index=False)

    print(comp.round(3).to_string())
    print(f"\nBest model: {best_name}. Assets written under {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT)
    args = ap.parse_args()
    main(args.dataset, args.out)
