"""Paths, constants, theme colors and default thresholds for RoadPulse."""
from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------- paths
ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODELS_DIR: Final[Path] = ROOT / "models"
DATA_DIR: Final[Path] = ROOT / "data"
SAMPLE_DIR: Final[Path] = DATA_DIR / "sample_rides"
ASSETS_DIR: Final[Path] = ROOT / "assets"

MODEL_PATH: Final[Path] = MODELS_DIR / "road_quality_model.joblib"
REFERENCE_STATS_PATH: Final[Path] = MODELS_DIR / "reference_stats.json"
TRAINING_WINDOWS_PATH: Final[Path] = DATA_DIR / "training_windows.csv"
CV_RESULTS_PATH: Final[Path] = DATA_DIR / "cv_results.csv"
PER_CLASS_METRICS_PATH: Final[Path] = DATA_DIR / "per_class_metrics.csv"
MODEL_COMPARISON_PATH: Final[Path] = DATA_DIR / "model_comparison.csv"

# ---------------------------------------------------------------- signal constants
GRAVITY: Final[float] = 9.81
WINDOW_SIZE: Final[int] = 40          # samples (2 s at 20 Hz)
STEP: Final[int] = 20                 # samples (1 s)
SAMPLING_HZ: Final[int] = 20
WINDOW_SECONDS: Final[float] = WINDOW_SIZE / SAMPLING_HZ
STEP_SECONDS: Final[float] = STEP / SAMPLING_HZ

# ---------------------------------------------------------------- classes
SMOOTH, SPEED_BREAKER, POTHOLE = 0, 1, 2
CLASS_NAMES: Final[list[str]] = ["Smooth", "Speed breaker", "Pothole"]
EVENT_CLASSES: Final[tuple[int, int]] = (SPEED_BREAKER, POTHOLE)
CLASS_MEAN_DURATION_S: Final[dict[int, float]] = {SPEED_BREAKER: 1.05, POTHOLE: 0.38}

# ---------------------------------------------------------------- features (order matters)
FEATURE_ORDER: Final[list[str]] = [
    "z_std", "z_mean", "z_max", "z_min", "z_range", "z_absmax", "z_skew", "z_kurt",
    "x_std", "y_mean", "y_std", "gx_std", "gy_std", "gz_std", "gx_absmax",
    "speed_mean", "speed_min", "speed_range",
    "pow_0_2hz", "pow_2_5hz", "pow_5_10hz",
]

FEATURE_LABELS: Final[dict[str, str]] = {
    "z_std": "vertical vibration (spread)",
    "z_mean": "average vertical push",
    "z_max": "peak upward jolt",
    "z_min": "peak downward dip",
    "z_range": "vertical jolt range",
    "z_absmax": "largest vertical jolt",
    "z_skew": "vibration asymmetry",
    "z_kurt": "spikiness of the vibration",
    "x_std": "sideways vibration",
    "y_mean": "braking / acceleration push",
    "y_std": "forward-back vibration",
    "gx_std": "x-axis rotation (spread)",
    "gy_std": "y-axis rotation (spread)",
    "gz_std": "turning rotation (spread)",
    "gx_absmax": "sharpest x-axis rotation",
    "speed_mean": "average speed",
    "speed_min": "slowest speed",
    "speed_range": "speed change",
    "pow_0_2hz": "slow-wave energy share (0-2 Hz)",
    "pow_2_5hz": "mid-frequency energy share (2-5 Hz)",
    "pow_5_10hz": "fast-vibration energy share (5-10 Hz)",
}

# ---------------------------------------------------------------- raw input schema
SENSOR_COLUMNS: Final[list[str]] = [
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
]
REQUIRED_COLUMNS: Final[list[str]] = ["time_s", "speed_kmh", *SENSOR_COLUMNS]
GPS_COLUMNS: Final[list[str]] = ["lat", "lon"]
OPTIONAL_COLUMNS: Final[list[str]] = ["ride_id", "route_id", "label"]
TEMPLATE_COLUMNS: Final[list[str]] = [
    "ride_id", "route_id", "time_s", "lat", "lon", "speed_kmh",
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "label",
]

# ---------------------------------------------------------------- theme "Asphalt & Signage"
COLORS: Final[dict[str, str]] = {
    "bg": "#0B0F14",
    "panel": "#131A22",
    "border": "#1F2A36",
    "text": "#E6EDF3",
    "muted": "#8B98A5",
    "speed_breaker": "#FFB020",
    "pothole": "#FF5A4F",
    "smooth": "#2DD4BF",
    "accent": "#FFB020",
}
CLASS_COLORS: Final[dict[int, str]] = {
    SMOOTH: COLORS["smooth"],
    SPEED_BREAKER: COLORS["speed_breaker"],
    POTHOLE: COLORS["pothole"],
}
FONT_HEADING: Final[str] = "Space Grotesk"
FONT_MONO: Final[str] = "JetBrains Mono"
FONT_BODY: Final[str] = "Inter"

# ---------------------------------------------------------------- operating modes
# Per-class probability thresholds: a window is flagged as class c when p_c >= threshold[c].
DEFAULT_THRESHOLDS: Final[dict[int, float]] = {SPEED_BREAKER: 0.50, POTHOLE: 0.50}
OPERATING_MODES: Final[dict[str, dict]] = {
    "Balanced": {
        "thresholds": {SPEED_BREAKER: 0.50, POTHOLE: 0.50},
        "tagline": "Default trade-off",
        "description": "Flags a window when the model is at least 50% sure.",
    },
    "Driver Alert": {
        "thresholds": {SPEED_BREAKER: 0.80, POTHOLE: 0.80},
        "tagline": "High precision",
        "description": "Only alerts when the model is very sure, so fewer false alarms but some events are missed.",
    },
    "Municipal Audit": {
        "thresholds": {SPEED_BREAKER: 0.25, POTHOLE: 0.25},
        "tagline": "High recall",
        "description": "Flags anything plausible so few defects are missed, at the cost of more false alarms to review.",
    },
}

# ---------------------------------------------------------------- events, matching, consensus
EVENT_MERGE_GAP_S: Final[float] = STEP_SECONDS          # consecutive windows start 1 s apart
MATCH_TOLERANCE_S: Final[float] = 1.0
SEVERITY_REF_PEAK: Final[dict[int, float]] = {SPEED_BREAKER: 4.5, POTHOLE: 8.0}   # m/s^2
SEVERITY_BANDS: Final[tuple[float, float]] = (35.0, 70.0)                          # Low | Medium | High
EVENT_PEAK_HALF_WINDOW_S: Final[float] = 0.75
CONSENSUS_RADIUS_M: Final[float] = 15.0
MIN_CONFIRMATIONS: Final[int] = 2
EARTH_RADIUS_M: Final[float] = 6_371_000.0

# ---------------------------------------------------------------- trust meter
TRUST_BANDS: Final[tuple[float, float]] = (75.0, 50.0)   # High >= 75, Medium >= 50, else Low
SIMULATED_NOTICE: Final[str] = (
    "The model was trained on SIMULATED sensor data. Scores describe how well it recovers "
    "simulated patterns, not real-world accuracy."
)
