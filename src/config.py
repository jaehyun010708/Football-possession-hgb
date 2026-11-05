"""Single source of truth for paths, match split and frozen baseline settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "pysport/idsse-data"
VALID_MATCH = "J03WPY"
COMPETITION = {
    "J03WMX": "DFL-COM-000001",
    "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
TRAIN_MATCHES = [mid for mid in COMPETITION if mid != VALID_MATCH]
LABEL_NAMES = ["H", "A", "D"]
VERSION = "v1"
SAMPLE_FPS = 5.0
MARKOV_STRENGTH = 0.25
CANONICAL_LENGTH_M, CANONICAL_WIDTH_M = 105.0, 68.0
RAW_DIR = ROOT / "data" / "external" / "idsse"
FEATURE_DIR = RAW_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs" / "idsse_temporal"
BALL_OUTPUT_DIR = ROOT / "outputs" / "idsse_ball_hgb_v1"
MODEL_DIR = ROOT / "models"
PLAYER_MODEL = MODEL_DIR / "idsse_temporal_v1.joblib"
OFFICIAL_METRIC = ROOT / "evaluation" / "possession_metrics.py"
