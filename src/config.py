"""Shared project constants and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VIZ_DATA_DIR = DATA_DIR / "viz"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_CHARTS_DIR = OUTPUTS_DIR / "charts"
OUTPUTS_MAPS_DIR = OUTPUTS_DIR / "maps"
OUTPUTS_REPORTS_DIR = OUTPUTS_DIR / "reports"

DEFAULT_TESTING_MONTH = "2025-05"

LONDON_BBOX = {
    "longitude_min": -0.55,
    "longitude_max": 0.35,
    "latitude_min": 51.25,
    "latitude_max": 51.75,
}

RAW_COLUMN_RENAME_MAP = {
    "Crime ID": "crime_id",
    "Month": "month",
    "Reported by": "reported_by",
    "Falls within": "falls_within",
    "Longitude": "longitude",
    "Latitude": "latitude",
    "Location": "location",
    "LSOA code": "lsoa_code",
    "LSOA name": "lsoa_name",
    "Crime type": "crime_type",
    "Last outcome category": "last_outcome_category",
    "Context": "context",
}
