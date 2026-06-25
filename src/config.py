"""Shared project constants and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VIZ_DATA_DIR = DATA_DIR / "viz"

# ---------------------------------------------------------------------------
# Single source of truth for which months the pipeline processes.
# Set to a list to process specific months, or None to process every
# YYYY-MM folder found under data/raw/.
#
# Examples:
#   ["2025-05"]             # single month
#   ["2025-05", "2025-06"]  # specific months
#   None                    # all YYYY-MM folders under data/raw/
# ---------------------------------------------------------------------------
PIPELINE_MONTHS: list[str] | None = None

# Default month used as the initial selection in the Streamlit app.
# Falls back to "2025-05" when PIPELINE_MONTHS is None (process-all mode).
DEFAULT_TESTING_MONTH = PIPELINE_MONTHS[0] if PIPELINE_MONTHS else "2025-05"

# Geographic bounds for filtering points and framing the map.
# Defaults to Greater London; update lat/long limits if using data from another area.
AREA_BBOX = {
    "longitude_min": -0.55,
    "longitude_max": 0.35,
    "latitude_min": 51.25,
    "latitude_max": 51.75,
}

# ---------------------------------------------------------------------------
# Hex grid + 3D map parameters. Single source of truth shared by the pipeline
# aggregation (data/viz/crime_hex_3d_month.parquet) and the Streamlit app.
# Tune the 3D map appearance here.
#
# HEX_RADIUS_METERS controls how points are binned into cells. Both pipeline
# and app read it from here so they stay aligned — do not redefine elsewhere.
# ---------------------------------------------------------------------------
HEX_RADIUS_METERS = 100         # hex cell size (binning geometry)
HEX_COVERAGE = 0.88             # rendered column fill fraction (0-1)
HEX_COLUMN_DISK_SIDES = 6       # 6 -> hexagonal columns

# Height encoding: elevation = crime_count ^ power * scale (meters)
HEX_ELEVATION_POWER = 0.65
HEX_ELEVATION_SCALE = 400

# Colour encoding: normalized sqrt(count), gamma stretches the low end so
# small differences stay visible; alpha is the fill opacity byte (0-255).
HEX_COLOR_GAMMA = 0.62
HEX_FILL_ALPHA = 230

# Camera framing for the tilted 3D view.
HEX_CAMERA_PITCH = 55
HEX_CAMERA_BEARING = -10
HEX_CAMERA_ZOOM = 10

# Streamlit / PyDeck display height in pixels (default PyDeck height is 500).
HEX_MAP_HEIGHT = 1000

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
