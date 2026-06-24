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

# ---------------------------------------------------------------------------
# Single source of truth for which months run through the pipeline.
# Change this list in one place to control loading, profiling, cleaning, and
# validation. Every step reads from here (orchestrated by run_pipeline.py).
#
# Examples:
#   ["2025-05"]                      # one month (current default)
#   ["2025-05", "2025-06"]           # specific months
#   None                             # all YYYY-MM folders found under data/raw/
# ---------------------------------------------------------------------------
PIPELINE_MONTHS: list[str] | None = None

# Convenience single month, derived from PIPELINE_MONTHS. Used for single-month
# defaults and the "testing month" highlight in the raw file inventory. Falls
# back to "2025-05" when PIPELINE_MONTHS is None (process-all mode).
DEFAULT_TESTING_MONTH = PIPELINE_MONTHS[0] if PIPELINE_MONTHS else "2025-05"

# ---------------------------------------------------------------------------
# Raw profile report output mode (step 2 in run_pipeline.py).
#
#   "separate"  -> one file per month (default)
#   "combined"  -> one aggregated profile across all selected months/files
#   "both"      -> write separate per-month and aggregated combined reports
# ---------------------------------------------------------------------------
REPORT_OUTPUT_MODE = "combined"

LONDON_BBOX = {
    "longitude_min": -0.55,
    "longitude_max": 0.35,
    "latitude_min": 51.25,
    "latitude_max": 51.75,
}

# ---------------------------------------------------------------------------
# Hex grid + 3D map parameters. Single source of truth shared by the pipeline
# aggregation (data/viz/crime_hex_3d_month.parquet) and the rendering layers
# (standalone HTML map and the Streamlit app). Tune the 3D map from here.
#
# HEX_RADIUS_METERS is the only value that changes how points are binned into
# cells. The precomputed aggregation and the live render stay aligned because
# both read it from here, so do not redefine it elsewhere.
# ---------------------------------------------------------------------------
HEX_RADIUS_METERS = 100  # hex cell size (binning geometry)
HEX_COVERAGE = 0.88  # rendered column fill fraction (0-1)
HEX_COLUMN_DISK_SIDES = 6  # 6 -> hexagonal columns

# Height encoding: elevation = crime_count^power * scale (meters)
HEX_ELEVATION_POWER = 0.65
HEX_ELEVATION_SCALE = 400

# Colour encoding: normalized sqrt(count), gamma stretches the low end so small
# differences stay visible; alpha is the fill opacity byte (0-255).
HEX_COLOR_GAMMA = 0.62
HEX_FILL_ALPHA = 230

# Camera framing for the tilted 3D view.
HEX_CAMERA_PITCH = 55
HEX_CAMERA_BEARING = -20
HEX_CAMERA_ZOOM = 9.2

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
