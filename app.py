"""London Crime Pulse Explorer — Streamlit app.

Reads the precomputed visual-ready aggregation
(``data/viz/crime_hex_3d_month.parquet``) produced by ``run_pipeline.py``.
Switching month or crime type filters and sums the small aggregated table —
no raw data is loaded at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DEFAULT_TESTING_MONTH, HEX_MAP_HEIGHT, VIZ_DATA_DIR
from src.transformation.aggregate_hex_grid import (
    CRIME_COUNT_COLUMN,
    CRIME_TYPE_COLUMN,
    hex_month_type_output_path,
)
from src.transformation.hex_grid import HEX_Q_COLUMN, HEX_R_COLUMN
from src.visualization.hexagon_3d_map import build_hexagon_map, encode_hex_cells

MONTH_COLUMN = "month"
LONGITUDE_COLUMN = "longitude"
LATITUDE_COLUMN = "latitude"

ALL_CRIME_TYPES_LABEL = "All"
ALL_MONTHS_LABEL = "All"

MAP_FILTER_CSS = """
<style>
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) {
        margin-bottom: 0.15rem;
    }
    h3.crime-map-heading {
        margin-top: 0.35rem;
        margin-bottom: 0.2rem;
        padding: 0;
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.2;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(14, 17, 23, 0.92);
        border-color: rgba(255, 255, 255, 0.18) !important;
        border-bottom: none !important;
        border-radius: 10px 10px 0 0 !important;
        margin-top: 0 !important;
        padding-top: 0.2rem;
    }
    .map-filter-label {
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0 0 -0.35rem 0;
    }
    .map-filter-label--crime-type {
        margin: 0.325rem 0 0.35rem 0;
    }
    div[data-testid="stPills"] {
        padding-top: 0.15rem;
    }
    div[data-testid="stSelectSlider"] {
        padding-top: 0;
        padding-bottom: 0.175rem;
    }
    div[data-testid="stSelectSlider"] div[role="slider"] {
        background: transparent !important;
        border: 2px solid rgba(255, 255, 255, 0.92) !important;
        box-shadow: none !important;
        color: #ffffff !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        min-width: 5.5rem;
        padding: 0.1rem 0.45rem;
    }
    div[data-testid="stSelectSlider"] div[data-testid="stThumbValue"] {
        color: #ffffff !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stPills"] button {
        border-color: rgba(255, 255, 255, 0.28);
        color: rgba(255, 255, 255, 0.82);
        font-size: 0.78rem;
        white-space: nowrap;
    }
    div[data-testid="stPills"] button[aria-pressed="true"] {
        background: rgba(255, 255, 255, 0.14);
        border-color: rgba(255, 255, 255, 0.85);
        color: #ffffff;
    }
    div[data-testid="stPydeckChart"] {
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 0 0 10px 10px;
        overflow: hidden;
    }
</style>
"""


@st.cache_data
def load_parquet(parquet_path: str) -> pd.DataFrame:
    """Load a Parquet file. The path is the cache key so regenerating via the
    pipeline automatically invalidates the cache."""
    return pd.read_parquet(Path(parquet_path))


def sorted_unique_values(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype("string").unique().tolist())


def month_key_to_label(month: str) -> str:
    """Convert a YYYY-MM key to a human-readable label, e.g. 'May 2025'."""
    try:
        return pd.Timestamp(month + "-01").strftime("%b %Y")
    except Exception:
        return month


def month_slider_label(month: str) -> str:
    if month == ALL_MONTHS_LABEL:
        return "All months"
    return month_key_to_label(month)


def read_filter_selection(default_month: str) -> tuple[str, str]:
    """Read current filter values from widget session state."""
    selected_month = st.session_state.get("month_slider", default_month)
    selected_crime_type = st.session_state.get("crime_type_pills", ALL_CRIME_TYPES_LABEL)
    if selected_crime_type is None:
        selected_crime_type = ALL_CRIME_TYPES_LABEL
    return selected_month, selected_crime_type


def build_display_hex(
    hex_viz_df: pd.DataFrame,
    month_choice: str,
    crime_type_choice: str,
) -> pd.DataFrame:
    """Filter the precomputed hex aggregation by month + crime type, sum counts
    per hex cell, and attach colour + elevation for rendering."""
    df = hex_viz_df
    if month_choice != ALL_MONTHS_LABEL:
        df = df[df[MONTH_COLUMN] == month_choice]
    if crime_type_choice != ALL_CRIME_TYPES_LABEL:
        df = df[df[CRIME_TYPE_COLUMN] == crime_type_choice]

    if df.empty:
        return df.iloc[0:0].copy()

    grouped = (
        df.groupby(
            [HEX_Q_COLUMN, HEX_R_COLUMN, LONGITUDE_COLUMN, LATITUDE_COLUMN],
            as_index=False,
        )[CRIME_COUNT_COLUMN]
        .sum()
    )
    return encode_hex_cells(grouped)


st.set_page_config(
    page_title="London Crime Explorer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("London Crime Explorer")
st.caption(
    "London Crime Explorer is a focused 3D map of real London street-level crime data. The final app turns raw monthly police CSV files into a single visual-ready hex aggregation, then lets users explore crime intensity by month and crime type through a Streamlit interface with a large tilted PyDeck map, summary metrics, a month slider, and crime-type controls."
    "\n\nThe runtime is intentionally lightweight: the app reads one precomputed Parquet file and renders the selected slice as extruded hex columns where height and colour represent crime volume."
)

st.subheader("Dataset status")

hex_viz_path = hex_month_type_output_path(VIZ_DATA_DIR)
if not hex_viz_path.is_file():
    st.error(
        "Visual-ready 3D hex dataset not found.\n\n"
        f"Expected file: `{hex_viz_path}`\n\n"
        "Run `python run_pipeline.py` to generate the aggregated map source."
    )
    st.stop()

hex_viz_df = load_parquet(str(hex_viz_path))
available_months = sorted(
    hex_viz_df[MONTH_COLUMN].dropna().astype("string").unique().tolist()
)

if not available_months:
    st.error("The hex dataset has no months. Re-run the pipeline.")
    st.stop()

month_options = [ALL_MONTHS_LABEL, *available_months] if len(available_months) > 1 else list(available_months)

default_month = (
    DEFAULT_TESTING_MONTH
    if DEFAULT_TESTING_MONTH in month_options
    else month_options[0]
)

crime_types = sorted_unique_values(hex_viz_df[CRIME_TYPE_COLUMN])
crime_type_options = [ALL_CRIME_TYPES_LABEL, *crime_types]

selected_month, selected_crime_type = read_filter_selection(default_month)
hex_df = build_display_hex(hex_viz_df, selected_month, selected_crime_type)

crimes_total = int(hex_viz_df[CRIME_COUNT_COLUMN].sum())
crimes_selected = int(hex_df[CRIME_COUNT_COLUMN].sum()) if not hex_df.empty else 0
hex_cell_count = len(hex_df) if not hex_df.empty else 0

total_col, selected_col, hex_col = st.columns(3)
total_col.metric("Crimes total", f"{crimes_total:,}")
selected_col.metric("Selected crimes", f"{crimes_selected:,}")
hex_col.metric("Hex cells", f"{hex_cell_count:,}")

st.markdown('<h3 class="crime-map-heading">Crime map</h3>', unsafe_allow_html=True)

st.markdown(MAP_FILTER_CSS, unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<p class="map-filter-label">Month</p>', unsafe_allow_html=True)
    selected_month = st.select_slider(
        "Month",
        options=month_options,
        value=default_month,
        format_func=month_slider_label,
        label_visibility="collapsed",
        key="month_slider",
    )

    st.markdown(
        '<p class="map-filter-label map-filter-label--crime-type">Crime type</p>',
        unsafe_allow_html=True,
    )
    selected_crime_type = st.pills(
        "Crime type",
        options=crime_type_options,
        default=ALL_CRIME_TYPES_LABEL,
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        key="crime_type_pills",
    )

if selected_crime_type is None:
    selected_crime_type = ALL_CRIME_TYPES_LABEL

if hex_df.empty:
    st.info(
        "No hex cells for the current month / crime type selection. "
        f"Try **{ALL_MONTHS_LABEL}** month or **{ALL_CRIME_TYPES_LABEL}** crime type."
    )
else:
    hex_map = build_hexagon_map(hex_df)
    if hex_map is None:
        st.info("No hex cells to display for the current filter.")
    else:
        st.pydeck_chart(hex_map, width="stretch", height=HEX_MAP_HEIGHT)
