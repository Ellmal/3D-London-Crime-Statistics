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

from src.config import DEFAULT_TESTING_MONTH, VIZ_DATA_DIR
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
        return pd.Timestamp(month + "-01").strftime("%B %Y")
    except Exception:
        return month


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
    page_title="London Crime Pulse Explorer",
    page_icon="🗺️",
    layout="wide",
)

st.title("London Crime Pulse Explorer")
st.caption(
    "Exploratory 3D crime map for London — pick a month and crime type to "
    "update the hex columns and summary metrics."
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

default_month_index = (
    month_options.index(DEFAULT_TESTING_MONTH)
    if DEFAULT_TESTING_MONTH in month_options
    else 0
)

crime_types = sorted_unique_values(hex_viz_df[CRIME_TYPE_COLUMN])

with st.sidebar:
    st.header("Map & filters")

    selected_month = st.selectbox(
        "Month",
        options=month_options,
        index=default_month_index,
        help=(
            "Switches the 3D map and metrics to the chosen month. "
            f"**{ALL_MONTHS_LABEL}** sums every available month."
        ),
    )

    selected_crime_type = st.selectbox(
        "Crime type",
        options=[ALL_CRIME_TYPES_LABEL, *crime_types],
        help="Filters the map and metrics.",
    )

hex_df = build_display_hex(hex_viz_df, selected_month, selected_crime_type)

crimes_total = int(hex_viz_df[CRIME_COUNT_COLUMN].sum())
crimes_selected = int(hex_df[CRIME_COUNT_COLUMN].sum()) if not hex_df.empty else 0
hex_cell_count = len(hex_df) if not hex_df.empty else 0

total_col, selected_col, hex_col = st.columns(3)
total_col.metric("Crimes total", f"{crimes_total:,}")
selected_col.metric("Selected crimes", f"{crimes_selected:,}")
hex_col.metric("Hex cells", f"{hex_cell_count:,}")

month_labels = [month_key_to_label(m) for m in available_months]
detail_col, type_col = st.columns(2)

with detail_col:
    st.markdown("**Months**")
    st.write(", ".join(month_labels) if month_labels else "—")

with type_col:
    st.markdown("**Crime types**")
    st.write(", ".join(crime_types) if crime_types else "—")

st.subheader("Crime map")

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
        st.pydeck_chart(hex_map, width="stretch")
