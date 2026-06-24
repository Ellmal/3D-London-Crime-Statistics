"""London Crime Pulse Explorer — Streamlit app.

The 3D hexagon layer reads the precomputed visual-ready aggregation
(``data/viz/crime_hex_3d_month.parquet``) produced by ``run_pipeline.py``, so
switching month or crime type only filters + sums a small table instead of
re-binning raw points. Row-level layers (raw points, charts, preview) load the
processed Parquet for the selected month (or the combined file for "All").
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from src.cleaning.clean_crime_data import (
    HAS_VALID_COORDINATES_COLUMN,
    IS_WITHIN_LONDON_BBOX_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    PROCESSED_COMBINED_FILENAME,
)
from src.config import (
    DEFAULT_TESTING_MONTH,
    LONDON_BBOX,
    OUTPUTS_REPORTS_DIR,
    PROCESSED_DATA_DIR,
    VIZ_DATA_DIR,
)
from src.transformation.aggregate_crime_grid import (
    CRIME_COUNT_COLUMN,
    hex_month_type_output_path,
)
from src.transformation.hex_grid import HEX_Q_COLUMN, HEX_R_COLUMN
from src.validation.validate_crime_data import (
    LSOA_CODE_COLUMN,
    LSOA_NAME_COLUMN,
    REPORT_FILENAME_TEMPLATE,
    TOP_LSOA_COUNT,
    processed_parquet_path,
)
from src.visualization.hexagon_3d_map import build_hexagon_map, encode_hex_cells

CRIME_TYPE_COLUMN = "crime_type"
REPORTED_BY_COLUMN = "reported_by"
MONTH_COLUMN = "month"
MONTH_LABEL_COLUMN = "month_label"
LOCATION_COLUMN = "location"
FALLS_WITHIN_COLUMN = "falls_within"
LAST_OUTCOME_CATEGORY_COLUMN = "last_outcome_category"
PREVIEW_ROW_COUNT = 200

MAP_LAYER_HEXAGONS = "3D hexagons"
MAP_LAYER_POINTS = "Raw points"
MAP_LAYER_OPTIONS = [MAP_LAYER_HEXAGONS, MAP_LAYER_POINTS]

MAP_POINT_LIMITS: dict[str, int | None] = {
    "5,000": 5_000,
    "25,000": 25_000,
    "All": None,
}
MAP_TOOLTIP_COLUMNS = [
    MONTH_COLUMN,
    CRIME_TYPE_COLUMN,
    LOCATION_COLUMN,
    LSOA_NAME_COLUMN,
    LSOA_CODE_COLUMN,
    REPORTED_BY_COLUMN,
    FALLS_WITHIN_COLUMN,
    LAST_OUTCOME_CATEGORY_COLUMN,
]
ALL_CRIME_TYPES_LABEL = "All"
ALL_MONTHS_LABEL = "All"


@st.cache_data
def load_parquet(parquet_path: str) -> pd.DataFrame:
    """Load a Parquet file. The path is the cache key, so regenerating the
    file via the pipeline invalidates the cache."""
    return pd.read_parquet(Path(parquet_path))


def sorted_unique_values(series: pd.Series) -> list[str]:
    """Return sorted unique non-null string values for display."""
    return sorted(series.dropna().astype("string").unique().tolist())


def filter_by_crime_type(df: pd.DataFrame, crime_type: str) -> pd.DataFrame:
    """Return rows for one crime type, or the full dataframe for ``All``."""
    if crime_type == ALL_CRIME_TYPES_LABEL:
        return df
    return df.loc[df[CRIME_TYPE_COLUMN].astype("string") == crime_type].copy()


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


def processed_path_for(month_choice: str) -> Path:
    """Resolve the row-level Parquet path for the selected month (or combined)."""
    if month_choice == ALL_MONTHS_LABEL:
        return PROCESSED_DATA_DIR / PROCESSED_COMBINED_FILENAME
    return processed_parquet_path(month_choice, PROCESSED_DATA_DIR)


def quality_report_path(month: str) -> Path:
    return OUTPUTS_REPORTS_DIR / REPORT_FILENAME_TEMPLATE.format(month=month)


@st.cache_data
def load_quality_report(report_path: str) -> str | None:
    """Load the pipeline data quality report text, if it exists."""
    path = Path(report_path)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def value_counts_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a single-column dataframe of value counts for bar charts."""
    counts = df[column].value_counts(dropna=False).sort_values(ascending=False)
    counts.index = counts.index.map(lambda value: str(value) if pd.notna(value) else "(missing)")
    return counts.rename("count").to_frame()


def top_lsoa_counts_frame(df: pd.DataFrame, limit: int = TOP_LSOA_COUNT) -> pd.DataFrame:
    """Return top LSOAs by crime count, aligned with the pipeline report."""
    lsoa_df = df.dropna(subset=[LSOA_CODE_COLUMN])
    if lsoa_df.empty:
        return pd.DataFrame(columns=["count"])

    top_lsoas = (
        lsoa_df.groupby([LSOA_CODE_COLUMN, LSOA_NAME_COLUMN], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(limit)
    )
    labels = [
        f"{lsoa_code} | {lsoa_name if pd.notna(lsoa_name) else '(no name)'}"
        for lsoa_code, lsoa_name in top_lsoas.index
    ]
    return pd.DataFrame({"count": top_lsoas.values}, index=labels)


def filter_map_points(
    df: pd.DataFrame,
    *,
    within_london_bbox: bool = True,
) -> pd.DataFrame:
    """Keep rows that are safe to plot on the map."""
    mask = df[HAS_VALID_COORDINATES_COLUMN]
    if within_london_bbox:
        mask = mask & df[IS_WITHIN_LONDON_BBOX_COLUMN]
    return df.loc[mask].copy()


def limit_map_points(df: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    """Cap the number of points sent to the map for performance."""
    if max_points is None or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=42)


def prepare_map_tooltip_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return map rows with string tooltip fields and explicit lat/lon columns."""
    map_df = df[MAP_TOOLTIP_COLUMNS + [LONGITUDE_COLUMN, LATITUDE_COLUMN]].copy()
    for column in MAP_TOOLTIP_COLUMNS:
        map_df[column] = map_df[column].astype("string").fillna("")
    return map_df


def build_crime_point_map(map_df: pd.DataFrame) -> pdk.Deck | None:
    """Build a PyDeck scatter map centred on London."""
    if map_df.empty:
        return None

    center_latitude = (LONDON_BBOX["latitude_min"] + LONDON_BBOX["latitude_max"]) / 2
    center_longitude = (LONDON_BBOX["longitude_min"] + LONDON_BBOX["longitude_max"]) / 2

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=[LONGITUDE_COLUMN, LATITUDE_COLUMN],
        get_fill_color=[220, 50, 70, 140],
        get_radius=25,
        radius_min_pixels=2,
        radius_max_pixels=8,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=center_latitude,
        longitude=center_longitude,
        zoom=9,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<b>{crime_type}</b><br/>"
            "Month: {month}<br/>"
            "Location: {location}<br/>"
            "LSOA: {lsoa_name} ({lsoa_code})<br/>"
            "Reported by: {reported_by}<br/>"
            "Falls within: {falls_within}<br/>"
            "Outcome: {last_outcome_category}"
        ),
        "style": {
            "backgroundColor": "#1e1e1e",
            "color": "white",
        },
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="dark",
    )


st.set_page_config(
    page_title="London Crime Pulse Explorer",
    page_icon="🗺️",
    layout="wide",
)

st.title("London Crime Pulse Explorer")
st.caption(
    "Exploratory 3D crime map for London — pick a month and crime type to "
    "update the hex columns, raw points, and summary metrics."
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
available_months = sorted(hex_viz_df[MONTH_COLUMN].dropna().astype("string").unique().tolist())

if not available_months:
    st.error("The hex dataset has no months. Re-run the pipeline.")
    st.stop()

if len(available_months) > 1:
    month_options = [ALL_MONTHS_LABEL, *available_months]
else:
    month_options = list(available_months)

default_month_index = (
    month_options.index(DEFAULT_TESTING_MONTH)
    if DEFAULT_TESTING_MONTH in month_options
    else 0
)

with st.sidebar:
    st.header("Map & filters")

    selected_map_layer = st.radio(
        "Map layer",
        options=MAP_LAYER_OPTIONS,
        index=0,
        help=(
            "**3D hexagons** — precomputed aggregated intensity. "
            "**Raw points** — individual crimes for debugging."
        ),
    )

    selected_month = st.selectbox(
        "Month",
        options=month_options,
        index=default_month_index,
        help=(
            "Switches the 3D map, charts, and metrics to the chosen month. "
            f"**{ALL_MONTHS_LABEL}** sums every available month."
        ),
    )

dataset_path = processed_path_for(selected_month)
if not dataset_path.is_file():
    st.error(
        f"Processed dataset not found for **{selected_month}**.\n\n"
        f"Expected file: `{dataset_path}`\n\n"
        "Run `python run_pipeline.py` to generate the cleaned Parquet file."
    )
    st.stop()

df = load_parquet(str(dataset_path))

month_labels = (
    sorted_unique_values(df[MONTH_LABEL_COLUMN])
    if MONTH_LABEL_COLUMN in df.columns
    else sorted_unique_values(df[MONTH_COLUMN])
)
crime_types = sorted_unique_values(df[CRIME_TYPE_COLUMN])
police_forces = sorted_unique_values(df[REPORTED_BY_COLUMN])

with st.sidebar:
    selected_crime_type = st.selectbox(
        "Crime type",
        options=[ALL_CRIME_TYPES_LABEL, *crime_types],
        help="Filters charts, metrics, and the active map layer.",
    )

    within_london_bbox = st.toggle(
        "Limit to London bounding box",
        value=True,
        help=(
            "Applies to the **raw points** layer. When off, any row with valid "
            "coordinates is shown (may include points outside London). The 3D "
            "hex layer is always London-only (precomputed)."
        ),
    )

    max_points: int | None = None
    if selected_map_layer == MAP_LAYER_POINTS:
        point_limit_label = st.select_slider(
            "Max points on map",
            options=list(MAP_POINT_LIMITS.keys()),
            value="5,000",
            help="Caps how many raw points are drawn. Does not apply to 3D hexagons.",
        )
        max_points = MAP_POINT_LIMITS[point_limit_label]

filtered_df = filter_by_crime_type(df, selected_crime_type)
map_points_df = filter_map_points(filtered_df, within_london_bbox=within_london_bbox)

if selected_map_layer == MAP_LAYER_HEXAGONS:
    hex_df = build_display_hex(hex_viz_df, selected_month, selected_crime_type)
else:
    hex_df = pd.DataFrame()

hex_cell_count = len(hex_df) if not hex_df.empty else 0
if selected_map_layer == MAP_LAYER_HEXAGONS:
    crimes_on_map = int(hex_df[CRIME_COUNT_COLUMN].sum()) if not hex_df.empty else 0
else:
    crimes_on_map = len(map_points_df)

status_col, filtered_col, map_ready_col, map_metric_col, forces_col = st.columns(5)
status_col.metric("Status", "Loaded")
filtered_col.metric("Filtered rows", f"{len(filtered_df):,}")
map_ready_col.metric("Map-ready points", f"{len(map_points_df):,}")
if selected_map_layer == MAP_LAYER_HEXAGONS:
    map_metric_col.metric("Hex cells", f"{hex_cell_count:,}")
else:
    displayed_points = min(len(map_points_df), max_points or len(map_points_df))
    map_metric_col.metric("Points drawn", f"{displayed_points:,}")
forces_col.metric("Police forces", len(police_forces))

if selected_map_layer == MAP_LAYER_HEXAGONS:
    st.caption(f"Source: `{hex_viz_path}` · Month: **{selected_month}**")
else:
    st.caption(f"Source: `{dataset_path}` · Month: **{selected_month}**")

active_filters: list[str] = []
if selected_month != ALL_MONTHS_LABEL:
    active_filters.append(f"month = **{selected_month}**")
if selected_crime_type != ALL_CRIME_TYPES_LABEL:
    active_filters.append(f"crime type = **{selected_crime_type}**")
if selected_map_layer == MAP_LAYER_POINTS and not within_london_bbox:
    active_filters.append("London bbox **off** (includes out-of-bbox coordinates)")
if selected_map_layer == MAP_LAYER_POINTS and max_points is not None:
    active_filters.append(f"raw points capped at **{max_points:,}**")

if active_filters:
    st.info("Active filters: " + " · ".join(active_filters))

detail_col, forces_col_detail = st.columns(2)

with detail_col:
    st.markdown("**Available months**")
    st.write(", ".join(available_months) if available_months else "—")

    st.markdown("**Available crime types**")
    st.write(", ".join(crime_types) if crime_types else "—")

with forces_col_detail:
    st.markdown("**Available police forces**")
    st.write(", ".join(police_forces) if police_forces else "—")

    st.markdown("**Month labels (selection)**")
    st.write(", ".join(month_labels) if month_labels else "—")

st.subheader("Data quality report")
if selected_month == ALL_MONTHS_LABEL:
    st.info(
        "Data quality reports are generated per month. Select a single month to "
        "view its report."
    )
else:
    report_path = quality_report_path(selected_month)
    report_text = load_quality_report(str(report_path))
    if report_text is None:
        st.warning(
            f"Data quality report not found for **{selected_month}**.\n\n"
            f"Expected file: `{report_path}`\n\n"
            "Run `python run_pipeline.py` to generate the report."
        )
    else:
        with st.expander("View pipeline data quality report", expanded=False):
            st.text(report_text)
        st.caption(f"Source: `{report_path}`")

st.subheader("Summary charts")

if filtered_df.empty:
    st.warning("No rows match the selected filters.")
else:
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("**Crimes by crime type**")
        st.bar_chart(value_counts_frame(filtered_df, CRIME_TYPE_COLUMN), width="stretch")

    with chart_right:
        st.markdown("**Crimes by police force**")
        st.bar_chart(value_counts_frame(filtered_df, REPORTED_BY_COLUMN), width="stretch")

    st.markdown(f"**Top {TOP_LSOA_COUNT} LSOAs by crime count**")
    st.bar_chart(top_lsoa_counts_frame(filtered_df), width="stretch")

    map_summary_col, hex_summary_col = st.columns(2)
    with map_summary_col:
        st.metric("Crimes represented on map", f"{crimes_on_map:,}")
    with hex_summary_col:
        if selected_map_layer == MAP_LAYER_HEXAGONS and not hex_df.empty:
            st.metric(
                "Crimes per hex (median)",
                f"{hex_df[CRIME_COUNT_COLUMN].median():.0f}",
            )

st.subheader("Crime map")

if selected_map_layer == MAP_LAYER_HEXAGONS:
    if hex_df.empty:
        st.info(
            "No hex cells for the current month / crime type selection. "
            f"Try **{ALL_MONTHS_LABEL}** month or **{ALL_CRIME_TYPES_LABEL}** crime type."
        )
    else:
        st.caption(
            f"**3D hexagons** — {hex_cell_count:,} cells representing "
            f"{crimes_on_map:,} crimes"
            + (f" · month **{selected_month}**" if selected_month != ALL_MONTHS_LABEL else " · **all months**")
            + (
                f" · **{selected_crime_type}** only"
                if selected_crime_type != ALL_CRIME_TYPES_LABEL
                else ""
            )
            + ". Hover a column for the crime count."
        )
        hex_map = build_hexagon_map(hex_df)
        if hex_map is None:
            st.info("No hex cells to display for the current filter.")
        else:
            st.pydeck_chart(hex_map, width="stretch")
elif map_points_df.empty:
    st.info(
        "No map-ready points for the current filters. "
        f"Try turning on **Limit to London bounding box** off, or choosing "
        f"**{ALL_CRIME_TYPES_LABEL}**."
    )
else:
    display_map_df = prepare_map_tooltip_data(
        limit_map_points(map_points_df, max_points)
    )
    st.caption(
        f"**Raw points** — showing {len(display_map_df):,} of "
        f"{len(map_points_df):,} map-ready points"
        + (
            f" (**{selected_crime_type}** only)"
            if selected_crime_type != ALL_CRIME_TYPES_LABEL
            else ""
        )
        + "."
    )
    point_map = build_crime_point_map(display_map_df)
    if point_map is None:
        st.info("No map-ready crime points to display for the current filter.")
    else:
        st.pydeck_chart(point_map, width="stretch")

st.subheader("Cleaned data preview")
if filtered_df.empty:
    st.warning("No rows to preview for the current filter.")
else:
    st.dataframe(filtered_df.head(PREVIEW_ROW_COUNT), width="stretch")
    st.caption(
        f"Showing first {min(PREVIEW_ROW_COUNT, len(filtered_df)):,} of "
        f"{len(filtered_df):,} filtered rows."
    )
