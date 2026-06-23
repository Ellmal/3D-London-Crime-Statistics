"""Build a standalone 2D crime intensity map from the spatial grid summary."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pydeck as pdk

from src.config import (
    DEFAULT_TESTING_MONTH,
    LONDON_BBOX,
    OUTPUTS_MAPS_DIR,
    PIPELINE_MONTHS,
    VIZ_DATA_DIR,
)
from src.ingestion.load_crime_files import resolve_load_months
from src.transformation.aggregate_crime_grid import (
    CENTER_LATITUDE_COLUMN,
    CENTER_LONGITUDE_COLUMN,
    CRIME_COUNT_COLUMN,
    TOP_CRIME_TYPE_COLUMN,
    UNIQUE_LSOA_COUNT_COLUMN,
    output_parquet_path,
)

COLOR_R_COLUMN = "color_r"
COLOR_G_COLUMN = "color_g"
COLOR_B_COLUMN = "color_b"
COLOR_A_COLUMN = "color_a"

OUTPUT_FILENAME_TEMPLATE = "london_crime_2d_intensity_{month}.html"

# Sequential scale: low (cool blue) -> high (warm orange-red)
COLOR_LOW = (35, 70, 180)
COLOR_MID = (90, 180, 220)
COLOR_HIGH = (255, 210, 80)
COLOR_PEAK = (230, 60, 45)

CIRCLE_RADIUS_METERS = 90
CIRCLE_RADIUS_PIXELS = 7
CIRCLE_FILL_ALPHA = 200


def output_html_path(
    month: str,
    maps_dir: Path = OUTPUTS_MAPS_DIR,
) -> Path:
    return maps_dir / OUTPUT_FILENAME_TEMPLATE.format(month=month)


def load_grid_summary(
    month: str,
    viz_dir: Path = VIZ_DATA_DIR,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the aggregated grid summary for one month."""
    parquet_path = output_parquet_path(month, viz_dir)
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Grid summary not found: {parquet_path}\n"
            "Run `python run_pipeline.py` or "
            "`python -m src.transformation.aggregate_crime_grid` first."
        )

    if verbose:
        print(f"Loading {parquet_path} ...")
    return pd.read_parquet(parquet_path)


def _lerp_rgb(
    t: float,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(
        int(start[i] + (end[i] - start[i]) * t)
        for i in range(3)
    )


def crime_count_to_rgb(normalized: float) -> tuple[int, int, int]:
    """Map a 0-1 intensity value to a sequential blue -> yellow -> red colour."""
    if normalized <= 0.5:
        return _lerp_rgb(normalized * 2, COLOR_LOW, COLOR_MID)
    if normalized <= 0.85:
        return _lerp_rgb((normalized - 0.5) / 0.35, COLOR_MID, COLOR_HIGH)
    return _lerp_rgb((normalized - 0.85) / 0.15, COLOR_HIGH, COLOR_PEAK)


def apply_intensity_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add RGBA columns from crime_count using a log-scaled sequential palette."""
    counts = df[CRIME_COUNT_COLUMN].astype(float)
    log_counts = counts.map(math.log1p)
    min_log = float(log_counts.min())
    max_log = float(log_counts.max())

    if max_log == min_log:
        normalized = pd.Series(0.0, index=df.index)
    else:
        normalized = (log_counts - min_log) / (max_log - min_log)

    colors = normalized.map(crime_count_to_rgb)
    map_df = df.copy()
    map_df[COLOR_R_COLUMN] = colors.map(lambda rgb: rgb[0])
    map_df[COLOR_G_COLUMN] = colors.map(lambda rgb: rgb[1])
    map_df[COLOR_B_COLUMN] = colors.map(lambda rgb: rgb[2])
    map_df[COLOR_A_COLUMN] = CIRCLE_FILL_ALPHA
    return map_df


def prepare_map_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return grid rows with colour channels and string tooltip fields."""
    map_df = apply_intensity_colors(df)
    map_df[TOP_CRIME_TYPE_COLUMN] = (
        map_df[TOP_CRIME_TYPE_COLUMN].astype("string").fillna("(unknown)")
    )
    map_df[CRIME_COUNT_COLUMN] = map_df[CRIME_COUNT_COLUMN].astype(int)
    map_df[UNIQUE_LSOA_COUNT_COLUMN] = map_df[UNIQUE_LSOA_COUNT_COLUMN].astype(int)
    return map_df


def build_intensity_map(map_df: pd.DataFrame) -> pdk.Deck | None:
    """Build a flat PyDeck map with colour-encoded crime intensity per grid cell."""
    if map_df.empty:
        return None

    center_latitude = (LONDON_BBOX["latitude_min"] + LONDON_BBOX["latitude_max"]) / 2
    center_longitude = (LONDON_BBOX["longitude_min"] + LONDON_BBOX["longitude_max"]) / 2

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=[CENTER_LONGITUDE_COLUMN, CENTER_LATITUDE_COLUMN],
        get_fill_color=[COLOR_R_COLUMN, COLOR_G_COLUMN, COLOR_B_COLUMN, COLOR_A_COLUMN],
        get_radius=CIRCLE_RADIUS_METERS,
        radius_min_pixels=CIRCLE_RADIUS_PIXELS,
        radius_max_pixels=CIRCLE_RADIUS_PIXELS,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=center_latitude,
        longitude=center_longitude,
        zoom=9,
        pitch=0,
        bearing=0,
    )

    tooltip = {
        "html": (
            "<b>{crime_count}</b> crimes<br/>"
            "Top type: {top_crime_type}<br/>"
            "LSOAs in cell: {unique_lsoa_count}"
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


def save_intensity_map(
    deck: pdk.Deck,
    month: str,
    maps_dir: Path = OUTPUTS_MAPS_DIR,
    *,
    verbose: bool = True,
) -> Path:
    """Write the map to a standalone HTML file."""
    maps_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_html_path(month, maps_dir)
    deck.to_html(str(output_path))
    if verbose:
        print(f"  Wrote map -> {output_path}")
    return output_path


def render_intensity_map(
    month: str,
    viz_dir: Path = VIZ_DATA_DIR,
    maps_dir: Path = OUTPUTS_MAPS_DIR,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """Load grid summary, build map, and save HTML for one month."""
    grid_df = load_grid_summary(month, viz_dir, verbose=verbose)
    map_df = prepare_map_data(grid_df)
    deck = build_intensity_map(map_df)
    if deck is None:
        raise ValueError(f"No grid cells to map for {month}.")

    output_path = save_intensity_map(deck, month, maps_dir, verbose=verbose)
    return map_df, output_path


def main() -> None:
    months = resolve_load_months(month_filter=PIPELINE_MONTHS)
    if not months:
        months = [DEFAULT_TESTING_MONTH]

    print("2D crime intensity map")
    print(f"Months: {', '.join(months)}")
    print("Encoding: colour = crime_count (log-scaled), fixed circle size")

    for month in months:
        print(f"\nRendering {month} ...")
        map_df, output_path = render_intensity_map(month)

        print()
        print(f"Summary for {month}")
        print("-" * 40)
        print(f"  Grid cells plotted: {len(map_df):,}")
        print(f"  Total crimes represented: {int(map_df[CRIME_COUNT_COLUMN].sum()):,}")
        print(f"  Mean crimes per cell: {map_df[CRIME_COUNT_COLUMN].mean():.2f}")
        print(f"  Max crimes in one cell: {int(map_df[CRIME_COUNT_COLUMN].max()):,}")
        print(f"  Output: {output_path}")

    print()
    print("Open the HTML file in a browser to explore the map.")


if __name__ == "__main__":
    main()
