"""Build a standalone 3D hexagon extrusion map from cleaned crime points."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk

from src.cleaning.clean_crime_data import (
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
)
from src.config import (
    DEFAULT_TESTING_MONTH,
    LONDON_BBOX,
    OUTPUTS_MAPS_DIR,
    PIPELINE_MONTHS,
)
from src.ingestion.load_crime_files import resolve_load_months
from src.transformation.aggregate_crime_grid import (
    CRIME_COUNT_COLUMN,
    load_filtered_points,
    output_parquet_path,
)

OUTPUT_FILENAME_TEMPLATE = "london_crime_3d_hex_{month}.html"

HEX_Q_COLUMN = "hex_q"
HEX_R_COLUMN = "hex_r"
ELEVATION_COLUMN = "elevation"
COLOR_R_COLUMN = "color_r"
COLOR_G_COLUMN = "color_g"
COLOR_B_COLUMN = "color_b"
COLOR_A_COLUMN = "color_a"

# Smooth gradient: dark (low) -> bright (high). Values are interpolation stops, not bands.
GRADIENT_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, (38, 26, 52)),
    (0.10, (55, 36, 62)),
    (0.22, (78, 48, 68)),
    (0.35, (105, 62, 72)),
    (0.48, (138, 82, 78)),
    (0.60, (172, 108, 88)),
    (0.72, (205, 140, 105)),
    (0.84, (235, 180, 130)),
    (0.94, (252, 220, 165)),
    (1.0, (255, 248, 215)),
]

# ~350 m hexagons suit London's extent
HEX_RADIUS_METERS = 100
HEX_COVERAGE = 0.88
COLUMN_DISK_SIDES = 6

# count^power * scale → meters; power 0.65 stretches outliers more than sqrt (0.5)
ELEVATION_POWER = 0.65
ELEVATION_SCALE = 400
FILL_ALPHA = 230
# Power < 1 stretches the low end across more of the gradient for visible differences
COLOR_GAMMA = 0.62

CAMERA_PITCH = 55
CAMERA_BEARING = -20
CAMERA_ZOOM = 9.2

REF_LATITUDE = (LONDON_BBOX["latitude_min"] + LONDON_BBOX["latitude_max"]) / 2
METERS_PER_DEGREE_LATITUDE = 110_540.0
METERS_PER_DEGREE_LONGITUDE = 111_320.0 * math.cos(math.radians(REF_LATITUDE))
SQRT3 = math.sqrt(3.0)


def output_html_path(
    month: str,
    maps_dir: Path = OUTPUTS_MAPS_DIR,
) -> Path:
    return maps_dir / OUTPUT_FILENAME_TEMPLATE.format(month=month)


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


def normalized_value_to_rgb(t: float) -> tuple[int, int, int]:
    """Map a 0-1 value to a smoothly interpolated dark -> bright colour."""
    t = max(0.0, min(1.0, t))
    for index in range(len(GRADIENT_STOPS) - 1):
        start_t, start_rgb = GRADIENT_STOPS[index]
        end_t, end_rgb = GRADIENT_STOPS[index + 1]
        if t <= end_t or index == len(GRADIENT_STOPS) - 2:
            span = end_t - start_t
            segment_t = 0.0 if span <= 0 else (t - start_t) / span
            return _lerp_rgb(segment_t, start_rgb, end_rgb)
    return GRADIENT_STOPS[-1][1]


def lon_lat_to_meters(longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = longitude * METERS_PER_DEGREE_LONGITUDE
    y = latitude * METERS_PER_DEGREE_LATITUDE
    return x, y


def meters_to_lon_lat(x: float, y: float) -> tuple[float, float]:
    return x / METERS_PER_DEGREE_LONGITUDE, y / METERS_PER_DEGREE_LATITUDE


def cube_round(q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round fractional axial hex coordinates to the nearest cell."""
    s = -q - r
    rq = np.round(q)
    rr = np.round(r)
    rs = np.round(s)

    dq = np.abs(rq - q)
    dr = np.abs(rr - r)
    ds = np.abs(rs - s)

    fix_q = (dq > dr) & (dq > ds)
    fix_r = (dr > ds) & ~fix_q

    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    return rq.astype(np.int64), rr.astype(np.int64)


def hex_cell_center(q: int, r: int, radius_meters: float) -> tuple[float, float]:
    x = radius_meters * 1.5 * q
    y = radius_meters * SQRT3 * (r + q / 2.0)
    return meters_to_lon_lat(x, y)


def aggregate_points_to_hex(
    point_df: pd.DataFrame,
    radius_meters: float = HEX_RADIUS_METERS,
) -> pd.DataFrame:
    """Bin crime points into hex cells and attach smooth colour + elevation fields."""
    longitude = point_df[LONGITUDE_COLUMN].to_numpy(dtype=float)
    latitude = point_df[LATITUDE_COLUMN].to_numpy(dtype=float)

    x, y = lon_lat_to_meters(longitude, latitude)
    q = (2.0 / 3.0 * x) / radius_meters
    r = (-1.0 / 3.0 * x + SQRT3 / 3.0 * y) / radius_meters
    hex_q, hex_r = cube_round(q, r)

    binned = pd.DataFrame({HEX_Q_COLUMN: hex_q, HEX_R_COLUMN: hex_r})
    grouped = (
        binned.groupby([HEX_Q_COLUMN, HEX_R_COLUMN], as_index=False)
        .size()
        .rename(columns={"size": CRIME_COUNT_COLUMN})
    )

    centers = grouped.apply(
        lambda row: hex_cell_center(
            int(row[HEX_Q_COLUMN]),
            int(row[HEX_R_COLUMN]),
            radius_meters,
        ),
        axis=1,
    )
    grouped[LONGITUDE_COLUMN] = centers.map(lambda pair: pair[0])
    grouped[LATITUDE_COLUMN] = centers.map(lambda pair: pair[1])

    counts = grouped[CRIME_COUNT_COLUMN].astype(float)
    sqrt_counts = np.sqrt(counts)
    min_sqrt = float(sqrt_counts.min())
    max_sqrt = float(sqrt_counts.max())
    if max_sqrt == min_sqrt:
        normalized = pd.Series(0.0, index=grouped.index)
    else:
        normalized = (sqrt_counts - min_sqrt) / (max_sqrt - min_sqrt)

    color_input = normalized.pow(COLOR_GAMMA)
    colors = color_input.map(normalized_value_to_rgb)
    grouped[COLOR_R_COLUMN] = colors.map(lambda rgb: int(rgb[0]))
    grouped[COLOR_G_COLUMN] = colors.map(lambda rgb: int(rgb[1]))
    grouped[COLOR_B_COLUMN] = colors.map(lambda rgb: int(rgb[2]))
    grouped[COLOR_A_COLUMN] = FILL_ALPHA

    grouped[ELEVATION_COLUMN] = np.power(counts, ELEVATION_POWER)
    grouped[CRIME_COUNT_COLUMN] = grouped[CRIME_COUNT_COLUMN].astype(int)
    return grouped


def build_hexagon_map(hex_df: pd.DataFrame) -> pdk.Deck | None:
    """Build a tilted 3D map with one hex column per aggregated cell."""
    if hex_df.empty:
        return None

    center_latitude = (LONDON_BBOX["latitude_min"] + LONDON_BBOX["latitude_max"]) / 2
    center_longitude = (LONDON_BBOX["longitude_min"] + LONDON_BBOX["longitude_max"]) / 2

    layer = pdk.Layer(
        "ColumnLayer",
        data=hex_df,
        get_position=[LONGITUDE_COLUMN, LATITUDE_COLUMN],
        get_elevation=ELEVATION_COLUMN,
        get_fill_color=[COLOR_R_COLUMN, COLOR_G_COLUMN, COLOR_B_COLUMN, COLOR_A_COLUMN],
        radius=HEX_RADIUS_METERS,
        coverage=HEX_COVERAGE,
        disk_resolution=COLUMN_DISK_SIDES,
        extruded=True,
        pickable=True,
        auto_highlight=True,
        elevation_scale=ELEVATION_SCALE,
        opacity=0.5,
        material=True,
    )

    view_state = pdk.ViewState(
        latitude=center_latitude,
        longitude=center_longitude,
        zoom=CAMERA_ZOOM,
        pitch=CAMERA_PITCH,
        bearing=CAMERA_BEARING,
    )

    tooltip = {
        "html": "<b>{crime_count}</b> crimes in hexagon",
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


def save_hexagon_map(
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


def render_hexagon_map(
    month: str,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """Load crime points, aggregate to hex cells, build map, and save HTML."""
    point_df = load_filtered_points(month, verbose=verbose)
    hex_df = aggregate_points_to_hex(point_df)
    deck = build_hexagon_map(hex_df)
    if deck is None:
        raise ValueError(f"No crime points to map for {month}.")

    output_path = save_hexagon_map(deck, month, verbose=verbose)
    return hex_df, output_path


def print_validation_summary(month: str, hex_count: int, point_total: int) -> None:
    """Compare hex totals against the grid summary from Mini 8."""
    parquet_path = output_parquet_path(month)
    if not parquet_path.is_file():
        print("  Grid summary not found; skipping cross-check.")
        return

    grid_df = pd.read_parquet(parquet_path)
    grid_total = int(grid_df[CRIME_COUNT_COLUMN].sum())
    delta = point_total - grid_total
    print(f"  Grid summary total (Mini 8): {grid_total:,}")
    if delta == 0:
        print("  Point count matches grid summary.")
    else:
        print(f"  Delta vs grid summary: {delta:+,}")
    print(f"  Hex cells rendered: {hex_count:,}")


def main() -> None:
    months = resolve_load_months(month_filter=PIPELINE_MONTHS)
    if not months:
        months = [DEFAULT_TESTING_MONTH]

    print("3D hexagon crime map")
    print(f"Months: {', '.join(months)}")
    print(
        "Encoding: hex columns, sqrt-scaled smooth gradient (dark low -> bright high), "
        f"radius={HEX_RADIUS_METERS}m, pitch={CAMERA_PITCH}°, "
        f"height=count^{ELEVATION_POWER}×{ELEVATION_SCALE}"
    )

    for month in months:
        print(f"\nRendering {month} ...")
        hex_df, output_path = render_hexagon_map(month)
        point_total = int(hex_df[CRIME_COUNT_COLUMN].sum())

        print()
        print(f"Summary for {month}")
        print("-" * 40)
        print(f"  Crime points represented: {point_total:,}")
        print_validation_summary(month, len(hex_df), point_total)
        print(f"  Min / max crimes per hex: {hex_df[CRIME_COUNT_COLUMN].min()} / {hex_df[CRIME_COUNT_COLUMN].max()}")
        print(f"  Hex radius: {HEX_RADIUS_METERS} m")
        print(f"  Elevation scale: {ELEVATION_SCALE}")
        print(f"  Output: {output_path}")

    print()
    print("Open the HTML file in a browser to explore the 3D map.")


if __name__ == "__main__":
    main()
