"""Colour/elevation encoding and PyDeck rendering for the 3D hexagon crime map.

Hex geometry lives in :mod:`src.transformation.hex_grid`; tunable visual
parameters (radius, elevation scale, colour gamma, camera) live in
:mod:`src.config`. This module only handles encoding and rendering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk

from src.config import (
    HEX_CAMERA_BEARING as CAMERA_BEARING,
    HEX_CAMERA_PITCH as CAMERA_PITCH,
    HEX_CAMERA_ZOOM as CAMERA_ZOOM,
    HEX_MAP_HEIGHT,
    HEX_COLOR_GAMMA as COLOR_GAMMA,
    HEX_COLUMN_DISK_SIDES as COLUMN_DISK_SIDES,
    HEX_COVERAGE,
    HEX_ELEVATION_POWER as ELEVATION_POWER,
    HEX_ELEVATION_SCALE as ELEVATION_SCALE,
    HEX_FILL_ALPHA as FILL_ALPHA,
    HEX_RADIUS_METERS,
    AREA_BBOX,
)
from src.transformation.aggregate_hex_grid import CRIME_COUNT_COLUMN

LONGITUDE_COLUMN = "longitude"
LATITUDE_COLUMN = "latitude"
ELEVATION_COLUMN = "elevation"
COLOR_R_COLUMN = "color_r"
COLOR_G_COLUMN = "color_g"
COLOR_B_COLUMN = "color_b"
COLOR_A_COLUMN = "color_a"

# Smooth gradient: near-black (low) -> vivid red (high). Values are interpolation stops.
GRADIENT_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, (18, 12, 14)),
    (0.10, (32, 16, 18)),
    (0.22, (52, 20, 24)),
    (0.35, (78, 26, 30)),
    (0.48, (110, 32, 34)),
    (0.60, (148, 38, 36)),
    (0.72, (188, 44, 38)),
    (0.84, (225, 48, 40)),
    (0.94, (248, 52, 42)),
    (1.0, (255, 55, 45)),
]


def _lerp_rgb(
    t: float,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(start[i] + (end[i] - start[i]) * t) for i in range(3))


def normalized_value_to_rgb(t: float) -> tuple[int, int, int]:
    """Map a 0-1 value to a smoothly interpolated dark -> red colour."""
    t = max(0.0, min(1.0, t))
    for index in range(len(GRADIENT_STOPS) - 1):
        start_t, start_rgb = GRADIENT_STOPS[index]
        end_t, end_rgb = GRADIENT_STOPS[index + 1]
        if t <= end_t or index == len(GRADIENT_STOPS) - 2:
            span = end_t - start_t
            segment_t = 0.0 if span <= 0 else (t - start_t) / span
            return _lerp_rgb(segment_t, start_rgb, end_rgb)
    return GRADIENT_STOPS[-1][1]


def encode_hex_cells(hex_df: pd.DataFrame) -> pd.DataFrame:
    """Attach smooth colour + elevation fields to aggregated hex cells.

    Colour is a sqrt-normalized, gamma-stretched gradient; elevation is
    ``count^power``. Normalisation is relative to the current filter so
    colours reflect the selected slice, not the full dataset.
    """
    if hex_df.empty:
        return hex_df

    grouped = hex_df.copy()
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

    center_latitude = (AREA_BBOX["latitude_min"] + AREA_BBOX["latitude_max"]) / 2
    center_longitude = (AREA_BBOX["longitude_min"] + AREA_BBOX["longitude_max"]) / 2

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
        height=HEX_MAP_HEIGHT,
    )
