"""Pure hexagonal grid geometry shared by aggregation and rendering.

All hex binning math lives here so the precomputed visual-ready aggregation
(``data/viz/crime_hex_3d_month.parquet``) and the live 3D render always use an
identical grid. The hex size is read from ``src.config.HEX_RADIUS_METERS`` and
must not be redefined elsewhere.

This module is deliberately free of any plotting dependency (no pydeck) so it
can be imported by the pipeline without pulling in the rendering stack.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import AREA_BBOX, HEX_RADIUS_METERS

LONGITUDE_COLUMN = "longitude"
LATITUDE_COLUMN = "latitude"
HEX_Q_COLUMN = "hex_q"
HEX_R_COLUMN = "hex_r"

# Local equirectangular projection centred on AREA_BBOX. Good enough for binning
# at city scale and far cheaper than a full geographic projection.
REF_LATITUDE = (AREA_BBOX["latitude_min"] + AREA_BBOX["latitude_max"]) / 2
METERS_PER_DEGREE_LATITUDE = 110_540.0
METERS_PER_DEGREE_LONGITUDE = 111_320.0 * math.cos(math.radians(REF_LATITUDE))
SQRT3 = math.sqrt(3.0)


def lon_lat_to_meters(
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat degrees to local meters (equirectangular)."""
    x = longitude * METERS_PER_DEGREE_LONGITUDE
    y = latitude * METERS_PER_DEGREE_LATITUDE
    return x, y


def meters_to_lon_lat(x: float, y: float) -> tuple[float, float]:
    """Inverse of :func:`lon_lat_to_meters` for a single point."""
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


def hex_cell_center(
    q: int,
    r: int,
    radius_meters: float = HEX_RADIUS_METERS,
) -> tuple[float, float]:
    """Return the (lon, lat) centre of a single hex cell."""
    x = radius_meters * 1.5 * q
    y = radius_meters * SQRT3 * (r + q / 2.0)
    return meters_to_lon_lat(x, y)


def hex_centers(
    hex_q: np.ndarray,
    hex_r: np.ndarray,
    radius_meters: float = HEX_RADIUS_METERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized hex-centre lon/lat for arrays of axial coordinates."""
    q = np.asarray(hex_q, dtype=float)
    r = np.asarray(hex_r, dtype=float)
    x = radius_meters * 1.5 * q
    y = radius_meters * SQRT3 * (r + q / 2.0)
    longitude = x / METERS_PER_DEGREE_LONGITUDE
    latitude = y / METERS_PER_DEGREE_LATITUDE
    return longitude, latitude


def assign_hex_columns(
    df: pd.DataFrame,
    radius_meters: float = HEX_RADIUS_METERS,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``hex_q`` / ``hex_r`` cell indices added.

    Expects ``longitude`` and ``latitude`` columns. Points are snapped to the
    nearest pointy-top hex using the same projection and rounding everywhere.
    """
    longitude = df[LONGITUDE_COLUMN].to_numpy(dtype=float)
    latitude = df[LATITUDE_COLUMN].to_numpy(dtype=float)

    x, y = lon_lat_to_meters(longitude, latitude)
    q = (2.0 / 3.0 * x) / radius_meters
    r = (-1.0 / 3.0 * x + SQRT3 / 3.0 * y) / radius_meters
    hex_q, hex_r = cube_round(q, r)

    out = df.copy()
    out[HEX_Q_COLUMN] = hex_q
    out[HEX_R_COLUMN] = hex_r
    return out
