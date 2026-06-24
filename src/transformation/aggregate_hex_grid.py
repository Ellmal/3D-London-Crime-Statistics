"""Aggregate cleaned crime points into the hex/month/crime-type visual-ready dataset.

The single output file ``data/viz/crime_hex_3d_month.parquet`` is the only
artifact produced by the final pipeline and is all the Streamlit app needs at
runtime.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning.clean_crime_data import (
    HAS_VALID_COORDINATES_COLUMN,
    IS_WITHIN_LONDON_BBOX_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    MONTH_COLUMN,
    ROW_ID_COLUMN,
)
from src.config import VIZ_DATA_DIR
from src.transformation.hex_grid import (
    HEX_Q_COLUMN,
    HEX_R_COLUMN,
    assign_hex_columns,
    hex_centers,
)

CRIME_TYPE_COLUMN = "crime_type"
CRIME_COUNT_COLUMN = "crime_count"

HEX_MONTH_TYPE_OUTPUT_FILENAME = "crime_hex_3d_month.parquet"

GROUP_COLUMNS = [MONTH_COLUMN, CRIME_TYPE_COLUMN, HEX_Q_COLUMN, HEX_R_COLUMN]


def hex_month_type_output_path(viz_dir: Path = VIZ_DATA_DIR) -> Path:
    return viz_dir / HEX_MONTH_TYPE_OUTPUT_FILENAME


def filter_london_points(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Keep rows with valid coordinates inside the London bounding box."""
    mask = df[HAS_VALID_COORDINATES_COLUMN] & df[IS_WITHIN_LONDON_BBOX_COLUMN]
    filtered = df.loc[mask].copy()
    if verbose:
        print(
            f"  {len(df):,} total rows -> {len(filtered):,} with valid London coordinates"
        )
    return filtered


def aggregate_crime_hex_month_type(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize points into one row per month / crime type / hex cell."""
    output_columns = [*GROUP_COLUMNS, LONGITUDE_COLUMN, LATITUDE_COLUMN, CRIME_COUNT_COLUMN]
    if df.empty:
        return pd.DataFrame(columns=output_columns)

    binned = assign_hex_columns(df)
    summary = (
        binned.groupby(GROUP_COLUMNS, as_index=False)
        .agg(crime_count=(ROW_ID_COLUMN, "count"))
        .rename(columns={"crime_count": CRIME_COUNT_COLUMN})
    )

    longitude, latitude = hex_centers(summary[HEX_Q_COLUMN], summary[HEX_R_COLUMN])
    summary[LONGITUDE_COLUMN] = longitude
    summary[LATITUDE_COLUMN] = latitude

    return summary[output_columns]


def build_hex_month_type(
    cleaned_df: pd.DataFrame,
    viz_dir: Path = VIZ_DATA_DIR,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Path, int]:
    """Filter to London points, aggregate to hex grid, and save.

    Returns ``(summary, output_path, input_row_count)``.
    """
    points = filter_london_points(cleaned_df, verbose=verbose)
    input_rows = len(points)
    summary = aggregate_crime_hex_month_type(points)

    viz_dir.mkdir(parents=True, exist_ok=True)
    output_path = hex_month_type_output_path(viz_dir)
    summary.to_parquet(output_path, index=False)
    if verbose:
        print(f"  Wrote {len(summary):,} rows -> {output_path}")

    return summary, output_path, input_rows


def verify_hex_summary(
    summary: pd.DataFrame,
    input_row_count: int,
    *,
    verbose: bool = True,
) -> list[str]:
    """Check the hex aggregation. Returns issue messages (empty list if all pass)."""
    issues: list[str] = []

    if input_row_count == 0:
        if not summary.empty:
            issues.append("Expected empty summary for zero input rows.")
        return issues

    if summary.empty:
        issues.append("Aggregated hex dataset is empty.")
        return issues

    duplicate_cells = summary.duplicated(subset=GROUP_COLUMNS).sum()
    if duplicate_cells:
        issues.append(f"Found {duplicate_cells:,} duplicate month/type/hex cells.")

    if summary[CRIME_COUNT_COLUMN].sum() != input_row_count:
        issues.append(
            "crime_count totals do not match input row count "
            f"({int(summary[CRIME_COUNT_COLUMN].sum()):,} vs {input_row_count:,})."
        )

    if verbose and not issues:
        print("  Hex verification passed.")
    elif verbose:
        for issue in issues:
            print(f"  Hex verification issue: {issue}")

    return issues
