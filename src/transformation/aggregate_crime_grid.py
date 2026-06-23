"""Aggregate cleaned crime points into a spatial grid summary for visualization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning.clean_crime_data import (
    HAS_VALID_COORDINATES_COLUMN,
    IS_WITHIN_LONDON_BBOX_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    MONTH_COLUMN,
    PROCESSED_FILENAME_TEMPLATE,
    ROW_ID_COLUMN,
)
from src.config import (
    PIPELINE_MONTHS,
    PROCESSED_DATA_DIR,
    VIZ_DATA_DIR,
)
from src.ingestion.load_crime_files import resolve_load_months

CRIME_TYPE_COLUMN = "crime_type"
LSOA_CODE_COLUMN = "lsoa_code"

LAT_GRID_COLUMN = "lat_grid"
LON_GRID_COLUMN = "lon_grid"
CRIME_COUNT_COLUMN = "crime_count"
UNIQUE_CRIME_TYPE_COUNT_COLUMN = "unique_crime_type_count"
TOP_CRIME_TYPE_COLUMN = "top_crime_type"
UNIQUE_LSOA_COUNT_COLUMN = "unique_lsoa_count"
CENTER_LATITUDE_COLUMN = "center_latitude"
CENTER_LONGITUDE_COLUMN = "center_longitude"

GRID_DECIMAL_PLACES = 3
OUTPUT_FILENAME_TEMPLATE = "crime_grid_summary_{month}.parquet"

GROUP_COLUMNS = [MONTH_COLUMN, LAT_GRID_COLUMN, LON_GRID_COLUMN]


def processed_parquet_path(
    month: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    return processed_dir / PROCESSED_FILENAME_TEMPLATE.format(month=month)


def output_parquet_path(
    month: str,
    viz_dir: Path = VIZ_DATA_DIR,
) -> Path:
    return viz_dir / OUTPUT_FILENAME_TEMPLATE.format(month=month)


def load_filtered_points(
    month: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load one processed month and keep rows with valid London coordinates."""
    parquet_path = processed_parquet_path(month, processed_dir)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Processed file not found: {parquet_path}")

    if verbose:
        print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)

    mask = df[HAS_VALID_COORDINATES_COLUMN] & df[IS_WITHIN_LONDON_BBOX_COLUMN]
    filtered = df.loc[mask].copy()
    if verbose:
        print(f"  {len(df):,} total rows -> {len(filtered):,} with valid London coordinates")
    return filtered


def add_grid_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Snap each point to a simple lat/lon grid using rounded coordinates."""
    df[LAT_GRID_COLUMN] = df[LATITUDE_COLUMN].round(GRID_DECIMAL_PLACES)
    df[LON_GRID_COLUMN] = df[LONGITUDE_COLUMN].round(GRID_DECIMAL_PLACES)
    return df


def top_crime_type_per_group(series: pd.Series) -> str | None:
    """Return the most frequent crime type in a grid cell."""
    counts = series.dropna().value_counts()
    if counts.empty:
        return None
    return str(counts.index[0])


def aggregate_crime_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize point-level crime data into one row per month/grid cell."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                *GROUP_COLUMNS,
                CRIME_COUNT_COLUMN,
                UNIQUE_CRIME_TYPE_COUNT_COLUMN,
                TOP_CRIME_TYPE_COLUMN,
                UNIQUE_LSOA_COUNT_COLUMN,
                CENTER_LATITUDE_COLUMN,
                CENTER_LONGITUDE_COLUMN,
            ]
        )

    df = add_grid_columns(df)

    summary = (
        df.groupby(GROUP_COLUMNS, as_index=False)
        .agg(
            crime_count=(ROW_ID_COLUMN, "count"),
            unique_crime_type_count=(CRIME_TYPE_COLUMN, "nunique"),
            unique_lsoa_count=(LSOA_CODE_COLUMN, "nunique"),
            center_latitude=(LAT_GRID_COLUMN, "first"),
            center_longitude=(LON_GRID_COLUMN, "first"),
        )
        .rename(
            columns={
                "crime_count": CRIME_COUNT_COLUMN,
                "unique_crime_type_count": UNIQUE_CRIME_TYPE_COUNT_COLUMN,
                "unique_lsoa_count": UNIQUE_LSOA_COUNT_COLUMN,
                "center_latitude": CENTER_LATITUDE_COLUMN,
                "center_longitude": CENTER_LONGITUDE_COLUMN,
            }
        )
    )

    top_types = (
        df.groupby(GROUP_COLUMNS)[CRIME_TYPE_COLUMN]
        .apply(top_crime_type_per_group)
        .reset_index(name=TOP_CRIME_TYPE_COLUMN)
    )
    summary = summary.merge(top_types, on=GROUP_COLUMNS, how="left")

    column_order = [
        MONTH_COLUMN,
        LAT_GRID_COLUMN,
        LON_GRID_COLUMN,
        CRIME_COUNT_COLUMN,
        UNIQUE_CRIME_TYPE_COUNT_COLUMN,
        TOP_CRIME_TYPE_COLUMN,
        UNIQUE_LSOA_COUNT_COLUMN,
        CENTER_LATITUDE_COLUMN,
        CENTER_LONGITUDE_COLUMN,
    ]
    return summary[column_order]


def save_grid_summary(
    df: pd.DataFrame,
    month: str,
    viz_dir: Path = VIZ_DATA_DIR,
    *,
    verbose: bool = True,
) -> Path:
    """Write the grid summary for one month to the viz data layer."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_parquet_path(month, viz_dir)
    df.to_parquet(output_path, index=False)
    if verbose:
        print(f"  Wrote {len(df):,} grid cells -> {output_path}")
    return output_path


def aggregate_month(
    month: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
    viz_dir: Path = VIZ_DATA_DIR,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Path, int]:
    """Load, aggregate, and save one month. Returns summary, path, input row count."""
    points = load_filtered_points(month, processed_dir, verbose=verbose)
    input_rows = len(points)
    summary = aggregate_crime_grid(points)
    output_path = save_grid_summary(summary, month, viz_dir, verbose=verbose)
    return summary, output_path, input_rows


def aggregate_months(
    months: list[str] | None = PIPELINE_MONTHS,
    processed_dir: Path = PROCESSED_DATA_DIR,
    viz_dir: Path = VIZ_DATA_DIR,
    *,
    verbose: bool = True,
) -> dict[str, Path]:
    """Aggregate each selected month. Returns month -> output path."""
    resolved = resolve_load_months(processed_dir.parent / "raw", months)
    if not resolved:
        return {}

    written: dict[str, Path] = {}
    for month in resolved:
        if verbose:
            print(f"\nAggregating grid for {month} ...")
        _, output_path, _ = aggregate_month(
            month,
            processed_dir,
            viz_dir,
            verbose=verbose,
        )
        written[month] = output_path
    return written


def verify_summary(
    summary: pd.DataFrame,
    input_row_count: int,
    *,
    verbose: bool = True,
) -> list[str]:
    """Check success criteria. Returns a list of issue messages (empty if all pass)."""
    issues: list[str] = []

    if summary.empty:
        issues.append("Aggregated dataset is empty.")
        return issues

    if len(summary) >= input_row_count:
        issues.append(
            f"Expected fewer rows than input points ({len(summary):,} >= {input_row_count:,})."
        )

    duplicate_cells = summary.duplicated(subset=GROUP_COLUMNS).sum()
    if duplicate_cells:
        issues.append(f"Found {duplicate_cells:,} duplicate grid cells.")

    if summary[CRIME_COUNT_COLUMN].sum() != input_row_count:
        issues.append(
            "crime_count totals do not match input row count "
            f"({int(summary[CRIME_COUNT_COLUMN].sum()):,} vs {input_row_count:,})."
        )

    if verbose and not issues:
        print("  Verification passed.")
    elif verbose:
        for issue in issues:
            print(f"  Verification issue: {issue}")

    return issues


def main() -> None:
    months = resolve_load_months()
    if not months:
        print("No months selected. Edit PIPELINE_MONTHS in src/config.py.")
        raise SystemExit(1)

    print("Spatial grid aggregation")
    print(f"Months: {', '.join(months)}")
    print(f"Grid resolution: round(coordinate, {GRID_DECIMAL_PLACES})")

    all_issues: list[str] = []
    for month in months:
        summary, output_path, input_rows = aggregate_month(month)

        print()
        print(f"Summary for {month}")
        print("-" * 40)
        print(f"  Input points: {input_rows:,}")
        print(f"  Grid cells: {len(summary):,}")
        print(f"  Compression ratio: {input_rows / len(summary):.1f}x")
        print(f"  Total crimes in grid: {int(summary[CRIME_COUNT_COLUMN].sum()):,}")
        print(f"  Mean crimes per cell: {summary[CRIME_COUNT_COLUMN].mean():.2f}")
        print(f"  Max crimes in one cell: {int(summary[CRIME_COUNT_COLUMN].max()):,}")
        print(f"  Output: {output_path}")

        all_issues.extend(verify_summary(summary, input_rows))

    if all_issues:
        print()
        print("Aggregation finished with verification issues:")
        for issue in all_issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    print()
    print("Aggregation complete.")


if __name__ == "__main__":
    main()
