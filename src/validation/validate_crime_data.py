"""Generate a data quality report for cleaned London crime pipeline output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.cleaning.clean_crime_data import (
    HAS_COORDINATES_COLUMN,
    HAS_VALID_COORDINATES_COLUMN,
    IS_WITHIN_LONDON_BBOX_COLUMN,
    PROCESSED_FILENAME_TEMPLATE,
    ROW_ID_COLUMN,
    clean_months,
)
from src.config import (
    OUTPUTS_REPORTS_DIR,
    PIPELINE_MONTHS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.ingestion.load_crime_files import (
    SOURCE_FILE_COLUMN,
    load_months,
    resolve_load_months,
)
from src.validation.profile_raw_data import count_missing_text

REPORT_FILENAME_TEMPLATE = "data_quality_report_{month}.txt"

CRIME_ID_COLUMN = "crime_id"
LSOA_CODE_COLUMN = "lsoa_code"
LSOA_NAME_COLUMN = "lsoa_name"
CRIME_TYPE_COLUMN = "crime_type"
LONGITUDE_COLUMN = "longitude"
LATITUDE_COLUMN = "latitude"

TOP_LSOA_COUNT = 20


def processed_parquet_path(
    month: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    return processed_dir / PROCESSED_FILENAME_TEMPLATE.format(month=month)


def load_cleaned_month(
    month: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
    *,
    raw_dir: Path = RAW_DATA_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load cleaned data from Parquet, or clean from raw if the file is missing."""
    parquet_path = processed_parquet_path(month, processed_dir)
    if parquet_path.is_file():
        if verbose:
            print(f"Loading cleaned data from {parquet_path} ...")
        return pd.read_parquet(parquet_path)

    if verbose:
        print(f"Processed file not found: {parquet_path}")
        print("Cleaning from raw data instead ...")
    cleaned = clean_months([month], raw_dir, verbose=verbose)
    return cleaned[cleaned["source_month_folder"] == month].copy()


def format_min_max(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return f"{numeric.min():.6f} / {numeric.max():.6f}"
    return "(no valid values)"


def build_overall_section(
    cleaned_df: pd.DataFrame,
    *,
    raw_row_count: int,
    file_count: int,
) -> list[str]:
    lines: list[str] = []
    cleaned_count = len(cleaned_df)

    missing_longitude = int(cleaned_df[LONGITUDE_COLUMN].isna().sum())
    missing_latitude = int(cleaned_df[LATITUDE_COLUMN].isna().sum())
    rows_with_coordinates = int(cleaned_df[HAS_COORDINATES_COLUMN].sum())
    rows_with_invalid_coordinates = int(
        cleaned_df[HAS_COORDINATES_COLUMN].sum()
        - cleaned_df[HAS_VALID_COORDINATES_COLUMN].sum()
    )
    rows_outside_london_bbox = int(
        cleaned_df[HAS_VALID_COORDINATES_COLUMN].sum()
        - cleaned_df[IS_WITHIN_LONDON_BBOX_COLUMN].sum()
    )
    rows_map_ready = int(cleaned_df[IS_WITHIN_LONDON_BBOX_COLUMN].sum())

    duplicate_row_ids = int(
        cleaned_df[ROW_ID_COLUMN].duplicated(keep=False).sum()
    )
    unique_row_ids = int(cleaned_df[ROW_ID_COLUMN].nunique())

    lines.append("Overall")
    lines.append("-" * 60)
    lines.append(f"  Total raw rows: {raw_row_count:,}")
    lines.append(f"  Total cleaned rows: {cleaned_count:,}")
    if raw_row_count != cleaned_count:
        lines.append(
            f"  Raw vs cleaned row difference: {raw_row_count - cleaned_count:,}"
        )
    lines.append(f"  Number of files loaded: {file_count:,}")
    lines.append("")
    lines.append("  Coordinates")
    lines.append(f"    Rows with coordinates: {rows_with_coordinates:,}")
    lines.append(f"    Rows missing longitude: {missing_longitude:,}")
    lines.append(f"    Rows missing latitude: {missing_latitude:,}")
    lines.append(f"    Rows with invalid coordinates: {rows_with_invalid_coordinates:,}")
    lines.append(f"    Rows outside London bounding box: {rows_outside_london_bbox:,}")
    lines.append(f"    Rows map-ready (valid coords within London): {rows_map_ready:,}")
    if cleaned_count:
        map_ready_pct = 100.0 * rows_map_ready / cleaned_count
        lines.append(f"    Map-ready share of cleaned rows: {map_ready_pct:.2f}%")
    lines.append("")
    lines.append("  Identifiers and geography")
    lines.append(
        f"    Missing Crime ID count: {count_missing_text(cleaned_df[CRIME_ID_COLUMN]):,}"
    )
    lines.append(
        f"    Missing LSOA code count: {count_missing_text(cleaned_df[LSOA_CODE_COLUMN]):,}"
    )
    lines.append(f"    Unique row_id values: {unique_row_ids:,}")
    lines.append(f"    Duplicate row_id values: {duplicate_row_ids:,}")
    lines.append("")
    return lines


def build_by_file_section(cleaned_df: pd.DataFrame) -> list[str]:
    lines: list[str] = ["By file", "-" * 60]

    if cleaned_df.empty:
        lines.append("  (no data)")
        lines.append("")
        return lines

    grouped = cleaned_df.groupby(SOURCE_FILE_COLUMN, sort=True)
    for source_file, file_df in grouped:
        missing_coords = int((~file_df[HAS_COORDINATES_COLUMN]).sum())
        crime_type_count = int(file_df[CRIME_TYPE_COLUMN].nunique(dropna=True))

        lines.append(f"  {source_file}")
        lines.append(f"    Row count: {len(file_df):,}")
        lines.append(f"    Missing coordinate count: {missing_coords:,}")
        lines.append(
            f"    Latitude min/max: {format_min_max(file_df[LATITUDE_COLUMN])}"
        )
        lines.append(
            f"    Longitude min/max: {format_min_max(file_df[LONGITUDE_COLUMN])}"
        )
        lines.append(f"    Unique crime type count: {crime_type_count:,}")
        lines.append("")

    return lines


def build_by_crime_type_section(cleaned_df: pd.DataFrame) -> list[str]:
    lines: list[str] = ["By crime type", "-" * 60]

    if cleaned_df.empty:
        lines.append("  (no data)")
        lines.append("")
        return lines

    counts = cleaned_df[CRIME_TYPE_COLUMN].value_counts(dropna=False).sort_values(
        ascending=False
    )
    total = len(cleaned_df)

    for crime_type, count in counts.items():
        label = crime_type if pd.notna(crime_type) else "(missing)"
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"  {label}: {count:,} ({pct:.2f}%)")

    lines.append("")
    return lines


def build_by_lsoa_section(cleaned_df: pd.DataFrame) -> list[str]:
    lines: list[str] = ["By LSOA", "-" * 60]

    if cleaned_df.empty:
        lines.append("  (no data)")
        lines.append("")
        return lines

    lsoa_df = cleaned_df.dropna(subset=[LSOA_CODE_COLUMN]).copy()
    unique_lsoas = int(lsoa_df[LSOA_CODE_COLUMN].nunique())
    lines.append(f"  Number of unique LSOAs: {unique_lsoas:,}")
    lines.append("")
    lines.append(f"  Top {TOP_LSOA_COUNT} LSOAs by crime count:")

    if lsoa_df.empty:
        lines.append("    (no LSOA codes present)")
        lines.append("")
        return lines

    top_lsoas = (
        lsoa_df.groupby([LSOA_CODE_COLUMN, LSOA_NAME_COLUMN], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(TOP_LSOA_COUNT)
    )

    for (lsoa_code, lsoa_name), count in top_lsoas.items():
        name_label = lsoa_name if pd.notna(lsoa_name) else "(no name)"
        lines.append(f"    {lsoa_code} | {name_label}: {count:,}")

    lines.append("")
    return lines


def build_data_quality_report(
    cleaned_df: pd.DataFrame,
    *,
    month: str,
    raw_row_count: int,
    file_count: int,
) -> str:
    """Build the full text report for one cleaned month."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "London Crime Data Quality Report",
        "=" * 60,
        f"Month: {month}",
        f"Generated: {generated_at}",
        "",
    ]

    lines.extend(
        build_overall_section(
            cleaned_df,
            raw_row_count=raw_row_count,
            file_count=file_count,
        )
    )
    lines.extend(build_by_file_section(cleaned_df))
    lines.extend(build_by_crime_type_section(cleaned_df))
    lines.extend(build_by_lsoa_section(cleaned_df))

    lines.append("Notes")
    lines.append("-" * 60)
    lines.append(
        "  Map-ready rows have valid coordinates inside the London bounding box."
    )
    lines.append(
        "  Rows outside the London bbox still have valid global coordinates."
    )
    lines.append(
        "  Out-of-bounds rows are kept in the cleaned dataset; filter them when"
    )
    lines.append("  building map-ready exports.")

    return "\n".join(lines)


def validate_month(
    month: str,
    *,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = OUTPUTS_REPORTS_DIR,
    verbose: bool = True,
) -> tuple[str, Path]:
    """Validate one month and write the report. Returns report text and path."""
    if verbose:
        print(f"Validating month: {month}")

    raw_df = load_months([month], raw_dir, verbose=verbose)
    raw_row_count = len(raw_df)
    file_count = int(raw_df[SOURCE_FILE_COLUMN].nunique()) if not raw_df.empty else 0

    cleaned_df = load_cleaned_month(
        month,
        processed_dir,
        raw_dir=raw_dir,
        verbose=verbose,
    )
    if cleaned_df.empty:
        raise ValueError(f"No cleaned data available for month: {month}")

    report = build_data_quality_report(
        cleaned_df,
        month=month,
        raw_row_count=raw_row_count,
        file_count=file_count,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / REPORT_FILENAME_TEMPLATE.format(month=month)
    output_path.write_text(report, encoding="utf-8")

    if verbose:
        print(f"Report saved to: {output_path}")

    return report, output_path


def validate_months(
    months: list[str] | None = PIPELINE_MONTHS,
    *,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = OUTPUTS_REPORTS_DIR,
    verbose: bool = True,
) -> dict[str, Path]:
    """Validate one or more months and write reports."""
    resolved_months = resolve_load_months(raw_dir, months)
    written: dict[str, Path] = {}

    for month in resolved_months:
        _report, output_path = validate_month(
            month,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            output_dir=output_dir,
            verbose=verbose,
        )
        written[month] = output_path

    return written


def main() -> None:
    months = resolve_load_months(month_filter=PIPELINE_MONTHS)
    if not months:
        print(
            "No months selected for validation. "
            "Edit PIPELINE_MONTHS in src/config.py."
        )
        raise SystemExit(1)

    print(f"Validating months: {', '.join(months)}")
    written_reports = validate_months(months)

    for month in months:
        report = written_reports[month].read_text(encoding="utf-8")
        print()
        print(report)

    if len(months) == 1 and PIPELINE_MONTHS is not None:
        print(
            "\nTo validate more months, edit PIPELINE_MONTHS in src/config.py."
        )


if __name__ == "__main__":
    main()
