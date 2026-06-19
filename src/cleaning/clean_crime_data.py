"""Clean and standardize raw London crime data into a processed dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    LONDON_BBOX,
    PIPELINE_MONTHS,
    PROCESSED_DATA_DIR,
    RAW_COLUMN_RENAME_MAP,
    RAW_DATA_DIR,
)
from src.ingestion.load_crime_files import (
    SOURCE_FILE_COLUMN,
    SOURCE_MONTH_FOLDER_COLUMN,
    SOURCE_ROW_NUMBER_COLUMN,
    load_months,
    resolve_load_months,
)

LONGITUDE_COLUMN = "longitude"
LATITUDE_COLUMN = "latitude"
MONTH_COLUMN = "month"
FALLS_WITHIN_COLUMN = "falls_within"

DATA_SOURCE_NAME_COLUMN = "data_source_name"
HAS_COORDINATES_COLUMN = "has_coordinates"
HAS_VALID_COORDINATES_COLUMN = "has_valid_coordinates"
IS_WITHIN_LONDON_BBOX_COLUMN = "is_within_london_bbox"
MONTH_START_DATE_COLUMN = "month_start_date"
YEAR_COLUMN = "year"
MONTH_NUMBER_COLUMN = "month_number"
MONTH_LABEL_COLUMN = "month_label"
ROW_ID_COLUMN = "row_id"

ROW_ID_PREFIX = "LCS"

LATITUDE_MIN, LATITUDE_MAX = -90.0, 90.0
LONGITUDE_MIN, LONGITUDE_MAX = -180.0, 180.0

PROCESSED_FILENAME_TEMPLATE = "london_crime_clean_{month}.parquet"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw column headers to snake_case using the shared mapping."""
    return df.rename(columns=RAW_COLUMN_RENAME_MAP)


def build_data_source_name(falls_within: pd.Series) -> pd.Series:
    """Build uppercase acronyms from the first letter of each word in falls_within.

    Examples: "City of London Police" -> "COLP",
              "Metropolitan Police Service" -> "MPS".
    Blank or missing values become <NA>.
    """
    as_text = falls_within.astype("string")
    stripped = as_text.str.strip()
    empty_mask = as_text.isna() | (stripped == "")

    def initials(words: object) -> str | None:
        if not isinstance(words, list) or not words:
            return None
        acronym = "".join(word[0].upper() for word in words if word)
        return acronym or None

    acronyms = stripped.str.split().apply(initials).astype("string")
    return acronyms.mask(empty_mask, pd.NA)


def add_numeric_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert longitude and latitude to numeric, coercing bad values to NaN."""
    df[LONGITUDE_COLUMN] = pd.to_numeric(df[LONGITUDE_COLUMN], errors="coerce")
    df[LATITUDE_COLUMN] = pd.to_numeric(df[LATITUDE_COLUMN], errors="coerce")
    return df


def add_coordinate_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag coordinate presence, validity, and whether the point is in London."""
    longitude = df[LONGITUDE_COLUMN]
    latitude = df[LATITUDE_COLUMN]

    has_coordinates = longitude.notna() & latitude.notna()

    in_global_range = (
        latitude.between(LATITUDE_MIN, LATITUDE_MAX)
        & longitude.between(LONGITUDE_MIN, LONGITUDE_MAX)
    )
    has_valid_coordinates = has_coordinates & in_global_range

    in_london = (
        longitude.between(LONDON_BBOX["longitude_min"], LONDON_BBOX["longitude_max"])
        & latitude.between(LONDON_BBOX["latitude_min"], LONDON_BBOX["latitude_max"])
    )
    is_within_london_bbox = has_valid_coordinates & in_london

    df[HAS_COORDINATES_COLUMN] = has_coordinates
    df[HAS_VALID_COORDINATES_COLUMN] = has_valid_coordinates
    df[IS_WITHIN_LONDON_BBOX_COLUMN] = is_within_london_bbox
    return df


def add_data_source_name(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the short police-force acronym from falls_within."""
    if FALLS_WITHIN_COLUMN in df.columns:
        df[DATA_SOURCE_NAME_COLUMN] = build_data_source_name(df[FALLS_WITHIN_COLUMN])
    else:
        df[DATA_SOURCE_NAME_COLUMN] = pd.Series(pd.NA, index=df.index, dtype="string")
    return df


def add_month_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the YYYY-MM month string into date parts."""
    month_start = pd.to_datetime(
        df[MONTH_COLUMN].astype("string") + "-01", errors="coerce"
    )
    df[MONTH_START_DATE_COLUMN] = month_start
    df[YEAR_COLUMN] = month_start.dt.year.astype("Int64")
    df[MONTH_NUMBER_COLUMN] = month_start.dt.month.astype("Int64")
    df[MONTH_LABEL_COLUMN] = month_start.dt.strftime("%B %Y").astype("string")
    return df


def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable row id that does not depend on the (often missing) crime_id.

    Format: LCS/<data_source_name>/<month>/<source_row_number>
    """
    data_source = df[DATA_SOURCE_NAME_COLUMN].astype("string").fillna("UNK")
    month = df[MONTH_COLUMN].astype("string").fillna("UNK")
    row_number = df[SOURCE_ROW_NUMBER_COLUMN].astype("string")

    df[ROW_ID_COLUMN] = (
        ROW_ID_PREFIX + "/" + data_source + "/" + month + "/" + row_number
    )
    return df


def clean_crime_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on a loaded raw crime dataframe."""
    if df.empty:
        return df

    df = standardize_columns(df)
    df = add_numeric_coordinates(df)
    df = add_data_source_name(df)
    df = add_coordinate_flags(df)
    df = add_month_fields(df)
    df = add_row_id(df)
    return df


def save_processed(
    df: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
    *,
    verbose: bool = True,
) -> dict[str, Path]:
    """Write one Parquet file per source month folder. Returns month -> path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for month, month_df in df.groupby(SOURCE_MONTH_FOLDER_COLUMN, sort=True):
        output_path = output_dir / PROCESSED_FILENAME_TEMPLATE.format(month=month)
        month_df.to_parquet(output_path, index=False)
        written[str(month)] = output_path
        if verbose:
            print(f"  Wrote {len(month_df):,} rows -> {output_path}")

    return written


def clean_months(
    months: list[str] | None = PIPELINE_MONTHS,
    raw_dir: Path = RAW_DATA_DIR,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load raw months via the ingestion loader, then clean them."""
    raw_df = load_months(months, raw_dir, verbose=verbose)
    if raw_df.empty:
        return raw_df

    if verbose:
        print("\nCleaning loaded data ...")
    return clean_crime_data(raw_df)


def main() -> None:
    months = resolve_load_months()
    if not months:
        print("No months selected for cleaning. Edit PIPELINE_MONTHS in src/config.py.")
        raise SystemExit(1)

    df = clean_months()

    if df.empty:
        print("\nNo data to clean.")
        raise SystemExit(1)

    print()
    print("Clean summary")
    print("-" * 40)
    print(f"  Months: {', '.join(months)}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows with coordinates: {int(df[HAS_COORDINATES_COLUMN].sum()):,}")
    print(f"  Rows with valid coordinates: {int(df[HAS_VALID_COORDINATES_COLUMN].sum()):,}")
    print(f"  Rows within London bbox: {int(df[IS_WITHIN_LONDON_BBOX_COLUMN].sum()):,}")
    print(f"  Unique row_id values: {df[ROW_ID_COLUMN].nunique():,}")
    print()
    print("Rows by data_source_name:")
    counts = df[DATA_SOURCE_NAME_COLUMN].value_counts(dropna=False).sort_index()
    for name, count in counts.items():
        label = name if pd.notna(name) else "(NaN)"
        print(f"  {label}: {count:,}")

    print()
    print("Writing processed Parquet files:")
    written = save_processed(df)
    if not written:
        print("  (nothing written)")


if __name__ == "__main__":
    main()
