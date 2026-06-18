"""Load raw monthly crime CSV files into a combined Pandas dataframe."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_TESTING_MONTH, RAW_DATA_DIR

MONTH_LABEL_PATTERN = re.compile(r"^\d{4}-\d{2}$")
STREET_FILE_GLOB = "*-street.csv"

SOURCE_FILE_COLUMN = "source_file"
SOURCE_MONTH_FOLDER_COLUMN = "source_month_folder"
SOURCE_ROW_NUMBER_COLUMN = "source_row_number"
FALLS_WITHIN_COLUMN = "Falls within"
DATA_SOURCE_NAME_COLUMN = "data_source_name"

# ---------------------------------------------------------------------------
# Month filter - adjust this to control which months are loaded.
#
# Examples:
#   ["2025-05"]                      # May 2025 only (current default)
#   ["2025-05", "2025-06"]           # specific months
#   None                             # all month folders found under data/raw/
# ---------------------------------------------------------------------------
LOAD_MONTHS: list[str] | None = [DEFAULT_TESTING_MONTH]


def resolve_load_months(
    raw_dir: Path = RAW_DATA_DIR,
    month_filter: list[str] | None = LOAD_MONTHS,
) -> list[str]:
    """Return sorted month labels to load.

    Uses ``LOAD_MONTHS`` by default. When ``month_filter`` is ``None``, every
    ``YYYY-MM`` folder under ``raw_dir`` is included.
    """
    if month_filter is not None:
        return sorted(month_filter)

    if not raw_dir.exists():
        return []

    months = [
        path.name
        for path in raw_dir.iterdir()
        if path.is_dir() and MONTH_LABEL_PATTERN.match(path.name)
    ]
    return sorted(months)


def get_street_csv_files(month: str, raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Return sorted ``*-street.csv`` paths for a month folder."""
    month_dir = raw_dir / month
    if not month_dir.is_dir():
        return []
    return sorted(month_dir.glob(STREET_FILE_GLOB))


def build_data_source_name(falls_within: pd.Series) -> pd.Series:
    """Build uppercase acronyms from the first letter of each word in Falls within."""
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


def load_street_csv(
    csv_path: Path,
    month: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load one street CSV and attach source metadata columns."""
    if verbose:
        print(f"  Loading {csv_path.name} ...")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    df[SOURCE_FILE_COLUMN] = csv_path.name
    df[SOURCE_MONTH_FOLDER_COLUMN] = month
    # Line 1 is the header row in the source file.
    df[SOURCE_ROW_NUMBER_COLUMN] = df.index + 2
    if FALLS_WITHIN_COLUMN in df.columns:
        df[DATA_SOURCE_NAME_COLUMN] = build_data_source_name(df[FALLS_WITHIN_COLUMN])
    else:
        df[DATA_SOURCE_NAME_COLUMN] = pd.NA
    return df


def load_month(
    month: str = DEFAULT_TESTING_MONTH,
    raw_dir: Path = RAW_DATA_DIR,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and combine all ``*-street.csv`` files for a single month."""
    month_dir = raw_dir / month

    if verbose:
        print(f"Scanning {month_dir} for {STREET_FILE_GLOB} files ...")

    if not month_dir.is_dir():
        if verbose:
            print(f"  Month folder not found: {month_dir}")
        return pd.DataFrame()

    street_files = get_street_csv_files(month, raw_dir)
    non_street_files = sorted(
        path
        for path in month_dir.iterdir()
        if path.is_file() and not path.name.endswith("-street.csv")
    )

    if verbose and non_street_files:
        ignored = ", ".join(path.name for path in non_street_files)
        print(f"  Ignoring non-street files: {ignored}")

    if not street_files:
        if verbose:
            print(f"  No {STREET_FILE_GLOB} files found in {month_dir}")
        return pd.DataFrame()

    if verbose:
        print(f"  Found {len(street_files)} street file(s) for {month}")

    frames: list[pd.DataFrame] = []
    for csv_path in street_files:
        frames.append(load_street_csv(csv_path, month, verbose=verbose))

    combined = pd.concat(frames, ignore_index=True)

    if verbose:
        print(
            f"  Combined {len(street_files)} file(s) -> "
            f"{len(combined):,} rows for {month}"
        )

    return combined


def load_months(
    months: list[str] | None = LOAD_MONTHS,
    raw_dir: Path = RAW_DATA_DIR,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and combine street CSVs across one or more months.

    Uses ``LOAD_MONTHS`` by default. Pass ``None`` for ``months`` to load every
    month folder under ``raw_dir``.
    """
    resolved_months = resolve_load_months(raw_dir, months)

    if verbose:
        if not resolved_months:
            print("No month folders selected for loading.")
            return pd.DataFrame()

        label = ", ".join(resolved_months)
        print(f"Loading months: {label}")

    frames: list[pd.DataFrame] = []
    for month in resolved_months:
        if verbose:
            print()
        month_df = load_month(month, raw_dir, verbose=verbose)
        if not month_df.empty:
            frames.append(month_df)

    if not frames:
        if verbose:
            print("\nNo rows loaded.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if verbose:
        print(
            f"\nLoaded {len(resolved_months)} month(s) -> "
            f"{len(combined):,} total rows"
        )

    return combined


def main() -> None:
    months = resolve_load_months()
    if not months:
        print("No months selected for loading. Edit LOAD_MONTHS in load_crime_files.py.")
        raise SystemExit(1)

    df = load_months()

    if df.empty:
        print("\nNo data loaded.")
        raise SystemExit(1)

    print()
    print("Load summary")
    print("-" * 40)
    print(f"  Months: {', '.join(months)}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Source files: {df[SOURCE_FILE_COLUMN].nunique()}")
    print()
    print("Rows by source file:")
    for source_file, count in df[SOURCE_FILE_COLUMN].value_counts().sort_index().items():
        print(f"  {source_file}: {count:,}")
    print()
    print("Rows by data_source_name:")
    for name, count in df[DATA_SOURCE_NAME_COLUMN].value_counts(dropna=False).sort_index().items():
        label = name if pd.notna(name) else "(NaN)"
        print(f"  {label}: {count:,}")
    print()
    print("Sample metadata columns:")
    metadata_cols = [
        SOURCE_FILE_COLUMN,
        SOURCE_MONTH_FOLDER_COLUMN,
        SOURCE_ROW_NUMBER_COLUMN,
        DATA_SOURCE_NAME_COLUMN,
    ]
    print(df[metadata_cols].head(3).to_string(index=False))

    if len(months) == 1 and LOAD_MONTHS is not None:
        print("\nTo load more months, edit LOAD_MONTHS in load_crime_files.py.")


if __name__ == "__main__":
    main()
