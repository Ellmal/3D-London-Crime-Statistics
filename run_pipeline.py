"""London Crime Pipeline — builds the 3D hex map source from raw crime data.

Pipeline steps:
  1. Load + clean raw monthly street CSVs
  2. Aggregate to hex / month / crime-type
     -> data/viz/crime_hex_3d_month.parquet  (Streamlit app runtime source)

Safe to rerun: the output Parquet is overwritten on each run.
Raw CSV files are never modified.

Usage:
    python run_pipeline.py

To change which months are processed, edit PIPELINE_MONTHS in src/config.py.
"""

from __future__ import annotations

from src.cleaning.clean_crime_data import clean_months
from src.config import PIPELINE_MONTHS, RAW_DATA_DIR, VIZ_DATA_DIR
from src.ingestion.load_crime_files import resolve_load_months
from src.transformation.aggregate_hex_grid import build_hex_month_type, verify_hex_summary


def print_step(number: int, title: str) -> None:
    print()
    print("=" * 70)
    print(f"STEP {number}: {title}")
    print("=" * 70)


def main() -> None:
    months = resolve_load_months(RAW_DATA_DIR, PIPELINE_MONTHS)
    if not months:
        print("No months selected. Edit PIPELINE_MONTHS in src/config.py.")
        raise SystemExit(1)

    print("London Crime Pipeline")
    print(f"Months: {', '.join(months)}")

    print_step(1, "Load + clean raw files")
    cleaned_df = clean_months(months, raw_dir=RAW_DATA_DIR)
    if cleaned_df.empty:
        print("No data loaded or cleaned; aborting.")
        raise SystemExit(1)
    print(f"  {len(cleaned_df):,} cleaned rows across {len(months)} month(s)")

    print_step(2, "Aggregate to hex grid")
    summary, output_path, input_rows = build_hex_month_type(cleaned_df, VIZ_DATA_DIR)

    issues = verify_hex_summary(summary, input_rows)
    if issues:
        print("\nHex aggregation verification failed:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    print()
    print("=" * 70)
    print("Pipeline complete.")
    print(f"  Months processed: {', '.join(months)}")
    print(f"  Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
