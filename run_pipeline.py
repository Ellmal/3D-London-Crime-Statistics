"""End-to-end London crime pipeline orchestrator.

Single source of truth for which months run: ``PIPELINE_MONTHS`` in
``src/config.py``. This script reads it once and passes ``months=`` explicitly
to every step, so there is one place to change the month filter.

Steps (in order):
  1. Inventory raw files          (src.ingestion.list_raw_files)
  2. Profile raw files            (src.validation.profile_raw_data)
  3. Load + clean raw files       (src.cleaning.clean_crime_data)*
  4. Write processed Parquet      (src.cleaning.clean_crime_data.save_processed)
  5. Generate data quality report (src.validation.validate_crime_data)
  6. Spatial grid aggregation    (src.transformation.aggregate_crime_grid)

* Loading (src.ingestion.load_crime_files) is folded into cleaning:
  ``clean_months`` calls ``load_months`` internally, so running a separate load
  pass would re-read the same CSVs. We load+clean once, then write the result.

Safe to rerun: processed Parquet files, viz summaries, and reports are
overwritten on each run; raw files are only read.

Usage:
    python run_pipeline.py
"""

from __future__ import annotations

from src.cleaning.clean_crime_data import clean_months, save_processed
from src.config import (
    OUTPUTS_REPORTS_DIR,
    PIPELINE_MONTHS,
    RAW_DATA_DIR,
    REPORT_OUTPUT_MODE,
)
from src.ingestion.list_raw_files import build_inventory
from src.ingestion.load_crime_files import resolve_load_months
from src.transformation.aggregate_crime_grid import aggregate_month, verify_summary
from src.validation.profile_raw_data import profile_months
from src.validation.validate_crime_data import validate_months

INVENTORY_REPORT_FILENAME = "raw_file_inventory.txt"


def print_step(number: int, title: str) -> None:
    print()
    print("=" * 70)
    print(f"STEP {number}: {title}")
    print("=" * 70)


def run_inventory() -> None:
    """Inventory all expected raw months and highlight the testing month.

    Note: this is a full inventory (every expected month), not filtered to
    PIPELINE_MONTHS. Missing future months are expected, so issues here do not
    abort the pipeline.
    """
    print_step(1, "Inventory raw files")
    OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report, issues = build_inventory()
    output_path = OUTPUTS_REPORTS_DIR / INVENTORY_REPORT_FILENAME
    output_path.write_text(report, encoding="utf-8")
    print(f"Inventory report saved to: {output_path}")
    print(f"Inventory issues noted (expected for future months): {len(issues)}")


def run_profile(months: list[str]) -> None:
    print_step(2, "Profile raw files")
    written = profile_months(
        months,
        raw_dir=RAW_DATA_DIR,
        output_mode=REPORT_OUTPUT_MODE,
    )
    for key, path in written.items():
        print(f"  Profile [{key}] saved to: {path}")


def run_clean_and_save(months: list[str]):
    print_step(3, "Load + clean raw files")
    cleaned_df = clean_months(months, raw_dir=RAW_DATA_DIR)
    if cleaned_df.empty:
        print("No cleaned data produced; aborting.")
        raise SystemExit(1)

    print_step(4, "Write processed Parquet")
    written = save_processed(cleaned_df)
    if not written:
        print("No processed files written; aborting.")
        raise SystemExit(1)
    return cleaned_df


def run_validate(months: list[str]) -> None:
    print_step(5, "Generate data quality report")
    written = validate_months(months, raw_dir=RAW_DATA_DIR)
    for month, path in written.items():
        print(f"  Report [{month}] saved to: {path}")


def run_aggregate(months: list[str]) -> None:
    print_step(6, "Spatial grid aggregation")
    all_issues: list[str] = []
    for month in months:
        summary, output_path, input_rows = aggregate_month(month)
        all_issues.extend(verify_summary(summary, input_rows))
        print(f"  Grid summary [{month}] saved to: {output_path}")

    if all_issues:
        print("\nGrid aggregation verification failed:")
        for issue in all_issues:
            print(f"  - {issue}")
        raise SystemExit(1)


def main() -> None:
    months = resolve_load_months(RAW_DATA_DIR, PIPELINE_MONTHS)
    if not months:
        print("No months selected. Edit PIPELINE_MONTHS in src/config.py.")
        raise SystemExit(1)

    print("London Crime Pipeline")
    print(f"Months to process (from PIPELINE_MONTHS): {', '.join(months)}")

    run_inventory()
    run_profile(months)
    run_clean_and_save(months)
    run_validate(months)
    run_aggregate(months)

    print()
    print("=" * 70)
    print("Pipeline complete.")
    print(f"  Processed months: {', '.join(months)}")
    print("  To change which months run, edit PIPELINE_MONTHS in src/config.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
