"""Profile raw monthly crime CSV files before any cleaning or transformation."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from typing import Literal

from src.config import DEFAULT_TESTING_MONTH, OUTPUTS_REPORTS_DIR, RAW_DATA_DIR

MONTH_LABEL_PATTERN = re.compile(r"^\d{4}-\d{2}$")

SAMPLE_ROW_COUNT = 5
COMBINED_REPORT_FILENAME = "raw_data_profile_combined.txt"

ReportOutputMode = Literal["separate", "combined", "both"]

# ---------------------------------------------------------------------------
# Month filter - adjust this to control which months are profiled.
#
# Examples:
#   ["2025-05"]                      # May 2025 only (current default)
#   ["2025-05", "2025-06"]           # specific months
#   None                             # all month folders found under data/raw/
# ---------------------------------------------------------------------------
PROFILE_MONTHS: list[str] | None = [DEFAULT_TESTING_MONTH]

# ---------------------------------------------------------------------------
# Report output mode - controls how profile text is saved.
#
#   "separate"  -> one file per month (default)
#   "combined"  -> one aggregated profile across all selected months/files
#   "both"      -> write separate per-month and aggregated combined reports
# ---------------------------------------------------------------------------
REPORT_OUTPUT_MODE: ReportOutputMode = "separate"


def resolve_profile_months(
    raw_dir: Path = RAW_DATA_DIR,
    month_filter: list[str] | None = PROFILE_MONTHS,
) -> list[str]:
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


def get_month_csv_files(month: str, raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    month_dir = raw_dir / month
    if not month_dir.is_dir():
        return []
    return sorted(month_dir.glob("*.csv"))


def count_missing_text(series: pd.Series) -> int:
    as_text = series.astype("string")
    return int(as_text.isna().sum() + (as_text.str.strip() == "").sum())


def format_unique_values(values: pd.Series, max_values: int = 50) -> list[str]:
    unique_values = sorted(values.dropna().astype(str).str.strip().unique())
    unique_values = [value for value in unique_values if value]
    lines = [f"    count: {len(unique_values)}"]
    if not unique_values:
        lines.append("    values: (none)")
        return lines

    if len(unique_values) <= max_values:
        lines.append("    values:")
        for value in unique_values:
            lines.append(f"      - {value}")
    else:
        lines.append(f"    values (first {max_values} of {len(unique_values)}):")
        for value in unique_values[:max_values]:
            lines.append(f"      - {value}")
        lines.append("      ...")

    return lines


def profile_dataframe(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    lines.append(f"  Row count: {len(df):,}")
    lines.append("  Columns:")
    for column in df.columns:
        lines.append(f"    - {column}")

    lines.append("  Missing values by column:")
    for column in df.columns:
        missing_count = count_missing_text(df[column])
        lines.append(f"    - {column}: {missing_count:,}")

    for column_name, label in [
        ("Month", "Unique Month values"),
        ("Reported by", "Unique Reported by values"),
        ("Falls within", "Unique Falls within values"),
        ("Crime type", "Unique Crime type values"),
    ]:
        if column_name in df.columns:
            lines.append(f"  {label}:")
            lines.extend(format_unique_values(df[column_name]))

    if "Longitude" in df.columns and "Latitude" in df.columns:
        longitude = pd.to_numeric(df["Longitude"], errors="coerce")
        latitude = pd.to_numeric(df["Latitude"], errors="coerce")

        lines.append("  Longitude / Latitude:")
        lines.append(f"    missing longitude: {int(longitude.isna().sum()):,}")
        lines.append(f"    missing latitude: {int(latitude.isna().sum()):,}")
        lines.append(
            "    missing either coordinate: "
            f"{int((longitude.isna() | latitude.isna()).sum()):,}"
        )

        if longitude.notna().any():
            lines.append(
                f"    longitude min/max: {longitude.min():.6f} / {longitude.max():.6f}"
            )
        else:
            lines.append("    longitude min/max: (no valid values)")

        if latitude.notna().any():
            lines.append(
                f"    latitude min/max: {latitude.min():.6f} / {latitude.max():.6f}"
            )
        else:
            lines.append("    latitude min/max: (no valid values)")

    if "Crime ID" in df.columns:
        lines.append(f"  Missing Crime ID count: {count_missing_text(df['Crime ID']):,}")

    if "LSOA code" in df.columns:
        lines.append(f"  Missing LSOA code count: {count_missing_text(df['LSOA code']):,}")

    lines.append(f"  Sample rows (first {SAMPLE_ROW_COUNT}):")
    sample = df.head(SAMPLE_ROW_COUNT)
    if sample.empty:
        lines.append("    (no rows)")
    else:
        for sample_line in sample.to_string(index=False).splitlines():
            lines.append(f"    {sample_line}")

    return lines


def compare_columns(file_frames: dict[str, pd.DataFrame]) -> list[str]:
    lines: list[str] = []
    column_sets = {name: list(df.columns) for name, df in file_frames.items()}

    if not column_sets:
        lines.append("  No files available for column comparison.")
        return lines

    reference_name, reference_columns = next(iter(column_sets.items()))
    reference_set = set(reference_columns)
    all_match = True

    lines.append(f"  Reference file: {reference_name}")
    lines.append(f"  Reference columns ({len(reference_columns)}):")
    for column in reference_columns:
        lines.append(f"    - {column}")

    for file_name, columns in column_sets.items():
        if file_name == reference_name:
            continue

        file_set = set(columns)
        if columns == reference_columns:
            lines.append(f"  {file_name}: same column order and names")
            continue

        all_match = False
        lines.append(f"  {file_name}: DIFFERS from reference")
        only_in_reference = sorted(reference_set - file_set)
        only_in_file = sorted(file_set - reference_set)
        if only_in_reference:
            lines.append(f"    missing from file: {only_in_reference}")
        if only_in_file:
            lines.append(f"    extra in file: {only_in_file}")
        if file_set == reference_set and columns != reference_columns:
            lines.append("    same columns, different order")

    if all_match and len(column_sets) > 1:
        lines.append("  Result: all files share the same schema.")

    return lines


def load_raw_file_frames(
    months: list[str],
    raw_dir: Path = RAW_DATA_DIR,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    file_frames: dict[str, pd.DataFrame] = {}
    issues: list[str] = []

    for month in months:
        csv_files = get_month_csv_files(month, raw_dir)
        if not csv_files:
            issues.append(f"No CSV files found for month: {month}")
            continue

        for csv_path in csv_files:
            file_key = f"{month}/{csv_path.name}"
            try:
                file_frames[file_key] = pd.read_csv(
                    csv_path,
                    dtype=str,
                    keep_default_na=True,
                )
            except Exception as exc:
                issues.append(f"Failed to read {file_key}: {exc}")

    return file_frames, issues


def build_aggregated_profile(
    months: list[str],
    raw_dir: Path = RAW_DATA_DIR,
) -> tuple[str, list[str]]:
    lines: list[str] = []
    issues: list[str] = []

    file_frames, load_issues = load_raw_file_frames(months, raw_dir)
    issues.extend(load_issues)

    lines.append("London Crime Raw Data Profile (Combined)")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Months profiled: {', '.join(months)}")
    lines.append(f"Month count: {len(months)}")
    lines.append("")

    if not file_frames:
        lines.append("No CSV files could be loaded.")
        if issues:
            lines.append("")
            lines.append("Issues")
            lines.append("-" * 60)
            for issue in issues:
                lines.append(f"  [ISSUE] {issue}")
        return "\n".join(lines), issues

    combined_df = pd.concat(file_frames.values(), ignore_index=True)

    lines.append("Source files")
    lines.append("-" * 60)
    for file_key in sorted(file_frames):
        lines.append(f"  - {file_key} ({len(file_frames[file_key]):,} rows)")
    lines.append(f"  Total files: {len(file_frames)}")
    lines.append("")

    lines.append("Aggregated profile (all files combined)")
    lines.append("-" * 60)
    lines.extend(profile_dataframe(combined_df))
    lines.append("")

    lines.append("Cross-file schema comparison")
    lines.append("-" * 60)
    lines.extend(compare_columns(file_frames))
    lines.append("")

    lines.append("Data quality notes (pre-cleaning)")
    lines.append("-" * 60)
    if len(file_frames) < 2:
        lines.append("  Only one file included in this combined profile.")
    elif all(
        list(df.columns) == list(next(iter(file_frames.values())).columns)
        for df in file_frames.values()
    ):
        lines.append("  Schema is consistent across all included files.")
    else:
        issues.append("Schema mismatch detected across included files")
        lines.append("  Schema mismatch detected - review before loading.")

    lines.append("  Review missing coordinates, Crime ID, and LSOA fields above.")
    lines.append("  Out-of-range coordinates should be flagged during cleaning.")
    lines.append("")

    lines.append("Summary")
    lines.append("-" * 60)
    lines.append(f"  Files profiled: {len(file_frames)}")
    lines.append(f"  Total rows: {len(combined_df):,}")
    lines.append(f"  Issues detected: {len(issues)}")

    if issues:
        lines.append("")
        lines.append("Issues")
        lines.append("-" * 60)
        for issue in issues:
            lines.append(f"  [ISSUE] {issue}")

    return "\n".join(lines), issues


def build_month_profile(
    month: str,
    raw_dir: Path = RAW_DATA_DIR,
) -> tuple[str, list[str], int]:
    lines: list[str] = []
    issues: list[str] = []

    lines.append("London Crime Raw Data Profile")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Month: {month}")
    lines.append(f"Raw data directory: {raw_dir / month}")
    lines.append("")

    csv_files = get_month_csv_files(month, raw_dir)
    if not csv_files:
        issues.append(f"No CSV files found for month: {month}")
        lines.append("No CSV files found.")
        return "\n".join(lines), issues, 0

    file_frames: dict[str, pd.DataFrame] = {}

    for csv_path in csv_files:
        lines.append("File")
        lines.append("-" * 60)
        lines.append(f"  Path: {csv_path}")
        lines.append(f"  Name: {csv_path.name}")
        lines.append("")

        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
        except Exception as exc:
            issues.append(f"Failed to read {csv_path.name}: {exc}")
            lines.append(f"  [ERROR] Could not read file: {exc}")
            lines.append("")
            continue

        file_frames[csv_path.name] = df
        lines.extend(profile_dataframe(df))
        lines.append("")

    lines.append("Cross-file schema comparison")
    lines.append("-" * 60)
    lines.extend(compare_columns(file_frames))
    lines.append("")

    lines.append("Data quality notes (pre-cleaning)")
    lines.append("-" * 60)
    if len(file_frames) < 2:
        lines.append("  Only one file profiled for this month.")
    elif all(
        list(df.columns) == list(next(iter(file_frames.values())).columns)
        for df in file_frames.values()
    ):
        lines.append("  Schema is consistent across files in this month.")
    else:
        issues.append(f"Schema mismatch detected for month: {month}")
        lines.append("  Schema mismatch detected - review before loading.")

    lines.append("  Review missing coordinates, Crime ID, and LSOA fields above.")
    lines.append("  Out-of-range coordinates should be flagged during cleaning.")
    lines.append("")

    lines.append("Summary")
    lines.append("-" * 60)
    lines.append(f"  Files profiled: {len(file_frames)}")
    total_rows = sum(len(df) for df in file_frames.values())
    lines.append(f"  Total rows: {total_rows:,}")
    lines.append(f"  Issues detected: {len(issues)}")

    if issues:
        lines.append("")
        lines.append("Issues")
        lines.append("-" * 60)
        for issue in issues:
            lines.append(f"  [ISSUE] {issue}")

    return "\n".join(lines), issues, total_rows


def profile_months(
    months: list[str],
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = OUTPUTS_REPORTS_DIR,
    output_mode: ReportOutputMode = REPORT_OUTPUT_MODE,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_reports: dict[str, Path] = {}

    if output_mode in ("separate", "both"):
        for month in months:
            separate_report, _issues, _total_rows = build_month_profile(month, raw_dir)
            output_path = output_dir / f"raw_data_profile_{month}.txt"
            output_path.write_text(separate_report, encoding="utf-8")
            written_reports[month] = output_path

    if output_mode in ("combined", "both"):
        combined_report, _issues = build_aggregated_profile(months, raw_dir)
        combined_path = output_dir / COMBINED_REPORT_FILENAME
        combined_path.write_text(combined_report, encoding="utf-8")
        written_reports["combined"] = combined_path

    return written_reports


def main() -> None:
    months = resolve_profile_months()
    if not months:
        print("No months selected for profiling.")
        raise SystemExit(1)

    if REPORT_OUTPUT_MODE not in ("separate", "combined", "both"):
        print(f"Invalid REPORT_OUTPUT_MODE: {REPORT_OUTPUT_MODE}")
        raise SystemExit(1)

    print(f"Profiling months: {', '.join(months)}")
    print(f"Report output mode: {REPORT_OUTPUT_MODE}")
    written_reports = profile_months(months)

    if "combined" in written_reports:
        report = written_reports["combined"].read_text(encoding="utf-8")
        print()
        print(report)
        print(f"\nCombined report saved to: {written_reports['combined']}")

    for month in months:
        if month not in written_reports:
            continue
        report = written_reports[month].read_text(encoding="utf-8")
        print()
        print(report)
        print(f"\nReport saved to: {written_reports[month]}")

    if len(months) == 1 and REPORT_OUTPUT_MODE == "separate":
        print("\nTo profile more months, edit PROFILE_MONTHS in profile_raw_data.py.")


if __name__ == "__main__":
    main()
