"""List and verify raw monthly crime CSV files."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.config import DEFAULT_TESTING_MONTH, OUTPUTS_REPORTS_DIR, RAW_DATA_DIR

MONTH_LABEL_PATTERN = re.compile(r"^\d{4}-\d{2}$")
MONTH_IN_FILENAME_PATTERN = re.compile(r"^(\d{4}-\d{2})-")

EXPECTED_MONTHS = [f"2025-{month:02d}" for month in range(5, 13)] + [
    f"2026-{month:02d}" for month in range(1, 5)
]

EXPECTED_STREET_FILES = {
    "city-of-london-street.csv",
    "metropolitan-street.csv",
}


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def expected_filename(month: str, force_suffix: str) -> str:
    return f"{month}-{force_suffix}"


def build_inventory(raw_dir: Path = RAW_DATA_DIR) -> tuple[str, list[str]]:
    lines: list[str] = []
    issues: list[str] = []
    found_months: set[str] = set()

    lines.append("London Crime Raw File Inventory")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Raw data directory: {raw_dir}")
    lines.append(f"Testing month: {DEFAULT_TESTING_MONTH}")
    lines.append("")

    if not raw_dir.exists():
        issues.append(f"Raw data directory does not exist: {raw_dir}")
        lines.append("ISSUES")
        lines.append("-" * 60)
        for issue in issues:
            lines.append(f"  [ERROR] {issue}")
        return "\n".join(lines), issues

    entries = sorted(raw_dir.iterdir(), key=lambda path: path.name)

    month_folders = [entry for entry in entries if entry.is_dir()]
    non_folder_entries = [entry for entry in entries if not entry.is_dir()]

    if non_folder_entries:
        lines.append("Unexpected items in raw directory (should be month folders only)")
        lines.append("-" * 60)
        for entry in non_folder_entries:
            issue = f"Unexpected file in raw root: {entry.name}"
            issues.append(issue)
            lines.append(f"  [WARN] {entry.name}")
        lines.append("")

    lines.append("Month folders")
    lines.append("-" * 60)

    if not month_folders:
        issues.append("No month folders found under data/raw/")
        lines.append("  (none)")
    else:
        for folder in month_folders:
            folder_month = folder.name
            folder_month_valid = bool(MONTH_LABEL_PATTERN.match(folder_month))

            if folder_month_valid:
                found_months.add(folder_month)
            else:
                issues.append(f"Invalid month folder name: {folder.name}")

            lines.append(f"Folder: {folder.name}")
            lines.append(
                f"  Detected month (folder): {folder_month if folder_month_valid else 'INVALID'}"
            )

            csv_files = sorted(folder.glob("*.csv"))
            other_files = sorted(
                path for path in folder.iterdir() if path.is_file() and path.suffix.lower() != ".csv"
            )

            if not csv_files and not other_files:
                issues.append(f"Empty month folder: {folder.name}")
                lines.append("  CSV files: (none)")
            else:
                lines.append("  CSV files:")
                present_suffixes: set[str] = set()

                for csv_path in csv_files:
                    size = csv_path.stat().st_size
                    file_month = detect_month_from_filename(csv_path.name)
                    suffix = csv_path.name.removeprefix(f"{file_month}-") if file_month else None

                    lines.append(f"    - {csv_path.name}")
                    lines.append(f"        size: {format_size(size)}")
                    lines.append(
                        f"        detected month (file): {file_month or 'NOT DETECTED'}"
                    )

                    if not file_month:
                        issues.append(f"Could not detect month from filename: {csv_path.name}")
                    elif file_month != folder_month:
                        issues.append(
                            f"Month mismatch in {folder.name}: folder={folder_month}, "
                            f"file={file_month} ({csv_path.name})"
                        )

                    if not csv_path.name.endswith("-street.csv"):
                        issues.append(f"Unexpected CSV naming pattern: {csv_path.name}")
                    elif suffix in EXPECTED_STREET_FILES:
                        present_suffixes.add(suffix)
                    else:
                        issues.append(f"Unexpected street file suffix: {csv_path.name}")

                for other_path in other_files:
                    issues.append(f"Non-CSV file in {folder.name}: {other_path.name}")
                    lines.append(f"    - {other_path.name} [non-CSV, ignored for loading]")

                if folder_month_valid:
                    for force_suffix in EXPECTED_STREET_FILES:
                        expected_name = expected_filename(folder_month, force_suffix)
                        if force_suffix not in present_suffixes:
                            issues.append(
                                f"Missing expected file: {folder.name}/{expected_name}"
                            )

            lines.append("")

    lines.append("Expected months checklist")
    lines.append("-" * 60)
    for month in EXPECTED_MONTHS:
        status = "FOUND" if month in found_months else "MISSING"
        marker = " " if status == "FOUND" else "!"
        lines.append(f"  [{marker}] {month}: {status}")
        if status == "MISSING":
            issues.append(f"Missing expected month folder: {month}")

    lines.append("")
    lines.append(f"Testing month ({DEFAULT_TESTING_MONTH})")
    lines.append("-" * 60)
    testing_folder = raw_dir / DEFAULT_TESTING_MONTH
    if DEFAULT_TESTING_MONTH in found_months:
        lines.append(f"  [OK] Folder exists: data/raw/{DEFAULT_TESTING_MONTH}/")
        for force_suffix in EXPECTED_STREET_FILES:
            expected_name = expected_filename(DEFAULT_TESTING_MONTH, force_suffix)
            expected_path = testing_folder / expected_name
            if expected_path.exists():
                lines.append(
                    f"  [OK] {expected_name} ({format_size(expected_path.stat().st_size)})"
                )
            else:
                lines.append(f"  [!] Missing: {expected_name}")
                issues.append(f"Missing testing file: {DEFAULT_TESTING_MONTH}/{expected_name}")
    else:
        lines.append(f"  [!] Folder missing: data/raw/{DEFAULT_TESTING_MONTH}/")
        issues.append(f"Testing month folder missing: {DEFAULT_TESTING_MONTH}")

    lines.append("")
    lines.append("Summary")
    lines.append("-" * 60)
    lines.append(f"  Month folders found: {len(found_months)}")
    lines.append(f"  Expected months: {len(EXPECTED_MONTHS)}")
    lines.append(f"  Issues detected: {len(issues)}")

    lines.append("")
    lines.append("Issues")
    lines.append("-" * 60)
    if issues:
        for issue in issues:
            lines.append(f"  [ISSUE] {issue}")
    else:
        lines.append("  None - raw file organization looks good.")

    return "\n".join(lines), issues


def detect_month_from_filename(filename: str) -> str | None:
    match = MONTH_IN_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    month = match.group(1)
    return month if MONTH_LABEL_PATTERN.match(month) else None


def main() -> None:
    OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report, issues = build_inventory()
    output_path = OUTPUTS_REPORTS_DIR / "raw_file_inventory.txt"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {output_path}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
