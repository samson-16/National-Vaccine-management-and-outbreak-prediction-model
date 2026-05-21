"""Check which training-data source files have been collected.

This script reads config/data_catalog.json and reports whether the expected
local files or folders exist. It intentionally does not download anything,
because several key sources require request-copy or account workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "config" / "data_catalog.json"


def path_exists(pattern: str) -> bool:
    target = ROOT / pattern
    if any(char in pattern for char in "*?["):
        return any(ROOT.glob(pattern))
    return target.exists()


def summarize_source(source: dict[str, Any]) -> dict[str, Any]:
    expected_files = source.get("expected_files", [])
    present = [pattern for pattern in expected_files if path_exists(pattern)]
    missing = [pattern for pattern in expected_files if not path_exists(pattern)]
    return {
        "id": source["id"],
        "priority": source.get("priority", ""),
        "access": source.get("access", ""),
        "present": present,
        "missing": missing,
        "ready": len(expected_files) > 0 and len(missing) == 0,
        "url": source.get("source_url", ""),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["status", "priority", "id", "access", "expected"]
    widths = {header: len(header) for header in headers}
    formatted_rows = []

    for row in rows:
        status = "READY" if row["ready"] else "MISSING"
        expected = "; ".join(row["present"] + row["missing"])
        formatted = {
            "status": status,
            "priority": row["priority"],
            "id": row["id"],
            "access": row["access"],
            "expected": expected,
        }
        formatted_rows.append(formatted)
        for header in headers:
            widths[header] = max(widths[header], len(formatted[header]))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    print(header_line)
    print("  ".join("-" * widths[header] for header in headers))
    for row in formatted_rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check expected raw data files.")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, type=Path)
    parser.add_argument(
        "--priority",
        help="Only show one priority, for example required_mvp or recommended_mvp.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    with args.catalog.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)

    rows = [summarize_source(source) for source in catalog["sources"]]
    if args.priority:
        rows = [row for row in rows if row["priority"] == args.priority]

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)
        missing_required = [
            row
            for row in rows
            if row["priority"] in {"required_mvp", "required_geospatial"}
            and not row["ready"]
        ]
        if missing_required:
            print()
            print("Next step: collect the required MVP files listed above.")
            print("For APHI files, use data/source_requests/aphi_request_message.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
