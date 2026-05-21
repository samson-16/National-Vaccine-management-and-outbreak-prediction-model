"""Filter the MVP measles dataset and optionally drop provenance flag columns."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "measles_woreda_month_ml_ready_mixed.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "measles_filtered.csv"

FLAG_COLUMNS = [
    "real_flag",
    "weak_label_flag",
    "synthetic_flag",
    "imputed_flag",
    "guessed_flag",
    "external_country_flag",
    "ethiopia_flag",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter the measles MVP dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--remove-synthetic", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--real-weak-only", action="store_true")
    parser.add_argument("--ethiopia-only", action="store_true")
    parser.add_argument("--drop-flag-columns", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.remove_synthetic and "synthetic_flag" in df.columns:
        df = df[df["synthetic_flag"] == 0]
    if args.real_only and "label_type" in df.columns:
        df = df[df["label_type"] == "real_ephi"]
    if args.real_weak_only and "label_type" in df.columns:
        df = df[df["label_type"].isin(["real_ephi", "weak_public_report"])]
    if args.ethiopia_only and "ethiopia_flag" in df.columns:
        df = df[df["ethiopia_flag"] == 1]
    if args.drop_flag_columns:
        df = df.drop(columns=[column for column in FLAG_COLUMNS if column in df.columns])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(args.output, index=False)
        output = args.output
    except PermissionError:
        output = args.output.with_name(args.output.stem + "_new" + args.output.suffix)
        df.to_csv(output, index=False)
        print(f"Warning: {args.output} is locked. Wrote {output} instead.", file=sys.stderr)
    print(f"Wrote {output} with {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
