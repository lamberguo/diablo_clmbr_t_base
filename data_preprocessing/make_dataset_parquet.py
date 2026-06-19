#!/usr/bin/env python3
"""
Create CLMBR-compatible parquet dataset from user CSV.

Output layout:
  dataset/data/data.parquet

Required semantic columns:
  - patient id
  - event time
  - clinical code
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert user CSV to dataset/data/data.parquet")
    p.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to user CSV file.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Dataset root directory. Output is <output_dir>/data/data.parquet",
    )
    p.add_argument("--patient_id_col", type=str, default="patient_id")
    p.add_argument("--time_col", type=str, default="event_time")
    p.add_argument("--code_col", type=str, default="code")
    p.add_argument("--numeric_value_col", type=str, default="")
    p.add_argument("--text_value_col", type=str, default="")
    p.add_argument(
        "--time_format",
        type=str,
        default="",
        help="Optional datetime format passed to pandas.to_datetime.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    for col in [args.patient_id_col, args.time_col, args.code_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {input_csv}")

    out = pd.DataFrame()
    out["subject_id"] = pd.to_numeric(df[args.patient_id_col], errors="raise").astype("int64")
    out["time"] = pd.to_datetime(
        df[args.time_col],
        format=args.time_format if args.time_format else None,
        errors="raise",
    )
    out["code"] = df[args.code_col].astype(str)

    if args.numeric_value_col:
        if args.numeric_value_col not in df.columns:
            raise ValueError(f"numeric_value_col not found: {args.numeric_value_col}")
        out["numeric_value"] = pd.to_numeric(df[args.numeric_value_col], errors="coerce")
    else:
        out["numeric_value"] = pd.NA

    if args.text_value_col:
        if args.text_value_col not in df.columns:
            raise ValueError(f"text_value_col not found: {args.text_value_col}")
        out["text_value"] = df[args.text_value_col].astype(str)
    else:
        out["text_value"] = pd.NA

    out = out.sort_values(["subject_id", "time"]).reset_index(drop=True)

    output_root = Path(args.output_dir).resolve()
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "data.parquet"
    out.to_parquet(parquet_path, index=False)
    print(f"Saved parquet: {parquet_path}")
    print(f"Rows: {len(out):,}, Patients: {out['subject_id'].nunique():,}")


if __name__ == "__main__":
    main()

