import argparse
import os
import pandas as pd

EXPECTED_COLS = [
    "Id","ProductId","UserId","ProfileName","HelpfulnessNumerator",
    "HelpfulnessDenominator","Score","Time","Summary","Text"
]

NUMERIC_COLS = ["Id","HelpfulnessNumerator","HelpfulnessDenominator","Score","Time"]



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Path/URI to Reviews.csv")
    ap.add_argument("--output_parquet", required=True, help="Path/URI to output .parquet file")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"[ingest] read: rows={len(df):,}, cols={list(df.columns)}")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    created_cols = []
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
            created_cols.append(c)

    df = df[EXPECTED_COLS]

    na_stats = df[["Text","Summary","Score","Time"]].isna().sum().to_dict()
    print(f"[ingest] created_missing_cols={created_cols}")
    print(f"[ingest] NA stats (Text/Summary/Score/Time) = {na_stats}")

    out_dir = os.path.dirname(args.output_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_parquet(args.output_parquet, index=False, engine="pyarrow", compression="snappy")
    print(f"[ingest] rows={len(df):,} -> {args.output_parquet}")
