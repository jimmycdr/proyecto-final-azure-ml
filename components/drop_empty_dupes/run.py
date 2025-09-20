import argparse
import os
import pandas as pd

def is_empty(s: str) -> bool:
    if s is None:
        return True
    s = str(s).strip()
    return len(s) == 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)

    before = len(df)

    df = df[~df["Text"].map(is_empty)]
    after_empty = len(df)

    dupe_keys = ["UserId", "ProductId", "Text"]
    df = df.drop_duplicates(subset=dupe_keys, keep="first")
    after_dupes = len(df)

    out_dir = os.path.dirname(args.output_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_parquet(args.output_parquet, index=False, engine="pyarrow", compression="snappy")

    print(f"[drop_empty_dupes] before={before:,} after_empty={after_empty:,} after_dupes={after_dupes:,} keys={dupe_keys} -> {args.output_parquet}")
