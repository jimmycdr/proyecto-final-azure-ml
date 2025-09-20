import argparse, os, json
import pandas as pd

DEFAULT_COLS = [
  "Id","ProductId","UserId","Score","Summary","Text",
  "HelpfulnessNumerator","HelpfulnessDenominator","Time"
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)

    cols = DEFAULT_COLS

    keep = [c for c in cols if c in df.columns]
    df = df[keep]

    out_dir = os.path.dirname(args.output_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_parquet(args.output_parquet, index=False)
    print(f"[select_cols] kept={keep} rows={len(df)} -> {args.output_parquet}")
