import argparse, os
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--neg_threshold", type=int, default=2)
    ap.add_argument("--pos_threshold", type=int, default=4)
    ap.add_argument("--output_parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)

    # ensure Score is numeric
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

    # filter out neutral
    df = df[df["Score"].notna()]
    df = df[(df["Score"] <= args.neg_threshold) | (df["Score"] >= args.pos_threshold)]

    # build label
    df["label"] = (df["Score"] >= args.pos_threshold).astype("int8")

    # basic report
    n = len(df)
    pos = int(df["label"].sum())
    neg = n - pos
    print(f"[make_label] rows={n} pos={pos} neg={neg} pos_rate={pos/n:.3f}")

    out_dir = os.path.dirname(args.output_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_parquet(args.output_parquet, index=False)
