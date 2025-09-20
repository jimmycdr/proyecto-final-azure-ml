import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--output_train", required=True)
    ap.add_argument("--output_test", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)
    if args.label_col not in df.columns:
        raise ValueError(f"label_col '{args.label_col}' not found. Columns: {df.columns.tolist()}")
    y = df[args.label_col]

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    os.makedirs(os.path.dirname(args.output_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_test), exist_ok=True)
    train_df.to_parquet(args.output_train, index=False)
    test_df.to_parquet(args.output_test, index=False)
    print(f"[split_stratified] train={len(train_df)} test={len(test_df)} stratified by '{args.label_col}'")
