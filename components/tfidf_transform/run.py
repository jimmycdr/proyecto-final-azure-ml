import argparse, os, pickle
import pandas as pd
from scipy import sparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--test_parquet", required=True)
    ap.add_argument("--vectorizer_pkl", required=True)
    ap.add_argument("--text_col", default="text_clean")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--output_X_train", required=True)
    ap.add_argument("--output_X_test", required=True)
    ap.add_argument("--output_y_train", required=True)
    ap.add_argument("--output_y_test", required=True)
    args = ap.parse_args()

    with open(args.vectorizer_pkl, "rb") as f:
        vec = pickle.load(f)

    dtr = pd.read_parquet(args.train_parquet)
    dte = pd.read_parquet(args.test_parquet)

    Xtr = vec.transform(dtr[args.text_col].fillna("").astype(str).tolist())
    Xte = vec.transform(dte[args.text_col].fillna("").astype(str).tolist())

    out_dir_tr = os.path.dirname(args.output_X_train)
    out_dir_te = os.path.dirname(args.output_X_test)
    if out_dir_tr: os.makedirs(out_dir_tr, exist_ok=True)
    if out_dir_te: os.makedirs(out_dir_te, exist_ok=True)

    with open(args.output_X_train, "wb") as f:
        sparse.save_npz(f, Xtr)
    with open(args.output_X_test, "wb") as f:
        sparse.save_npz(f, Xte)

    ytr = dtr[[args.label_col]]
    yte = dte[[args.label_col]]
    ytr.to_parquet(args.output_y_train, index=False)
    yte.to_parquet(args.output_y_test, index=False)

    print(f"[tfidf_transform] X_train={Xtr.shape} X_test={Xte.shape} y_train={len(ytr)} y_test={len(yte)}")
