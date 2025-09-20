import argparse, os, json, pickle
import numpy as np
import pandas as pd

def sigmoid(x): return 1/(1+np.exp(-x))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--vectorizer_pkl", required=True)
    ap.add_argument("--model_pkl", required=True)
    ap.add_argument("--text_col", default="text_clean")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output_predictions_parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)
    with open(args.vectorizer_pkl,"rb") as f: vec = pickle.load(f)
    with open(args.model_pkl,"rb") as f: mdl = pickle.load(f)

    X = vec.transform(df[args.text_col].fillna("").astype(str).tolist())

    if hasattr(mdl, "predict_proba"):
        y_prob = mdl.predict_proba(X)[:,1]
    else:
        margin = mdl.decision_function(X)
        y_prob = sigmoid(margin)

    y_pred = (y_prob >= args.threshold).astype(int)

    out = pd.DataFrame({"y_prob": y_prob, "y_pred": y_pred})
    if args.label_col in df.columns:
        out["y_true"] = df[args.label_col].astype(int).values


    os.makedirs(os.path.dirname(args.output_predictions_parquet), exist_ok=True)
    out.to_parquet(args.output_predictions_parquet, index=False)
    print(f"[predict_generic] saved -> {args.output_predictions_parquet} rows={len(out)}")
