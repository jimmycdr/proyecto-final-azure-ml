import argparse, os, pickle, pandas as pd
from scipy.sparse import csr_matrix

def makedirs_safe(p): 
    d = os.path.dirname(p); 
    if d: os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--vectorizer_pkl", required=True)
    ap.add_argument("--model_pkl", required=True)
    ap.add_argument("--text_col", default="text_clean")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output_predictions_parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)
    with open(args.vectorizer_pkl, "rb") as f: vec = pickle.load(f)
    with open(args.model_pkl, "rb") as f: model = pickle.load(f)

    texts = df[args.text_col].fillna("").astype(str).str.strip().tolist()
    X = vec.transform(texts)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        margin = model.decision_function(X)
        import numpy as np
        prob = 1.0 / (1.0 + np.exp(-margin))

    import numpy as np
    yhat = (prob >= args.threshold).astype(int)

    out = df.copy()
    out["pred_prob"] = prob
    out["pred_label"] = yhat

    makedirs_safe(args.output_predictions_parquet)
    out.to_parquet(args.output_predictions_parquet, index=False)

    print(f"[predict_batch] rows={len(out)} threshold={args.threshold} -> {args.output_predictions_parquet}")
