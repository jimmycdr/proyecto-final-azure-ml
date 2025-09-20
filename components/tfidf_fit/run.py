import argparse, os, pickle, json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_ngram(ngram: str):
    parts = [int(x.strip()) for x in ngram.split(",")]
    if len(parts) == 1:
        return (parts[0], parts[0])
    return (parts[0], parts[1])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--text_col", default="text_clean")
    ap.add_argument("--max_features", type=int, default=30000)
    ap.add_argument("--ngram_range", type=str, default="1,2", help="format: '1,2'")
    ap.add_argument("--min_df", type=int, default=3)
    ap.add_argument("--output_vectorizer", required=True)
    ap.add_argument("--output_report", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.train_parquet)
    texts = df[args.text_col].fillna("").astype(str).tolist()

    vec = TfidfVectorizer(max_features=args.max_features,
                          ngram_range=parse_ngram(args.ngram_range),
                          min_df=args.min_df)
    vec.fit(texts)

    os.makedirs(os.path.dirname(args.output_vectorizer), exist_ok=True)
    with open(args.output_vectorizer, "wb") as f:
        pickle.dump(vec, f)

    report = {
        "vocab_size": len(vec.vocabulary_),
        "max_features": args.max_features,
        "ngram_range": args.ngram_range,
        "min_df": args.min_df,
        "text_col": args.text_col,
    }
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w") as f:
        json.dump(report, f)
    print(f"[tfidf_fit] vocab_size={report['vocab_size']} saved -> {args.output_vectorizer}")
