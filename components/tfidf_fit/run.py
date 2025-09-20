import argparse, os, pickle, json, math
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_range(txt: str):
    parts = [int(x.strip()) for x in txt.split(",")]
    return (parts[0], parts[0] if len(parts) == 1 else parts[1])

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit TF-IDF vectorizer and emit report.")
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--text_col", default="text_clean")
    ap.add_argument("--max_features", type=int, default=30000)
    ap.add_argument("--ngram_range", type=str, default="1,2", help="word ngrams: '1,2'")
    ap.add_argument("--min_df", type=int, default=3)
    ap.add_argument("--max_df", type=float, default=0.9, help="drop terms present in > max_df of docs (0<max_df<=1)")
    ap.add_argument("--stop_lang", choices=["none", "english", "spanish"], default="none")
    ap.add_argument("--analyzer", choices=["word", "char_wb"], default="word",
                    help="use 'char_wb' with --char_ngram_range to add robustness")
    ap.add_argument("--char_ngram_range", type=str, default="3,5", help="char ngrams: '3,5' (when analyzer=char_wb)")
    ap.add_argument("--output_vectorizer", required=True)
    ap.add_argument("--output_report", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.train_parquet)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available: {list(df.columns)[:10]} ...")
    texts = df[args.text_col].fillna("").astype(str).tolist()
    n_docs = len(texts)

    stop_words = None if args.stop_lang == "none" else args.stop_lang
    if args.analyzer == "word":
        ngram = parse_range(args.ngram_range)
        analyzer = "word"
    else:
        ngram = parse_range(args.char_ngram_range)
        analyzer = "char_wb"

    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words=stop_words,            # 'english' o 'spanish' si lo pides
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,                # tf -> 1 + log(tf)
        dtype=np.float32,
        token_pattern=r"(?u)\b\w\w+\b" if analyzer == "word" else None,
    )

    vec.fit(texts)

    ensure_dir(args.output_vectorizer)
    with open(args.output_vectorizer, "wb") as f:
        pickle.dump(vec, f)

    # ------- Reporte enriquecido -------
    # Invertir el vocabulario para alinear con idf_
    inv_vocab = [None] * len(vec.vocabulary_)
    for term, idx in vec.vocabulary_.items():
        inv_vocab[idx] = term

    idf = getattr(vec, "idf_", None)
    idf_stats = {}
    common_terms = []
    rare_terms = []
    removed_stopwords = int(getattr(vec, "stop_words_", None) is not None and len(vec.stop_words_) or 0)

    if idf is not None and len(idf) == len(inv_vocab):
        idf = idf.astype(float)
        idf_stats = {
            "min": float(np.min(idf)),
            "median": float(np.median(idf)),
            "max": float(np.max(idf)),
        }
        # Estimar DF desde la fórmula de sklearn (smooth_idf=True por defecto):
        # idf = log((1 + n_docs) / (1 + df)) + 1  =>  df ≈ (1 + n_docs)/exp(idf - 1) - 1
        def df_from_idf(idf_val):
            return max(0, int(round((1 + n_docs) / math.exp(idf_val - 1.0) - 1)))

        pairs = [{"term": inv_vocab[i], "idf": float(idf[i]), "doc_freq_est": df_from_idf(idf[i])}
                 for i in range(len(inv_vocab))]

        # Más comunes = idf bajo; más raros = idf alto
        common_terms = sorted(pairs, key=lambda x: x["idf"])[:20]
        rare_terms   = sorted(pairs, key=lambda x: x["idf"], reverse=True)[:20]

    report = {
        "docs": n_docs,
        "vocab_size": len(vec.vocabulary_),
        "removed_stopwords_count": removed_stopwords,
        "params": {
            "analyzer": analyzer,
            "word_ngram_range": args.ngram_range if analyzer == "word" else None,
            "char_ngram_range": args.char_ngram_range if analyzer != "word" else None,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "stop_lang": args.stop_lang,
            "lowercase": True,
            "strip_accents": "unicode",
            "sublinear_tf": True,
            "dtype": "float32"
        },
        "idf_stats": idf_stats,
        "top_common_terms_by_idf": common_terms,  # 20
        "top_rare_terms_by_idf": rare_terms       # 20
    }

    ensure_dir(args.output_report)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[tfidf_fit] docs={n_docs} | vocab_size={report['vocab_size']} | saved -> {args.output_vectorizer}")
