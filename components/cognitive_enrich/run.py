import argparse, os, pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def chunker(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)
    text = (df["text_clean"] if "text_clean" in df.columns
            else (df["Summary"].fillna("").astype(str) + ". " + df["Text"].fillna("").astype(str)))

    client = TextAnalyticsClient(endpoint=args.endpoint, credential=AzureKeyCredential(args.key))

    sentiments, posc, neuc, negc, langs = [], [], [], [], []
    
    for batch in chunker(text.tolist(), 10):
        # Sentiment
        sres = client.analyze_sentiment(batch)
        for d in sres:
            if d.is_error:
                sentiments += [None]; posc += [None]; neuc += [None]; negc += [None]
            else:
                sentiments += [d.sentiment]
                posc += [d.confidence_scores.positive]
                neuc += [d.confidence_scores.neutral]
                negc += [d.confidence_scores.negative]
        # Language
        lres = client.detect_language(batch)
        for d in lres:
            langs += [None if d.is_error else d.primary_language.iso6391_name]

    df["sentiment_cs"] = sentiments
    df["cs_pos"] = posc; df["cs_neu"] = neuc; df["cs_neg"] = negc
    df["language"] = langs

    out_dir = os.path.dirname(args.output_parquet)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)
    print("[cognitive_enrich] added: sentiment_cs, cs_pos, cs_neu, cs_neg, language")
