import argparse, os, re
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
WS_RE = re.compile(r"\s+")

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = t.lower()
    t = URL_RE.sub(" ", t)
    t = HTML_RE.sub(" ", t)
    t = t.replace("\n", " ").replace("\r", " ")
    t = WS_RE.sub(" ", t).strip()
    return t

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)

    df["text_raw"] = (df["Summary"].fillna("").astype(str) + ". " + df["Text"].fillna("").astype(str)).str.strip()
    df["text_clean"] = df["text_raw"].map(normalize)

    df = df[df["text_clean"].str.strip().ne("")]

    out_dir = os.path.dirname(args.output_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_parquet(args.output_parquet, index=False)
    print(f"[clean_text] rows={len(df)} -> {args.output_parquet}")
    dropped = len(df) - len(df[df["text_clean"].str.strip().ne("")])
    print(f"[clean_text] rows={len(df)}, dropped_empty={dropped} -> {args.output_parquet}")
