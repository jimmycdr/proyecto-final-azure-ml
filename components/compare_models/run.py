import argparse, json, os
import math

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

def _safe(x, default):
    if x is None:
        return default
    try:
        xf = float(x)
        return default if math.isnan(xf) else xf
    except Exception:
        return default

def readj(p):
    with open(p) as f:
        return json.load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_a_json", required=True)  # LR
    ap.add_argument("--metrics_b_json", required=True)  # SVM
    ap.add_argument("--name_a", default="lr")
    ap.add_argument("--name_b", default="svm")
    ap.add_argument("--output_selection_json", required=True)
    ap.add_argument("--output_winner_txt", required=False, default="")
    args = ap.parse_args()

    A = readj(args.metrics_a_json).get("metrics", {})
    B = readj(args.metrics_b_json).get("metrics", {})

    def score_tuple(m):
        return (
            _safe(m.get("f1"), 0.0),
            _safe(m.get("roc_auc"), -1.0),
            _safe(m.get("pr_auc"), -1.0),
        )

    keyA, keyB = score_tuple(A), score_tuple(B)
    winner = args.name_a if keyA >= keyB else args.name_b

    sel = {
        "by": "f1->roc_auc->pr_auc",
        "winner": winner,
        "scores": {
            args.name_a: {
                "f1": _safe(A.get("f1"), 0.0),
                "roc_auc": _safe(A.get("roc_auc"), -1.0),
                "pr_auc": _safe(A.get("pr_auc"), -1.0),
            },
            args.name_b: {
                "f1": _safe(B.get("f1"), 0.0),
                "roc_auc": _safe(B.get("roc_auc"), -1.0),
                "pr_auc": _safe(B.get("pr_auc"), -1.0),
            },
        },
    }

    d = os.path.dirname(args.output_selection_json)
    if d: os.makedirs(d, exist_ok=True)
    with open(args.output_selection_json, "w") as f:
        json.dump(sel, f, indent=2)

    if args.output_winner_txt:
        dw = os.path.dirname(args.output_winner_txt)
        if dw: os.makedirs(dw, exist_ok=True)
        with open(args.output_winner_txt, "w") as f:
            f.write(winner)

    print(f"[compare_models] winner={winner} "
          f"{args.name_a}_f1={sel['scores'][args.name_a]['f1']:.4f} "
          f"{args.name_b}_f1={sel['scores'][args.name_b]['f1']:.4f}")

    if _HAS_MLFLOW:
        mlflow.log_param("compare_by", sel["by"])
        mlflow.log_metric(f"f1_{args.name_a}", sel["scores"][args.name_a]["f1"])
        mlflow.log_metric(f"f1_{args.name_b}", sel["scores"][args.name_b]["f1"])
        mlflow.log_metric(f"roc_auc_{args.name_a}", sel["scores"][args.name_a]["roc_auc"])
        mlflow.log_metric(f"roc_auc_{args.name_b}", sel["scores"][args.name_b]["roc_auc"])
        mlflow.log_metric(f"pr_auc_{args.name_a}", sel["scores"][args.name_a]["pr_auc"])
        mlflow.log_metric(f"pr_auc_{args.name_b}", sel["scores"][args.name_b]["pr_auc"])
        mlflow.log_param("winner", winner)
