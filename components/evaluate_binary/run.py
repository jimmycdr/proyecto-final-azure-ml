import argparse, os, json
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

def makedirs(p): d=os.path.dirname(p); 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_parquet", required=True)
    ap.add_argument("--y_true_col", default="y_true")
    ap.add_argument("--y_pred_col", default="y_pred")
    ap.add_argument("--y_prob_col", default="y_prob")
    ap.add_argument("--output_metrics_json", required=True)
    ap.add_argument("--output_confusion_csv", required=True)
    ap.add_argument("--output_roc_csv", required=True)
    ap.add_argument("--output_pr_csv", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.preds_parquet)
    y_true = df[args.y_true_col].astype(int).values
    y_pred = df[args.y_pred_col].astype(int).values
    y_prob = df[args.y_prob_col].astype(float).values

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    try: roc = roc_auc_score(y_true, y_prob)
    except: roc = None
    pr_auc = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)

    # log a MLflow (pestaña Metrics)
    mlflow.log_metrics({
        "eval_accuracy": acc, "eval_precision": prec, "eval_recall": rec,
        "eval_f1": f1, "eval_roc_auc": -1.0 if roc is None else float(roc),
        "eval_pr_auc": pr_auc
    })

    metrics = {
        "metrics": {
            "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "roc_auc": None if roc is None else float(roc),
            "pr_auc": float(pr_auc)
        }
    }

    for p in [args.output_metrics_json, args.output_confusion_csv, args.output_roc_csv, args.output_pr_csv]:
        d=os.path.dirname(p)
        if d: os.makedirs(d, exist_ok=True)

    with open(args.output_metrics_json, "w") as f: json.dump(metrics, f, indent=2)
    pd.DataFrame(cm, index=["true_neg(0)","true_pos(1)"], columns=["pred_neg(0)","pred_pos(1)"]).to_csv(args.output_confusion_csv, index=True)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(args.output_roc_csv, index=False)
    pd.DataFrame({"precision": prec_c, "recall": rec_c}).to_csv(args.output_pr_csv, index=False)

    print(f"[evaluate_binary] acc={acc:.4f} f1={f1:.4f} roc_auc={roc} pr_auc={pr_auc:.4f}")
