import argparse, os, json, pickle
import numpy as np
import pandas as pd
import mlflow 
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef, log_loss,            
    precision_recall_fscore_support
)

def makedirs_safe(path: str):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # inputs
    ap.add_argument("--X_train_npz", required=True)
    ap.add_argument("--y_train_parquet", required=True)
    ap.add_argument("--X_test_npz", required=True)
    ap.add_argument("--y_test_parquet", required=True)
    ap.add_argument("--y_col", default="label")

    # hyperparams
    ap.add_argument("--solver", default="liblinear", choices=["liblinear","saga"])
    ap.add_argument("--penalty", default="l2", choices=["l1","l2","elasticnet"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--class_weight", default="balanced", choices=["balanced","none"])
    ap.add_argument("--l1_ratio", type=float, default=0.0, help="Solo para penalty=elasticnet + solver=saga")

    # outputs
    ap.add_argument("--output_model_pkl", required=True)
    ap.add_argument("--output_metrics_json", required=True)
    ap.add_argument("--output_predictions_parquet", required=True)
    ap.add_argument("--output_confusion_csv", required=True)
    ap.add_argument("--output_roc_csv", required=True)
    ap.add_argument("--output_pr_csv", required=True)
    args = ap.parse_args()

    for p in [args.X_train_npz, args.X_test_npz, args.y_train_parquet, args.y_test_parquet]:
        print("[train] expecting:", p)
        if not os.path.exists(p):
            d = os.path.dirname(p)
            print("[train][ERROR] not found:", p)
            if d and os.path.isdir(d):
                print("[train][DEBUG] ls", d, "->", os.listdir(d))
            raise FileNotFoundError(p)

    # --- carga datos ---
    Xtr = load_npz(args.X_train_npz)
    Xte = load_npz(args.X_test_npz)
    ytr = pd.read_parquet(args.y_train_parquet)[args.y_col].astype("int8").to_numpy()
    yte = pd.read_parquet(args.y_test_parquet)[args.y_col].astype("int8").to_numpy()

    # --- valida binario ---
    classes = np.unique(ytr)
    if not set(classes).issubset({0,1}):
        raise ValueError(f"train_lr: se esperaba problema binario con labels 0/1; encontrados {classes}")

    # --- configura modelo ---
    cw = None if args.class_weight == "none" else "balanced"
    if args.solver == "liblinear" and args.penalty == "elasticnet":
        raise ValueError("elasticnet requiere solver='saga'")
    if args.penalty == "l1" and args.solver not in ("liblinear","saga"):
        raise ValueError("l1 soportado por liblinear/saga")
    if args.penalty == "elasticnet" and args.solver != "saga":
        raise ValueError("elasticnet requiere solver='saga'")

    lr = LogisticRegression(
        solver=args.solver,
        penalty=args.penalty,
        C=args.C,
        class_weight=cw,
        max_iter=args.max_iter,
        n_jobs=None if args.solver=="liblinear" else None,
        l1_ratio=args.l1_ratio if (args.penalty=="elasticnet" and args.solver=="saga") else None,
        random_state=42
    )

    # --- entrena ---
    lr.fit(Xtr, ytr)

    # --- predice ---
    y_pred = lr.predict(Xte)
    if hasattr(lr, "predict_proba"):
        y_prob = lr.predict_proba(Xte)[:,1]
    else:
        margin = lr.decision_function(Xte)
        y_prob = 1/(1+np.exp(-margin))

    # --- métricas base ---
    acc = accuracy_score(yte, y_pred)
    f1  = f1_score(yte, y_pred)
    prec= precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred)
    try:    roc = roc_auc_score(yte, y_prob)
    except ValueError: roc = None
    pr_auc = average_precision_score(yte, y_prob)

    # --- métricas extra (para Metrics) ---
    bal_acc = balanced_accuracy_score(yte, y_pred)
    mcc     = matthews_corrcoef(yte, y_pred)
    try:    ll = log_loss(yte, y_prob)
    except ValueError: ll = None

    # --- logging MLflow: params + metrics escalares ---
    mlflow.log_params({
        "model": "LogisticRegression",
        "solver": args.solver,
        "penalty": args.penalty,
        "C": args.C,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "l1_ratio": args.l1_ratio if args.penalty=="elasticnet" else None,
        "n_features": int(Xtr.shape[1]),
    })
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": -1.0 if roc is None else float(roc),
        "pr_auc": pr_auc,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "log_loss": -1.0 if ll is None else float(ll),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
    })

    # métricas por clase (0/1)
    pcls, rcls, fcls, _ = precision_recall_fscore_support(yte, y_pred, labels=[0,1], zero_division=0)
    mlflow.log_metrics({
        "precision_0": pcls[0], "recall_0": rcls[0], "f1_0": fcls[0],
        "precision_1": pcls[1], "recall_1": rcls[1], "f1_1": fcls[1],
    })

    # barrido de umbral para ver curvas en Metrics (via step)
    for i, t in enumerate(np.linspace(0.1, 0.9, 17)):
        y_hat_t = (y_prob >= t).astype(int)
        mlflow.log_metric("f1_at_threshold",       f1_score(yte, y_hat_t), step=i)
        mlflow.log_metric("precision_at_threshold",precision_score(yte, y_hat_t, zero_division=0), step=i)
        mlflow.log_metric("recall_at_threshold",   recall_score(yte, y_hat_t), step=i)

    # --- tablas/artefactos en Outputs + logs ---
    cm = confusion_matrix(yte, y_pred, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["true_neg(0)","true_pos(1)"], columns=["pred_neg(0)","pred_pos(1)"])
    fpr, tpr, _ = roc_curve(yte, y_prob)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    prec_c, rec_c, _ = precision_recall_curve(yte, y_prob)
    pr_df = pd.DataFrame({"precision": prec_c, "recall": rec_c})

    for p in [args.output_model_pkl, args.output_metrics_json, args.output_predictions_parquet,
              args.output_confusion_csv, args.output_roc_csv, args.output_pr_csv]:
        makedirs_safe(p)

    with open(args.output_model_pkl, "wb") as f:
        pickle.dump(lr, f)

    metrics = {
        "solver": args.solver,
        "penalty": args.penalty,
        "C": args.C,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "l1_ratio": args.l1_ratio if args.penalty=="elasticnet" else None,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_features": int(Xtr.shape[1]),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": None if roc is None else float(roc),
            "pr_auc": float(pr_auc),
            "balanced_accuracy": float(bal_acc),
            "mcc": float(mcc),
            "log_loss": None if ll is None else float(ll)
        }
    }
    with open(args.output_metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    preds = pd.DataFrame({"y_true": yte.astype(int), "y_pred": y_pred.astype(int), "y_prob": y_prob.astype(float)})
    preds.to_parquet(args.output_predictions_parquet, index=False)
    cm_df.to_csv(args.output_confusion_csv, index=True)
    roc_df.to_csv(args.output_roc_csv, index=False)
    pr_df.to_csv(args.output_pr_csv, index=False)

    # sube artefactos a MLflow (aparecen en Outputs + logs)
    mlflow.log_artifact(args.output_confusion_csv,        artifact_path="tables")
    mlflow.log_artifact(args.output_roc_csv,              artifact_path="tables")
    mlflow.log_artifact(args.output_pr_csv,               artifact_path="tables")
    mlflow.log_artifact(args.output_metrics_json,         artifact_path="json")
    mlflow.log_artifact(args.output_predictions_parquet,  artifact_path="preds")

    dens_tr = Xtr.nnz/(Xtr.shape[0]*Xtr.shape[1])
    dens_te = Xte.nnz/(Xte.shape[0]*Xte.shape[1])
    print(f"[train_lr] Xtr={Xtr.shape} dens={dens_tr:.4f} | Xte={Xte.shape} dens={dens_te:.4f} "
          f"| acc={acc:.4f} f1={f1:.4f} roc_auc={roc} pr_auc={pr_auc:.4f}")
