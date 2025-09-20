import argparse, os, json, pickle
import numpy as np
import pandas as pd
import mlflow
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef, log_loss,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def makedirs_safe(p):
    d = os.path.dirname(p)
    if d: os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--X_train_npz", required=True)
    ap.add_argument("--y_train_parquet", required=True)
    ap.add_argument("--X_test_npz", required=True)
    ap.add_argument("--y_test_parquet", required=True)
    ap.add_argument("--y_col", default="label")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--class_weight", default="balanced", choices=["balanced","none"])
    ap.add_argument("--calibration_cv", type=int, default=3)  # 0 para desactivar
    ap.add_argument("--output_model_pkl", required=True)
    ap.add_argument("--output_metrics_json", required=True)
    ap.add_argument("--output_predictions_parquet", required=True)
    ap.add_argument("--output_confusion_csv", required=True)
    ap.add_argument("--output_roc_csv", required=True)
    ap.add_argument("--output_pr_csv", required=True)
    args = ap.parse_args()

    # --- carga datos ---
    Xtr = load_npz(args.X_train_npz)
    Xte = load_npz(args.X_test_npz)
    ytr = pd.read_parquet(args.y_train_parquet)[args.y_col].astype("int8").to_numpy()
    yte = pd.read_parquet(args.y_test_parquet)[args.y_col].astype("int8").to_numpy()

    # --- modelo ---
    cw = None if args.class_weight == "none" else "balanced"
    base = LinearSVC(C=args.C, class_weight=cw, max_iter=args.max_iter, dual=True)

    if args.calibration_cv and args.calibration_cv > 0:
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=args.calibration_cv)
        model_name = "LinearSVC+Calibrated"
    else:
        clf = base
        model_name = "LinearSVC"

    clf.fit(Xtr, ytr)

    # --- predicciones ---
    y_pred = clf.predict(Xte)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(Xte)[:, 1]
    else:
        margin = clf.decision_function(Xte)
        y_prob = 1.0 / (1.0 + np.exp(-margin))

    # --- métricas base ---
    acc = accuracy_score(yte, y_pred)
    f1  = f1_score(yte, y_pred)
    prec= precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred)
    try:    roc = roc_auc_score(yte, y_prob)
    except ValueError: roc = None
    pr_auc = average_precision_score(yte, y_prob)

    # --- métricas extra ---
    bal_acc = balanced_accuracy_score(yte, y_pred)
    mcc     = matthews_corrcoef(yte, y_pred)
    try:    ll = log_loss(yte, y_prob)
    except ValueError: ll = None

    # --- log en MLflow: params + metrics ---
    mlflow.log_params({
        "model": model_name,
        "C": args.C,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "calibration_cv": args.calibration_cv,
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

    # por clase (0/1)
    pcls, rcls, fcls, _ = precision_recall_fscore_support(yte, y_pred, labels=[0,1], zero_division=0)
    mlflow.log_metrics({
        "precision_0": pcls[0], "recall_0": rcls[0], "f1_0": fcls[0],
        "precision_1": pcls[1], "recall_1": rcls[1], "f1_1": fcls[1],
    })

    # barrido de umbral (curvas en Metrics por 'step')
    for i, t in enumerate(np.linspace(0.1, 0.9, 17)):
        y_hat_t = (y_prob >= t).astype(int)
        mlflow.log_metric("f1_at_threshold",        f1_score(yte, y_hat_t), step=i)
        mlflow.log_metric("precision_at_threshold", precision_score(yte, y_hat_t, zero_division=0), step=i)
        mlflow.log_metric("recall_at_threshold",    recall_score(yte, y_hat_t), step=i)

    # --- tablas/artefactos ---
    cm = confusion_matrix(yte, y_pred, labels=[0,1])
    fpr, tpr, _ = roc_curve(yte, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(yte, y_prob)

    cm_df  = pd.DataFrame(cm, index=["true_neg(0)","true_pos(1)"], columns=["pred_neg(0)","pred_pos(1)"])
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    pr_df  = pd.DataFrame({"precision": prec_c, "recall": rec_c})

    for p in [args.output_model_pkl, args.output_metrics_json, args.output_predictions_parquet,
              args.output_confusion_csv, args.output_roc_csv, args.output_pr_csv]:
        makedirs_safe(p)

    with open(args.output_model_pkl, "wb") as f:
        pickle.dump(clf, f)

    metrics = {
        "model": model_name,
        "C": args.C, "class_weight": args.class_weight, "max_iter": args.max_iter,
        "calibration_cv": args.calibration_cv,
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]), "n_features": int(Xtr.shape[1]),
        "metrics": {
            "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "roc_auc": None if roc is None else float(roc), "pr_auc": float(pr_auc),
            "balanced_accuracy": float(bal_acc), "mcc": float(mcc),
            "log_loss": None if ll is None else float(ll)
        }
    }
    with open(args.output_metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({"y_true": yte, "y_pred": y_pred, "y_prob": y_prob}).to_parquet(args.output_predictions_parquet, index=False)
    cm_df.to_csv(args.output_confusion_csv, index=True)
    roc_df.to_csv(args.output_roc_csv, index=False)
    pr_df.to_csv(args.output_pr_csv, index=False)

    # figuras → artifacts
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(yte, y_pred, ax=ax_cm)
    mlflow.log_figure(fig_cm, "plots/confusion_matrix.png"); plt.close(fig_cm)

    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(yte, y_prob, ax=ax_roc)
    mlflow.log_figure(fig_roc, "plots/roc_curve.png"); plt.close(fig_roc)

    fig_pr, ax_pr = plt.subplots()
    PrecisionRecallDisplay.from_predictions(yte, y_prob, ax=ax_pr)
    mlflow.log_figure(fig_pr, "plots/pr_curve.png"); plt.close(fig_pr)

    # sube también los archivos generados
    mlflow.log_artifact(args.output_confusion_csv,        artifact_path="tables")
    mlflow.log_artifact(args.output_roc_csv,              artifact_path="tables")
    mlflow.log_artifact(args.output_pr_csv,               artifact_path="tables")
    mlflow.log_artifact(args.output_metrics_json,         artifact_path="json")
    mlflow.log_artifact(args.output_predictions_parquet,  artifact_path="preds")

    print(f"[train_svm] acc={acc:.4f} f1={f1:.4f} roc_auc={roc} pr_auc={pr_auc:.4f}")
