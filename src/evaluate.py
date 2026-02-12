from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix,
    classification_report
)

def _to_binary01(y):
    y = np.asarray(y)
    # Many versions use {-1, 1}. Convert to {0,1}.
    uniq = set(np.unique(y).tolist())
    if uniq == {-1, 1}:
        return (y == 1).astype(int)
    # If already {0,1} or {1,0}, keep
    return y.astype(int)

def evaluate_binary(y_true, y_pred, y_proba):
    y_true = _to_binary01(y_true)
    y_pred = _to_binary01(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return metrics, cm, report
