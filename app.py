from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from joblib import load

from src.utils import MODEL_DIR, ARTIFACTS_DIR, load_json, infer_target_column


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Phishing Detection ML App", layout="wide")

st.title("🛡️ Web Page Phishing Detection")
st.caption("Upload CSV → select model → view metrics → confusion matrix/report → get predictions.")


# -----------------------------
# Load precomputed metrics/reports
# -----------------------------
metrics_path = MODEL_DIR / "metrics.json"
reports_path = MODEL_DIR / "reports.json"

metrics = load_json(metrics_path) if metrics_path.exists() else {}
reports = load_json(reports_path) if reports_path.exists() else {}


# -----------------------------
# Helpers
# -----------------------------
def get_report_for_model(reports_dict: dict, key: str) -> dict:
    """
    Supports both formats:
    A) reports[key] = {"confusion_matrix": ..., "classification_report": ...}
    B) reports["confusion_matrices"][key] and reports["classification_reports"][key]
    """
    if not reports_dict:
        return {}

    # Format A: per-model dict at top level
    if key in reports_dict and isinstance(reports_dict[key], dict):
        return reports_dict[key]

    # Format B: split dicts
    out = {}
    cms = reports_dict.get("confusion_matrices", {})
    crs = reports_dict.get("classification_reports", {})
    if isinstance(cms, dict) and key in cms:
        out["confusion_matrix"] = cms[key]
    if isinstance(crs, dict) and key in crs:
        out["classification_report"] = crs[key]

    # Optional extras (labels, etc.)
    labels = reports_dict.get("labels")
    if labels:
        out["labels"] = labels

    return out

def align_features_for_model(X: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    Rename common phishing feature column variants and align to expected columns.
    Missing expected columns are created with 0.
    Extra columns are dropped.
    """
    rename_map = {
        # your sample -> training expected
        "length_url": "url_length",
        "nb_dots": "n_dots",
        "nb_hyphens": "n_hypens",
        "nb_at": "n_at",
        "nb_qm": "n_questionmark",
        "nb_and": "n_and",
        "nb_percent": "n_percent",
        "nb_slash": "n_slash",
        "nb_underscore": "n_underline",
        "nb_eq": "n_equal",
        "nb_tilde": "n_tilde",
        "nb_plus": "n_plus",
        "nb_exclamation": "n_exclamation",
        "nb_comma": "n_comma",
        "nb_space": "n_space",
        "nb_star": "n_asterisk",
        "nb_dollar": "n_dollar",
        # sometimes:
        "nb_redirection": "n_redirection",
        "nb_hashtag": "n_hastag",   # common misspelling in datasets
        "nb_hastag": "n_hastag",
    }

    X2 = X.copy()
    X2 = X2.rename(columns={c: rename_map.get(c, c) for c in X2.columns})

    # Create missing expected columns as 0
    for c in expected_cols:
        if c not in X2.columns:
            X2[c] = 0

    # Keep only expected columns in the right order
    X2 = X2[expected_cols]
    return X2

def plot_confusion_matrix(cm, labels=None):
    cm = np.array(cm)
    if labels is None:
        labels = ["Legit", "Phishing"]

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    st.pyplot(fig)


def pretty_model_table(metrics_dict: dict) -> pd.DataFrame:
    dfm = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={"index": "model"})
    # Optional: order columns if present
    preferred = ["model", "accuracy", "auc", "precision", "recall", "f1", "mcc"]
    cols = [c for c in preferred if c in dfm.columns] + [c for c in dfm.columns if c not in preferred]
    return dfm[cols]


# -----------------------------
# Model map
# -----------------------------
model_map = {
    "Logistic Regression": "lr",
    "Decision Tree": "dt",
    "KNN": "knn",
    "Naive Bayes": "nb",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload CSV (test data preferred)", type=["csv"])
    model_name = st.selectbox("Choose a model", list(model_map.keys()))
    show_all = st.checkbox("Show comparison table (all models)", value=True)

key = model_map[model_name]
artifact_file = ARTIFACTS_DIR / f"{key}.joblib"


# -----------------------------
# Top layout (metrics + model details)
# -----------------------------
colA, colB = st.columns([1.1, 1])

with colA:
    if show_all and metrics:
        st.subheader("Model comparison")
        dfm = pretty_model_table(metrics)
        st.dataframe(dfm, use_container_width=True)
    else:
        st.info("Metrics table hidden or metrics.json not found yet.")

with colB:
    st.subheader(f"Selected: {model_name}")

    if metrics and key in metrics:
        m = metrics[key]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
        c2.metric("AUC", f"{m.get('auc', 0):.4f}")
        c3.metric("MCC", f"{m.get('mcc', 0):.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Precision", f"{m.get('precision', 0):.4f}")
        c5.metric("Recall", f"{m.get('recall', 0):.4f}")
        c6.metric("F1", f"{m.get('f1', 0):.4f}")
    else:
        st.warning("No metrics found for this model. Train models first.")

    # Confusion matrix + report (from training) shown even without upload
    rep = get_report_for_model(reports, key)
    with st.expander("📌 Confusion matrix & classification report (from training)", expanded=True):
        cm = rep.get("confusion_matrix") or rep.get("cm")
        if cm is not None:
            plot_confusion_matrix(cm, labels=rep.get("labels"))
        else:
            st.info("Confusion matrix not found in reports.json for this model.")

        cr = rep.get("classification_report") or rep.get("report")
        if cr is not None:
            st.subheader("Classification report")
            if isinstance(cr, dict):
                st.json(cr)
            else:
                st.code(str(cr))
        else:
            st.info("Classification report not found in reports.json for this model.")


st.divider()


# -----------------------------
# Prediction section
# -----------------------------
if uploaded is None:
    st.write("📎 Upload a CSV to run predictions.")
else:
    if not artifact_file.exists():
        st.error(f"Missing model artifact: {artifact_file}. Train models first.")
    else:
        df = pd.read_csv(uploaded)

        st.subheader("Uploaded data preview")
        st.dataframe(df.head(20), use_container_width=True)

        # If label exists, drop it for prediction
        target = infer_target_column(df.columns.tolist())
        has_label = target in df.columns

        X = df.drop(columns=[target]) if has_label else df

        pipe = load(artifact_file)

        # Align columns to what the model expects
        if hasattr(pipe, "feature_names_in_"):
            expected = list(pipe.feature_names_in_)
            X_for_pred = align_features_for_model(X, expected)
        else:
            # fallback: try as-is
            X_for_pred = X

        preds = pipe.predict(X_for_pred)


        # Some models might not support predict_proba; guard it
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_for_pred)[:, 1]
            except Exception:
                proba = None


        out = df.copy()
        out["pred_label"] = preds
        if proba is not None:
            out["pred_proba"] = proba

        st.subheader("Predictions")
        st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
