from __future__ import annotations

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
st.caption("Upload CSV → pick a model → see metrics + confusion matrix → generate predictions.")


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
        # sample -> training expected
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
        "nb_redirection": "n_redirection",
        "nb_hashtag": "n_hastag",
        "nb_hastag": "n_hastag",
    }

    X2 = X.copy()
    X2 = X2.rename(columns={c: rename_map.get(c, c) for c in X2.columns})

    missing = [c for c in expected_cols if c not in X2.columns]
    for c in missing:
        X2[c] = 0

    extra = [c for c in X2.columns if c not in expected_cols]
    if extra:
        X2 = X2.drop(columns=extra)

    return X2[expected_cols]


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
    preferred = ["model", "accuracy", "auc", "precision", "recall", "f1", "mcc"]
    cols = [c for c in preferred if c in dfm.columns] + [c for c in dfm.columns if c not in preferred]
    return dfm[cols]


def render_metric_cards(m: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
    c2.metric("AUC", f"{m.get('auc', 0):.4f}")
    c3.metric("MCC", f"{m.get('mcc', 0):.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Precision", f"{m.get('precision', 0):.4f}")
    c5.metric("Recall", f"{m.get('recall', 0):.4f}")
    c6.metric("F1", f"{m.get('f1', 0):.4f}")


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
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🧰 Controls")

    uploaded = st.file_uploader("Upload CSV (test data preferred)", type=["csv"])

    model_name = st.selectbox("Choose a model", list(model_map.keys()))
    key = model_map[model_name]
    artifact_file = ARTIFACTS_DIR / f"{key}.joblib"

    st.divider()
    st.subheader("Status")
    st.write("✅ metrics.json" if metrics else "⚠️ metrics.json missing")
    st.write("✅ reports.json" if reports else "⚠️ reports.json missing")
    st.write("✅ model artifact" if artifact_file.exists() else "⚠️ model artifact missing")

    if uploaded is not None:
        st.caption(f"Uploaded: `{uploaded.name}`")


# -----------------------------
# Tabs layout
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏁 Leaderboard", "🔍 Model Insights", "⚙️ Predict"])


# -----------------------------
# Tab 1: Leaderboard
# -----------------------------
with tab1:
    st.subheader("Model comparison")

    if metrics:
        dfm = pretty_model_table(metrics)
        st.dataframe(dfm, width="stretch")

        # highlight best by accuracy (simple)
        if "accuracy" in dfm.columns:
            best_row = dfm.sort_values("accuracy", ascending=False).head(1)
            best_model = best_row["model"].iloc[0]
            best_acc = best_row["accuracy"].iloc[0]
            st.success(f"Best by Accuracy: **{best_model}** (accuracy={best_acc:.4f}) ✅")
    else:
        st.info("No metrics found yet. Train models to populate metrics.json.")


# -----------------------------
# Tab 2: Model Insights
# -----------------------------
with tab2:
    st.subheader(f"Selected model: {model_name}")

    if metrics and key in metrics:
        render_metric_cards(metrics[key])
    else:
        st.warning("No metrics found for this model.")

    st.divider()

    rep = get_report_for_model(reports, key)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Confusion matrix")
        cm = rep.get("confusion_matrix") or rep.get("cm")
        if cm is not None:
            plot_confusion_matrix(cm, labels=rep.get("labels"))
        else:
            st.info("Confusion matrix not found in reports.json.")

    with right:
        st.subheader("Classification report")
        cr = rep.get("classification_report") or rep.get("report")
        if cr is not None:
            if isinstance(cr, dict):
                st.json(cr)
            else:
                st.code(str(cr))
        else:
            st.info("Classification report not found in reports.json.")


# -----------------------------
# Tab 3: Predict
# -----------------------------
with tab3:
    st.subheader("Run predictions on an uploaded CSV")

    if uploaded is None:
        st.info("📎 Upload a CSV from the sidebar to run predictions.")
    elif not artifact_file.exists():
        st.error(f"Missing model artifact: {artifact_file}. Train models first.")
    else:
        df = pd.read_csv(uploaded)

        st.subheader("Uploaded data preview")
        st.dataframe(df.head(20), width="stretch")

        target = infer_target_column(df.columns.tolist())
        has_label = target in df.columns
        X = df.drop(columns=[target]) if has_label else df

        pipe = load(artifact_file)

        if hasattr(pipe, "feature_names_in_"):
            expected = list(pipe.feature_names_in_)
            X_for_pred = align_features_for_model(X, expected)
        else:
            X_for_pred = X

        with st.spinner("Predicting..."):
            preds = pipe.predict(X_for_pred)

            proba = None
            if hasattr(pipe, "predict_proba"):
                try:
                    proba = pipe.predict_proba(X_for_pred)[:, 1]
                except Exception:
                    proba = None

        out = df.copy()
        out["pred_label"] = preds
        if proba is not None:
            out["pred_proba"] = proba

        st.subheader("Predictions output")
        st.dataframe(out.head(50), width="stretch")

        st.download_button(
            "⬇️ Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        st.caption("Tip: If your CSV has different column names, the app auto-maps common variants.")
