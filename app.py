from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import load

from src.utils import MODEL_DIR, ARTIFACTS_DIR, load_json, infer_target_column

st.set_page_config(page_title="Phishing Detection ML App", layout="wide")

st.title("🛡️ Web Page Phishing Detection")
st.caption("Upload CSV → select model → view metrics → get predictions.")

# Load precomputed metrics (from training)
metrics_path = MODEL_DIR / "metrics.json"
reports_path = MODEL_DIR / "reports.json"
metrics = load_json(metrics_path) if metrics_path.exists() else {}
reports = load_json(reports_path) if reports_path.exists() else {}

model_map = {
    "Logistic Regression": "lr",
    "Decision Tree": "dt",
    "KNN": "knn",
    "Naive Bayes": "nb",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload CSV (test data preferred)", type=["csv"])
    model_name = st.selectbox("Choose a model", list(model_map.keys()))
    show_all = st.checkbox("Show comparison table (all models)", value=True)

key = model_map[model_name]
artifact_file = ARTIFACTS_DIR / f"{key}.joblib"

colA, colB = st.columns([1.1, 1])

with colA:
    if show_all and metrics:
        st.subheader("Model comparison")
        dfm = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
        st.dataframe(dfm, use_container_width=True)
    else:
        st.info("Train models first to populate metrics.json (we’ll do this in Kaggle/Jupyter).")

with colB:
    st.subheader(f"Selected: {model_name}")
    if metrics and key in metrics:
        m = metrics[key]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']:.4f}")
        c2.metric("AUC", f"{m['auc']:.4f}")
        c3.metric("MCC", f"{m['mcc']:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Precision", f"{m['precision']:.4f}")
        c5.metric("Recall", f"{m['recall']:.4f}")
        c6.metric("F1", f"{m['f1']:.4f}")
    else:
        st.warning("No metrics yet for this model. Train first.")

st.divider()

if uploaded is None:
    st.write("📎 Upload a CSV to run predictions.")
else:
    if not artifact_file.exists():
        st.error(f"Missing model artifact: {artifact_file}. Train models first.")
    else:
        df = pd.read_csv(uploaded)
        st.subheader("Uploaded data preview")
        st.dataframe(df.head(20), use_container_width=True)

        # If label exists, we can compute metrics in-app later (optional).
        target = infer_target_column(df.columns.tolist())
        has_label = target in df.columns

        X = df.drop(columns=[target]) if has_label else df

        pipe = load(artifact_file)
        preds = pipe.predict(X)
        proba = pipe.predict_proba(X)[:, 1]

        out = df.copy()
        out["pred_label"] = preds
        out["pred_proba"] = proba

        st.subheader("Predictions")
        st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

        # Show saved confusion matrix/report if available
        if reports and "confusion_matrices" in reports and key in reports["confusion_matrices"]:
            st.subheader("Saved Confusion Matrix (from training)")
            st.write(reports["confusion_matrices"][key])

        if reports and "classification_reports" in reports and key in reports["classification_reports"]:
            st.subheader("Saved Classification Report (summary)")
            st.json(reports["classification_reports"][key])
