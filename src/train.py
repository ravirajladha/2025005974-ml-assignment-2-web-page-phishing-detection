from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils import ensure_dirs, ARTIFACTS_DIR, MODEL_DIR, save_json, infer_target_column
from src.evaluate import evaluate_binary

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0  # force dense output (helps GaussianNB)
    )
    return pre

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--target", default=None, help="Target column name (optional)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    df = pd.read_csv(args.data)
    target = args.target or infer_target_column(df.columns.tolist())

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {df.columns.tolist()}")

    y = df[target]
    X = df.drop(columns=[target])

    # Convert labels if needed (e.g., {-1,1})
    if set(pd.unique(y).tolist()) == {-1, 1}:
        y = (y == 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    pre = build_preprocessor(X_train)

    models = {
        "lr": LogisticRegression(max_iter=2000, n_jobs=None),
        "dt": DecisionTreeClassifier(random_state=args.seed),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "nb": GaussianNB(),
        "rf": RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1
        ),
        "xgb": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.seed,
            eval_metric="logloss"
        ),
    }

    all_metrics = {}
    all_reports = {}
    all_confusions = {}

    for key, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics, cm, report = evaluate_binary(y_test, y_pred, y_proba)

        all_metrics[key] = metrics
        all_confusions[key] = cm
        # keep report lighter (optional)
        all_reports[key] = {
            "0": report.get("0", {}),
            "1": report.get("1", {}),
            "accuracy": report.get("accuracy", None),
            "macro avg": report.get("macro avg", {}),
            "weighted avg": report.get("weighted avg", {}),
        }

        dump(pipe, ARTIFACTS_DIR / f"{key}.joblib")

    save_json(MODEL_DIR / "metrics.json", all_metrics)
    save_json(MODEL_DIR / "reports.json", {"confusion_matrices": all_confusions, "classification_reports": all_reports})

    print("Saved ✅")
    print("Artifacts:", ARTIFACTS_DIR)
    print("metrics.json:", MODEL_DIR / "metrics.json")

if __name__ == "__main__":
    main()
