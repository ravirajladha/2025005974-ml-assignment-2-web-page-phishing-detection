# Web Page Phishing Detection (ML Assignment 2)

## a) Problem Statement
Build and compare multiple machine learning classification models to detect whether a web page is phishing or legitimate. Provide an interactive Streamlit app for model comparison and predictions.

## b) Dataset Description
- Dataset: Web Page Phishing Detection
- Type: Binary classification (phishing vs legitimate)
- Source: (Kaggle/UCI link here)
- Minimum requirements: >= 12 features and >= 500 instances (satisfied)
- Preprocessing summary:
  - (e.g., handled missing values, encoding if any, scaling if used, train-test split)

## c) Models Used + Metrics Comparison
All models trained/evaluated on the same dataset split.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8563 | 0.9308 | 0.8658 | 0.7152 | 0.7834 | 0.6842 |
| Decision Tree | 0.8829 | 0.9149 | 0.8665 | 0.8013 | 0.8326 | 0.7442 |
| kNN | 0.8878 | 0.9430 | 0.8567 | 0.8302 | 0.8432 | 0.7562 |
| Naive Bayes | 0.7022 | 0.8995 | 0.8915 | 0.2056 | 0.3341 | 0.3320 |
| Random Forest (Ensemble) | 0.8822 | 0.9554 | 0.8598 | 0.8075 | 0.8328 | 0.7429 |
| XGBoost (Ensemble) | **0.8922** | **0.9609** | 0.8515 | **0.8521** | **0.8518** | **0.7671** |

Best model by Accuracy: **XGBoost (0.8922)**

## Observations (Model-wise)
| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with high AUC; lower recall suggests some phishing pages are missed compared to tree ensembles. |
| Decision Tree | Better recall than LR but can overfit; performance is good but slightly behind ensembles. |
| kNN | Competitive accuracy/F1; performance depends on feature scaling and chosen k. |
| Naive Bayes | High precision but extremely low recall here; it predicts phishing very conservatively (misses many phishing cases). |
| Random Forest | Strong overall metrics and AUC; more stable than single tree. |
| XGBoost | Best overall balance (accuracy/F1/MCC) and highest AUC; captures non-linear patterns well. |

## How to Run (Local / Virtual Lab)
```bash
pip install -r requirements.txt
streamlit run app.py
