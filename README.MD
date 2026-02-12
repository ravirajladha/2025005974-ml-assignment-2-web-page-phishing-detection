# Web Page Phishing Detection (ML Assignment 2)

## Problem
Classify a web page/URL as Phishing vs Legitimate using classical ML models.

## Dataset
Web Page Phishing Detection Dataset (Kaggle). ~11,430 rows and 87 features. Target column commonly `Result`. 

## Models
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- XGBoost

## Metrics
Accuracy, AUC, Precision, Recall, F1, MCC

## How to Train
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/train.py --data path/to/your_dataset.csv
