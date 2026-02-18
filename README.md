# Fake Job Posting Detection using Machine Learning

## 📌 Problem Statement
Online job portals often contain fraudulent job postings designed to scam applicants. The objective of this project is to build a machine learning model that can accurately detect fraudulent job listings while minimizing missed fraud cases.

---

## 📊 Dataset Overview

The dataset contains structured and textual information about job postings, including:

- Job title
- Company profile
- Description
- Requirements
- Benefits
- Location
- Employment type
- Industry
- Fraudulent label (Target variable)

The dataset is **highly imbalanced**, with only ~4.8% fraudulent postings.

Because of this imbalance, **accuracy is not a reliable metric** for evaluation.

---

## ⚖️ Evaluation Strategy

Since the dataset is imbalanced:

- Accuracy can be misleading (95% accuracy possible even if fraud is never detected).
- We prioritize:
  - **Recall** (to minimize missed fraud cases)
  - **Precision**
  - **F1 Score** (balance between precision and recall)

In fraud detection, missing a fraudulent job (False Negative) is more harmful than incorrectly flagging a real job.

---

## 🧹 Data Cleaning

- Text columns were cleaned and missing values replaced with empty strings.
- Categorical missing values were replaced with `"Unknown"` to preserve interpretability.
- High-cardinality categorical columns were carefully handled to avoid dimensional explosion.

---

## 🏗 Feature Engineering

### 1. Text Features
- Combined relevant text columns into a single `full_text` feature.
- Applied **TF-IDF Vectorization** (Top 5000 features).
- This converts text into numerical form while emphasizing important words.

### 2. Categorical & Binary Features
- Encoded using One-Hot Encoding.
- Binary features kept as numeric.

Final feature matrix shape:  
`X_train: (14304, 5201)`  
`X_test:  (3576, 5201)`

---

## 🤖 Model Selection

### Why Logistic Regression?
Logistic Regression was chosen as the baseline model because:

- Performs well with **high-dimensional sparse data** (like TF-IDF).
- Efficient and fast to train.
- Less prone to overfitting compared to tree-based models in sparse text data.
- Provides interpretable feature weights.
- Widely used industry baseline for NLP classification tasks.

### Why Not Random Forest?
Random Forest was not chosen as the primary model because:
- It does not perform optimally with extremely sparse high-dimensional TF-IDF features.
- Memory intensive with thousands of features.
- Can overfit noisy text patterns.

### Why Not Deep Learning?
Deep learning was not used because:
- Dataset size is moderate.
- Simpler linear models already provide strong performance.
- Objective was to build an interpretable and efficient baseline model.

---

## 📈 Baseline Model Results (Logistic Regression)
- Recall (Fraud): 0.91  
- Precision (Fraud): 0.60  
- F1 Score (Fraud): 0.72  
The model prioritizes recall to ensure fraudulent jobs are not missed.

---

## 🚀 Next Steps

- Train and evaluate Linear SVM.
- Compare models using Precision, Recall, and F1-score.
- Perform threshold tuning if necessary.
- Finalize best-performing model.

---

## 💡 Key Learning

This project demonstrates handling:
- Imbalanced datasets
- Text vectorization (TF-IDF)
- High-dimensional sparse features
- Model selection based on problem context
- Evaluation beyond accuracy
