# AI-Based Customer Churn Prediction — Full Project Report

---

## 1. Problem Statement

Customer churn (when a customer stops using a company's service) is a major problem for telecom companies. Every customer who leaves means **lost recurring revenue**, and acquiring a new customer costs **5–7x more** than retaining an existing one.

The goal of this project is to build a **Machine Learning model** that can predict whether a customer is likely to churn based on their account details (tenure, charges, contract type, services subscribed, etc.). If the company knows a customer is at risk, they can proactively offer discounts, better plans, or customer support to retain them.

**Dataset:** Telco Customer Churn dataset from IBM (7043 customers, 21 features).
- **Target variable:** `Churn` (Yes/No)
- **Features:** Gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

---

## 2. Why We Selected This Problem

| Reason | Explanation |
|--------|-------------|
| **Real-world relevance** | Every telecom company faces churn — solving it has direct business value |
| **Clear classification problem** | Binary outcome (churn / no churn) is perfect for ML beginners to advanced |
| **Rich dataset** | Mix of numeric, binary, and multi-category features — covers all preprocessing techniques |
| **Imbalanced classes** | ~26.5% churn rate — teaches handling of class imbalance & model comparison |
| **Deployable** | Easy to wrap in a simple web app for real-time prediction |

---

## 3. Project Workflow (End-to-End Flow)

```
                   ┌─────────────────────────────────┐
                   │        DATASET (CSV)              │
                   │  7043 customers, 21 columns      │
                   └──────────┬──────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      1. DATA LOADING          │
              │    pd.read_csv("dataset.csv") │
              └──────────┬───────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────┐
            │     2. EXPLORATORY ANALYSIS     │
            │  • info() → dtypes, null counts │
            │  • describe() → stats           │
            │  • head() → sample rows         │
            └──────────┬─────────────────────┘
                       │
                       ▼
          ┌───────────────────────────────────┐
          │       3. DATA CLEANING            │
          │  • TotalCharges → numeric (coerce)│
          │  • Drop null rows (11 rows)       │
          │  • Drop customerID (unique ID)    │
          └──────────┬───────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────┐
        │     4. FEATURE ENGINEERING          │
        │  • Binary cols → 0/1 mapping        │
        │  • Multi-category cols → one-hot    │
        │  (get_dummies, drop_first=True)     │
        └──────────┬─────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────────┐
        │     5. TRAIN/TEST SPLIT             │
        │  X_train, X_test, y_train, y_test   │
        │  test_size=0.2, random_state=42     │
        └──────────┬─────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────────┐
        │     6. FEATURE SCALING              │
        │  StandardScaler on:                 │
        │  tenure, MonthlyCharges, TotalCharges│
        │  Fit on X_train, transform on X_test│
        └──────────┬─────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────────┐
        │     7. MODEL TRAINING               │
        │  ┌───────────────────────┐          │
        │  │ Logistic Regression   │          │
        │  └───────────────────────┘          │
        │  ┌───────────────────────┐          │
        │  │ Random Forest (100    │          │
        │  │ estimators)           │          │
        │  └───────────────────────┘          │
        └──────────┬─────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────────┐
        │     8. MODEL COMPARISON             │
        │  • Accuracy: Both ~78.68%           │
        │  • Recall (churn): LR 51% > RF 46%  │
        │  • Winner: LOGISTIC REGRESSION      │
        └──────────┬─────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────────┐
        │     9. SAVE MODEL                   │
        │  model.pkl, scaler.pkl,             │
        │  feature_columns.pkl                │
        └──────────┬─────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────────┐
        │     10. DEPLOY (Streamlit App)      │
        │  app.py → streamlit run app.py      │
        │  Real-time prediction in browser    │
        └─────────────────────────────────────┘
```

---

## 4. Detailed Code Walkthrough

### 4.1 Import Libraries
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
```

**What this does:** Imports all necessary libraries — Pandas for data handling, Scikit-learn for ML models & preprocessing, Seaborn/Matplotlib for visualization, Joblib for saving the model.

**Alternatives:** Could use TensorFlow/Keras for deep learning, or XGBoost/LightGBM for boosted tree models. For this dataset size, sklearn is sufficient and simpler.

---

### 4.2 Load Data
```python
data = pd.read_csv("dataset.csv")
print(data.head())
print(data.info())
print(data.describe())
```

**What this does:** Reads the CSV file into a DataFrame, then displays first 5 rows, column info (dtypes, non-null counts), and summary statistics.

**Alternatives:** Could load from a SQL database (`pd.read_sql`), API, or cloud storage (S3, GCS).

---

### 4.3 Fix TotalCharges Column
```python
# Why? New customers have blank spaces (" ") instead of numbers, making the whole column 'object' type.
# pd.to_numeric forces spaces to become NaN so we can handle them, avoiding StandardScaler crash later!
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
# errors='coerce' turns any non-numeric data (like blank spaces or text) into NaN instead of crashing

data.dropna(inplace=True)
```

**What this does:** The `TotalCharges` column has blank strings for new customers, making it an `object` dtype. `pd.to_numeric` converts it to float, turning empty strings into `NaN`. Then we drop those 11 rows.

**Alternatives:** Instead of dropping, we could fill NaN values with `0` (new customers) or with the median/mean of TotalCharges. Dropping is safe since it's only 11 out of 7043 rows.

---

### 4.4 Drop Irrelevant Column & Encode Features
```python
data.drop(columns=['customerID'], inplace=True)

# Binary columns → 0/1
main_mapping = {'No': 0, 'Yes': 1, 'Female': 0, 'Male': 1}
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    data[col] = data[col].map(main_mapping)

# Multi-category columns → one-hot encoding
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
data = pd.get_dummies(data, columns=multi_cols, drop_first=True)
```

**What this does:**
- `customerID` is a unique identifier with no predictive value → drop it.
- Binary columns (Yes/No, Male/Female) are mapped directly to 0 and 1 — simple and efficient.
- Multi-category columns get **one-hot encoded**: each category becomes its own column with 0/1 values. `drop_first=True` drops one category per column to avoid multicollinearity (dummy variable trap).

**Why one-hot instead of LabelEncoder for multi-category?** LabelEncoder would assign numbers 0, 1, 2, ... implying an order (e.g., "DSL < Fiber optic < No"), which is misleading. One-hot treats all categories equally.

**Alternatives:** Could use `pd.get_dummies` without `drop_first` and use regularization (L1/L2) to handle multicollinearity. Could also use `LabelEncoder` + `OrdinalEncoder` if categories have a natural order (they don't here).

---

### 4.5 Split Features & Target
```python
X = data.drop(columns=['Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**What this does:** Separates the target (`Churn`) from features. Splits into 80% training, 20% testing. `random_state=42` ensures reproducibility.

**Alternatives:** Could use `StratifiedShuffleSplit` to maintain the same churn ratio in train/test sets (helps with imbalance). For very large datasets, could use 90/10 or 99/1 split.

---

### 4.6 Scale Numeric Features
```python
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
```

**What this does:** `StandardScaler` subtracts the mean and divides by standard deviation, so all numeric features have mean=0 and std=1. This is critical for Logistic Regression (gradient descent converges faster) and distance-based models.

**Important:** `fit_transform` on training, only `transform` on testing — this prevents **data leakage** (the test set shouldn't influence training statistics).

**Alternatives:** `MinMaxScaler` (scales to 0–1 range), `RobustScaler` (uses median/IQR — better for outliers), or `Normalizer`. StandardScaler is the safest default.

---

### 4.7 Train Logistic Regression
```python
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(classification_report(y_test, y_pred_log))
```

**What this does:** Creates a Logistic Regression model, trains it on X_train/y_train, and predicts on X_test. Outputs accuracy and a detailed classification report (precision, recall, f1-score).

**How Logistic Regression works:** It calculates the probability that a customer churns using the formula: `p = 1 / (1 + e^-(b0 + b1*x1 + b2*x2 + ...))`. If probability > 0.5, predict churn=1 (Yes); else predict churn=0 (No).

**Alternatives:** Could adjust `class_weight='balanced'` to handle the ~26.5% churn imbalance. Could tune `C` (inverse regularization strength) via cross-validation. Could also add L1 (Lasso) regularization for feature selection.

---

### 4.8 Train Random Forest (For Comparison)
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))
```

**What this does:** Creates a Random Forest with 100 decision trees, trains, and predicts.

**How Random Forest works:** It builds many decision trees on random subsets of data and features, then averages their predictions. This reduces overfitting compared to a single decision tree.

**Alternatives:** Could tune `n_estimators` (more trees = better but slower), `max_depth`, `min_samples_split`, or use `class_weight='balanced'`. Could also try Gradient Boosting (XGBoost, LightGBM, CatBoost) for potentially higher accuracy.

---

### 4.9 Model Comparison — Why Logistic Regression Wins

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | **78.68%** | **78.68%** |
| Precision (Churn) | **0.62** | 0.64 |
| Recall (Churn) | **0.51** | 0.46 |
| F1-Score (Churn) | **0.56** | 0.54 |

**Decision Rationale:**
Both models give the same accuracy, but for churn prediction, **Recall matters most**. Recall measures: "Of all customers who actually churned, how many did we catch?" Higher recall = fewer missed churners = less revenue lost.

Logistic Regression catches **51%** of churners vs Random Forest's **46%** — a 5% improvement in catching leavers. In a business context, this could save millions.

**Alternatives considered:**
- **XGBoost/LightGBM** — Usually better performance, but more complex to tune
- **Neural Networks** — Overkill for this dataset size (7K rows); prone to overfitting
- **SVM** — Works well but doesn't give interpretable probabilities
- **K-Nearest Neighbors** — No training needed, but poor with high-dimensional one-hot data

Logistic Regression was chosen because:
1. **Simple & interpretable** — You can see which features matter most
2. **Good enough performance** — 78.68% accuracy with 51% churn recall
3. **Fast to train** — Seconds vs minutes for ensemble methods
4. **Probabilistic output** — Gives churn probability, not just a label

---

### 4.10 Confusion Matrix Visualization
```python
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

**What this does:** Plots a heatmap showing True Positives, True Negatives, False Positives, False Negatives — helps visualize where the model makes mistakes.

---

### 4.11 Save Model for Deployment
```python
joblib.dump(log_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
```

**What this does:** Saves the trained Logistic Regression model, the fitted scaler (so new data is scaled the same way), and the list of feature column names (to ensure the app creates identical columns).

**Alternatives:** Could use `pickle` instead of `joblib` (joblib is more efficient for large numpy arrays). Could save to cloud storage (S3, GCS) or MLflow model registry.

---

### 4.12 Streamlit Web App (app.py)

The app lets users input customer details via a sidebar form and get instant predictions.

```python
# Simplified flow of app.py:
# 1. Load model.pkl, scaler.pkl, feature_columns.pkl
# 2. Build input from sidebar selections
# 3. Apply same encoding & scaling as training
# 4. Predict churn probability
# 5. Display result (Will Churn / Will NOT Churn) with probability
```

**Key logic:**
```python
input_df = pd.get_dummies(input_df, columns=multi_cols, drop_first=True)
# Add missing columns that model was trained on
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]  # Same column order as training
input_df[num_cols] = scaler.transform(input_df[num_cols])  # Scale numbers
prob = model.predict_proba(input_df)[0][1]  # Get churn probability
```

**What this does:** The app mirrors the training preprocessing exactly. One-hot encoding may produce fewer columns than training (if the user didn't select every category), so we add missing columns with value 0. Then we reorder columns to match training order exactly.

**Alternatives to Streamlit:** Flask/Django (full web framework), Gradio (easy ML demos), FastAPI (REST API), or embed in a mobile app via TensorFlow Lite.

---

## 5. Further Improvements (What Could Be Done Better)

| Area | Current Approach | Possible Improvement |
|------|-----------------|---------------------|
| **Data** | Drop 11 rows with missing TotalCharges | Fill with 0 or tenure×MonthlyCharges |
| **Imbalance** | No handling (~26.5% churn) | SMOTE oversampling, class_weight='balanced', or threshold tuning |
| **Features** | Manual one-hot encoding | Feature selection (SelectKBest, RFE) or PCA for dimensionality reduction |
| **Model** | Logistic Regression (78.68% acc) | XGBoost, Hyperparameter tuning (GridSearchCV), Ensemble stacking |
| **Validation** | Single train/test split | K-Fold Cross-Validation for more reliable performance estimate |
| **Deployment** | Local Streamlit | Deploy to Streamlit Cloud, Heroku, Docker, or AWS/GCP |
| **Threshold** | Default 0.5 | Adjust threshold based on cost analysis (cost of FP vs FN) |
| **Monitoring** | None | Track model drift, retrain periodically on new data |

---

## 6. Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.13** | Core programming language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computations |
| **Scikit-learn** | ML models, preprocessing, metrics |
| **Joblib** | Model serialization |
| **Seaborn / Matplotlib** | Visualization |
| **Streamlit** | Web deployment |

---

## 7. Conclusion

This project successfully built an **AI-based Customer Churn Prediction system** that:

1. Takes raw Telco customer data (7043 records, 21 features)
2. Cleans, encodes, and scales it properly
3. Trains and compares two ML models (Logistic Regression & Random Forest)
4. Selects **Logistic Regression** as the winner based on higher churn recall (51% vs 46%)
5. Deploys the model through an interactive **Streamlit web app**

The final model achieves **78.68% accuracy** and can predict churn probability in real-time, enabling telecom companies to take proactive retention actions and reduce revenue loss.

**"Catch them before they leave."**
