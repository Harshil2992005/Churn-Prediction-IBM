# 📋 Project Flow — Customer Churn Prediction

> **Purpose:** Step-by-step documentation of everything we did in this project. Use this file to remember the exact flow, code logic, and reasoning behind each decision.

---

## 🔁 Complete Flow Overview

```
dataset.csv → Load → Clean → Encode → Split → Scale → Train (LR + RF) → Compare → Save → Deploy
```

---

## 📌 Step 1: Problem Understanding

**What:** Predict if a telecom customer will churn (Yes/No) based on their account details.

**Why:** Customer churn causes revenue loss. Early detection → proactive retention → save money.

**Dataset:** Telco Customer Churn (IBM) — 7043 rows, 21 columns.

---

## 📌 Step 2: Data Loading

**File:** `train.py` (lines 1-13)

```python
data = pd.read_csv("dataset.csv")
```

**What we did:**
- Loaded CSV using Pandas.
- Checked `data.head()` to see sample rows.
- Checked `data.info()` for column dtypes and null counts.
- Checked `data.describe()` for statistical summary.
- Checked `data.isnull().sum()` — no null values initially (but TotalCharges has blank spaces).

---

## 📌 Step 3: Data Cleaning

**File:** `train.py` (lines 15-24)

### 3.1 Fix TotalCharges Column
```python
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)
```

**Why:** New customers have blank spaces `" "` in TotalCharges, making it an `object` dtype instead of numeric. `pd.to_numeric` with `errors='coerce'` converts blanks to NaN. Then we drop those 11 rows.

**Alternative considered:** Could fill NaN with 0 or median instead of dropping.

### 3.2 Drop customerID
```python
data.drop(columns=['customerID'], inplace=True)
```

**Why:** customerID is a unique identifier — no predictive value.

---

## 📌 Step 4: Feature Encoding

**File:** `train.py` (lines 27-37)

### 4.1 Binary Columns (0/1 Mapping)
```python
main_mapping = {'No': 0, 'Yes': 1, 'Female': 0, 'Male': 1}
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    data[col] = data[col].map(main_mapping)
```

**Why:** Direct mapping is simpler than LabelEncoder for 2-value columns.

### 4.2 Multi-Category Columns (One-Hot Encoding)
```python
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
data = pd.get_dummies(data, columns=multi_cols, drop_first=True)
```

**Why:** One-hot encoding treats all categories equally. LabelEncoder would assign 0,1,2... implying order.

**Why drop_first=True:** Avoids multicollinearity (dummy variable trap).

**Alternative:** Could use `pd.get_dummies` without `drop_first` and rely on regularization.

---

## 📌 Step 5: Train/Test Split

**File:** `train.py` (lines 39-44)

```python
X = data.drop(columns=['Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**What:** 80% training, 20% testing. `random_state=42` for reproducibility.

**Alternative:** Could use `StratifiedShuffleSplit` to maintain churn ratio across splits.

---

## 📌 Step 6: Feature Scaling

**File:** `train.py` (lines 46-51)

```python
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
```

**What:** StandardScaler makes features have mean=0, std=1. Critical for Logistic Regression (gradient descent) and distance-based models.

**Important:** `fit_transform` on training → learns mean & std. `transform` on test → applies same scaling. This prevents data leakage.

**Alternatives:** MinMaxScaler, RobustScaler, Normalizer.

---

## 📌 Step 7: Model Training

**File:** `train.py` (lines 53-67)

### 7.1 Logistic Regression
```python
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
```

**How it works:** `p = 1 / (1 + e^-(b0 + b1*x1 + ...))`. If p > 0.5 → churn=1, else churn=0.

**Pros:** Simple, interpretable, gives probabilities, fast.
**Cons:** Assumes linear decision boundary.

### 7.2 Random Forest
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

**How it works:** Builds 100 decision trees on random data subsets, averages their predictions.

**Pros:** Handles non-linearity, feature importance available.
**Cons:** Slower, less interpretable, prone to overfitting.

---

## 📌 Step 8: Model Comparison

**File:** `train.py` (lines 77-91)

### Results

| Metric | Logistic Regression | Random Forest |
|--------|:------------------:|:-------------:|
| Accuracy | 78.68% | 78.68% |
| Precision (churn) | 0.62 | 0.64 |
| **Recall (churn)** | **0.51** | **0.46** |
| F1-Score (churn) | **0.56** | 0.54 |

### Decision: Logistic Regression Wins

**Why not Random Forest?** Same accuracy, but LR has **higher recall** (51% vs 46%). In churn prediction, recall is king — catching a leaver saves more than falsely flagging a loyal customer.

**Why not other models?**
- XGBoost/LightGBM — better but more complex, overkill for 7K rows
- Neural Networks — prone to overfitting on this dataset size
- SVM — doesn't give interpretable probabilities
- KNN — poor with high-dimensional one-hot data

---

## 📌 Step 9: Save Model Artifacts

**File:** `train.py` (lines 93-96)

```python
joblib.dump(log_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
```

**What gets saved:**
- `model.pkl` — Trained Logistic Regression
- `scaler.pkl` — Fitted StandardScaler (for consistent scaling of new input)
- `feature_columns.pkl` — Column names in exact training order

**Why:** Deployment needs these 3 files to preprocess input exactly as training did.

---

## 📌 Step 10: Streamlit Web App

**File:** `app.py` (98 lines)

### How it works:
1. **Load** model.pkl, scaler.pkl, feature_columns.pkl
2. **Sidebar UI** — User fills in customer details (dropdowns, sliders, number inputs)
3. **Build DataFrame** — Convert sidebar values into a single-row DataFrame
4. **One-hot encode** — Same `pd.get_dummies` with same columns
5. **Add missing columns** — If user's input doesn't have all categories, add with value 0
6. **Reorder columns** — Match training order exactly
7. **Scale** — Apply scaler.transform on numeric columns
8. **Predict** — `model.predict_proba()` → get churn probability
9. **Display** — Show ✅ "Will NOT Churn" or ❌ "WILL Churn" with probability

### Key Code Snippet:
```python
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]
input_df[num_cols] = scaler.transform(input_df[num_cols])
prob = model.predict_proba(input_df)[0][1]
```

---

## 📌 Important Concepts Used

| Concept | Where | Why |
|---------|-------|-----|
| **Data Leakage** | Scaling (train vs test) | Test data must never influence training |
| **Multicollinearity** | One-hot encoding | `drop_first=True` prevents correlated dummy columns |
| **Class Imbalance** | Model comparison | ~26.5% churn → recall > accuracy for business value |
| **Overfitting** | RF vs LR | LR is simpler, less prone to overfitting on this data |
| **Serialization** | Model saving | Joblib preserves sklearn objects for deployment |

---

## 📌 Commands to Remember

```bash
# Train model
python train.py

# Run web app
streamlit run app.py

# View dependencies
cat requirements.txt

# Activate venv
.\churn\Scripts\activate
```

---

## 📌 File-by-File Summary

| File | Role | Key Functions/Classes Used |
|------|------|---------------------------|
| `train.py` | Training pipeline | pd.read_csv, StandardScaler, LogisticRegression, RandomForestClassifier, joblib.dump |
| `app.py` | Web deployment | streamlit, joblib.load, pd.get_dummies, model.predict_proba |
| `dataset.csv` | Raw data | 21 columns, 7043 rows → 7032 after cleaning |
| `model.pkl` | Trained model | LogisticRegression object |
| `scaler.pkl` | Scaler | StandardScaler (mean, std for tenure, MonthlyCharges, TotalCharges) |
| `feature_columns.pkl` | Column order | List of 30 column names after encoding |

---

## 📌 If I Were to Do This Again...

1. **Handle imbalance** — Use SMOTE or `class_weight='balanced'` to improve churn recall
2. **Feature selection** — Remove low-importance features to reduce noise
3. **Hyperparameter tuning** — GridSearchCV for optimal C, n_estimators, max_depth
4. **K-Fold CV** — More reliable performance estimate than single split
5. **Threshold tuning** — Find optimal probability cutoff based on cost of FP vs FN
6. **Try XGBoost** — Usually outperforms RF on tabular data
7. **Add more features** — Engineer features like `avg_charge_per_month = TotalCharges / tenure`
