# 🔮 AI-Based Customer Churn Prediction

An intelligent Machine Learning system that predicts whether a telecom customer is likely to churn (leave the service). Built with **Logistic Regression** and deployed via **Streamlit** for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41%2B-red)
![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Problem Statement

Telecom companies lose **26.5%** of their customers on average. Every churned customer means lost recurring revenue, and acquiring new customers costs **5–7x more** than retaining existing ones.

The goal: **Predict which customers are about to leave** so retention teams can proactively offer discounts, better plans, or support.

### Why This Matters
| Challenge | Impact |
|-----------|--------|
| Revenue Loss | Each churned customer = $50–100+ monthly lost |
| High Acquisition Cost | Marketing to new customers is expensive |
| Late Action | By the time you know, customer is already gone |
| Data Overload | Manual analysis of thousands of accounts is impossible |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📊 Data Preprocessing** | Auto-cleaning, encoding, and scaling of raw CSV data |
| **🤖 Model Training** | Logistic Regression & Random Forest with comparison |
| **📈 Performance Metrics** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **⚡ Real-Time Prediction** | Streamlit web app for instant churn probability |
| **🛡️ Anti-Data Leakage** | Scaler fitted on train set, only transformed on test set |
| **🔒 Lightweight** | No GPU needed, runs on any laptop |

---

## 🏗️ Project Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│   Dataset.csv   │────▶│  train.py        │────▶│  Model Artifacts     │
│  (7043 rows)    │     │  (Training Pipe) │     │  ┌────────────────┐  │
└─────────────────┘     └──────────────────┘     │  │ model.pkl      │  │
                                                    │  │ scaler.pkl     │  │
                                                    │  │ feature_cols   │  │
                                                    │  │ .pkl          │  │
                                                    │  └────────────────┘  │
                                                    └─────────┬──────────┘
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │  app.py          │
                                                    │  (Streamlit UI)  │
                                                    └──────────────────┘
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │  User Browser    │
                                                    │  127.0.0.1:8501 │
                                                    └──────────────────┘
```

### Data Pipeline Flow
```
CSV → Load → Clean (fix TotalCharges → drop nulls → drop customerID)
    → Encode (binary map + one-hot)
    → Split (80/20) → Scale (StandardScaler)
    → Train (LR + RF) → Compare → Save Best Model
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.10+ | Core programming |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **ML Models** | Scikit-learn | Logistic Regression, Random Forest |
| **Preprocessing** | Scikit-learn | LabelEncoder, StandardScaler, train_test_split |
| **Visualization** | Seaborn, Matplotlib | Confusion Matrix, Charts |
| **Serialization** | Joblib | Model persistence |
| **Frontend** | Streamlit | Web deployment UI |

---

## 📦 Installation

```bash
# 1. Clone & enter project
cd ml1

# 2. Create virtual environment (optional)
python -m venv churn
.\churn\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python train.py

# 5. Run the web app
streamlit run app.py
```

---

## 🚀 Usage

### Web App (Streamlit)
```
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

1. Fill in customer details in the **sidebar** (gender, tenure, charges, contract type, services, etc.)
2. Click **"Predict Churn"**
3. Get instant result: ✅ Will NOT Churn / ❌ WILL Churn (with probability %)

### Training Pipeline (CLI)
```bash
python train.py
```
This runs the full pipeline: loads data, cleans, encodes, scales, trains both models, compares them, and saves the best model.

---

## 📊 Model Performance

### Logistic Regression (Selected Model)

| Metric | Score |
|--------|-------|
| **Accuracy** | **78.68%** |
| Precision (Churn) | 0.62 |
| **Recall (Churn)** | **0.51** ✅ |
| F1-Score (Churn) | 0.56 |

### Random Forest (Comparison)

| Metric | Score |
|--------|-------|
| **Accuracy** | **78.68%** |
| Precision (Churn) | 0.64 |
| **Recall (Churn)** | **0.46** |
| F1-Score (Churn) | 0.54 |

### Why Logistic Regression Won

> Both models have equal accuracy (78.68%), but **Recall matters more** in churn prediction. Catching a leaving customer saves more money than falsely flagging a loyal one. Logistic Regression catches **51%** of actual churners vs Random Forest's **46%** — a meaningful 5% improvement.

---

## 📖 Mark-Based Answer System (for reference)

| Marks | Word Count | Response Type |
|-------|------------|---------------|
| 7 marks | 350-420 words | Detailed essay with examples |
| 4 marks | 200 words | Structured bullet points |
| 3 marks | 150 words | Concise with key points |
| 2 marks | 80-100 words | Short explanation |
| 1 mark | 30 words | One-liner answer |

---

## 📁 Project Structure

```
ml1/
├── app.py                                  # Streamlit web application
├── train.py                                # Full ML training pipeline
├── dataset.csv                             # Telco customer churn dataset
├── model.pkl                               # Trained Logistic Regression model
├── scaler.pkl                              # Fitted StandardScaler
├── feature_columns.pkl                     # Training column names
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
├── PROJECT_FLOW.md                         # Step-by-step flow guide
├── REPORT.md                               # Detailed project report
├── customer-churn-prediction.ipynb         # Jupyter notebook (exploration)
├── generate_flow.py                        # Flow diagram generator
├── .gitignore                              # Git ignore rules
├── LICENSE                                 # MIT License
└── churn/                                  # Virtual environment (ignored)
```

---

## 🔐 Privacy & Security

- **Local Processing**: All training and prediction runs locally on your machine
- **No Data Upload**: Everything stays on your computer
- **Open Source**: Full code visible, no hidden data collection
- **Model Artifacts**: `.pkl` files are portable, no cloud dependency

---

## 🧩 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: dataset.csv` | Ensure you're in the project root directory |
| Streamlit won't start | Run `pip install streamlit` then retry |
| Model loads but wrong prediction | Make sure all 3 `.pkl` files are present |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

---

## 🚀 Future Improvements

| Area | Current | Planned |
|------|---------|---------|
| **Model** | Logistic Regression | XGBoost, LightGBM, Hyperparameter Tuning |
| **Data** | Raw CSV with 7032 rows | SMOTE oversampling, Feature Engineering |
| **Validation** | Single 80/20 split | K-Fold Cross Validation |
| **Deployment** | Local Streamlit | Streamlit Cloud, Docker, AWS |
| **Threshold** | Default 0.5 | Cost-based threshold optimization |
| **Monitoring** | None | Model drift detection, retraining pipeline |

---

## 🙏 Acknowledgments

- **Dataset:** IBM Telco Customer Churn Dataset
- **Libraries:** Scikit-learn, Pandas, Streamlit, Joblib, Seaborn

---

**Made with ❤️ for learning and real-world ML application**
