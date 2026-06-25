
# STEP 2: STREAMLIT APP (app.py)
# Run with:  streamlit run app.py


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD SAVED FILES 
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# PAGE CONFIG 
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title(" Customer Churn Prediction")
st.markdown("Enter customer details below to predict if they will churn.")

# SIDEBAR: CUSTOMER INFO 
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
total = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

# ---------- BUILD INPUT DATAFRAME ----------
input_data = {
    'gender': 0 if gender == "Female" else 1,
    'SeniorCitizen': senior,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service == "Yes" else 0,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': 1 if paperless == "Yes" else 0,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly,
    'TotalCharges': total
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# One-Hot Encode (same way as training)
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
input_df = pd.get_dummies(input_df, columns=multi_cols, drop_first=True)

# Add any missing columns that the model was trained on
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[feature_columns]

# Scale numeric columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[num_cols] = scaler.transform(input_df[num_cols])

#PREDICT 
st.divider()
if st.button(" Predict Churn", type="primary", use_container_width=True):
    prob = model.predict_proba(input_df)[0][1]  # Probability of churn (Yes)
    pred = model.predict(input_df)[0]

    st.subheader("Result")
    if pred == 1:
        st.error(f"❌ Customer WILL Churn (Probability: {prob:.1%})")
    else:
        st.success(f"✅ Customer will NOT Churn (Probability: {prob:.1%})")
