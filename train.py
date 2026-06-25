import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
import joblib

#load CSV File 
data = pd.read_csv("dataset.csv") 

#NOTE:Fix hidden text trap in TotalCharges
# Why? New customers have blank spaces (" ") instead of numbers, making the whole column an 'object' type.
# pd.to_numeric forces spaces to become NaN so we can handle them, avoiding a StandardScaler crash later!
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
# errors='coerce' turns any non-numeric data (like blank spaces or text) into NaN instead of crashing the code.

data.dropna(inplace=True)


data.drop(columns=['customerID'], inplace=True)


main_mapping = {'No': 0, 'Yes': 1, 'Female': 0, 'Male': 1}
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

for col in binary_cols:
    data[col] = data[col].map(main_mapping)

# pd.get_dummies splits multi-option columns into separate 0 and 1 columns.
# Why? It treats all categories equally, so the model doesn't think one category is "greater than" another.
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
data = pd.get_dummies(data, columns=multi_cols, drop_first=True)

3# 'Churn' column is dropped from X and moved to y. 
# All other cleaned, mapped, scaled, and dummy columns remain in X to train the model.
X = data.drop(columns=['Churn'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## StandardScaler brings columns like tenure and charges to the same scale (Mean=0, SD=1).
# We fit_transform on X_train to learn patterns, but ONLY transform on X_test to prevent Data Leakage.
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# LOGISTIC REGRESSION 
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("RESULT OF LOGISTIC REGRESSION")
print(f"Accuracy Score: {accuracy_score(y_test,y_pred_log):.4f}")
print("\nClassification Report:\n" , classification_report(y_test,y_pred_log))

# RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf=rf_model.predict(X_test)
print("RESULT OF RANDOMFOREST")
print(f"Accuracy Score: {accuracy_score (y_test,y_pred_rf):.4f}")
print("nClassification Report:\n" , classification_report(y_test,y_pred_rf))

# CONFUSION MATRIX
cm = confusion_matrix(y_test ,y_pred_rf)
sns.heatmap(cm, annot = True, fmt = "d" ,cmap = "Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# FINAL MODEL COMPARISON & SELECTION CRITERIA (Why Logistic Regression Wins)
# 
# 1. ACCURACY: Both models give the exact same overall accuracy of 78.68%.
#    This means both predict roughly 79 out of 100 customers correctly.
# 
# 2. THE CHURN CHALLENGE (Class 1): Our main goal is to catch customers who will LEAVE.
#    - Logistic Regression Recall = 51% (Catches 51% of actual churners)
#    - Random Forest Recall = 46% (Catches only 46% of actual churners)
# 
# 3. BUSINESS IMPACT: In Telecom Churn, missing a leaving customer costs more money
#    than accidentally giving a discount to a loyal customer. 
#    Therefore, we need HIGHER RECALL to catch as many churners as possible.
# 
# FINAL VERDICT: We choose Logistic Regression because its higher Recall (51% vs 46%)
# and higher F1-Score (0.56 vs 0.54) make it better at solving the actual business problem.

#  SAVE LOGISTIC REGRESSION MODEL
joblib.dump(log_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\nSaved: model.pkl, scaler.pkl, feature_columns.pkl")
