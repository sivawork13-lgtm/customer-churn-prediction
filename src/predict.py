import pandas as pd
import pickle
import numpy as np

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/random_forest_baseline.pkl"
ENCODER_PATH = "models/encoders.pkl"

# -----------------------------
# Load trained model
# -----------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Load encoders
# -----------------------------
with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

# -----------------------------
# New unseen customer data
# -----------------------------
customer_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 45,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 42.3,
    "TotalCharges": 1840.75
}

# Convert to DataFrame
input_df = pd.DataFrame([customer_data])

# -----------------------------
# Encode categorical columns
# -----------------------------
for col, encoder in encoders.items():
    input_df[col] = encoder.transform(input_df[col])

# -----------------------------
# Make prediction
# -----------------------------
prob_churn = model.predict_proba(input_df)[0][1]  # probability of churn
prediction = int(prob_churn >= 0.4)  # use your tuned threshold

# -----------------------------
# Output result
# -----------------------------
print("Churn probability:", round(prob_churn, 3))

if prediction == 1:
    print("Prediction: CHURN")
else:
    print("Prediction: NOT CHURN")
