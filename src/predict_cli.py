import pandas as pd
import pickle

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/random_forest_baseline.pkl"
ENCODER_PATH = "models/encoders.pkl"
THRESHOLD = 0.4   # tuned threshold

# -----------------------------
# Load model
# -----------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Load encoders
# -----------------------------
with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

# -----------------------------
# Take input from user (CLI)
# -----------------------------
print("\n--- Enter Customer Details ---\n")

customer_data = {
    "gender": input("Gender (Male/Female): "),
    "SeniorCitizen": int(input("Senior Citizen (0 or 1): ")),
    "Partner": input("Partner (Yes/No): "),
    "Dependents": input("Dependents (Yes/No): "),
    "tenure": int(input("Tenure (months): ")),
    "PhoneService": input("Phone Service (Yes/No): "),
    "MultipleLines": input("Multiple Lines (Yes/No/No phone service): "),
    "InternetService": input("Internet Service (DSL/Fiber optic/No): "),
    "OnlineSecurity": input("Online Security (Yes/No/No internet service): "),
    "OnlineBackup": input("Online Backup (Yes/No/No internet service): "),
    "DeviceProtection": input("Device Protection (Yes/No/No internet service): "),
    "TechSupport": input("Tech Support (Yes/No/No internet service): "),
    "StreamingTV": input("Streaming TV (Yes/No/No internet service): "),
    "StreamingMovies": input("Streaming Movies (Yes/No/No internet service): "),
    "Contract": input("Contract (Month-to-month/One year/Two year): "),
    "PaperlessBilling": input("Paperless Billing (Yes/No): "),
    "PaymentMethod": input(
        "Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): "
    ),
    "MonthlyCharges": float(input("Monthly Charges: ")),
    "TotalCharges": float(input("Total Charges: "))
}

# -----------------------------
# Convert to DataFrame
# -----------------------------
input_df = pd.DataFrame([customer_data])

# -----------------------------
# Encode categorical features
# -----------------------------
for col, encoder in encoders.items():
    input_df[col] = encoder.transform(input_df[col])

# -----------------------------
# Predict churn
# -----------------------------
churn_prob = model.predict_proba(input_df)[0][1]
prediction = int(churn_prob >= THRESHOLD)

# -----------------------------
# Output result
# -----------------------------
print("\n--- Prediction Result ---")
print(f"Churn Probability: {churn_prob:.3f}")

if prediction == 1:
    print("Prediction: CHURN")
else:
    print("Prediction: NOT CHURN")
