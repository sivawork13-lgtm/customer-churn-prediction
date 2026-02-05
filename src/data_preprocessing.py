
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_DATA_PATH = "dataset/cleaned_telco_churn.csv"
ENCODER_PATH = "models/encoders.pkl"

# create models folder if not exists
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
print("Initial shape:", df.shape)

# -----------------------------
# Drop unnecessary column
# -----------------------------
df.drop(columns=["customerID"], inplace=True)

# -----------------------------
# Handle TotalCharges issue
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0.0, inplace=True)

# -----------------------------
# Encode target variable
# -----------------------------
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print("\nClass distribution:")
print(df["Churn"].value_counts())

# -----------------------------
# Encode categorical features
# -----------------------------
categorical_columns = df.select_dtypes(include="object").columns

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# Save encoders
# -----------------------------
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(encoders, f)

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv(CLEAN_DATA_PATH, index=False)

print("\nPreprocessing completed successfully.")
print(f"Cleaned data saved to: {CLEAN_DATA_PATH}")
print(f"Encoders saved to: {ENCODER_PATH}")
df.head(20).to_csv("dataset/preview.csv", index=False)
