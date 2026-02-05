import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import os

# -----------------------------
# Paths
# -----------------------------
CLEAN_DATA_PATH = "dataset/cleaned_telco_churn.csv"
MODEL_PATH = "models/random_forest_baseline.pkl"

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv(CLEAN_DATA_PATH)
print("Loaded data shape:", df.shape)

# -----------------------------
# Handle any remaining NaN values (safety check)
# -----------------------------
print("\nChecking missing values:")
print(df.isnull().sum())

df = df.fillna(0)

# -----------------------------
# Split features and target
# -----------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

# -----------------------------
# Apply SMOTE (TRAIN ONLY)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# -----------------------------
# Train baseline Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_smote, y_train_smote)

# -----------------------------
# Evaluate on test data
# -----------------------------
y_pred = rf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# -----------------------------
# Threshold tuning
# -----------------------------
y_prob = rf.predict_proba(X_test)[:, 1]  # probability of churn (class 1)

THRESHOLD = 0.4   # try 0.4 first
y_pred_custom = (y_prob >= THRESHOLD).astype(int)
print(f"\n=== Custom Threshold ({THRESHOLD}) ===")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))
# -----------------------------
# Save baseline model
# -----------------------------
os.makedirs("models", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(rf, f)

print("\nBaseline Random Forest model saved.")
