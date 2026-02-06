import streamlit as st
import pandas as pd
import pickle

# =============================
# Configuration
# =============================
MODEL_PATH = "models/random_forest_baseline.pkl"
ENCODER_PATH = "models/encoders.pkl"
THRESHOLD = 0.4

# =============================
# Load model & encoders (cached)
# =============================
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_artifacts()

# =============================
# UI Header
# =============================
st.title("📊 Customer Churn Prediction")
st.markdown(
    "Predict whether a customer is likely to **churn** based on their profile, services, and billing details."
)

st.divider()

# =============================
# Customer Profile
# =============================
st.subheader("👤 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        [ "Male", "Female"]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:
    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    step=1
)

st.divider()

# =============================
# Services Information
# =============================
st.subheader("📡 Services Information")

col3, col4 = st.columns(2)

with col3:
    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col4:
    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

st.divider()

# =============================
# Billing Information
# =============================
st.subheader("💳 Billing Information")

Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0
)

st.divider()

# =============================
# Predict Button
# =============================
if st.button("🚀 Predict Churn"):

    required_fields = [
        gender, Partner, Dependents, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies, Contract,
        PaperlessBilling, PaymentMethod, SeniorCitizen
    ]

    if any("-- Select" in str(val) for val in required_fields):
        st.warning("⚠️ Please select all required fields before prediction.")
        st.stop()

    # Prepare input
    customer_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }

    input_df = pd.DataFrame([customer_data])

    # Encode categorical features
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # Predict
    churn_prob = model.predict_proba(input_df)[0][1]
    prediction = int(churn_prob >= THRESHOLD)

    # =============================
    # Output
    # =============================
    st.subheader("📈 Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{churn_prob:.2%}"
    )

    if prediction == 1:
        st.error("🚨 High Risk: Customer is likely to **CHURN**")
    else:
        st.success("✅ Low Risk: Customer is likely to **STAY**")
