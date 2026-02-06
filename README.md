# Customer Churn Prediction

This project predicts customer churn using the Telco Customer Churn dataset.

## Problem Statement
Customer churn leads to revenue loss. The objective is to identify customers who are likely to churn so that preventive actions can be taken.

## Dataset
Telco Customer Churn dataset (Kaggle)

## Approach
- Data cleaning and preprocessing
- Label encoding for categorical features
- Handling class imbalance using SMOTE
- Random Forest classifier
- Threshold tuning to improve churn recall

## Results
- Churn recall improved from ~0.59 to ~0.71
- Accuracy ≈ 0.75
- Macro-average recall ≈ 0.74

## Project Structure
Customer_prediction/
├── src/
│ ├── data_preprocessing.py
│ ├── train_baseline.py
│ └── predict.py
├── dataset/
├── models/
└── README.md


## How to Run
1️⃣ Setup Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2️⃣ Data Preprocessing
python src/data_preprocessing.py

3️⃣ Train Model
python src/train_baseline.py

4️⃣ CLI Prediction
python src/predict_cli.py

5️⃣ Web App (Streamlit)
streamlit run src/app.py