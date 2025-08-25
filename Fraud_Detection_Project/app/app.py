# ===============================
# Credit Card Fraud Detection App
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# Load models and scaler
# -------------------------------
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/final_fraud_model.pkl")

# -------------------------------
# App Title & Description
# -------------------------------
st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
This app predicts whether a credit card transaction is **Fraudulent** or **Legit**.
- Single transaction prediction
- Bulk CSV prediction
""")

# -------------------------------
# Single Transaction Prediction
# -------------------------------
st.header("🔹 Enter Transaction Details (Single Transaction)")

# Generate 28 V columns + Amount input
transaction_input = []
for i in range(1, 29):
    transaction_input.append(st.number_input(f"V{i}", value=0.0, format="%.4f"))

transaction_input.append(st.number_input("Transaction Amount", value=0.0, format="%.2f"))

if st.button("Predict Transaction"):
    try:
        # Scale amount
        amount_scaled = scaler.transform([[transaction_input[-1]]])[0][0]
        features = transaction_input[:-1] + [amount_scaled]

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        result = "🚨 Fraud" if prediction == 1 else "✅ Legit"
        st.success(f"Prediction: {result}")

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# Bulk CSV Prediction
# -------------------------------
st.header("📁 Upload CSV for Bulk Prediction")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Preprocess CSV
        if 'Amount' in df.columns:
            df['Amount_scaled'] = scaler.transform(df[['Amount']])
            df = df.drop(['Amount', 'Time'], axis=1, errors='ignore')

        predictions = model.predict(df)
        df['Prediction'] = ["🚨 Fraud" if p == 1 else "✅ Legit" for p in predictions]

        # Save predictions
        os.makedirs("predictions_output", exist_ok=True)
        output_path = os.path.join("predictions_output", "predictions_output.csv")
        df.to_csv(output_path, index=False)

        st.success("✅ Predictions Completed!")
        st.download_button("Download Predictions CSV", df.to_csv(index=False).encode('utf-8'), "predictions_output.csv")

        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error processing file: {e}")
