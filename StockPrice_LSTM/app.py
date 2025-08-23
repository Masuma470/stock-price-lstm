# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("📈 Stock Price Trend Prediction with LSTM")

# Sidebar input
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

if st.button("Fetch & Predict"):
    # Fetch data
    df = yf.download(ticker, start=start_date, end=end_date)
    
    st.subheader("Historical Closing Prices")
    st.line_chart(df['Close'])
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_close = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    
    # Prepare X (last 60 days)
    X = []
    y = []
    for i in range(60, len(scaled_close)):
        X.append(scaled_close[i-60:i,0])
        y.append(scaled_close[i,0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Load trained model
    model = load_model("lstm_model.h5")  # Make sure this file exists
    
    # Predict
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    # Plot predictions vs actual
    st.subheader("Predicted vs Actual Closing Prices")
    plt.figure(figsize=(12,6))
    plt.plot(df['Close'].values[60:], color='blue', label='Actual')
    plt.plot(predictions, color='red', label='Predicted')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)
