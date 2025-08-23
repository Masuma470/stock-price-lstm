import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# ------------------------------
# App Title
# ------------------------------
st.title("📈 Stock Price Trend Prediction with LSTM")
st.markdown("""
This app allows you to:  
- Enter one or more stock tickers (e.g., AAPL, MSFT, TSLA)  
- Select a start and end date for historical data  
- View scaled Close prices and daily % change  
- See LSTM predictions and combined trend plots  
- Optionally save the combined plot for reporting
""")

# ------------------------------
# Pre-filled Test Cases
# ------------------------------
test_cases = {
    "Test 1: AAPL 2 years": {"tickers": "AAPL", "start": "2021-01-01", "end": "2023-01-01", "forecast": 5},
    "Test 2: AAPL, MSFT, TSLA": {"tickers": "AAPL, MSFT, TSLA", "start": "2022-01-01", "end": "2023-01-01", "forecast": 7},
    "Test 3: AAPL + invalid ticker": {"tickers": "AAPL, XYZ123, MSFT", "start": "2021-01-01", "end": "2023-01-01", "forecast": 5},
    "Test 4: Start date after end date": {"tickers": "TSLA", "start": "2023-01-01", "end": "2020-01-01", "forecast": 5},
    "Test 5: Short dataset": {"tickers": "TSLA", "start": "2023-08-01", "end": "2023-08-15", "forecast": 3},
    "Test 6: Very short dataset": {"tickers": "GOOG", "start": "2020-06-01", "end": "2020-06-10", "forecast": 2},
    "Test 7: Empty ticker": {"tickers": "", "start": "2021-01-01", "end": "2023-01-01", "forecast": 5},
    "Test 8: Forecast days = 0": {"tickers": "AAPL", "start": "2021-01-01", "end": "2023-01-01", "forecast": 0}
}

selected_test = st.sidebar.selectbox("Select a Test Case", list(test_cases.keys()))

# Automatically fill inputs based on selection
if selected_test:
    test = test_cases[selected_test]
    tickers_input = test["tickers"]
    start_date = pd.to_datetime(test["start"])
    end_date = pd.to_datetime(test["end"])
    forecast_days = test["forecast"]
    st.sidebar.write(f"Running {selected_test}: {tickers_input}, {start_date.date()} to {end_date.date()}, Forecast Days: {forecast_days}")

# ------------------------------
# Fetch Data & Predict
# ------------------------------
if st.sidebar.button("Fetch Data & Predict"):
    if start_date > end_date:
        st.error("Start Date cannot be after End Date!")
    else:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker.")
        else:
            hist_colors = plt.cm.tab10.colors
            pred_colors = plt.cm.Set1.colors
            invalid_tickers = []
            fig_combined, ax_combined = plt.subplots(figsize=(12,6))
            st.header("📊 Stock Data & Predictions")
            cols = st.columns(len(tickers))

            for idx, ticker in enumerate(tickers):
                df = yf.download(ticker, start=start_date, end=end_date)

                if df.empty:
                    invalid_tickers.append(ticker)
                    continue

                with cols[idx]:
                    st.subheader(ticker)
                    st.dataframe(df.head())

                # Scale Close prices
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_close = scaler.fit_transform(df['Close'].values.reshape(-1,1))
                st.write(f"Scaled Close Prices for {ticker}:")
                st.write(scaled_close[:10])

                # Daily % Change
                df['Daily % Change'] = df['Close'].pct_change() * 100
                st.write(f"Daily % Change for {ticker}:")
                st.write(df['Daily % Change'].head(10))

                # Scaled plot
                fig_scaled, ax_scaled = plt.subplots(figsize=(10,4))
                ax_scaled.plot(scaled_close, color='blue', label=f"{ticker} Scaled Close")
                ax_scaled.set_title(f"{ticker} Scaled Close Prices")
                ax_scaled.set_xlabel("Days")
                ax_scaled.set_ylabel("Scaled Price (0-1)")
                ax_scaled.legend()
                st.pyplot(fig_scaled)

                # -----------------------------
                # LSTM Training
                # -----------------------------
                data_to_use = scaled_close
                sequence_length = min(50, len(data_to_use)-1)

                if sequence_length < 1:
                    st.warning(f"Not enough data for LSTM prediction for {ticker}.")
                    predicted_prices = np.array([])
                else:
                    epochs = 15 if len(data_to_use) > 200 else 3
                    if sequence_length < 50:
                        st.info(f"Note: sequence length reduced to {sequence_length} for {ticker}.")

                    # Prepare sequences
                    X, y = [], []
                    for i in range(sequence_length, len(data_to_use)):
                        X.append(data_to_use[i-sequence_length:i,0])
                        y.append(data_to_use[i,0])
                    X, y = np.array(X), np.array(y)
                    X = X.reshape((X.shape[0], X.shape[1],1))

                    # Build LSTM
                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
                    model.add(LSTM(50))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)

                    # -----------------------------
                    # Forecast next days
                    # -----------------------------
                    last_sequence = data_to_use[-sequence_length:]
                    predicted_list = []
                    current_seq = last_sequence.copy()

                    for i in range(forecast_days):
                        pred = model.predict(current_seq.reshape(1, sequence_length,1), verbose=0)[0,0]
                        predicted_list.append(pred)
                        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

                    # -----------------------------
                    # Handle zero forecast safely & show note
                    # -----------------------------
                    if forecast_days > 0 and len(predicted_list) > 0:
                        predicted_prices = scaler.inverse_transform(np.array(predicted_list).reshape(-1,1))
                        st.write(f"LSTM Predicted Close Prices for next {forecast_days} days for {ticker}:")
                        st.write(predicted_prices.flatten())

                        # Predicted line visibility warning/note
                        pred_range = float(predicted_prices.max() - predicted_prices.min())
                        hist_range = float(df['Close'].max() - df['Close'].min())
                        if hist_range > 0 and pred_range < 0.01 * hist_range:
                            st.info(f"Note: Predicted line for {ticker} is very small compared to historical prices; "
                                    "it may appear almost flat. This is normal for short-term forecasts or low volatility stocks.")
                    else:
                        predicted_prices = np.array([])
                        if forecast_days == 0:
                            st.info(f"No forecast requested for {ticker}.")

                # -----------------------------
                # Combined Plot
                # -----------------------------
                hist_dates = df.index
                ax_combined.plot(hist_dates, df['Close'], label=f"{ticker} Historical", color=hist_colors[idx % len(hist_colors)])

                if predicted_prices.size > 0:
                    forecast_dates = pd.date_range(start=hist_dates[-1]+pd.Timedelta(days=1), periods=forecast_days)
                    ax_combined.plot(forecast_dates, predicted_prices.flatten(), linestyle='--',
                                     color=pred_colors[idx % len(pred_colors)], label=f"{ticker} Predicted")

            if invalid_tickers:
                st.warning(f"No data found for these tickers: {', '.join(invalid_tickers)}")

            ax_combined.set_title(f"Stock Prices from {start_date} to {end_date} with LSTM Predictions")
            ax_combined.set_xlabel("Date")
            ax_combined.set_ylabel("Close Price (USD)")
            ax_combined.legend()
            ax_combined.grid(True)
            fig_combined.autofmt_xdate()
            st.pyplot(fig_combined)

            # ------------------------------
            # Save Combined Plot
            # ------------------------------
            os.makedirs("Dmoscreenshots", exist_ok=True)
            if st.button("Save Combined Plot"):
                filename = f"Dmoscreenshots/Stock_Plot_{start_date}_{end_date}.png"
                fig_combined.savefig(filename)
                st.success(f"Plot saved to {filename}")
