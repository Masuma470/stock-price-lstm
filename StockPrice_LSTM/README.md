# Stock Price Trend Prediction with LSTM & Streamlit Dashboard

## 📌 Introduction  
This project focuses on predicting future stock prices using **Long Short-Term Memory (LSTM)** neural networks. Stock price prediction is a classic time-series problem, and LSTM is well-suited because of its ability to capture sequential dependencies and long-term patterns in financial data.  

This project now includes an **interactive Streamlit app** that allows users to input stock tickers, select date ranges, and view scaled prices, daily % changes, and LSTM-based predictions in real time.

---

## 🎯 Abstract  
The objective of this project is to analyze stock market data, train an LSTM model, and evaluate its performance in predicting stock price trends. Alongside the LSTM model, technical indicators such as **Moving Averages (MA20, MA50)** and the **Relative Strength Index (RSI)** are integrated to provide deeper insights into market momentum and price behavior.  

---

## 🛠️ Tools & Libraries Used  
- **Python** – Programming Language  
- **Pandas & NumPy** – Data manipulation  
- **Matplotlib** – Data visualization  
- **Keras (TensorFlow backend)** – Deep Learning framework  
- **yFinance API** – Stock market data fetching  
- **Streamlit** – Interactive dashboard deployment  

---

## 🔄 Steps Involved in Building the Project  
1. **Fetch Data** – Stock price data collected using the Yahoo Finance API.  
2. **Preprocess Data** – Normalization and preparation of sequences for LSTM.  
3. **Build LSTM Model** – Constructed using Keras Sequential API.  
4. **Train & Validate Model** – Evaluated predictions against actual stock prices.  
5. **Plot Predictions vs Actual** – Visualized model accuracy with comparison graphs.  
6. **Technical Indicators** – Implemented Moving Averages (MA20, MA50) and RSI indicator.  
7. **Deploy Jupyter Notebook** – Analysis and plots generated in a notebook.  
8. **Deploy Interactive Dashboard** – Users can input tickers, date ranges, and see scaled prices, daily % changes, and LSTM predictions in an interactive Streamlit app.  

---

## 📊 Results & Insights  
- The **LSTM model** successfully captured overall stock price trends.  
- **Predictions** closely followed actual price movements, though with some lag.  
- **Moving Averages** helped identify support/resistance zones.  
- **RSI indicator** highlighted potential overbought/oversold market conditions.  
- **Streamlit app** allows users to visualize historical vs predicted trends interactively.

**Note:** For short-term forecasts or low-volatility stocks, the predicted line may appear almost flat compared to historical prices. This is normal.

---

## 🧪 Sample Inputs for Testing Streamlit App
| Test Case | Tickers             | Start Date  | End Date    | Forecast Days |
|-----------|-------------------|------------|------------|---------------|
| 1         | AAPL               | 2021-01-01 | 2023-01-01 | 5             |
| 2         | AAPL, MSFT, TSLA   | 2022-01-01 | 2023-01-01 | 7             |
| 3         | AAPL, XYZ123, MSFT | 2021-01-01 | 2023-01-01 | 5             |
| 4         | TSLA               | 2023-01-01 | 2020-01-01 | 5             |
| 5         | TSLA               | 2023-08-01 | 2023-08-15 | 3             |

---

## 🚀 Future Enhancements  
- Integrate more advanced models such as **GRU or Transformer-based models**.  
- Add **sentiment analysis** using news headlines or social media data.  
- Incorporate additional financial indicators for improved accuracy.  
- Deploy the model on **cloud platforms** (AWS/GCP/Azure) for real-time predictions.  

---

## 📂 Project Structure  
│── app.py # Streamlit app for stock prediction
│── StockPrice_LSTM.ipynb # Jupyter Notebook with full implementation
│── lstm_model.h5 # Trained LSTM model weights
│── StockPrice_LSTM_Report.pdf # 1–2 page project report
│── README.md # Project documentation
└── DemoScreenshots/ # Optional: Folder containing notebook plots & indicator visuals
├── predictions.png
├── moving_average.png
└── rsi.png


---

## 📂 Deliverables  
- `StockPrice_LSTM.ipynb` → Jupyter Notebook with full code  
- `lstm_model.h5` → Trained model weights  
- `StockPrice_LSTM_Report.pdf` → 1–2 page project report (Introduction, Abstract, Tools, Steps, Conclusion)  
- `README.md` → Project documentation (overview, workflow, results, future enhancements)  
- `app.py` → Streamlit app for interactive predictions  
- `/DemoScreenshots/` → Folder containing visual outputs (plots & indicators, optional)  

---

## 👩‍💻 Author  
**Syeda Masuma Fatima**  
📧 Email: masumafatima03@gmail.com  