# Stock Price Trend Prediction with LSTM  

## 📌 Introduction  
This project focuses on predicting future stock prices using **Long Short-Term Memory (LSTM)** neural networks. Stock price prediction is a classic time-series problem, and LSTM is well-suited because of its ability to capture sequential dependencies and long-term patterns in financial data.  

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
- **Streamlit** – Dashboard deployment  

---

## 🔄 Steps Involved in Building the Project  
1. **Fetch Data** – Stock price data collected using the Yahoo Finance API.  
2. **Preprocess Data** – Normalization and preparation of sequences for LSTM.  
3. **Build LSTM Model** – Constructed using Keras Sequential API.  
4. **Train & Validate Model** – Evaluated predictions against actual stock prices.  
5. **Plot Predictions vs Actual** – Visualized model accuracy with comparison graphs.  
6. **Technical Indicators** – Implemented Moving Averages (MA20, MA50) and RSI indicator.  
7. **Deploy Dashboard** – Interactive dashboard created using Streamlit for visualization.  

---

## 📊 Results & Insights  
- The **LSTM model** successfully captured overall stock price trends.  
- **Predictions** closely followed actual price movements, though with some lag.  
- **Moving Averages** helped identify support/resistance zones.  
- **RSI indicator** highlighted potential overbought/oversold market conditions.  

---

## 🚀 Future Enhancements  
- Integrate more advanced models such as **GRU, Transformer-based models**.  
- Add **sentiment analysis** using news headlines or social media data.  
- Incorporate additional financial indicators for improved accuracy.  
- Deploy the model on **cloud platforms** (AWS/GCP/Azure) for real-time predictions.  

---

## 📂 Project Structure  
StockPrice_LSTM/
│── StockPrice_LSTM.ipynb # Jupyter Notebook with full implementation
│── lstm_model.h5 # Trained LSTM model weights
│── Report.pdf # 1–2 page project report
│── README.md # Project documentation
│
└── DemoScreenshots/ # Folder containing plots & indicator visuals
├── predictions.png
├── moving_average.png
└── rsi.png


---

## 📂 Deliverables  
- `StockPrice_LSTM.ipynb` → Jupyter Notebook with full code  
- `lstm_model.h5` → Trained model weights  
- `Report.pdf` → 1–2 page project report (Introduction, Abstract, Tools, Steps, Conclusion)  
- `README.md` → Project documentation (overview, workflow, results, future enhancements)  
- `/DemoScreenshots/` → Folder containing visual outputs (plots & indicators)  

---

## 👩‍💻 Author  
**Syeda Masuma Fatima**  
📧 Email: masumafatima03@gmail.com  