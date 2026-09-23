# 📈 AI-Powered Stock Market Prediction & Analytics System

An end-to-end financial forecasting and market intelligence platform that predicts asset prices using machine learning and deep learning models[cite: 1]. The application ingests market data, calculates key technical indicators, trains time-series forecasting models, and serves interactive predictions through a web dashboard[cite: 1, 6].

---

## 📌 Features

- **Multi-Model Forecasting**: Implements predictive architectures tailored for sequential financial data, including LSTM, GRU, ARIMA, and XGBoost[cite: 1].
- **Technical Indicator Computation**: Generates technical analysis indicators (RSI, Moving Averages, MACD, Bollinger Bands) from price movements[cite: 1, 6].
- **Model Evaluation & Benchmarking**: Evaluates and compares predictions against test datasets using regression metrics (RMSE, MAE, MAPE)[cite: 1, 6].
- **Interactive UI**: Web interface for selecting tickers, adjusting lookback windows, visualizing forecast trajectories, and reviewing analytical insights[cite: 1, 6].

---

## 🛠️ Tech Stack

- **Core**: Python 3.10+
- **Time-Series & ML**: PyTorch, Scikit-Learn, XGBoost[cite: 1]
- **Data & Feature Engineering**: Pandas, NumPy[cite: 6]
- **Web App / Serving**: Streamlit / FastAPI, Requests[cite: 6]
- **Visualization**: Plotly / Matplotlib[cite: 6]

---

## 📂 Project Architecture

```text
AI-Stock-Price-predictor/
├── app.py                      # Main application interface and orchestration logic[cite: 6]
├── requirements.txt            # Python environment dependencies[cite: 6]
├── utils/
│   ├── data_utils.py           # Data fetching, normalization, and sequence generation[cite: 6]
│   ├── indicators.py           # Financial technical indicators calculation[cite: 6]
│   ├── metrics.py              # Performance evaluation routines (RMSE, MAE, etc.)[cite: 6]
│   ├── models.py               # ML/DL model definitions, training routines, and inference[cite: 6]
│   └── styles.py               # Custom UI layouts and design configurations[cite: 6]
└── README.md
