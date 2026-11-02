# Stock Price Prediction Using Machine Learning

## Project Overview
This project is a Stock Price Prediction application developed using Python and Machine Learning. It allows users to explore historical stock prices, view stock information, and forecast future stock prices using an AutoReg model.

## Features
- User login and registration
- Selection of multiple Indian and international companies
- Historical stock price visualization
- Future stock price prediction
- Interactive charts using Plotly
- Stock information dashboard
- Historical data fetched using Yahoo Finance

## Technologies Used
- Python
- Streamlit
- Pandas
- Yahoo Finance (yfinance)
- Plotly
- Statsmodels
- AutoReg Time Series Model

## Project Structure
```text
Stock_Price_Prediction_ML/
│
├── four.py
├── auth.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project folder:

```bash
cd Stock_Price_Prediction_ML
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run four.py
```

## Model Used
AutoReg (Autoregressive) is used to forecast future stock prices based on historical closing prices.

## Disclaimer
The predicted stock prices are estimates based on historical data. They are not guaranteed future prices or financial advice. This project is developed for educational purposes.

## Author
Hemavathi Sundaresan
