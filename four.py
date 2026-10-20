import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
from auth import login_page, register_page, logout

# Handle Authentication
query_params = st.experimental_get_query_params()
current_page = query_params.get("page", ["login"])[0]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if not st.session_state.logged_in:
    if current_page == "register":
        register_page()
    else:
        login_page()
    st.stop()

# Logout button
st.sidebar.button("Logout", on_click=logout)

# Set background image function
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background for the main page
set_background("https://img.freepik.com/free-photo/calculator-money-notepad-orange-background-copy-space_169016-24503.jpg")

# Define 20 companies
stock_list = {
    "Bajaj Electricals Ltd.": "BAJAJELEC.BO", "Reliance Industries": "RELIANCE.NS",
    "Tata Motors": "TATAMOTORS.NS", "HDFC Bank": "HDFCBANK.NS", "Infosys": "INFY.NS",
    "TCS": "TCS.NS", "Wipro": "WIPRO.NS", "Adani Enterprises": "ADANIENT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "Asian Paints": "ASIANPAINT.NS",
    "Apple Inc.": "AAPL", "Microsoft": "MSFT", "Tesla": "TSLA",
    "Amazon": "AMZN", "Google (Alphabet)": "GOOGL",
    "Meta (Facebook)": "META", "Netflix": "NFLX", "Nvidia": "NVDA",
    "JP Morgan": "JPM", "Coca-Cola": "KO"
}

# Function to fetch stock data
def fetch_stock_data(ticker, period="6mo", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df.reset_index() if not df.empty else None

# Function to fetch stock info
def fetch_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Current Price": info.get("currentPrice", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "PE Ratio": info.get("trailingPE", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A"),
        "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52-Week Low": info.get("fiftyTwoWeekLow", "N/A")
    }

# Function to plot historical stock data
def plot_historical_data(df, stock_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name="Stock Price", line=dict(color='blue', width=2)))
    fig.update_layout(title=f"Historical Stock Prices for {stock_name}",
                      xaxis_title="Date",
                      yaxis_title="Closing Price",
                      template="plotly_dark")
    st.plotly_chart(fig)

# Function to train AutoReg model and predict future prices
def train_and_predict(df, forecast_days):
    df = df[['Date', 'Close']].dropna()
    df.set_index('Date', inplace=True)
    
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    model = AutoReg(train, lags=5).fit()
    
    # Predict test set
    preds = model.predict(start=len(train), end=len(df)-1, dynamic=False)
    
    # Forecast future stock prices
    future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='B')[1:]  # Business days only
    future_preds = model.predict(start=len(df), end=len(df) + forecast_days - 1, dynamic=False)
    
    return train, test, preds, future_dates, future_preds

# Function to plot predictions
def plot_stock_prediction(train, test, preds, future_dates, future_preds, stock_name):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name="Train Data", line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name="Test Data", line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=preds, mode='lines', name="Model Prediction", line=dict(color='green', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines+markers', name=f"Next {len(future_dates)} Days Forecast", line=dict(color='orange', width=3, dash='dot')))

    fig.update_layout(title=f"Stock Price Forecast for {stock_name} ({len(future_dates)} Days)",
                      xaxis_title="Date",
                      yaxis_title="Stock Price",
                      template="plotly_dark",
                      legend=dict(x=0, y=1))

    st.plotly_chart(fig)

# Streamlit App
def main():
    st.sidebar.header("Navigation")
    menu = st.sidebar.radio("Choose an option", ["📈 Stock Prediction", "ℹ️ Stock Info"])

    st.sidebar.header("User Input Features")
    stock_name = st.sidebar.selectbox("Choose a stock", list(stock_list.keys()))
    stock_ticker = stock_list[stock_name]
    period = st.sidebar.selectbox("Select historical data period", ["1mo", "6mo", "1y"])
    forecast_period = st.sidebar.selectbox("Select forecast period", ["1 day", "5 days", "1 month", "6 months"])

    # Convert forecast period to numeric days
    forecast_days = {"1 day": 1, "5 days": 5, "1 month": 30, "6 months": 180}[forecast_period]
    
    st.title("📊 Stock Analysis Dashboard")

    if menu == "📈 Stock Prediction":
        df = fetch_stock_data(stock_ticker, period, "1d")
        if df is not None:
            plot_historical_data(df, stock_name)
            train, test, preds, future_dates, future_preds = train_and_predict(df, forecast_days)
            plot_stock_prediction(train, test, preds, future_dates, future_preds, stock_name)
        else:
            st.error("No data available for the selected stock.")

    elif menu == "ℹ️ Stock Info":
        st.subheader(f"🔍 Stock Information for {stock_name}")
        stock_info = fetch_stock_info(stock_ticker)
        st.write("### 📊 Stock Details")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Current Price", stock_info["Current Price"])
        col2.metric("🏦 Market Cap", stock_info["Market Cap"])
        col3.metric("📈 PE Ratio", stock_info["PE Ratio"])
        
        col1.metric("📉 52-Week Low", stock_info["52-Week Low"])
        col2.metric("📈 52-Week High", stock_info["52-Week High"])
        col3.metric("💵 Dividend Yield", stock_info["Dividend Yield"])

if __name__ == "__main__":
    main()
