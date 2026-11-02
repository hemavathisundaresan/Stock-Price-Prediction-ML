
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from auth import login_page, register_page, logout

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# AUTHENTICATION
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

current_page = st.query_params.get("page", "login")

if not st.session_state.logged_in:
    if current_page == "register":
        register_page()
    else:
        login_page()
    st.stop()

st.sidebar.button("Logout", on_click=logout)

# BACKGROUND
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(
    "https://img.freepik.com/free-photo/calculator-money-notepad-orange-background-copy-space_169016-24503.jpg"
)

# STOCK LIST
stock_list = {
    "Bajaj Electricals Ltd.": "BAJAJELEC.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Motors Passenger Vehicles": "TMPV.NS",
    "Tata Motors Commercial Vehicles": "TMCV.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Apple Inc.": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Google (Alphabet)": "GOOGL",
    "Meta (Facebook)": "META",
    "Netflix": "NFLX",
    "Nvidia": "NVDA",
    "JP Morgan": "JPM",
    "Coca-Cola": "KO"
}

# FETCH HISTORICAL STOCK DATA
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="6mo", interval="1d"):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            data = yf.Ticker(ticker).history(
                period=period,
                interval=interval,
                auto_adjust=True
            )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        date_column = next(
            (col for col in data.columns
             if str(col).lower() in ["date", "datetime"]),
            None
        )

        if date_column is None or "Close" not in data.columns:
            return pd.DataFrame()

        data = data.rename(columns={date_column: "Date"})

        data["Date"] = pd.to_datetime(
            data["Date"], errors="coerce"
        )

        data["Close"] = pd.to_numeric(
            data["Close"], errors="coerce"
        )

        data = data.dropna(subset=["Date", "Close"])
        data = data.sort_values("Date")
        data = data.drop_duplicates(subset=["Date"])

        return data[["Date", "Close"]].reset_index(drop=True)

    except Exception as error:
        print(f"Data fetching error for {ticker}: {error}")
        return pd.DataFrame()

# FETCH STOCK INFORMATION
@st.cache_data(ttl=300)
def fetch_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info

        return {
            "Current Price": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52-Week Low": info.get("fiftyTwoWeekLow", "N/A")
        }

    except Exception as error:
        print(f"Stock info error: {error}")
        return {
            "Current Price": "N/A",
            "Market Cap": "N/A",
            "PE Ratio": "N/A",
            "Dividend Yield": "N/A",
            "52-Week High": "N/A",
            "52-Week Low": "N/A"
        }


# PLOT HISTORICAL STOCK DATA
def plot_historical_data(data, stock_name):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines",
            name="Closing Price",
            line=dict(width=2)
        )
    )

    fig.update_layout(
        title=f"Historical Stock Prices - {stock_name}",
        xaxis_title="Date",
        yaxis_title="Adjusted Closing Price",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


# TRAIN AUTOREG MODEL AND PREDICT FUTURE PRICES
def train_and_predict(data, forecast_days):

    model_data = data[["Date", "Close"]].dropna().copy()

    model_data["Close"] = pd.to_numeric(
        model_data["Close"],
        errors="coerce"
    )

    model_data = model_data.dropna(subset=["Close"])

    if len(model_data) < 15:
        raise ValueError(
            "Not enough historical records. "
            "Please select a longer data period."
        )

    model_data = model_data.set_index("Date")

    train_size = int(len(model_data) * 0.8)

    train = model_data.iloc[:train_size].copy()
    test = model_data.iloc[train_size:].copy()

    if len(train) < 10:
        raise ValueError("Not enough training data.")

    lags = min(5, max(1, len(train) // 5))

    train_values = train["Close"].astype(float).reset_index(drop=True)

    model = AutoReg(
        train_values,
        lags=lags
    ).fit()

    # TEST DATA PREDICTION
    if len(test) > 0:

        predictions = model.predict(
            start=len(train_values),
            end=len(train_values) + len(test) - 1
        )

    else:
        predictions = pd.Series(dtype=float)

    # TRAIN FINAL MODEL USING ALL HISTORICAL DATA
    all_values = model_data["Close"].astype(float).reset_index(drop=True)

    final_model = AutoReg(
        all_values,
        lags=lags
    ).fit()

    # FUTURE PREDICTION
    future_predictions = final_model.predict(
        start=len(all_values),
        end=len(all_values) + forecast_days - 1
    )

    # GENERATE FUTURE BUSINESS DAYS
    last_date = model_data.index[-1]

    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1),
        periods=forecast_days
    )

    return (
        train,
        test,
        predictions,
        future_dates,
        future_predictions
    )


# PLOT STOCK PREDICTIONS
def plot_stock_prediction(
    train,
    test,
    predictions,
    future_dates,
    future_predictions,
    stock_name
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train.index,
            y=train["Close"],
            mode="lines",
            name="Training Data"
        )
    )

    if not test.empty:

        fig.add_trace(
            go.Scatter(
                x=test.index,
                y=test["Close"],
                mode="lines",
                name="Actual Test Data"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=test.index,
                y=list(predictions),
                mode="lines",
                name="Model Predictions",
                line=dict(dash="dash")
            )
        )

    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=list(future_predictions),
            mode="lines+markers",
            name="Future Forecast",
            line=dict(width=3, dash="dot")
        )
    )

    fig.update_layout(
        title=f"Stock Price Forecast - {stock_name}",
        xaxis_title="Date",
        yaxis_title="Adjusted Closing Price",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# MAIN DASHBOARD
def main():

    st.title("📈 Stock Price Prediction Using Machine Learning")

    st.write(
        "Analyze historical stock prices and predict future stock trends."
    )

    # STOCK SELECTION
    selected_stock = st.selectbox(
        "Select a Stock",
        list(stock_list.keys())
    )

    ticker = stock_list[selected_stock]

    # HISTORICAL PERIOD
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }

    selected_period = st.selectbox(
        "Select Historical Data Period",
        list(period_options.keys()),
        index=2
    )

    period = period_options[selected_period]

    # FORECAST PERIOD
    forecast_options = {
        "1 Day": 1,
        "5 Days": 5,
        "1 Month": 21,
        "3 Months": 63,
        "6 Months": 126
    }

    selected_forecast = st.selectbox(
        "Select Prediction Period",
        list(forecast_options.keys())
    )

    forecast_days = forecast_options[selected_forecast]

    # FETCH DATA
    with st.spinner("Fetching stock data..."):
        data = fetch_stock_data(ticker, period)

    if data.empty:
        st.error(
            "Unable to fetch stock data. "
            "Please try another stock or period."
        )
        return

    st.subheader(f"Stock Information - {selected_stock}")

    # DISPLAY STOCK INFORMATION
    stock_info = fetch_stock_info(ticker)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Price",
            stock_info["Current Price"]
        )

    with col2:
        st.metric(
            "52-Week High",
            stock_info["52-Week High"]
        )

    with col3:
        st.metric(
            "52-Week Low",
            stock_info["52-Week Low"]
        )

    st.write("### Historical Stock Price")

    plot_historical_data(data, selected_stock)

    # PREDICTION
    st.write("### Future Stock Price Prediction")

    if st.button("Predict Stock Price"):

        try:
            with st.spinner("Training model and predicting..."):

                (
                    train,
                    test,
                    predictions,
                    future_dates,
                    future_predictions
                ) = train_and_predict(data, forecast_days)

                plot_stock_prediction(
                    train,
                    test,
                    predictions,
                    future_dates,
                    future_predictions,
                    selected_stock
                )

                prediction_table = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Stock Price": list(future_predictions)
                })

                st.write("### Predicted Prices")

                st.dataframe(
                    prediction_table,
                    use_container_width=True
                )

                st.success("Prediction completed successfully!")

        except Exception as error:
            st.error(f"Prediction error: {error}")

    st.warning(
        "Note: Stock price predictions are estimates, "
        "not guaranteed future prices. "
        "This project is for educational purposes only."
    )


# RUN APPLICATION
if __name__ == "__main__":
    main()

