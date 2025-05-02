import numpy as np
import pandas as pd
import requests
import json
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objects as go

# Polygon API Key (replace with your key)
API_KEY = "eCrMkhNAOEkkMDgsbbFpUa3tWQXCGfpQ"

# Function to fetch stock data from Polygon.io
def fetch_polygon_data(ticker, start, end):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "results" in data:
        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    else:
        st.write(f"Error fetching data for {ticker}: {data}")
        return pd.DataFrame()
# Data Preparation function
def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out)
    X = np.array(df[[forecast_col]])
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    label.dropna(inplace=True)
    y = np.array(label)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response

# Read the first CSV file
df_companies = pd.read_csv('sp500_companies.csv')

# Read the second CSV file
df_stocks = pd.read_csv('sp500_stocks.csv')

# Combines the two DataFrames
df = pd.concat([df_companies, df_stocks])

# Cleans the data
df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop_duplicates()
df_cleaned['Industry'] = df_cleaned['Industry'].str.replace(r'\W+','',regex=True)
df_cleaned = df_cleaned.drop_duplicates()
columns_to_delete = ['Open', 'Volume']
df_cleaned = df_cleaned.drop(columns=columns_to_delete)

# Ticker symbol input from user
ticker_symbol1 = st.text_input("Enter the first ticker symbol: ")
ticker_symbol2 = st.text_input("Enter the second ticker symbol: ")

# Select the rows from the data frame where the 'Symbol' column matches the input ticker symbol
selected_rows1 = df_companies[df_companies['Symbol'] == ticker_symbol1]
selected_row1 = df_stocks[df_stocks['Symbol'] == ticker_symbol1]

selected_rows2 = df_companies[df_companies['Symbol'] == ticker_symbol2]
selected_row2 = df_stocks[df_stocks['Symbol'] == ticker_symbol2]

# Display the selected rows based on the user input
st.write(selected_rows1)
st.write(selected_row1)

st.write(selected_rows2)
st.write(selected_row2)

try:
    # Fetch stock data
    df_stock_data1 = fetch_polygon_data(ticker_symbol1, "2019-01-01", "2024-04-01")
    df_stock_data2 = fetch_polygon_data(ticker_symbol2, "2019-01-01", "2024-04-01")

    # Create stock price chart
    st.write("### Stock Price Chart")
    fig = go.Figure(data=[go.Line(x=df_stock_data1.index, y=df_stock_data1['c'], name=ticker_symbol1),
                          go.Line(x=df_stock_data2.index, y=df_stock_data2['c'], name=ticker_symbol2)])
    st.plotly_chart(fig, use_container_width=True)

    # Create stock statistics table
    st.write("## Stock Statistics")
    st.dataframe(df_stock_data1.describe())
    st.dataframe(df_stock_data2.describe())

    # Stock Prediction
    forecast_col = 'c'  # Closing price
    forecast_out = 1
    test_size = 0.2
    def prepare_data(df, forecast_col, forecast_out, test_size):
        label = df[forecast_col].shift(-forecast_out)
        X = np.array(df[[forecast_col]])
        X = preprocessing.scale(X)
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
        label.dropna(inplace=True)
        y = np.array(label)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        return [X_train, X_test, Y_train, Y_test, X_lately]

    X_train1, X_test1, Y_train1, Y_test1, X_lately1 = prepare_data(df_stock_data1, forecast_col, forecast_out, test_size)
    X_train2, X_test2, Y_train2, Y_test2, X_lately2 = prepare_data(df_stock_data2, forecast_col, forecast_out, test_size)

    learner1 = LinearRegression()
    learner2 = LinearRegression()

    learner1.fit(X_train1, Y_train1)
    learner2.fit(X_train2, Y_train2)

    score1 = learner1.score(X_test1, Y_test1)
    score2 = learner2.score(X_test2, Y_test2)

    forecast1 = learner1.predict(X_lately1)
    forecast2 = learner2.predict(X_lately2)

    st.write("## Stock Prediction")
    st.write(f"**Ticker 1:** {ticker_symbol1} - Test Score: {score1}, Forecast: {forecast1}")
    st.write(f"**Ticker 2:** {ticker_symbol2} - Test Score: {score2}, Forecast: {forecast2}")

except Exception as e:
    st.write(f"Error fetching data: {e}")
