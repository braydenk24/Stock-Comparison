import numpy as np
import pandas as pd
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

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
# Pulls sotck info from yahoo finance to create charts comparing the two stocks from 2019 to 2024 April 1st 
    df_stock_data1 = yf.download(ticker_symbol1, start="2019-01-01", end="2024-04-01")
    df_stock_data2 = yf.download(ticker_symbol2, start="2019-01-01", end="2024-04-01")

# Creates the stock price charts 
    st.write("### Stock Price Chart")
    st.write(f"**Ticker 1:** {ticker_symbol1}")
    st.write(f"**Ticker 2:** {ticker_symbol2}")
    fig = go.Figure(data=[go.Line(x=df_stock_data1.index, y=df_stock_data1['Close'], name=ticker_symbol1),
                          go.Line(x=df_stock_data2.index, y=df_stock_data2['Close'], name=ticker_symbol2)])
    st.plotly_chart(fig, use_container_width=True)

# Creates the table stock statistics 
    st.write("## Stock Statistics")
    st.write(f"**Ticker 1:** {ticker_symbol1}")
    st.dataframe(df_stock_data1.describe())
    st.write(f"**Ticker 2:** {ticker_symbol2}")
    st.dataframe(df_stock_data2.describe())

# Creates the stock volume comparison chart
    st.write("## Stock Volume")
    fig = go.Figure(data=[go.Line(x=df_stock_data1.index, y=df_stock_data1['Volume'], name=ticker_symbol1),
                          go.Line(x=df_stock_data2.index, y=df_stock_data2['Volume'], name=ticker_symbol2)])
    st.plotly_chart(fig, use_container_width=True)

# Stock Prediction
    forecast_col = 'Close'
    forecast_out = 1
    test_size = 0.2
    X_train1, X_test1, Y_train1, Y_test1, X_lately1 = prepare_data(df_stock_data1, forecast_col, forecast_out, test_size)
    X_train2, X_test2, Y_train2, Y_test2, X_lately2 = prepare_data(df_stock_data2, forecast_col, forecast_out, test_size)

    # Creates two instances of a linear regression model, one for each stock 
    learner1 = LinearRegression()
    learner2 = LinearRegression()

    # Trains the linear regression models using the training data 
    learner1.fit(X_train1, Y_train1)
    learner2.fit(X_train2, Y_train2)

    # Creates the score value 
    score1 = learner1.score(X_test1, Y_test1)
    score2 = learner2.score(X_test2, Y_test2)

    # Uses the trained models to generate predictions 
    forecast1 = learner1.predict(X_lately1)
    forecast2 = learner2.predict(X_lately2)

    st.write("## Stock Prediction")
    st.write(f"**Ticker 1:** {ticker_symbol1} - Test Score: {score1}, Forecast: {forecast1}")
    st.write(f"**Ticker 2:** {ticker_symbol2} - Test Score: {score2}, Forecast: {forecast2}")


except Exception as e:
    st.write(f"Error fetching data for {ticker_symbol1}: {e}")