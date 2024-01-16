import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Download data
start = '2010-01-01'
end = '2024-01-01'
st.title('Stock Price Movement Prediction')
user_input = st.text_input('Enter Stock Ticker: ', 'AAPL')
df = yf.download(user_input, start=start, end=end)

st.subheader('Popular Stoks: GOOG, META, AMZN, NFLX, TSLA, SNAP, SBIN.NS, ICICIBANK.NS')

# Describing Data
st.subheader('Data from 2010 - 2024')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# Visualizations
st.subheader('Closing Price vs Time Chart with 100MA')
m100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(m100)
plt.plot(df.Close)
st.pyplot(fig)

# Visualizations
st.subheader('Closing Price vs Time Chart with 200MA')
m100 = df.Close.rolling(100).mean()
m200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(m100, 'r', label='100MA')
plt.plot(m200, 'g', label='200MA')
plt.plot(df.Close, 'b', label='Closing Price')
plt.legend()
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load model
model = load_model('keras_model.h5')

# Training Data
past_100_days = data_training.tail(100)

# Testing Data
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Predictions VS Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'r', label='Original Price')
plt.plot(y_predicted, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Thankyou so much for Watching...')
st.subheader('Created By : Harish Prasad Semwal...')
