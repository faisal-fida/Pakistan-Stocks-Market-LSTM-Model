# Project Documentation: Pakistan Stock Market LSTM Model


## Project Overview
The objective of this project is to develop a Long Short-Term Memory (LSTM) model to predict the closing prices of stocks listed on the Pakistan Stock Exchange (PSX). The model utilizes historical stock data to forecast future prices, providing valuable insights for investors and stakeholders.

## Data Collection
The project uses the Yahoo Finance API to download historical stock data. The following stock symbols were considered:
- PPL.KA (Pakistan Petroleum Limited)
- ACPL.KA (Attock Cement Pakistan Limited)
- PSO.KA (Pakistan State Oil)

The data includes the following columns: Date, Open, High, Low, Close, Adj Close, and Volume.

### Data Collection Code
```python
import yfinance as yf

def download_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

start_date = '2022-01-01'
end_date = '2024-03-12'
df = download_stock_data('PPL.KA', start_date, end_date)
```

## Data Preprocessing
The data preprocessing steps include filtering the 'Close' price, scaling the data to a range of 0 to 1 using MinMaxScaler, and creating sequences for model training.

### Preprocessing Code
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data

data = df.filter(['Close']).values
data = data.reshape(-1, 1)
scaled_data = preprocess_data(data)
```

## Model Development
The LSTM model is constructed using the Keras library. The model architecture includes two LSTM layers and two Dense layers.

### Model Code
```python
from keras.models import Sequential
from keras.layers import Dense, LSTM

def create_sequences(data, seq_len):
    x = []
    y = []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    return model
```

## Model Training
The model is compiled using the Adam optimizer and mean squared error loss function. It is trained on the preprocessed data for a specified number of epochs.

### Training Code
```python
def train_model(model, x_train, y_train, batch_size, epochs):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return model
```

## Model Testing
The model's performance is evaluated on the test data. Predictions are inverse transformed to obtain the original scale.

### Testing Code
```python
def test_model(model, x_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions
```

## Results Visualization
The results, including the actual and predicted closing prices, are visualized using Matplotlib.

### Visualization Code
```python
import matplotlib.pyplot as plt

def visualize_results(train_data, valid_data, predictions):
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train_data['Close'])
    plt.plot(valid_data[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
```

## Conclusion
The LSTM model successfully predicts the closing prices of selected stocks on the Pakistan Stock Exchange. The predictions provide a useful tool for making informed investment decisions. Future improvements can include experimenting with different model architectures and incorporating additional features such as trading volume and macroeconomic indicators.
