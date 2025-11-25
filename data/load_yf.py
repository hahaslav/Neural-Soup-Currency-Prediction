import numpy as np
from curl_cffi import requests
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pandas as pd


session = requests.Session(impersonate="chrome")
session.verify = False

ticker = 'EURUAH=X'
stock_data = yf.download(
                        tickers=ticker,
                        interval="1d",
                        start="2020-11-25",
                        end="2025-11-25",
                        progress=False,
                        session=session,
                    )


stock_data.to_csv("yf_eur_uah_stock_data.csv", index=True, encoding="utf-8")


def prepare_data(df, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i:0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler
