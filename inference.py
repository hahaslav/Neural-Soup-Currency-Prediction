import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from forex_prediction.core import predict_rate
from utils.plotting import plot_future_rate
from constants import CHECKPOINT_PATH
from model import RatePredictor
from torch.serialization import safe_globals


model = RatePredictor()
scaler = MinMaxScaler()

with safe_globals([MinMaxScaler]):
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

model.load_state_dict(checkpoint['model_state_dict'])
scaler = checkpoint['scaler']

stock_data = pd.read_csv("data/yf_usd_uah_stock_data.csv", skiprows=[1])


n_future_days = 15
preds = predict_rate(model, stock_data, scaler, window_size=60, n_days=n_future_days)

plot_future_rate(stock_data, preds, n_future_days)
