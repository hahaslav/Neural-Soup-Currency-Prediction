from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch


def prepare_data(df, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values.reshape(-1, 1))
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def predict_rate(model, data_df, scaler, window_size=60, n_days=10):
    """
    Predict next n_days using the trained LSTM model.

    Args:
        model: trained LSTM model
        data_df: pandas DataFrame with "Close" column
        scaler: fitted MinMaxScaler
        window_size: number of past timesteps used for input
        n_days: number of future days to predict

    Returns:
        predictions: list of predicted prices (in original scale)
    """
    model.eval()

    last_window = data_df["Close"].values[-window_size:]
    last_window_scaled = scaler.transform(last_window.reshape(-1, 1)).flatten()

    predictions_scaled = []

    for _ in range(n_days):
        input_seq = torch.tensor(last_window_scaled.reshape(1, window_size, 1), dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(input_seq).item()

        predictions_scaled.append(pred_scaled)

        last_window_scaled = np.append(last_window_scaled[1:], pred_scaled)

    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    return predictions
