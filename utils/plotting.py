import matplotlib.pyplot as plt


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.show()


def plot_future_rate(stock_data, preds, n_future_days):
    plt.figure(figsize=(10, 5))

    actual_x = stock_data.index[-100:]
    actual_y = stock_data["Close"].values[-100:]

    plt.plot(actual_x, actual_y, label="Actual", linestyle='-', color='blue', marker='o', markersize=2)

    last_index = actual_x[-1]
    pred_x = range(last_index, last_index + n_future_days + 1)
    pred_y = [actual_y[-1]] + list(preds)

    plt.plot(pred_x, pred_y, label="Predicted", marker='o', linestyle='-', color='red', markersize=2)

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Forex Rate Prediction")
    plt.legend()
    plt.show()