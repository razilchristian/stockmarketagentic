from sklearn.linear_model import LinearRegression
import numpy as np


def predict_price(data):

    prices = data["Close"].values

    if len(prices) < 5:
        return float(prices[-1])

    # Time index
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices

    model = LinearRegression()
    model.fit(X, y)

    # Predict next day
    next_day = np.array([[len(prices)]])
    predicted_close = model.predict(next_day)[0]

    # Estimate volatility
    volatility = np.std(prices[-10:]) if len(prices) >= 10 else np.std(prices)

    predicted_high = predicted_close + volatility
    predicted_low = predicted_close - volatility
    predicted_open = prices[-1]

    return {
        "open": float(predicted_open),
        "high": float(predicted_high),
        "low": float(predicted_low),
        "close": float(predicted_close)
    }