from sklearn.linear_model import LinearRegression
import numpy as np

def predict_price(data):

    prices = data["Close"].values
    X = np.arange(len(prices)).reshape(-1,1)
    y = prices

    model = LinearRegression()
    model.fit(X,y)

    next_day = np.array([[len(prices)]])
    prediction = model.predict(next_day)

    return float(prediction[0])