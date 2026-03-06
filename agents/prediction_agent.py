from services.stock_service import get_stock_data
from models.predictor import predict_price
from utils.gemini_client import ask_gemini


def run_prediction_agent(symbol):

    # Tool 1 → Fetch stock data
    data = get_stock_data(symbol)

    if data is None or data.empty:
        return {"error": "Stock data unavailable"}

    # Tool 2 → ML prediction
    prediction = predict_price(data)

    current_price = float(data["Close"].iloc[-1])

    predicted_close = prediction["close"]

    # Tool 3 → AI reasoning
    prompt = f"""
You are a financial AI agent.

Stock symbol: {symbol}

Current price: {current_price}
Predicted close price: {predicted_close}

Explain:
1. Is the stock bullish or bearish?
2. Should the user BUY, HOLD, or SELL?

Answer in 2–3 sentences.
"""

    explanation = ask_gemini(prompt)

    return {
        "symbol": symbol,
        "current_price": current_price,
        "prediction": prediction,
        "analysis": explanation
    }