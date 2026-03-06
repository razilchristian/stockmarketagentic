from services.stock_service import get_stock_data
from models.predictor import predict_price
from utils.gemini_client import ask_gemini


def run_prediction_agent(symbol):

    # Step 1 — Tool: Fetch market data
    data = get_stock_data(symbol)

    if data is None or data.empty:
        return {"error": "Stock data unavailable"}

    current_price = float(data["Close"].iloc[-1])

    # Step 2 — Tool: ML prediction
    predicted_price = predict_price(data)

    # Step 3 — Agent reasoning
    prompt = f"""
You are an autonomous financial AI agent.

Your goal is to analyze stock movement and give a trading recommendation.

Stock Symbol: {symbol}

Current Price: {current_price}
Predicted Next Price: {predicted_price}

Tasks:
1. Determine if the stock trend is Bullish, Bearish, or Neutral.
2. Provide a short explanation (2–3 sentences).
3. Give a recommendation: Buy / Hold / Sell.

Respond in JSON format:

{{
 "trend": "",
 "recommendation": "",
 "analysis": ""
}}
"""

    explanation = ask_gemini(prompt)

    return {
        "symbol": symbol,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "agent_analysis": explanation
    }