from services.stock_service import get_stock_data
from models.predictor import predict_price
from utils.gemini_client import ask_gemini

def run_prediction_agent(symbol):

    data = get_stock_data(symbol)

    predicted_price = predict_price(data)

    current_price = data["Close"].iloc[-1]

    prompt = f"""
    Current stock price: {current_price}
    Predicted next price: {predicted_price}

    Explain whether the stock looks bullish or bearish.
    """

    explanation = ask_gemini(prompt)

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "analysis": explanation
    }