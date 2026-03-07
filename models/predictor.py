# predictor.py
# Stock price prediction module with Gemini AI and Linear Regression fallback

import os
import json
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import google.genai as genai
from google.genai import types

# Initialize Gemini client (make sure GEMINI_API_KEY is set in environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Try multiple model names in order of preference
PREFERRED_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash", 
    "gemini-pro",
    "gemini-1.0-pro"
]

# Find the first working model
GEMINI_MODEL = None
if client:
    for model_name in PREFERRED_MODELS:
        try:
            # Test the model with a simple query
            test_response = client.models.generate_content(
                model=model_name,
                contents="OK"
            )
            if hasattr(test_response, 'text'):
                GEMINI_MODEL = model_name
                print(f"✓ Using Gemini model: {GEMINI_MODEL}")
                break
        except Exception as e:
            continue
    
    if not GEMINI_MODEL:
        print("⚠ No working Gemini model found - using Linear Regression only")
else:
    print("⚠ Gemini client not initialized - using Linear Regression only")

def predict_price(data, symbol="UNKNOWN", use_gemini=True):
    """
    Predict stock prices using either Gemini AI or Linear Regression fallback
    
    Args:
        data: DataFrame with 'Close' prices and other OHLC data
        symbol: Stock symbol for context
        use_gemini: Whether to try Gemini first (falls back to Linear Regression if fails)
    
    Returns:
        Dictionary with open, high, low, close predictions
    """
    
    # Get the latest price
    latest_price = float(data["Close"].iloc[-1])
    
    # Calculate basic statistics for context
    if len(data) >= 20:
        recent_prices = data["Close"].tail(10).tolist()
        volatility = float(data["Close"].pct_change().std() * np.sqrt(252) * 100)
        ma_20 = float(data["Close"].tail(20).mean())
        ma_50 = float(data["Close"].tail(50).mean()) if len(data) >= 50 else latest_price
        day_high = float(data["High"].tail(1).max()) if 'High' in data.columns else latest_price * 1.02
        day_low = float(data["Low"].tail(1).min()) if 'Low' in data.columns else latest_price * 0.98
    else:
        recent_prices = data["Close"].tolist()
        volatility = 20.0  # Default volatility
        ma_20 = latest_price
        ma_50 = latest_price
        day_high = latest_price * 1.02
        day_low = latest_price * 0.98
    
    # Try Gemini first if requested and available
    if use_gemini and client and GEMINI_MODEL:
        try:
            gemini_prediction = _predict_with_gemini(symbol, latest_price, recent_prices, volatility, ma_20, ma_50, day_high, day_low)
            if gemini_prediction:
                print(f"✓ Using Gemini AI prediction for {symbol}")
                return gemini_prediction
        except Exception as e:
            print(f"Gemini prediction failed, falling back to Linear Regression: {e}")
    
    # Fallback to Linear Regression
    print(f"Using Linear Regression fallback for {symbol}")
    return _predict_with_linear_regression(data)

def _predict_with_gemini(symbol, latest_price, recent_prices, volatility, ma_20, ma_50, day_high, day_low):
    """
    Use Gemini to generate intelligent price predictions
    """
    
    prompt = f"""
    As a senior financial analyst, predict tomorrow's stock prices for {symbol} with confidence intervals.

    CURRENT MARKET DATA:
    - Current Price: ${latest_price:.2f}
    - Today's High: ${day_high:.2f}
    - Today's Low: ${day_low:.2f}
    - Volatility (annual): {volatility:.2f}%
    - 20-day MA: ${ma_20:.2f}
    - 50-day MA: ${ma_50:.2f}

    Recent price history (last 10 days): {[round(p, 2) for p in recent_prices]}

    Based on technical analysis and market patterns, predict tomorrow's:

    1. OPENING PRICE (with 90% confidence interval)
    2. DAY HIGH (with 90% confidence interval)
    3. DAY LOW (with 90% confidence interval)
    4. CLOSING PRICE (with 90% confidence interval)

    IMPORTANT RULES:
    - All predictions MUST be within ±10% of the current price (${latest_price:.2f})
    - OPEN price should be close to today's close
    - HIGH should be >= OPEN and CLOSE
    - LOW should be <= OPEN and CLOSE
    - Be realistic based on the recent price history

    Return ONLY a valid JSON object with this exact structure (no other text):
    {{
        "open": {{"value": float, "lower_bound": float, "upper_bound": float}},
        "high": {{"value": float, "lower_bound": float, "upper_bound": float}},
        "low": {{"value": float, "lower_bound": float, "upper_bound": float}},
        "close": {{"value": float, "lower_bound": float, "upper_bound": float}}
    }}
    """
    
    try:
        # Call Gemini with the selected model
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        if hasattr(response, "text"):
            text = response.text.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                predictions = json.loads(json_match.group())
                
                # Validate the predictions
                for key in ['open', 'high', 'low', 'close']:
                    if key not in predictions:
                        raise ValueError(f"Missing {key} in predictions")
                    
                    # Ensure predictions are within reasonable range (±15% of current price)
                    value = predictions[key]['value']
                    if abs(value - latest_price) > latest_price * 0.15:
                        # Adjust to be within range
                        direction = 1 if value > latest_price else -1
                        predictions[key]['value'] = round(latest_price * (1 + direction * 0.1), 2)
                    
                    # Ensure bounds are present
                    if 'lower_bound' not in predictions[key]:
                        predictions[key]['lower_bound'] = round(predictions[key]['value'] * 0.98, 2)
                    if 'upper_bound' not in predictions[key]:
                        predictions[key]['upper_bound'] = round(predictions[key]['value'] * 1.02, 2)
                
                # Add confidence scores
                for key in predictions:
                    predictions[key]['confidence'] = 85  # Default confidence
                
                return predictions
            else:
                print("No JSON found in Gemini response")
                return None
        else:
            print("Gemini response has no text attribute")
            return None
            
    except Exception as e:
        print(f"Error in Gemini prediction: {e}")
        return None

def _predict_with_linear_regression(data):
    """
    Fallback method using Linear Regression
    """
    prices = data["Close"].values

    if len(prices) < 5:
        close_value = float(prices[-1])
    else:
        # Time index
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices

        model = LinearRegression()
        model.fit(X, y)

        # Predict next day
        next_day = np.array([[len(prices)]])
        close_value = model.predict(next_day)[0]

    # Ensure prediction is within reasonable range of last price
    last_price = prices[-1]
    max_change = last_price * 0.05  # Max 5% change
    close_value = max(last_price - max_change, min(last_price + max_change, close_value))

    # Calculate volatility for bounds
    if len(prices) >= 10:
        volatility = np.std(prices[-10:])
    else:
        volatility = np.std(prices) if len(prices) > 1 else last_price * 0.02

    # Generate predictions
    predicted_close = float(close_value)
    predicted_open = float(prices[-1])
    predicted_high = max(predicted_open, predicted_close) + volatility * 0.5
    predicted_low = min(predicted_open, predicted_close) - volatility * 0.5

    # Calculate confidence based on data quality
    confidence = min(85, int(50 + len(prices) / 2))

    return {
        "open": round(predicted_open, 2),
        "high": round(predicted_high, 2),
        "low": round(predicted_low, 2),
        "close": round(predicted_close, 2)
    }

# Optional: Test function
if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample data
    dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
    sample_data = pd.DataFrame({
        'Close': [100 + i * 0.5 + np.random.randn() * 2 for i in range(30)],
        'High': [102 + i * 0.5 + np.random.randn() * 2 for i in range(30)],
        'Low': [98 + i * 0.5 + np.random.randn() * 2 for i in range(30)]
    }, index=dates)
    
    # Test prediction
    result = predict_price(sample_data, symbol="AAPL", use_gemini=True)
    print(json.dumps(result, indent=2))