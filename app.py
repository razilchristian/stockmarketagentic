# AlphaAnalytics Agentic AI Backend - Fixed Version

import os
import yfinance as yf
import numpy as np
import pandas as pd
import random  # <-- FIXED: Added missing import
import json
import re
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from email_service  import send_prediction_email
from dotenv import load_dotenv
load_dotenv()

# NEW: Use the new google.genai package instead of deprecated generativeai
import google.genai as genai
from google.genai import types

# ---------------------------------
# GEMINI CONFIGURATION (UPDATED)
# ---------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the new client
client = genai.Client(api_key=GEMINI_API_KEY)

# Use the latest available model
GEMINI_MODEL = "gemini-1.5-flash"  # or "gemini-1.5-pro" if you have access


# ---------------------------------
# FLASK APP
# ---------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------

def validate_stock_symbol(symbol):
    pattern = r'^[A-Z0-9\.\-\^]{1,10}$'
    return bool(re.match(pattern, symbol.upper()))

def get_next_trading_day():
    today = datetime.now()
    if today.weekday() == 4:  # Friday
        return (today + timedelta(days=3)).strftime('%Y-%m-%d')
    if today.weekday() == 5:  # Saturday
        return (today + timedelta(days=2)).strftime('%Y-%m-%d')
    if today.weekday() == 6:  # Sunday
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    return (today + timedelta(days=1)).strftime('%Y-%m-%d')

def calculate_technical_indicators(data):
    """Calculate technical indicators for better analysis"""
    df = data.copy()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    
    # Price momentum
    df['Momentum'] = df['Close'].pct_change(periods=5) * 100
    df['Price_change'] = df['Close'].pct_change() * 100
    
    return df

# ---------------------------------
# ENHANCED STOCK DATA FETCH
# ---------------------------------

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        
        # Fetch more historical data for better analysis
        data = stock.history(period="1y")
        
        if data.empty:
            return None
        
        # Calculate technical indicators
        df = calculate_technical_indicators(data)
        
        current_price = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
        
        # Advanced volatility metrics
        daily_returns = data["Close"].pct_change().dropna()
        volatility = float(daily_returns.std() * np.sqrt(252) * 100) if len(daily_returns) > 0 else 0
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(daily_returns, 5) * 100 if len(daily_returns) > 0 else 0
        var_99 = np.percentile(daily_returns, 1) * 100 if len(daily_returns) > 0 else 0
        
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess_returns = daily_returns - 0.02/252
            sharpe_ratio = float(np.sqrt(252) * excess_returns.mean() / daily_returns.std())
        else:
            sharpe_ratio = 0
        
        # Recent price trend
        recent_prices = data["Close"].tail(10).tolist()
        
        # Volume analysis
        current_volume = int(data["Volume"].iloc[-1]) if not pd.isna(data["Volume"].iloc[-1]) else 0
        avg_volume = int(data["Volume"].tail(30).mean()) if len(data) >= 30 else current_volume
        volume_trend = "HIGH" if current_volume > avg_volume * 1.5 else "NORMAL" if current_volume > avg_volume * 0.8 else "LOW"
        
        # Support and Resistance levels
        recent_high = float(data["High"].tail(20).max()) if len(data) >= 20 else current_price * 1.05
        recent_low = float(data["Low"].tail(20).min()) if len(data) >= 20 else current_price * 0.95
        
        # Get current RSI, MACD
        current_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50
        current_macd = float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else 0
        current_signal = float(df['Signal'].iloc[-1]) if not pd.isna(df['Signal'].iloc[-1]) else 0
        
        return {
            "current_price": current_price,
            "change_percent": change_percent,
            "recent_prices": recent_prices,
            "volatility": volatility,
            "volume": current_volume,
            "avg_volume": avg_volume,
            "volume_trend": volume_trend,
            "day_high": recent_high,
            "day_low": recent_low,
            "support": recent_low * 0.98,
            "resistance": recent_high * 1.02,
            "rsi": current_rsi,
            "macd": current_macd,
            "signal": current_signal,
            "var_95": var_95,
            "var_99": var_99,
            "sharpe_ratio": sharpe_ratio,
            "ma_20": float(df['MA20'].iloc[-1]) if not pd.isna(df['MA20'].iloc[-1]) else current_price,
            "ma_50": float(df['MA50'].iloc[-1]) if not pd.isna(df['MA50'].iloc[-1]) else current_price,
            "bb_upper": float(df['BB_upper'].iloc[-1]) if not pd.isna(df['BB_upper'].iloc[-1]) else current_price * 1.1,
            "bb_lower": float(df['BB_lower'].iloc[-1]) if not pd.isna(df['BB_lower'].iloc[-1]) else current_price * 0.9,
            "momentum": float(df['Momentum'].iloc[-1]) if not pd.isna(df['Momentum'].iloc[-1]) else 0
        }
        
    except Exception as e:
        print("Stock fetch error:", e)
        return None

# ---------------------------------
# GEMINI ENHANCED PRICE PREDICTION (UPDATED)
# ---------------------------------

def generate_gemini_predictions(symbol, stock_data):
    """Use Gemini to generate intelligent predictions with confidence bands"""
    
    prompt = f"""
    As a senior financial analyst, predict tomorrow's stock prices for {symbol} with confidence bands.

    CURRENT MARKET DATA:
    - Current Price: ${stock_data['current_price']:.2f}
    - Today's Change: {stock_data['change_percent']:.2f}%
    - Volatility (annual): {stock_data['volatility']:.2f}%
    - RSI (14): {stock_data['rsi']:.1f}
    - MACD: {stock_data['macd']:.3f}
    - Signal Line: {stock_data['signal']:.3f}
    - 20-day MA: ${stock_data['ma_20']:.2f}
    - 50-day MA: ${stock_data['ma_50']:.2f}
    - Bollinger Upper: ${stock_data['bb_upper']:.2f}
    - Bollinger Lower: ${stock_data['bb_lower']:.2f}
    - Momentum (5-day): {stock_data['momentum']:.2f}%
    - Volume Trend: {stock_data['volume_trend']}
    - Support Level: ${stock_data['support']:.2f}
    - Resistance Level: ${stock_data['resistance']:.2f}
    - Value at Risk (95%): {stock_data['var_95']:.2f}%
    - Sharpe Ratio: {stock_data['sharpe_ratio']:.3f}

    Recent price history: {[round(p, 2) for p in stock_data['recent_prices']]}

    Based on technical analysis and market conditions, predict:

    1. OPENING PRICE (with 90% confidence interval)
    2. DAY HIGH (with 90% confidence interval)
    3. DAY LOW (with 90% confidence interval)
    4. CLOSING PRICE (with 90% confidence interval)

    Also provide:
    5. CONFIDENCE LEVEL (0-100%) for each prediction
    6. PRICE TREND (Bullish/Bearish/Neutral)
    7. TREND STRENGTH (0-100%)
    8. KEY SUPPORT LEVEL
    9. KEY RESISTANCE LEVEL
    10. RISK FACTORS (list 2-3 key risks)
    11. MARKET SENTIMENT (Bullish/Bearish/Neutral)
    12. RECOMMENDATION (Strong Buy/Buy/Hold/Sell/Strong Sell)

    Return as JSON with this exact structure:
    {{
        "open": {{"value": float, "lower_bound": float, "upper_bound": float, "confidence": int}},
        "high": {{"value": float, "lower_bound": float, "upper_bound": float, "confidence": int}},
        "low": {{"value": float, "lower_bound": float, "upper_bound": float, "confidence": int}},
        "close": {{"value": float, "lower_bound": float, "upper_bound": float, "confidence": int}},
        "trend": string,
        "trend_strength": int,
        "support": float,
        "resistance": float,
        "risk_factors": [string],
        "sentiment": string,
        "recommendation": string,
        "overall_confidence": int,
        "analysis_summary": string
    }}
    """
    
    try:
        # UPDATED: Use the new client API
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        if hasattr(response, "text"):
            # Extract JSON from response
            text = response.text
            # Find JSON between curly braces
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                predictions = json.loads(json_match.group())
                return predictions
            else:
                # Fallback to structured response
                return generate_fallback_predictions(stock_data)
        else:
            return generate_fallback_predictions(stock_data)
            
    except Exception as e:
        print("Gemini prediction error:", e)
        return generate_fallback_predictions(stock_data)

def generate_fallback_predictions(stock_data):
    """Fallback prediction logic if Gemini fails"""
    current = stock_data['current_price']
    volatility = stock_data['volatility'] / 100
    
    # More sophisticated fallback predictions
    trend_factor = 1 if stock_data['momentum'] > 0 else -1 if stock_data['momentum'] < 0 else 0
    rsi_factor = (stock_data['rsi'] - 50) / 50  # -1 to 1
    
    combined_factor = (trend_factor * 0.6 + rsi_factor * 0.4) * volatility
    
    # Predict close with confidence bands
    expected_change = combined_factor * current * 0.02
    close_value = current + expected_change
    
    # Confidence based on volatility
    confidence = max(50, min(90, int(100 - volatility * 2)))
    
    # Calculate bounds based on volatility
    bound_width = current * volatility * 0.5
    
    return {
        "open": {
            "value": round(current * (1 + random.uniform(-0.005, 0.005)), 2),
            "lower_bound": round(current * 0.98, 2),
            "upper_bound": round(current * 1.02, 2),
            "confidence": confidence
        },
        "high": {
            "value": round(max(current, close_value) * 1.01, 2),
            "lower_bound": round(max(current, close_value) * 0.99, 2),
            "upper_bound": round(max(current, close_value) * 1.03, 2),
            "confidence": confidence - 5
        },
        "low": {
            "value": round(min(current, close_value) * 0.99, 2),
            "lower_bound": round(min(current, close_value) * 0.97, 2),
            "upper_bound": round(min(current, close_value) * 1.01, 2),
            "confidence": confidence - 5
        },
        "close": {
            "value": round(close_value, 2),
            "lower_bound": round(close_value * (1 - volatility * 0.5), 2),
            "upper_bound": round(close_value * (1 + volatility * 0.5), 2),
            "confidence": confidence
        },
        "trend": "BULLISH" if combined_factor > 0.1 else "BEARISH" if combined_factor < -0.1 else "NEUTRAL",
        "trend_strength": min(100, int(abs(combined_factor) * 500)),
        "support": stock_data['support'],
        "resistance": stock_data['resistance'],
        "risk_factors": [
            f"Volatility at {stock_data['volatility']:.1f}%",
            f"RSI at {stock_data['rsi']:.1f} indicating {'overbought' if stock_data['rsi'] > 70 else 'oversold' if stock_data['rsi'] < 30 else 'neutral'} conditions",
            f"Volume {stock_data['volume_trend'].lower()} compared to average"
        ],
        "sentiment": "NEUTRAL",
        "recommendation": "HOLD",
        "overall_confidence": confidence,
        "analysis_summary": f"Technical analysis suggests {('bullish' if combined_factor > 0 else 'bearish' if combined_factor < 0 else 'neutral')} momentum with {confidence}% confidence."
    }

# ---------------------------------
# ENHANCED RISK ANALYSIS
# ---------------------------------

def generate_risk_analysis(stock_data, predictions):
    """Generate comprehensive risk analysis"""
    
    risks = []
    
    # Volatility risk
    if stock_data['volatility'] > 40:
        risks.append({
            "level": "HIGH",
            "type": "VOLATILITY RISK",
            "message": f"Extreme volatility ({stock_data['volatility']:.1f}%) detected",
            "impact": "Large price swings expected",
            "mitigation": "Consider reducing position size or using options"
        })
    elif stock_data['volatility'] > 25:
        risks.append({
            "level": "MEDIUM",
            "type": "VOLATILITY RISK",
            "message": f"Elevated volatility ({stock_data['volatility']:.1f}%)",
            "impact": "Moderate price fluctuations",
            "mitigation": "Set wider stop-losses"
        })
    
    # RSI risk
    if stock_data['rsi'] > 70:
        risks.append({
            "level": "MEDIUM",
            "type": "OVERSOLD/OVERBOUGHT",
            "message": f"RSI at {stock_data['rsi']:.1f} - Overbought conditions",
            "impact": "Potential pullback or consolidation",
            "mitigation": "Wait for RSI to cool down before buying"
        })
    elif stock_data['rsi'] < 30:
        risks.append({
            "level": "MEDIUM",
            "type": "OVERSOLD/OVERBOUGHT",
            "message": f"RSI at {stock_data['rsi']:.1f} - Oversold conditions",
            "impact": "Potential bounce or reversal",
            "mitigation": "Look for confirmation before selling"
        })
    
    # Volume risk
    if stock_data['volume_trend'] == "LOW":
        risks.append({
            "level": "LOW",
            "type": "LIQUIDITY RISK",
            "message": "Below average trading volume",
            "impact": "May have wider bid-ask spreads",
            "mitigation": "Use limit orders"
        })
    
    # Trend reversal risk
    if predictions and predictions.get('trend') == "BULLISH" and stock_data['momentum'] < -5:
        risks.append({
            "level": "MEDIUM",
            "type": "TREND REVERSAL",
            "message": "Bullish prediction but negative momentum",
            "impact": "Potential false signal",
            "mitigation": "Wait for confirmation"
        })
    
    # Value at Risk
    risks.append({
        "level": "INFO",
        "type": "VALUE AT RISK",
        "message": f"95% VaR: {stock_data['var_95']:.2f}% | 99% VaR: {stock_data['var_99']:.2f}%",
        "impact": f"Maximum expected loss: ${abs(stock_data['current_price'] * stock_data['var_95']/100):.2f} (95% confidence)",
        "mitigation": "Adjust position size accordingly"
    })
    
    # Calculate overall risk score (0-100)
    risk_score = min(100, int(
        stock_data['volatility'] * 1.5 +
        (max(0, stock_data['rsi'] - 70) * 2 if stock_data['rsi'] > 70 else max(0, 30 - stock_data['rsi']) * 2) +
        (20 if stock_data['volume_trend'] == "LOW" else 0)
    ))
    
    risk_level = "CRITICAL" if risk_score > 80 else "HIGH" if risk_score > 60 else "MEDIUM" if risk_score > 40 else "LOW"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risks": risks,
        "var_95": stock_data['var_95'],
        "var_99": stock_data['var_99'],
        "sharpe_ratio": stock_data['sharpe_ratio']
    }

# ---------------------------------
# CONFIDENCE BAND GENERATION
# ---------------------------------

def generate_confidence_bands(predictions, stock_data):
    """Generate confidence bands for visualization"""
    
    if not predictions:
        return None
    
    # Create confidence bands for different confidence levels
    bands = []
    
    # 90% confidence band
    bands.append({
        "level": 90,
        "upper": predictions['close']['upper_bound'],
        "lower": predictions['close']['lower_bound']
    })
    
    # 75% confidence band (narrower)
    price_range = predictions['close']['upper_bound'] - predictions['close']['lower_bound']
    bands.append({
        "level": 75,
        "upper": predictions['close']['value'] + price_range * 0.5,
        "lower": predictions['close']['value'] - price_range * 0.5
    })
    
    # 50% confidence band (even narrower)
    bands.append({
        "level": 50,
        "upper": predictions['close']['value'] + price_range * 0.25,
        "lower": predictions['close']['value'] - price_range * 0.25
    })
    
    return bands

# ---------------------------------
# ENHANCED PREDICTION API
# ---------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symbol = data.get("symbol", "AAPL").upper()

    if not validate_stock_symbol(symbol):
        return jsonify({"error": "Invalid symbol"}), 400

    stock_data = get_stock_data(symbol)

    if not stock_data:
        return jsonify({"error": "Stock data unavailable"}), 400

    # Generate Gemini-powered predictions
    predictions = generate_gemini_predictions(symbol, stock_data)
    
    # Generate comprehensive risk analysis
    risk_analysis = generate_risk_analysis(stock_data, predictions)
    
    # Generate confidence bands
    confidence_bands = generate_confidence_bands(predictions, stock_data)
    
    # Get AI analysis from Gemini
    analysis_prompt = f"""
    Based on the following data for {symbol}:
    - Current Price: ${stock_data['current_price']:.2f}
    - RSI: {stock_data['rsi']:.1f}
    - Volatility: {stock_data['volatility']:.1f}%
    - Momentum: {stock_data['momentum']:.2f}%
    - Risk Level: {risk_analysis['risk_level']}
    
    Provide a brief market analysis and trading recommendation in 2-3 sentences.
    """
    
    try:
        analysis_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=analysis_prompt
        )
        ai_analysis = analysis_response.text if hasattr(analysis_response, "text") else predictions.get('analysis_summary', 'Analysis complete.')
    except:
        ai_analysis = predictions.get('analysis_summary', 'Analysis complete.')

    response = {
        "symbol": symbol,
        "prediction_date": get_next_trading_day(),
        "current_price": stock_data["current_price"],
        "change_percent": stock_data["change_percent"],
        "volatility": stock_data["volatility"],
        "rsi": stock_data["rsi"],
        "volume_trend": stock_data["volume_trend"],
        "support": stock_data["support"],
        "resistance": stock_data["resistance"],
        "prediction": predictions,
        "confidence_bands": confidence_bands,
        "risk_analysis": risk_analysis,
        "ai_analysis": ai_analysis,
        "technical_indicators": {
            "ma_20": stock_data["ma_20"],
            "ma_50": stock_data["ma_50"],
            "bb_upper": stock_data["bb_upper"],
            "bb_lower": stock_data["bb_lower"],
            "macd": stock_data["macd"],
            "signal": stock_data["signal"],
            "momentum": stock_data["momentum"]
        }
    }
    try:
        send_prediction_email(
            "razilchristian@gmail.com",
            symbol,
            predictions,
            ai_analysis
        )
    except Exception as e:
        print("Email error:", e)
    return jsonify(response)

# ---------------------------------
# MARKET SUMMARY (Enhanced)
# ---------------------------------

@app.route("/api/market-summary")
def market_summary():
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    data = []

    for s in stocks:
        d = get_stock_data(s)
        if d:
            data.append({
                "symbol": s,
                "price": d["current_price"],
                "change": d["change_percent"],
                "volatility": d["volatility"],
                "rsi": d["rsi"],
                "volume_trend": d["volume_trend"]
            })

    prompt = f"""
    Analyze this market data:
    {json.dumps(data, indent=2)}
    
    Provide:
    1. Overall market sentiment
    2. Most volatile stock
    3. Best performing sector
    4. Risk outlook
    5. Trading recommendation for the day
    """
    
    try:
        res = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        text = res.text if hasattr(res, "text") else "Market showing mixed signals."
    except:
        text = "Market showing mixed signals with varying volatility levels."

    return jsonify({
        "market_summary": text,
        "market_data": data,
        "timestamp": datetime.now().isoformat()
    })

# ---------------------------------
# HEALTH CHECK
# ---------------------------------

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "version": "Gemini Enhanced AI v2.0",
        "ai_model": GEMINI_MODEL,
        "features": [
            "Confidence Bands",
            "Risk Analysis",
            "Technical Indicators",
            "Sentiment Analysis",
            "VaR Calculation"
        ]
    })

# ---------------------------------
# SERVE UI PAGES
# ---------------------------------

@app.route("/")
def index():
    return render_template("jeet.html")

@app.route("/<page>")
def pages(page):
    try:
        return render_template(f"{page}.html")
    except:
        return render_template("jeet.html")

# ---------------------------------
# RUN SERVER
# ---------------------------------

if __name__ == "__main__":
    print("="*60)
    print("AlphaAnalytics Gemini Enhanced AI Server")
    print("="*60)
    print("Features:")
    print("  • Gemini-Powered Predictions")
    print("  • Confidence Bands (50%/75%/90%)")
    print("  • Comprehensive Risk Analysis")
    print("  • Technical Indicators")
    print("  • Value at Risk (VaR) Calculation")
    print("="*60)
    app.run(host="0.0.0.0", port=5000, debug=True)