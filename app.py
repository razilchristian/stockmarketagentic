# AlphaAnalytics Agentic AI Backend - Complete Authentication System

import os
import yfinance as yf
import numpy as np
import pandas as pd
import random
import json
import re
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template, redirect, url_for, session, flash
from flask_cors import CORS
from email_service import send_prediction_email
from dotenv import load_dotenv
load_dotenv()

# NEW: Use the new google.genai package instead of deprecated generativeai
import google.genai as genai
from google.genai import types

# ---------------------------------
# GEMINI CONFIGURATION (UPDATED)
# ---------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_SENDER = "razilchristian@gmail.com"
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

# Initialize the new client
client = genai.Client(api_key=GEMINI_API_KEY)

# Use the latest available model
GEMINI_MODEL = "models/gemini-2.0-flash"


# ---------------------------------
# FLASK APP
# ---------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)  # For session management
app.permanent_session_lifetime = timedelta(days=7)  # Session lifetime
CORS(app)

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# Cache for stock data to reduce API calls
stock_cache = {}
CACHE_DURATION = 60  # seconds

# Simple user database (in production, use real database)
# This will persist during runtime, but will reset when server restarts
users = {
    "demo@alpha.com": {
        "password": "demo123",
        "first_name": "Demo",
        "last_name": "User",
        "created_at": datetime.now().isoformat()
    }
}

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
# ENHANCED STOCK DATA FETCH WITH CACHING
# ---------------------------------

def get_stock_data(symbol, force_refresh=False):
    """Fetch stock data with caching to reduce API calls"""
    
    # Check cache first
    if not force_refresh and symbol in stock_cache:
        cache_time, cache_data = stock_cache[symbol]
        if (datetime.now() - cache_time).seconds < CACHE_DURATION:
            print(f"Using cached data for {symbol}")
            return cache_data
    
    try:
        print(f"Fetching fresh data for {symbol} from yfinance...")
        stock = yf.Ticker(symbol)
        
        # Fetch historical data for analysis
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
        
        # Get 52-week high/low
        week_52_high = float(data["High"].tail(252).max()) if len(data) >= 252 else current_price * 1.2
        week_52_low = float(data["Low"].tail(252).min()) if len(data) >= 252 else current_price * 0.8
        
        # Get market cap and other info
        info = stock.info
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        
        result = {
            "symbol": symbol,
            "current_price": current_price,
            "change": change,
            "change_percent": change_percent,
            "recent_prices": recent_prices,
            "volatility": volatility,
            "volume": current_volume,
            "avg_volume": avg_volume,
            "volume_trend": volume_trend,
            "day_high": recent_high,
            "day_low": recent_low,
            "week_52_high": week_52_high,
            "week_52_low": week_52_low,
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
            "momentum": float(df['Momentum'].iloc[-1]) if not pd.isna(df['Momentum'].iloc[-1]) else 0,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in cache
        stock_cache[symbol] = (datetime.now(), result)
        
        return result
        
    except Exception as e:
        print(f"Stock fetch error for {symbol}:", e)
        return None

# ============================================
# AUTHENTICATION ROUTES
# ============================================

@app.route("/")
def landing():
    """Root URL - redirect to login if not authenticated, else to dashboard"""
    if "user" in session:
        return redirect(url_for('jeet'))
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page - GET shows form, POST processes login"""
    # If user is already logged in, redirect to dashboard
    if "user" in session:
        return redirect(url_for('jeet'))
    
    if request.method == "POST":
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            email = data.get("email")
            password = data.get("password")
            remember = data.get("remember", False)
        else:
            email = request.form.get("email")
            password = request.form.get("password")
            remember = request.form.get("remember") == "on"
        
        # Validate input
        if not email or not password:
            if request.is_json:
                return jsonify({"success": False, "error": "Email and password are required"}), 400
            else:
                flash("Email and password are required", "error")
                return render_template("login.html")
        
        # Check credentials
        if email in users and users[email]["password"] == password:
            session.permanent = remember
            session["user"] = {
                "email": email,
                "first_name": users[email]["first_name"],
                "last_name": users[email]["last_name"]
            }
            
            if request.is_json:
                return jsonify({"success": True, "redirect": "/jeet"})
            else:
                flash(f"Welcome back, {users[email]['first_name']}!", "success")
                return redirect(url_for('jeet'))
        else:
            if request.is_json:
                return jsonify({"success": False, "error": "Invalid email or password"}), 401
            else:
                flash("Invalid email or password", "error")
                return render_template("login.html")
    
    # GET request - show login page
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup page - GET shows form, POST processes signup"""
    # If user is already logged in, redirect to dashboard
    if "user" in session:
        return redirect(url_for('jeet'))
    
    if request.method == "POST":
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            first_name = data.get("first_name") or data.get("firstName")
            last_name = data.get("last_name") or data.get("lastName")
            email = data.get("email")
            password = data.get("password")
        else:
            first_name = request.form.get("first_name") or request.form.get("firstName")
            last_name = request.form.get("last_name") or request.form.get("lastName")
            email = request.form.get("email")
            password = request.form.get("password")
        
        # Validate input
        if not all([first_name, last_name, email, password]):
            if request.is_json:
                return jsonify({"success": False, "error": "All fields are required"}), 400
            else:
                flash("All fields are required", "error")
                return render_template("signup.html")
        
        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            if request.is_json:
                return jsonify({"success": False, "error": "Invalid email format"}), 400
            else:
                flash("Invalid email format", "error")
                return render_template("signup.html")
        
        # Validate password strength (minimum 6 characters)
        if len(password) < 6:
            if request.is_json:
                return jsonify({"success": False, "error": "Password must be at least 6 characters"}), 400
            else:
                flash("Password must be at least 6 characters", "error")
                return render_template("signup.html")
        
        # Check if user already exists
        if email in users:
            if request.is_json:
                return jsonify({"success": False, "error": "Email already registered"}), 400
            else:
                flash("Email already registered", "error")
                return render_template("signup.html")
        
        # Create new user
        users[email] = {
            "password": password,
            "first_name": first_name,
            "last_name": last_name,
            "created_at": datetime.now().isoformat()
        }
        
        # Auto login after signup
        session["user"] = {
            "email": email,
            "first_name": first_name,
            "last_name": last_name
        }
        
        if request.is_json:
            return jsonify({"success": True, "redirect": "/jeet"})
        else:
            flash(f"Welcome to AlphaAnalytics, {first_name}!", "success")
            return redirect(url_for('jeet'))
    
    # GET request - show signup page
    return render_template("signup.html")

@app.route("/logout")
def logout():
    """Logout user and clear session"""
    session.pop("user", None)
    flash("You have been logged out", "info")
    return redirect(url_for('login'))

# ============================================
# PROTECTED ROUTES (Require Authentication)
# ============================================

def login_required(f):
    """Decorator to require login for protected routes"""
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please login to access this page", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route("/dashboard")
@login_required
def dashboard():
    """Main dashboard - redirect to jeet.html"""
    return redirect(url_for('jeet'))

@app.route("/jeet")
@login_required
def jeet():
    """Main dashboard page"""
    return render_template("jeet.html", user=session.get("user"))

@app.route("/portfolio")
@login_required
def portfolio():
    """Portfolio page"""
    return render_template("portfolio.html", user=session.get("user"))

@app.route("/mystock")
@login_required
def mystock():
    """My Stock page"""
    return render_template("mystock.html", user=session.get("user"))

@app.route("/deposit")
@login_required
def deposit():
    """Deposit page"""
    return render_template("deposit.html", user=session.get("user"))

@app.route("/insight")
@login_required
def insight():
    """Insight page"""
    return render_template("insight.html", user=session.get("user"))

@app.route("/prediction")
@login_required
def prediction():
    """Prediction page"""
    return render_template("prediction.html", user=session.get("user"))

@app.route("/news")
@login_required
def news():
    """News page"""
    return render_template("news.html", user=session.get("user"))

@app.route("/videos")
@login_required
def videos():
    """Videos page"""
    return render_template("videos.html", user=session.get("user"))

@app.route("/superstars")
@login_required
def superstars():
    """Superstars page"""
    return render_template("superstars.html", user=session.get("user"))

@app.route("/alerts")
@login_required
def alerts():
    """Alerts page"""
    return render_template("alerts.html", user=session.get("user"))

@app.route("/help")
@login_required
def help():
    """Help page"""
    return render_template("help.html", user=session.get("user"))

@app.route("/profile")
@login_required
def profile():
    """Profile page"""
    return render_template("profile.html", user=session.get("user"))

# ============================================
# API ENDPOINTS (All require authentication)
# ============================================

@app.route("/api/live-quote/<symbol>", methods=["GET"])
@login_required
def live_quote(symbol):
    """Get REAL-TIME stock data from yfinance"""
    try:
        symbol = symbol.upper()
        
        if not validate_stock_symbol(symbol):
            return jsonify({"error": "Invalid symbol"}), 400
        
        # Force refresh to get latest data
        stock_data = get_stock_data(symbol)
        
        if not stock_data:
            return jsonify({"error": "Stock data unavailable"}), 404
        
        return jsonify({
            "symbol": symbol,
            "current_price": stock_data["current_price"],
            "change": stock_data["change"],
            "change_percent": stock_data["change_percent"],
            "day_high": stock_data["day_high"],
            "day_low": stock_data["day_low"],
            "volume": stock_data["volume"],
            "avg_volume": stock_data["avg_volume"],
            "rsi": stock_data["rsi"],
            "volatility": stock_data["volatility"],
            "week_52_high": stock_data["week_52_high"],
            "week_52_low": stock_data["week_52_low"],
            "market_cap": stock_data["market_cap"],
            "pe_ratio": stock_data["pe_ratio"],
            "timestamp": stock_data["timestamp"]
        })
        
    except Exception as e:
        print(f"Live quote error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch-quote", methods=["POST"])
@login_required
def batch_quote():
    """Get real-time data for multiple symbols at once"""
    try:
        data = request.get_json()
        symbols = data.get("symbols", [])
        
        if not symbols or len(symbols) > 20:
            return jsonify({"error": "Please provide 1-20 symbols"}), 400
        
        results = {}
        for symbol in symbols:
            symbol = symbol.upper()
            if validate_stock_symbol(symbol):
                stock_data = get_stock_data(symbol, force_refresh=False)
                if stock_data:
                    results[symbol] = {
                        "current_price": stock_data["current_price"],
                        "change_percent": stock_data["change_percent"],
                        "volume": stock_data["volume"]
                    }
        
        return jsonify({
            "quotes": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    """Generate price predictions for a stock"""
    data = request.get_json()
    symbol = data.get("symbol", "AAPL").upper()

    if not validate_stock_symbol(symbol):
        return jsonify({"error": "Invalid symbol"}), 400

    stock_data = get_stock_data(symbol, force_refresh=True)

    if not stock_data:
        return jsonify({"error": "Stock data unavailable"}), 400

    predictions = generate_gemini_predictions(symbol, stock_data)
    risk_analysis = generate_risk_analysis(stock_data, predictions)
    confidence_bands = generate_confidence_bands(predictions, stock_data)
    
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
        "change": stock_data["change"],
        "change_percent": stock_data["change_percent"],
        "volatility": stock_data["volatility"],
        "rsi": stock_data["rsi"],
        "volume_trend": stock_data["volume_trend"],
        "support": stock_data["support"],
        "resistance": stock_data["resistance"],
        "day_high": stock_data["day_high"],
        "day_low": stock_data["day_low"],
        "week_52_high": stock_data["week_52_high"],
        "week_52_low": stock_data["week_52_low"],
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
            EMAIL_SENDER,
            symbol,
            predictions,
            ai_analysis
        )
    except Exception as e:
        print("Email error:", e)
        
    return jsonify(response)

@app.route("/api/market-summary")
@login_required
def market_summary():
    """Get market summary for major stocks"""
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

@app.route("/api/agentic-analyze", methods=["POST"])
@login_required
def agentic_analyze():
    """Agentic goal-based stock analysis"""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "AAPL").upper()
        user_goal = data.get("goal", "Analyze this stock and provide recommendations")
        
        if not validate_stock_symbol(symbol):
            return jsonify({"error": "Invalid stock symbol"}), 400
        
        if not user_goal:
            return jsonify({"error": "Please provide your investment goal"}), 400
        
        result = agentic_stock_analysis(symbol, user_goal)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/agentic-tools", methods=["GET"])
@login_required
def get_agentic_tools():
    """Return available tools for the agentic system"""
    tools = [
        {
            "name": "get_stock_data",
            "description": "Fetches current stock price, technical indicators, volatility, and volume",
            "parameters": ["symbol"]
        },
        {
            "name": "predict_price",
            "description": "Generates AI-powered price predictions with confidence bands for next trading day",
            "parameters": ["symbol"]
        },
        {
            "name": "risk_analysis",
            "description": "Analyzes risks including Value at Risk (VaR), volatility, RSI, and trend reversal",
            "parameters": ["symbol"]
        },
        {
            "name": "send_email",
            "description": "Sends comprehensive analysis report via email",
            "parameters": ["email", "symbol", "analysis"]
        }
    ]
    
    return jsonify({
        "tools": tools,
        "version": "1.0",
        "description": "Agentic AI Trading Assistant Tools"
    })

@app.route("/api/health")
def health():
    """Health check endpoint (public)"""
    return jsonify({
        "status": "healthy",
        "version": "Gemini Enhanced AI v2.0 with Live Data",
        "ai_model": GEMINI_MODEL,
        "features": [
            "Live Real-time Data",
            "Agentic AI Planning",
            "Goal-based Analysis",
            "Confidence Bands",
            "Risk Analysis",
            "Technical Indicators",
            "Sentiment Analysis",
            "VaR Calculation",
            "Email Notifications",
            "User Authentication"
        ],
        "users_registered": len(users)
    })

# ============================================
# GEMINI FUNCTIONS
# ============================================

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
    - 52-Week High: ${stock_data['week_52_high']:.2f}
    - 52-Week Low: ${stock_data['week_52_low']:.2f}

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

    IMPORTANT: All predictions MUST be within ±10% of the current price (${stock_data['current_price']:.2f}).
    Example: If current price is $100, predictions should be between $90 and $110.

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
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        if hasattr(response, "text"):
            text = response.text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                predictions = json.loads(json_match.group())
                
                # Validate predictions are within reasonable range
                current = stock_data['current_price']
                for key in ['open', 'high', 'low', 'close']:
                    if key in predictions:
                        value = predictions[key]['value']
                        if abs(value - current) > current * 0.15:
                            print(f"Warning: {key} prediction {value} too far from current {current}, adjusting")
                            predictions[key]['value'] = round(current * (1 + (value - current) / current * 0.5), 2)
                
                return predictions
            else:
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
    
    print(f"Generating fallback predictions based on current price: ${current:.2f}")
    
    trend_factor = 1 if stock_data['momentum'] > 0 else -1 if stock_data['momentum'] < 0 else 0
    rsi_factor = (stock_data['rsi'] - 50) / 50
    
    combined_factor = (trend_factor * 0.6 + rsi_factor * 0.4) * volatility
    
    expected_change_pct = combined_factor * 2
    close_value = current * (1 + expected_change_pct / 100)
    close_value = max(current * 0.95, min(current * 1.05, close_value))
    
    confidence = max(65, min(95, int(95 - volatility * 1.5)))
    bound_pct = volatility * 0.5
    
    return {
        "open": {
            "value": round(current * (1 + random.uniform(-0.01, 0.01)), 2),
            "lower_bound": round(current * (1 - bound_pct), 2),
            "upper_bound": round(current * (1 + bound_pct), 2),
            "confidence": confidence
        },
        "high": {
            "value": round(max(current, close_value) * (1 + random.uniform(0.005, 0.015)), 2),
            "lower_bound": round(max(current, close_value) * 0.99, 2),
            "upper_bound": round(max(current, close_value) * (1 + bound_pct * 1.5), 2),
            "confidence": confidence - 5
        },
        "low": {
            "value": round(min(current, close_value) * (1 - random.uniform(0.005, 0.015)), 2),
            "lower_bound": round(min(current, close_value) * (1 - bound_pct * 1.5), 2),
            "upper_bound": round(min(current, close_value) * 1.01, 2),
            "confidence": confidence - 5
        },
        "close": {
            "value": round(close_value, 2),
            "lower_bound": round(close_value * (1 - bound_pct), 2),
            "upper_bound": round(close_value * (1 + bound_pct), 2),
            "confidence": confidence
        },
        "trend": "BULLISH" if combined_factor > 0.05 else "BEARISH" if combined_factor < -0.05 else "NEUTRAL",
        "trend_strength": min(90, int(abs(combined_factor) * 800)),
        "support": round(current * 0.96, 2),
        "resistance": round(current * 1.04, 2),
        "risk_factors": [
            f"Volatility at {stock_data['volatility']:.1f}%",
            f"RSI at {stock_data['rsi']:.1f} indicating {'overbought' if stock_data['rsi'] > 70 else 'oversold' if stock_data['rsi'] < 30 else 'neutral'} conditions",
            f"Volume {stock_data['volume_trend'].lower()} compared to average"
        ],
        "sentiment": "NEUTRAL",
        "recommendation": "HOLD",
        "overall_confidence": confidence,
        "analysis_summary": f"Technical analysis suggests {('bullish' if combined_factor > 0 else 'bearish' if combined_factor < 0 else 'neutral')} momentum with {confidence}% confidence. Current price: ${current:.2f}"
    }

def generate_risk_analysis(stock_data, predictions):
    """Generate comprehensive risk analysis"""
    
    risks = []
    
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
    
    if stock_data['rsi'] > 70:
        risks.append({
            "level": "MEDIUM",
            "type": "RSI SIGNAL",
            "message": f"RSI at {stock_data['rsi']:.1f} - Overbought conditions",
            "impact": "Potential pullback or consolidation",
            "mitigation": "Wait for RSI to cool down before buying"
        })
    elif stock_data['rsi'] < 30:
        risks.append({
            "level": "MEDIUM",
            "type": "RSI SIGNAL",
            "message": f"RSI at {stock_data['rsi']:.1f} - Oversold conditions",
            "impact": "Potential bounce or reversal",
            "mitigation": "Look for confirmation before selling"
        })
    
    if stock_data['volume_trend'] == "LOW":
        risks.append({
            "level": "LOW",
            "type": "LIQUIDITY RISK",
            "message": "Below average trading volume",
            "impact": "May have wider bid-ask spreads",
            "mitigation": "Use limit orders"
        })
    
    if predictions and predictions.get('trend') == "BULLISH" and stock_data['momentum'] < -5:
        risks.append({
            "level": "MEDIUM",
            "type": "TREND REVERSAL",
            "message": "Bullish prediction but negative momentum",
            "impact": "Potential false signal",
            "mitigation": "Wait for confirmation"
        })
    
    risks.append({
        "level": "INFO",
        "type": "VALUE AT RISK",
        "message": f"95% VaR: {stock_data['var_95']:.2f}% | 99% VaR: {stock_data['var_99']:.2f}%",
        "impact": f"Max expected loss (95% confidence): ${abs(stock_data['current_price'] * stock_data['var_95']/100):.2f}",
        "mitigation": "Adjust position size accordingly"
    })
    
    risk_score = min(100, int(
        stock_data['volatility'] * 1.2 +
        (max(0, stock_data['rsi'] - 70) * 1.5 if stock_data['rsi'] > 70 else max(0, 30 - stock_data['rsi']) * 1.5) +
        (15 if stock_data['volume_trend'] == "LOW" else 0)
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

def generate_confidence_bands(predictions, stock_data):
    """Generate confidence bands for visualization"""
    
    if not predictions:
        return None
    
    bands = []
    
    bands.append({
        "level": 90,
        "upper": predictions['close']['upper_bound'],
        "lower": predictions['close']['lower_bound']
    })
    
    price_range = predictions['close']['upper_bound'] - predictions['close']['lower_bound']
    bands.append({
        "level": 75,
        "upper": predictions['close']['value'] + price_range * 0.5,
        "lower": predictions['close']['value'] - price_range * 0.5
    })
    
    bands.append({
        "level": 50,
        "upper": predictions['close']['value'] + price_range * 0.25,
        "lower": predictions['close']['value'] - price_range * 0.25
    })
    
    return bands

def agentic_stock_analysis(symbol, user_goal):
    """
    AI Agent that plans and executes steps to achieve user's goal
    """
    
    planning_prompt = f"""
    You are a senior financial AI agent with access to market analysis tools.

    USER GOAL: {user_goal}
    STOCK SYMBOL: {symbol}

    AVAILABLE TOOLS:
    1. get_stock_data - Fetches current stock price, technical indicators, volatility, volume
    2. predict_price - Generates AI-powered price predictions with confidence bands
    3. risk_analysis - Analyzes risks including VaR, volatility, RSI, trend reversal
    4. send_email - Sends analysis report via email

    Based on the user's goal, create a step-by-step plan to achieve it.
    Consider:
    - What data needs to be gathered first?
    - What analysis is required?
    - What tools should be used and in what order?
    - What final output should be provided?

    Return the plan as a numbered list of steps, with each step specifying:
    - The tool to use
    - What data to extract
    - Why this step is necessary

    Example format:
    1. Use get_stock_data to fetch current market data for {symbol} - needed for baseline analysis
    2. Use predict_price to generate tomorrow's price predictions - core requirement
    3. Use risk_analysis to assess potential downsides - important for risk management
    4. Use send_email to deliver comprehensive report - final delivery
    """
    
    try:
        planning_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=planning_prompt
        )
        
        plan = planning_response.text if hasattr(planning_response, "text") else "Unable to generate plan"
        print("\n" + "="*60)
        print("AGENTIC AI PLANNING")
        print("="*60)
        print("User Goal:", user_goal)
        print("Symbol:", symbol)
        print("-"*40)
        print("Generated Plan:")
        print(plan)
        print("="*60 + "\n")
        
        stock_data = get_stock_data(symbol, force_refresh=True)
        
        if not stock_data:
            return {
                "error": f"Unable to fetch data for {symbol}",
                "plan": plan
            }
        
        predictions = generate_gemini_predictions(symbol, stock_data)
        risk = generate_risk_analysis(stock_data, predictions)
        
        analysis_prompt = f"""
        Based on the analysis for {symbol}:

        Current Price: ${stock_data['current_price']:.2f}
        Change: {stock_data['change_percent']:.2f}%
        Volatility: {stock_data['volatility']:.2f}%
        RSI: {stock_data['rsi']:.1f}
        Risk Level: {risk['risk_level']}
        
        Price Predictions:
        - Open: ${predictions['open']['value']:.2f} (Confidence: {predictions['open']['confidence']}%)
        - High: ${predictions['high']['value']:.2f} (Confidence: {predictions['high']['confidence']}%)
        - Low: ${predictions['low']['value']:.2f} (Confidence: {predictions['low']['confidence']}%)
        - Close: ${predictions['close']['value']:.2f} (Confidence: {predictions['close']['confidence']}%)
        
        Trend: {predictions['trend']} (Strength: {predictions['trend_strength']}%)
        Recommendation: {predictions['recommendation']}
        
        User's Goal: {user_goal}
        
        Provide a comprehensive analysis summary that:
        1. Addresses the user's specific goal
        2. Explains the key findings
        3. Gives actionable recommendations
        4. Highlights important risks
        """
        
        analysis_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=analysis_prompt
        )
        
        comprehensive_analysis = analysis_response.text if hasattr(analysis_response, "text") else predictions.get('analysis_summary', 'Analysis complete.')
        
        result = {
            "symbol": symbol,
            "user_goal": user_goal,
            "plan_executed": plan,
            "timestamp": datetime.now().isoformat(),
            "stock_data": {
                "current_price": stock_data['current_price'],
                "change_percent": stock_data['change_percent'],
                "volatility": stock_data['volatility'],
                "rsi": stock_data['rsi'],
                "volume_trend": stock_data['volume_trend'],
                "support": stock_data['support'],
                "resistance": stock_data['resistance']
            },
            "predictions": {
                "open": predictions['open'],
                "high": predictions['high'],
                "low": predictions['low'],
                "close": predictions['close'],
                "trend": predictions['trend'],
                "trend_strength": predictions['trend_strength'],
                "recommendation": predictions['recommendation'],
                "overall_confidence": predictions['overall_confidence']
            },
            "risk_analysis": {
                "risk_score": risk['risk_score'],
                "risk_level": risk['risk_level'],
                "var_95": risk['var_95'],
                "var_99": risk['var_99'],
                "sharpe_ratio": risk['sharpe_ratio'],
                "risk_factors": risk['risks']
            },
            "comprehensive_analysis": comprehensive_analysis
        }
        
        if "email" in user_goal.lower() or "mail" in user_goal.lower() or "notif" in user_goal.lower():
            try:
                send_prediction_email(
                    EMAIL_SENDER,
                    symbol,
                    predictions,
                    comprehensive_analysis
                )
                result["email_sent"] = True
                print(f"✓ Email notification sent for {symbol}")
            except Exception as e:
                print(f"✗ Email error: {e}")
                result["email_sent"] = False
                result["email_error"] = str(e)
        
        return result
        
    except Exception as e:
        print("Agentic analysis error:", e)
        return {
            "error": str(e),
            "symbol": symbol,
            "user_goal": user_goal
        }

# ============================================
# FALLBACK ROUTE FOR UNKNOWN PAGES
# ============================================

@app.route("/<path:path>")
def catch_all(path):
    """Catch-all route - redirect to login if not authenticated, otherwise show 404"""
    if "user" not in session:
        return redirect(url_for('login'))
    return render_template("404.html"), 404

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    # Get port FIRST before using it
    port = int(os.environ.get("PORT", 5000))
    
    print("="*60)
    print("AlphaAnalytics Gemini Enhanced AI Server")
    print("="*60)
    print("Features:")
    print("  • User Authentication System")
    print("  • LIVE REAL-TIME DATA from yfinance")
    print("  • Gemini-Powered Predictions")
    print("  • Confidence Bands (50%/75%/90%)")
    print("  • Comprehensive Risk Analysis")
    print("  • Technical Indicators")
    print("  • Value at Risk (VaR) Calculation")
    print("  • AGENTIC AI PLANNING")
    print("  • Goal-based Stock Analysis")
    print("  • Email Notifications")
    print("="*60)
    print("\nAuthentication Flow:")
    print("  • /              - Redirects to /login or /jeet based on session")
    print("  • GET  /login    - Login page")
    print("  • POST /login    - Process login (JSON or form data)")
    print("  • GET  /signup   - Signup page")
    print("  • POST /signup   - Process signup (JSON or form data)")
    print("  • GET  /logout   - Logout user")
    print("="*60)
    print("\nProtected Pages (require login):")
    print("  • /dashboard -> /jeet - Main Dashboard")
    print("  • /jeet          - Main Dashboard")
    print("  • /portfolio     - Portfolio page")
    print("  • /mystock       - My Stock page")
    print("  • /deposit       - Deposit page")
    print("  • /insight       - Insight page")
    print("  • /prediction    - Prediction page")
    print("  • /news          - News page")
    print("  • /videos        - Videos page")
    print("  • /superstars    - Superstars page")
    print("  • /alerts        - Alerts page")
    print("  • /help          - Help page")
    print("  • /profile       - Profile page")
    print("="*60)
    print(f"\nDemo Account:")
    print(f"  • Email:    demo@alpha.com")
    print(f"  • Password: demo123")
    print("="*60)
    print(f"\nServer starting on http://0.0.0.0:{port}")
    print("="*60)
    
    # Now run the app with the port
    app.run(host="0.0.0.0", port=port, debug=False)