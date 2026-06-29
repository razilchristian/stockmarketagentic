# AlphaAnalytics Agentic AI Backend - Complete Authentication System with Predictor Module

# ============================================
# RENDER DEPLOYMENT FIXES - ADD THESE FIRST!
# ============================================
import sys
import os
import time

# Force unbuffered output so Render sees logs immediately
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    
print("="*60)
print("🚀 ALPHAANALYTICS STARTING UP ON RENDER")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"PORT environment variable: {os.environ.get('PORT', 'NOT SET - will use 10000')}")
print(f"PID: {os.getpid()}")
print("="*60)
sys.stdout.flush()
time.sleep(1)

# ============================================
# STANDARD IMPORTS
# ============================================
print("📦 Importing yfinance...")
sys.stdout.flush()
import yfinance as yf
print("✓ yfinance imported")

print("📦 Importing numpy...")
sys.stdout.flush()
import numpy as np
print("✓ numpy imported")

print("📦 Importing pandas...")
sys.stdout.flush()
import pandas as pd
print("✓ pandas imported")

print("📦 Importing random, json, re, traceback...")
sys.stdout.flush()
import random
import json
import re
import traceback
print("✓ Standard libraries imported")

print("📦 Importing datetime...")
sys.stdout.flush()
from datetime import datetime, timedelta
print("✓ datetime imported")

print("📦 Importing requests...")
sys.stdout.flush()
import requests
print("✓ requests imported")

print("📦 Importing Flask and extensions...")
sys.stdout.flush()
from flask import Flask, jsonify, request, render_template, redirect, url_for, session, flash
from flask_cors import CORS
print("✓ Flask and CORS imported")

print("📦 Importing email_service...")
sys.stdout.flush()
try:
    from email_service import send_prediction_email
    print("✓ email_service imported (SendGrid)")
except ImportError:
    print("⚠ email_service not found, creating fallback")
    def send_prediction_email(*args, **kwargs):
        print("Email service not available")
        return None

print("📦 Loading environment variables...")
sys.stdout.flush()
from dotenv import load_dotenv
load_dotenv()
print("✓ Environment variables loaded")

# Import predictor from models folder
print("📦 Importing predictor module...")
sys.stdout.flush()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from models.predictor import predict_price
    print("✓ Predictor module imported successfully")
except Exception as e:
    print(f"❌ Failed to import predictor: {e}")
    traceback.print_exc()
    # Define a fallback predictor function
    def predict_price(data, symbol="UNKNOWN", use_gemini=False):
        print("⚠ Using fallback predictor")
        if hasattr(data, 'iloc') and len(data) > 0:
            try:
                latest = float(data["Close"].iloc[-1])
                return {
                    "open": round(latest, 2),
                    "high": round(latest * 1.02, 2),
                    "low": round(latest * 0.98, 2),
                    "close": round(latest * 1.01, 2)
                }
            except:
                pass
        return {"open": 100.00, "high": 105.00, "low": 95.00, "close": 102.00}

print("📦 Importing google.genai...")
sys.stdout.flush()
try:
    import google.genai as genai
    from google.genai import types
    print("✓ Google GenAI imported")
    GENAI_AVAILABLE = True
except ImportError:
    print("⚠ Google GenAI not available")
    GENAI_AVAILABLE = False
    genai = None

print("="*60)
print("✅ ALL IMPORTS COMPLETED SUCCESSFULLY")
print("="*60)
sys.stdout.flush()
time.sleep(1)

# ---------------------------------
# GEMINI CONFIGURATION WITH AUTO-DIAGNOSTICS
# ---------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "razilchristian@gmail.com")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Debug: Check if API keys are loaded
if GEMINI_API_KEY:
    print(f"✓ Gemini API Key loaded: {GEMINI_API_KEY[:6]}...")
else:
    print("❌ WARNING: GEMINI_API_KEY not found in environment variables")

if SENDGRID_API_KEY:
    print(f"✓ SendGrid API Key loaded: {SENDGRID_API_KEY[:6]}...")
else:
    print("❌ WARNING: SENDGRID_API_KEY not found in environment variables")

if ALPHA_VANTAGE_API_KEY:
    print(f"✓ Alpha Vantage API Key loaded: {ALPHA_VANTAGE_API_KEY[:6]}...")
else:
    print("❌ WARNING: ALPHA_VANTAGE_API_KEY not found in environment variables")

sys.stdout.flush()

# Initialize the new client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY and GENAI_AVAILABLE else None
GEMINI_MODEL = None

# ============================================
# AUTO-DIAGNOSTIC: Check available Gemini models
# ============================================
if client:
    try:
        print("="*60)
        print("🔍 DIAGNOSIS: Checking available Gemini models...")
        print("="*60)
        sys.stdout.flush()
        
        models = client.models.list()
        available_models = []
        
        print("Models available to your API key:\n")
        for model in models:
            model_name = model.name
            display_name = model_name.replace('models/', '')
            available_models.append(display_name)
            
            actions = getattr(model, 'supported_actions', [])
            actions_str = ', '.join(actions) if actions else 'generateContent, countTokens'
            print(f"  • {display_name}")
            print(f"    Supports: {actions_str}\n")
            sys.stdout.flush()
        
        if available_models:
            print(f"\n✅ Found {len(available_models)} available models")
            
            preferred_models = [
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-pro",
                "gemini-1.0-pro"
            ]
            
            for preferred in preferred_models:
                if preferred in available_models:
                    GEMINI_MODEL = preferred
                    print(f"\n✅ Selected model: {GEMINI_MODEL}")
                    sys.stdout.flush()
                    break
            
            if not GEMINI_MODEL and available_models:
                GEMINI_MODEL = available_models[0]
                print(f"\n⚠ No preferred model found, using: {GEMINI_MODEL}")
                sys.stdout.flush()
            
            if GEMINI_MODEL:
                try:
                    test_response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents="Say 'OK' in one word"
                    )
                    if hasattr(test_response, 'text'):
                        print(f"✓ Model test successful: {test_response.text}")
                except Exception as e:
                    print(f"⚠ Model test failed: {e}")
                    print("  The app will use predictor module as fallback")
                    client = None
                sys.stdout.flush()
        else:
            print("❌ No models found for this API key")
            client = None
            
        print("="*60)
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Error accessing Gemini API: {e}")
        print("\nThe app will continue using the predictor module for all predictions.")
        sys.stdout.flush()
        client = None
else:
    print("⚠ Gemini client not initialized - using predictor module only")
    sys.stdout.flush()

# ---------------------------------
# FLASK APP
# ---------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")

# ============================================
# SESSION CONFIGURATION - CRITICAL FOR DEPLOYMENT
# ============================================
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=7)

CORS(app, supports_credentials=True, origins=[
    "https://stockmarketagentic.onrender.com", 
    "http://localhost:5000",
    "http://127.0.0.1:5000"
])

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# Cache for stock data to reduce API calls
stock_cache = {}
CACHE_DURATION = 3600  # 1 hour cache

# Simple in-memory user database populated through signup.
users = {}

# Rate limiting for API calls
api_call_counts = {}
RATE_LIMIT = 10
RATE_WINDOW = 60

def check_rate_limit(client_ip):
    now = datetime.now()
    if client_ip in api_call_counts:
        count, timestamp = api_call_counts[client_ip]
        if (now - timestamp).seconds < RATE_WINDOW:
            if count >= RATE_LIMIT:
                return False
            api_call_counts[client_ip] = (count + 1, timestamp)
        else:
            api_call_counts[client_ip] = (1, now)
    else:
        api_call_counts[client_ip] = (1, now)
    return True

# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------

def validate_stock_symbol(symbol):
    pattern = r'^[A-Z0-9\.\-\^]{1,10}$'
    return bool(re.match(pattern, symbol.upper()))

def get_next_trading_day():
    today = datetime.now()
    if today.weekday() == 4:
        return (today + timedelta(days=3)).strftime('%Y-%m-%d')
    if today.weekday() == 5:
        return (today + timedelta(days=2)).strftime('%Y-%m-%d')
    if today.weekday() == 6:
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    return (today + timedelta(days=1)).strftime('%Y-%m-%d')

def get_current_user_email():
    user = session.get("user") or {}
    email = user.get("email")
    return email.strip().lower() if isinstance(email, str) and email.strip() else None

def queue_prediction_email(symbol, predictions, analysis):
    recipient_email = get_current_user_email()

    if not recipient_email:
        print(f"⚠️ Prediction email skipped for {symbol}: no logged-in user email")
        return {"queued": False, "recipient": None, "reason": "missing_user_email"}

    if not SENDGRID_API_KEY:
        print(f"⚠️ Prediction email skipped for {symbol}: SENDGRID_API_KEY not configured")
        return {"queued": False, "recipient": recipient_email, "reason": "sendgrid_not_configured"}

    try:
        send_prediction_email(recipient_email, symbol, predictions, analysis)
        print(f"📧 Prediction email queued for {symbol} to {recipient_email}")
        return {"queued": True, "recipient": recipient_email}
    except Exception as e:
        print(f"⚠️ Prediction email queue error for {symbol}: {e}")
        return {"queued": False, "recipient": recipient_email, "reason": "queue_error"}

def calculate_technical_indicators(data):
    df = data.copy()
    
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    df['Momentum'] = df['Close'].pct_change(periods=5) * 100
    
    return df

# ---------------------------------
# ALPHA VANTAGE DATA FETCHER
# ---------------------------------

def fetch_alpha_vantage_daily(symbol):
    """
    Fetches up to 1 year of daily OHLCV data from Alpha Vantage.
    Returns a pandas DataFrame with columns: Open, High, Low, Close, Volume
    or None if the fetch fails.
    """
    if not ALPHA_VANTAGE_API_KEY:
        print("⚠ Alpha Vantage API key not configured, skipping AV fetch.")
        return None

    try:
        print(f"📡 [Alpha Vantage] Fetching daily data for {symbol}...")
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={symbol}"
            f"&outputsize=full"
            f"&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        if "Time Series (Daily)" not in raw:
            info = raw.get("Information") or raw.get("Note") or "Unknown error"
            print(f"⚠ [Alpha Vantage] No data for {symbol}: {info}")
            return None

        ts = raw["Time Series (Daily)"]
        records = []
        cutoff = datetime.now() - timedelta(days=365)

        for date_str, vals in ts.items():
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt < cutoff:
                continue
            records.append({
                "Date": dt,
                "Open":   float(vals["1. open"]),
                "High":   float(vals["2. high"]),
                "Low":    float(vals["3. low"]),
                "Close":  float(vals["5. adjusted close"]),   # adjusted close
                "Volume": int(vals["6. volume"])
            })

        if not records:
            print(f"⚠ [Alpha Vantage] No recent records found for {symbol}.")
            return None

        df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
        df.set_index("Date", inplace=True)
        print(f"✅ [Alpha Vantage] Got {len(df)} rows for {symbol}")
        return df

    except Exception as e:
        print(f"❌ [Alpha Vantage] Error fetching {symbol}: {e}")
        return None


def fetch_alpha_vantage_quote(symbol):
    """
    Fetches the real-time global quote from Alpha Vantage.
    Returns a dict with price/change/volume info or None on failure.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return None

    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol}"
            f"&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        q = raw.get("Global Quote", {})
        if not q or "05. price" not in q:
            return None

        return {
            "price":          float(q.get("05. price", 0)),
            "change":         float(q.get("09. change", 0)),
            "change_percent": float(q.get("10. change percent", "0%").replace("%", "")),
            "volume":         int(q.get("06. volume", 0)),
            "high":           float(q.get("03. high", 0)),
            "low":            float(q.get("04. low", 0)),
            "prev_close":     float(q.get("08. previous close", 0)),
        }
    except Exception as e:
        print(f"❌ [Alpha Vantage] Quote error for {symbol}: {e}")
        return None


# ---------------------------------
# STOCK DATA FETCH WITH CACHING
# Primary source: yfinance  |  Enriched/fallback: Alpha Vantage
# ---------------------------------

def get_stock_data(symbol, force_refresh=False):
    if not force_refresh and symbol in stock_cache:
        cache_time, cache_data = stock_cache[symbol]
        if (datetime.now() - cache_time).seconds < CACHE_DURATION:
            print(f"✅ Using cached data for {symbol} (age: {(datetime.now() - cache_time).seconds}s)")
            return cache_data
    
    # ── 1. Try yfinance first ──────────────────────────────────────────────
    yf_data = None
    try:
        print(f"📡 [yfinance] Fetching fresh data for {symbol}...")
        time.sleep(1)
        stock = yf.Ticker(symbol)
        yf_data = stock.history(period="1y")

        if yf_data.empty:
            print(f"⚠ [yfinance] Empty result for {symbol}, will try Alpha Vantage.")
            yf_data = None
    except Exception as e:
        print(f"❌ [yfinance] Error for {symbol}: {e}")
        yf_data = None

    # ── 2. Try Alpha Vantage as fallback / supplement ─────────────────────
    av_hist  = None
    av_quote = None
    if yf_data is None or yf_data.empty:
        av_hist = fetch_alpha_vantage_daily(symbol)
        if av_hist is not None:
            # AV daily history doesn't always have a Volume column initialised properly
            if "Volume" not in av_hist.columns:
                av_hist["Volume"] = 0

    # Also grab a live AV quote to enrich yfinance data (price/change might be stale)
    if ALPHA_VANTAGE_API_KEY:
        av_quote = fetch_alpha_vantage_quote(symbol)

    # ── 3. Decide which historical dataset to use ─────────────────────────
    data = yf_data if (yf_data is not None and not yf_data.empty) else av_hist

    if data is None or (hasattr(data, 'empty') and data.empty):
        # Last resort: return expired cache
        if symbol in stock_cache:
            print(f"⚠️ Using expired cache for {symbol}")
            return stock_cache[symbol][1]
        print(f"❌ No data available for {symbol} from any source.")
        return None

    try:
        df = calculate_technical_indicators(data)

        # ── 4. Price & change ─────────────────────────────────────────────
        if av_quote and av_quote["price"] > 0:
            # Prefer live Alpha Vantage quote for the most up-to-date price
            current_price  = av_quote["price"]
            change         = av_quote["change"]
            change_percent = av_quote["change_percent"]
            print(f"  ↳ Using Alpha Vantage live quote: ${current_price:.2f} ({change_percent:+.2f}%)")
        else:
            current_price  = float(data["Close"].iloc[-1])
            prev_close     = float(data["Close"].iloc[-2]) if len(data) > 1 else current_price
            change         = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0

        # ── 5. Statistical metrics ────────────────────────────────────────
        daily_returns = data["Close"].pct_change().dropna()
        volatility    = float(daily_returns.std() * np.sqrt(252) * 100) if len(daily_returns) > 0 else 0

        var_95 = np.percentile(daily_returns, 5) * 100 if len(daily_returns) > 0 else 0
        var_99 = np.percentile(daily_returns, 1) * 100 if len(daily_returns) > 0 else 0

        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess_returns = daily_returns - 0.02 / 252
            sharpe_ratio   = float(np.sqrt(252) * excess_returns.mean() / daily_returns.std())
        else:
            sharpe_ratio = 0

        recent_prices = data["Close"].tail(10).tolist()

        # ── 6. Volume (AV live quote takes priority for current vol) ──────
        if av_quote and av_quote.get("volume", 0) > 0:
            current_volume = av_quote["volume"]
        else:
            current_volume = int(data["Volume"].iloc[-1]) if not pd.isna(data["Volume"].iloc[-1]) else 0

        avg_volume   = int(data["Volume"].tail(30).mean()) if len(data) >= 30 else current_volume
        volume_trend = (
            "HIGH"   if current_volume > avg_volume * 1.5 else
            "NORMAL" if current_volume > avg_volume * 0.8 else
            "LOW"
        )

        # ── 7. High / Low / 52-week ───────────────────────────────────────
        if av_quote and av_quote.get("high", 0) > 0:
            recent_high = av_quote["high"]
            recent_low  = av_quote["low"]
        else:
            recent_high = float(data["High"].tail(20).max())  if len(data) >= 20 else current_price * 1.05
            recent_low  = float(data["Low"].tail(20).min())   if len(data) >= 20 else current_price * 0.95

        week_52_high = float(data["High"].tail(252).max()) if len(data) >= 252 else current_price * 1.2
        week_52_low  = float(data["Low"].tail(252).min())  if len(data) >= 252 else current_price * 0.8

        # ── 8. Technical indicators ───────────────────────────────────────
        current_rsi    = float(df['RSI'].iloc[-1])    if not pd.isna(df['RSI'].iloc[-1])    else 50
        current_macd   = float(df['MACD'].iloc[-1])   if not pd.isna(df['MACD'].iloc[-1])   else 0
        current_signal = float(df['Signal'].iloc[-1]) if not pd.isna(df['Signal'].iloc[-1]) else 0

        # ── 9. Fundamental info (yfinance only) ───────────────────────────
        market_cap = 0
        pe_ratio   = 0
        if yf_data is not None and not yf_data.empty:
            try:
                info       = yf.Ticker(symbol).info
                market_cap = info.get('marketCap', 0)
                pe_ratio   = info.get('trailingPE', 0)
            except Exception:
                pass

        result = {
            "symbol":        symbol,
            "current_price": current_price,
            "change":        change,
            "change_percent":change_percent,
            "recent_prices": recent_prices,
            "volatility":    volatility,
            "volume":        current_volume,
            "avg_volume":    avg_volume,
            "volume_trend":  volume_trend,
            "day_high":      recent_high,
            "day_low":       recent_low,
            "week_52_high":  week_52_high,
            "week_52_low":   week_52_low,
            "support":       recent_low  * 0.98,
            "resistance":    recent_high * 1.02,
            "rsi":           current_rsi,
            "macd":          current_macd,
            "signal":        current_signal,
            "var_95":        var_95,
            "var_99":        var_99,
            "sharpe_ratio":  sharpe_ratio,
            "ma_20":   float(df['MA20'].iloc[-1])    if not pd.isna(df['MA20'].iloc[-1])    else current_price,
            "ma_50":   float(df['MA50'].iloc[-1])    if not pd.isna(df['MA50'].iloc[-1])    else current_price,
            "bb_upper":float(df['BB_upper'].iloc[-1])if not pd.isna(df['BB_upper'].iloc[-1])else current_price * 1.1,
            "bb_lower":float(df['BB_lower'].iloc[-1])if not pd.isna(df['BB_lower'].iloc[-1])else current_price * 0.9,
            "momentum":float(df['Momentum'].iloc[-1])if not pd.isna(df['Momentum'].iloc[-1])else 0,
            "market_cap":    market_cap,
            "pe_ratio":      pe_ratio,
            "data_source":   "yfinance+alphavantage" if (yf_data is not None and av_quote) else
                             ("alphavantage"         if av_hist is not None else "yfinance"),
            "timestamp":     datetime.now().isoformat()
        }

        stock_cache[symbol] = (datetime.now(), result)
        print(f"✅ Successfully cached fresh data for {symbol} [source: {result['data_source']}]")
        return result

    except Exception as e:
        print(f"❌ Data processing error for {symbol}: {e}")
        if symbol in stock_cache:
            print(f"⚠️ Using expired cache for {symbol}")
            return stock_cache[symbol][1]
        return None

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def generate_stock_predictions(symbol, stock_data):
    try:
        recent_prices = stock_data['recent_prices']
        
        data = pd.DataFrame({'Close': recent_prices})
        predictions = predict_price(data)
        
        formatted_predictions = {
            "open": {
                "value": predictions['open'],
                "lower_bound": round(predictions['open'] * 0.98, 2),
                "upper_bound": round(predictions['open'] * 1.02, 2),
                "confidence": 85
            },
            "high": {
                "value": predictions['high'],
                "lower_bound": round(predictions['high'] * 0.98, 2),
                "upper_bound": round(predictions['high'] * 1.02, 2),
                "confidence": 80
            },
            "low": {
                "value": predictions['low'],
                "lower_bound": round(predictions['low'] * 0.98, 2),
                "upper_bound": round(predictions['low'] * 1.02, 2),
                "confidence": 80
            },
            "close": {
                "value": predictions['close'],
                "lower_bound": round(predictions['close'] * 0.98, 2),
                "upper_bound": round(predictions['close'] * 1.02, 2),
                "confidence": 85
            },
            "trend": "NEUTRAL",
            "trend_strength": 50,
            "recommendation": "HOLD"
        }
        
        print(f"✓ Generated predictions for {symbol}")
        return formatted_predictions
        
    except Exception as e:
        print(f"Error using predictor: {e}")
        return generate_fallback_predictions(stock_data)

def generate_fallback_predictions(stock_data):
    current = stock_data['current_price']
    volatility = stock_data['volatility'] / 100
    
    trend_factor = 1 if stock_data['momentum'] > 0 else -1 if stock_data['momentum'] < 0 else 0
    rsi_factor = (stock_data['rsi'] - 50) / 50
    
    combined_factor = (trend_factor * 0.6 + rsi_factor * 0.4) * volatility
    expected_change_pct = combined_factor * 2
    close_value = current * (1 + expected_change_pct / 100)
    close_value = max(current * 0.95, min(current * 1.05, close_value))
    
    confidence = max(65, min(95, int(95 - volatility * 1.5)))
    
    return {
        "open": {"value": round(current * (1 + random.uniform(-0.01, 0.01)), 2),
                "lower_bound": round(current * 0.97, 2),
                "upper_bound": round(current * 1.03, 2),
                "confidence": confidence},
        "high": {"value": round(max(current, close_value) * 1.01, 2),
                "lower_bound": round(max(current, close_value) * 0.99, 2),
                "upper_bound": round(max(current, close_value) * 1.03, 2),
                "confidence": confidence - 5},
        "low": {"value": round(min(current, close_value) * 0.99, 2),
               "lower_bound": round(min(current, close_value) * 0.97, 2),
               "upper_bound": round(min(current, close_value) * 1.01, 2),
               "confidence": confidence - 5},
        "close": {"value": round(close_value, 2),
                 "lower_bound": round(close_value * 0.97, 2),
                 "upper_bound": round(close_value * 1.03, 2),
                 "confidence": confidence},
        "trend": "BULLISH" if expected_change_pct > 0.5 else "BEARISH" if expected_change_pct < -0.5 else "NEUTRAL",
        "trend_strength": min(90, int(abs(expected_change_pct) * 50)),
        "recommendation": "BUY" if expected_change_pct > 1 else "SELL" if expected_change_pct < -1 else "HOLD"
    }

# ============================================
# LOGIN REQUIRED DECORATOR
# ============================================

def login_required(f):
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith('/api/'):
                return jsonify({"error": "Authentication required"}), 401
            flash("Please login to access this page", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# ============================================
# AUTHENTICATION ROUTES
# ============================================

@app.route("/")
def landing():
    if "user" in session:
        return redirect(url_for('jeet'))
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for('jeet'))
    
    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
            email = data.get("email")
            password = data.get("password")
        else:
            email = request.form.get("email")
            password = request.form.get("password")
        
        if not email or not password:
            if request.is_json:
                return jsonify({"success": False, "error": "Email and password required"}), 400
            flash("Email and password required", "error")
            return render_template("login.html")
        
        if email in users and users[email]["password"] == password:
            session["user"] = {
                "email": email,
                "first_name": users[email]["first_name"],
                "last_name": users[email]["last_name"]
            }
            
            if request.is_json:
                return jsonify({"success": True, "redirect": "/jeet"})
            flash(f"Welcome back, {users[email]['first_name']}!", "success")
            return redirect(url_for('jeet'))
        else:
            if request.is_json:
                return jsonify({"success": False, "error": "Invalid credentials"}), 401
            flash("Invalid email or password", "error")
            return render_template("login.html")
    
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user" in session:
        return redirect(url_for('jeet'))
    
    if request.method == "POST":
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
        
        if not all([first_name, last_name, email, password]):
            if request.is_json:
                return jsonify({"success": False, "error": "All fields required"}), 400
            flash("All fields required", "error")
            return render_template("signup.html")
        
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            if request.is_json:
                return jsonify({"success": False, "error": "Invalid email"}), 400
            flash("Invalid email format", "error")
            return render_template("signup.html")
        
        if len(password) < 6:
            if request.is_json:
                return jsonify({"success": False, "error": "Password too short"}), 400
            flash("Password must be at least 6 characters", "error")
            return render_template("signup.html")
        
        if email in users:
            if request.is_json:
                return jsonify({"success": False, "error": "Email already registered"}), 400
            flash("Email already registered", "error")
            return render_template("signup.html")
        
        users[email] = {
            "password": password,
            "first_name": first_name,
            "last_name": last_name,
            "created_at": datetime.now().isoformat()
        }
        
        session["user"] = {
            "email": email,
            "first_name": first_name,
            "last_name": last_name
        }
        
        if request.is_json:
            return jsonify({"success": True, "redirect": "/jeet"})
        flash(f"Welcome to AlphaAnalytics, {first_name}!", "success")
        return redirect(url_for('jeet'))
    
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out", "info")
    return redirect(url_for('login'))

# ============================================
# PROTECTED ROUTES
# ============================================

@app.route("/dashboard")
@login_required
def dashboard():
    return redirect(url_for('jeet'))

@app.route("/jeet")
@login_required
def jeet():
    return render_template("jeet.html", user=session.get("user"))

@app.route("/portfolio")
@login_required
def portfolio():
    return render_template("portfolio.html", user=session.get("user"))

@app.route("/mystock")
@login_required
def mystock():
    return render_template("mystock.html", user=session.get("user"))

@app.route("/deposit")
@login_required
def deposit():
    return render_template("deposit.html", user=session.get("user"))

@app.route("/insight")
@login_required
def insight():
    return render_template("insight.html", user=session.get("user"))

@app.route("/prediction")
@login_required
def prediction():
    return render_template("prediction.html", user=session.get("user"))

@app.route("/news")
@login_required
def news():
    return render_template("news.html", user=session.get("user"))

@app.route("/videos")
@login_required
def videos():
    return render_template("videos.html", user=session.get("user"))

@app.route("/superstars")
@login_required
def superstars():
    return render_template("superstars.html", user=session.get("user"))

@app.route("/alerts")
@login_required
def alerts():
    return render_template("alerts.html", user=session.get("user"))

@app.route("/help")
@login_required
def help():
    return render_template("help.html", user=session.get("user"))

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=session.get("user"))

# ============================================
# API ENDPOINTS
# ============================================

@app.route("/api/live-quote/<symbol>", methods=["GET"])
@login_required
def live_quote(symbol):
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip):
        return jsonify({"error": "Rate limit exceeded"}), 429
    
    try:
        symbol = symbol.upper()
        if not validate_stock_symbol(symbol):
            return jsonify({"error": "Invalid symbol"}), 400
        
        stock_data = get_stock_data(symbol, force_refresh=False)
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
            "data_source": stock_data.get("data_source", "yfinance"),
            "timestamp": stock_data["timestamp"]
        })
        
    except Exception as e:
        print(f"Live quote error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch-quote", methods=["POST"])
@login_required
def batch_quote():
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip):
        return jsonify({"error": "Rate limit exceeded"}), 429
    
    try:
        data = request.get_json()
        symbols = data.get("symbols", [])
        
        if not symbols or len(symbols) > 10:
            return jsonify({"error": "Please provide 1-10 symbols"}), 400
        
        results = {}
        for symbol in symbols:
            symbol = symbol.upper()
            if validate_stock_symbol(symbol):
                stock_data = get_stock_data(symbol, force_refresh=False)
                if stock_data:
                    results[symbol] = {
                        "current_price": stock_data["current_price"],
                        "change_percent": stock_data["change_percent"],
                        "volume": stock_data["volume"],
                        "data_source": stock_data.get("data_source", "yfinance")
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
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip):
        return jsonify({"error": "Rate limit exceeded"}), 429
    
    data = request.get_json()
    symbol = data.get("symbol", "AAPL").upper()

    if not validate_stock_symbol(symbol):
        return jsonify({"error": "Invalid symbol"}), 400

    stock_data = get_stock_data(symbol, force_refresh=False)

    if not stock_data:
        return jsonify({"error": "Stock data unavailable"}), 400

    predictions = generate_stock_predictions(symbol, stock_data)
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
    
    ai_analysis = predictions.get('analysis_summary', f'Analysis for {symbol} using predictor model.')
    
    if client and GEMINI_MODEL:
        try:
            analysis_response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=analysis_prompt
            )
            if hasattr(analysis_response, "text"):
                ai_analysis = analysis_response.text
                print(f"✓ Gemini analysis successful for {symbol}")
        except Exception as e:
            print(f"Gemini analysis error: {e}")

    response_data = {
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
        "data_source": stock_data.get("data_source", "yfinance"),
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
    
    response_data["email"] = queue_prediction_email(symbol, predictions, ai_analysis)
    
    return jsonify(response_data)

@app.route("/api/market-summary")
@login_required
def market_summary():
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip):
        return jsonify({"error": "Rate limit exceeded"}), 429
    
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    data = []

    for s in stocks:
        d = get_stock_data(s, force_refresh=False)
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
    
    text = "Market showing mixed signals with varying volatility levels."
    
    if client and GEMINI_MODEL:
        try:
            res = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            if hasattr(res, "text"):
                text = res.text
                print(f"✓ Gemini market summary successful")
        except Exception as e:
            print(f"Market summary Gemini error: {e}")

    return jsonify({
        "market_summary": text,
        "market_data": data,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/agentic-analyze", methods=["POST"])
@login_required
def agentic_analyze():
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip):
        return jsonify({"error": "Rate limit exceeded"}), 429
    
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
    tools = [
        {
            "name": "get_stock_data",
            "description": "Fetches current stock price, technical indicators, volatility, and volume (yfinance + Alpha Vantage)",
            "parameters": ["symbol"]
        },
        {
            "name": "predict_price",
            "description": "Generates price predictions using models/predictor.py",
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
        "version": "1.1",
        "description": "Agentic AI Trading Assistant Tools (yfinance + Alpha Vantage)"
    })

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "version": "Gemini Enhanced AI v2.1 with Alpha Vantage + SendGrid",
        "ai_model": GEMINI_MODEL if GEMINI_MODEL else "None (using predictor)",
        "api_key_configured": bool(GEMINI_API_KEY),
        "gemini_working": bool(client and GEMINI_MODEL),
        "sendgrid_configured": bool(SENDGRID_API_KEY),
        "alpha_vantage_configured": bool(ALPHA_VANTAGE_API_KEY),
        "predictor_loaded": True,
        "cache_duration": f"{CACHE_DURATION} seconds",
        "users_registered": len(users)
    })

@app.route("/api/debug-simple", methods=["GET"])
def debug_simple():
    return jsonify({
        "status": "ok",
        "message": "Server is running",
        "time": datetime.now().isoformat(),
        "cache_size": len(stock_cache)
    })

# ============================================
# RISK ANALYSIS FUNCTIONS
# ============================================

def generate_risk_analysis(stock_data, predictions):
    risks = []
    
    if stock_data['volatility'] > 40:
        risks.append({
            "level": "HIGH",
            "type": "VOLATILITY RISK",
            "message": f"Extreme volatility ({stock_data['volatility']:.1f}%)",
            "impact": "Large price swings expected"
        })
    elif stock_data['volatility'] > 25:
        risks.append({
            "level": "MEDIUM",
            "type": "VOLATILITY RISK",
            "message": f"Elevated volatility ({stock_data['volatility']:.1f}%)",
            "impact": "Moderate price fluctuations"
        })
    
    if stock_data['rsi'] > 70:
        risks.append({
            "level": "MEDIUM",
            "type": "RSI SIGNAL",
            "message": f"RSI at {stock_data['rsi']:.1f} - Overbought",
            "impact": "Potential pullback"
        })
    elif stock_data['rsi'] < 30:
        risks.append({
            "level": "MEDIUM",
            "type": "RSI SIGNAL",
            "message": f"RSI at {stock_data['rsi']:.1f} - Oversold",
            "impact": "Potential bounce"
        })
    
    risk_score = min(100, int(
        stock_data['volatility'] * 1.2 +
        (max(0, stock_data['rsi'] - 70) * 1.5 if stock_data['rsi'] > 70 else max(0, 30 - stock_data['rsi']) * 1.5)
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
    if not predictions or 'close' not in predictions:
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
    if not client or not GEMINI_MODEL:
        stock_data = get_stock_data(symbol, force_refresh=False)
        if not stock_data:
            return {"error": f"Unable to fetch data for {symbol}"}
        
        predictions = generate_stock_predictions(symbol, stock_data)
        risk = generate_risk_analysis(stock_data, predictions)
        
        return {
            "symbol": symbol,
            "user_goal": user_goal,
            "plan_executed": "1. Use get_stock_data\n2. Use predictor module\n3. Use risk_analysis",
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
                "trend": predictions.get('trend', 'NEUTRAL'),
                "trend_strength": predictions.get('trend_strength', 50),
                "recommendation": predictions.get('recommendation', 'HOLD'),
                "overall_confidence": predictions.get('overall_confidence', predictions['close']['confidence'])
            },
            "risk_analysis": {
                "risk_score": risk['risk_score'],
                "risk_level": risk['risk_level'],
                "var_95": risk['var_95'],
                "var_99": risk['var_99'],
                "sharpe_ratio": risk['sharpe_ratio'],
                "risk_factors": risk['risks']
            },
            "comprehensive_analysis": predictions.get('analysis_summary', 'Analysis complete.')
        }
    
    planning_prompt = f"""
    You are a senior financial AI agent with access to market analysis tools.

    USER GOAL: {user_goal}
    STOCK SYMBOL: {symbol}

    AVAILABLE TOOLS:
    1. get_stock_data - Fetches current stock price, technical indicators, volatility, volume (yfinance + Alpha Vantage)
    2. predict_price - Generates price predictions using machine learning model
    3. risk_analysis - Analyzes risks including VaR, volatility, RSI, trend reversal
    4. send_email - Sends analysis report via email
    5. gemini_enhance - Enhances analysis with Gemini AI

    Based on the user's goal, create a step-by-step plan to achieve it.
    Return the plan as a numbered list of steps.
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
        
        stock_data = get_stock_data(symbol, force_refresh=False)
        
        if not stock_data:
            return {"error": f"Unable to fetch data for {symbol}", "plan": plan}
        
        predictions = generate_stock_predictions(symbol, stock_data)
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
        
        User's Goal: {user_goal}
        
        Provide a comprehensive analysis summary that addresses the user's goal and gives actionable recommendations.
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
                "trend": predictions.get('trend', 'NEUTRAL'),
                "trend_strength": predictions.get('trend_strength', 50),
                "recommendation": predictions.get('recommendation', 'HOLD'),
                "overall_confidence": predictions.get('overall_confidence', predictions['close']['confidence'])
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
        
        if "email" in user_goal.lower() or "mail" in user_goal.lower():
            result["email"] = queue_prediction_email(symbol, predictions, comprehensive_analysis)
        
        return result
        
    except Exception as e:
        print("Agentic analysis error:", e)
        return {"error": str(e), "symbol": symbol, "user_goal": user_goal}

# ============================================
# FALLBACK ROUTE
# ============================================

@app.route("/<path:path>")
def catch_all(path):
    if "user" not in session:
        return redirect(url_for('login'))
    return render_template("404.html"), 404

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = "0.0.0.0"
    
    print("="*60)
    print("AlphaAnalytics Gemini Enhanced AI Server")
    print("="*60)
    print(f"Starting server on port: {port}")
    print(f"Binding to: {host}")
    print("="*60)
    print("\n✓ Configuration Status:")
    print(f"  • Gemini API Key:        {'✅ Configured' if GEMINI_API_KEY else '❌ Missing'}")
    print(f"  • Gemini Model:          {GEMINI_MODEL if GEMINI_MODEL else '❌ Not available'}")
    print(f"  • SendGrid API Key:      {'✅ Configured' if SENDGRID_API_KEY else '❌ Missing'}")
    print(f"  • Alpha Vantage API Key: {'✅ Configured' if ALPHA_VANTAGE_API_KEY else '❌ Missing'}")
    print(f"  • Predictor Module:      ✅ Loaded")
    print(f"  • Cache Duration:        {CACHE_DURATION} seconds")
    print("="*60)
    print(f"\n🚀 Server starting on http://{host}:{port}")
    print("="*60)
    sys.stdout.flush()
    
    app.run(host=host, port=port, debug=False)
