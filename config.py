# config.py
# Configuration module for AlphaAnalytics - Handles Gemini AI setup and diagnostics

import os
import sys
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

# Load environment variables
load_dotenv()

# ============================================
# GEMINI CONFIGURATION WITH AUTO-DIAGNOSTICS
# ============================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "razilchristian@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Debug: Check if API key is loaded
if GEMINI_API_KEY:
    print(f"✓ Gemini API Key loaded: {GEMINI_API_KEY[:6]}...")
else:
    print("❌ WARNING: GEMINI_API_KEY not found in environment variables")

# Initialize the new client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_MODEL = None  # Will be set after checking available models

# ============================================
# AUTO-DIAGNOSTIC: Check available Gemini models
# ============================================
if client:
    try:
        print("="*60)
        print("🔍 DIAGNOSIS: Checking available Gemini models...")
        print("="*60)
        
        models = client.models.list()
        available_models = []
        
        print("Models available to your API key:\n")
        for model in models:
            model_name = model.name
            # Remove 'models/' prefix for cleaner display
            display_name = model_name.replace('models/', '')
            available_models.append(display_name)
            
            # Show which methods each model supports
            actions = getattr(model, 'supported_actions', [])
            actions_str = ', '.join(actions) if actions else 'generateContent, countTokens'
            print(f"  • {display_name}")
            print(f"    Supports: {actions_str}\n")
        
        if available_models:
            print(f"\n✅ Found {len(available_models)} available models")
            
            # Try to find a suitable model in order of preference
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
                    break
            
            # If none of the preferred models found, use the first available
            if not GEMINI_MODEL and available_models:
                GEMINI_MODEL = available_models[0]
                print(f"\n⚠ No preferred model found, using: {GEMINI_MODEL}")
            
            # Test the selected model
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
        else:
            print("❌ No models found for this API key")
            client = None
            
        print("="*60)
    except Exception as e:
        print(f"❌ Error accessing Gemini API: {e}")
        print("\nThis could mean:")
        print(" 1. Your API key is invalid or expired")
        print(" 2. The API key doesn't have Gemini API enabled")
        print(" 3. Billing is not enabled for your project")
        print(" 4. The API key is for a different Google service")
        print("\nThe app will continue using the predictor module for all predictions.")
        client = None
else:
    print("⚠ Gemini client not initialized - using predictor module only")

# ============================================
# EXPORT CONFIGURATION
# ============================================

# Export all configuration variables
__all__ = [
    'GEMINI_API_KEY',
    'EMAIL_SENDER',
    'EMAIL_PASSWORD',
    'client',
    'GEMINI_MODEL'
]