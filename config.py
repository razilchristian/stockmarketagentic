# config.py

import os
from dotenv import load_dotenv
import google.genai as genai

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-1.5-flash"