import google.genai as genai
from config import GEMINI_API_KEY, MODEL_NAME

client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini(prompt):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if hasattr(response, "text"):
            return response.text

        return "AI response unavailable."

    except Exception as e:
        print("Gemini error:", e)
        return "AI analysis temporarily unavailable."