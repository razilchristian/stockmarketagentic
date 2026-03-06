import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(MODEL_NAME)


def ask_gemini(prompt):

    try:
        response = model.generate_content(prompt)

        if hasattr(response, "text"):
            return response.text

        return "AI response unavailable."

    except Exception as e:
        print("Gemini error:", e)
        return "AI analysis temporarily unavailable."