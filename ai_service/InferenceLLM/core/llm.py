import os
from groq import Groq

class LLMTranslator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Get it from console.groq.com")
        self.client = Groq(api_key=self.api_key)
        print("Groq LLM ready (Llama 3 70B)")

    def translate(self, gloss):
        prompt = f"Translate the following ASL gloss into natural English. Output only the English sentence, nothing else.\nASL: {gloss}\nEnglish:"
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60
            )
            english = response.choices[0].message.content.strip()
            return True, english
        except Exception as e:
            print(f"Groq error: {e}")
            return False, f"(Translation failed: {gloss})"