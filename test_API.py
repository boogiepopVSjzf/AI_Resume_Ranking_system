import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

response = requests.get(url)
models = [m['name'] for m in response.json().get('models', [])]
print("Available models for your API Key:")
for m in models:
    print(f" - {m}")