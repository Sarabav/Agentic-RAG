import os
import openai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not set in .env file.")
    exit(1)

openai.api_key = api_key
try:
    resp = openai.Embedding.create(input="hello world", model="text-embedding-ada-002")
    print("✅ OpenAI embedding API call succeeded.")
    print("Embedding vector (first 5 values):", resp['data'][0]['embedding'][:5])
except Exception as e:
    print(f"❌ OpenAI embedding API call failed: {e}") 