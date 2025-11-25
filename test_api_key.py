import os
from dotenv import load_dotenv
from openai import OpenAI  # Import the OpenAI client

load_dotenv()  # Load your .env file

# Step 1: Print the raw key from .env (for debugging—delete this line after it works!)
api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded API key starts with: {api_key[:10]}...")  # Shows first 10 chars (e.g., "sk-abc123de...")
print(f"Full key length: {len(api_key)}")  # Should be ~51 chars for a valid key

# Step 2: Test the key with a simple API call
try:
    client = OpenAI(api_key=api_key)
    # Make a tiny test call (uses ~$0.0001 credit)
    response = client.models.list()  # Just lists available models—no big deal
    print("SUCCESS! Key works. First model:", response.data[0].id if response.data else "No models?")
except Exception as e:
    print(f"ERROR: {e}")