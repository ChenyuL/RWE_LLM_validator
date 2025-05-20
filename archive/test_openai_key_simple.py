#!/usr/bin/env python
# test_openai_key_simple.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key (first 10 chars): {api_key[:10]}...")

try:
    # Try a simple API call to OpenAI
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        max_tokens=10
    )
    
    print("OpenAI API call successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
