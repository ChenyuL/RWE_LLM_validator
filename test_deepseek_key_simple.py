#!/usr/bin/env python
# test_deepseek_key_simple.py

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("DEEPSEEK_API_KEY")
print(f"DeepSeek API Key (first 10 chars): {api_key[:10]}...")

try:
    # Try a simple API call to DeepSeek
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "temperature": 0.2,
        "max_tokens": 10
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
        print("DeepSeek API call successful!")
        print(f"Response: {result}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error: {e}")
