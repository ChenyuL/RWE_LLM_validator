#!/usr/bin/env python
# test_claude_key_simple.py

import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"Claude API Key (first 10 chars): {api_key[:10]}...")

try:
    # Try a simple API call to Claude
    client = Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hello, world!"}]
    )
    
    print("Claude API call successful!")
    print(f"Response: {response.content[0].text}")
except Exception as e:
    print(f"Error: {e}")
