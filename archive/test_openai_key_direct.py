#!/usr/bin/env python
# test_openai_key_direct.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key directly from the .env file
with open('.env', 'r') as f:
    env_content = f.read()

# Parse the .env file to extract the OpenAI API key
openai_api_key = None
for line in env_content.split('\n'):
    if line.startswith('OPENAI_API_KEY='):
        openai_api_key = line.split('=', 1)[1].strip()
        break

print(f"OpenAI API Key from .env file (first 10 chars): {openai_api_key[:10]}...")

# Check if there's an environment variable set that might be overriding the .env file
env_openai_api_key = os.environ.get('OPENAI_API_KEY')
if env_openai_api_key:
    print(f"WARNING: OPENAI_API_KEY is also set in the environment: {env_openai_api_key[:10]}...")
    print("This might be overriding the key in the .env file!")

try:
    # Try a simple API call to OpenAI with the key from .env
    client = OpenAI(api_key=openai_api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        max_tokens=10
    )
    
    print("OpenAI API call successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
