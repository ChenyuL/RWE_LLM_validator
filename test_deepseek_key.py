import os
from dotenv import load_dotenv
import requests

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
        "max_tokens": 10
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        print("DeepSeek API call successful!")
        print(f"Response: {response.json()['choices'][0]['message']['content']}")
    else:
        print(f"DeepSeek API call failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
