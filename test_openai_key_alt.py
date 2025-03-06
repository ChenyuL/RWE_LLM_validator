import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key (first 10 chars): {api_key[:10]}...")

try:
    # Create a client with explicit configuration
    client = OpenAI(
        api_key=api_key,
        # Try with organization ID if available
        organization=os.getenv("OPENAI_ORG_ID", None)
    )
    
    # Print available models to check if authentication works
    print("Attempting to list models...")
    models = client.models.list()
    print(f"Models available: {len(models.data)}")
    
    # Try a simple API call with a different model
    print("Attempting to create a completion...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        max_tokens=10
    )
    
    print("API call successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
    
    # Check if there's an ANTHROPIC_API_KEY to test if that works
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("\nTesting Anthropic API key...")
        try:
            from anthropic import Anthropic
            
            anthropic = Anthropic(api_key=anthropic_key)
            message = anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello, world!"}]
            )
            print("Anthropic API call successful!")
            print(f"Response: {message.content[0].text}")
        except Exception as e:
            print(f"Anthropic Error: {e}")
