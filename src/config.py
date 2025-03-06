# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project base directory (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory paths
DATA_DIR = os.path.join(BASE_DIR, "data")
GUIDELINES_PATH = os.path.join(DATA_DIR, "Guidelines")
PAPERS_PATH = os.path.join(DATA_DIR, "Papers")

# Output directory
OUTPUT_PATH = os.path.join(BASE_DIR, "output")

# LLM configurations
LLM_CONFIGS = {
    "openai": {
        "reasoner": {
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 4000
        },
        "extractor": {
            "model": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 4000
        },
        "validator": {
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2000
        }
    },
    "anthropic": {
        "reasoner": {
            "model": "claude-3-opus-20240229",
            "temperature": 0.1,
            "max_tokens": 4000
        },
        "extractor": {
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.2,
            "max_tokens": 4000
        },
        "validator": {
            "model": "claude-3-haiku-20240307",
            "temperature": 0.1,
            "max_tokens": 2000
        }
    },
    "deepseek": {
        "reasoner": {
            "model": "deepseek-chat",
            "temperature": 0.1,
            "max_tokens": 4000
        },
        "extractor": {
            "model": "deepseek-chat",
            "temperature": 0.2,
            "max_tokens": 4000
        },
        "validator": {
            "model": "deepseek-chat",
            "temperature": 0.1,
            "max_tokens": 2000
        }
    }
}

# PDF processing settings
PDF_SETTINGS = {
    "use_ocr": False,
    "chunk_size": 8000,
    "chunk_overlap": 500
}

# Validation settings
VALIDATION = {
    "min_confidence_threshold": 0.7,
    "disagreement_threshold": 0.5
}

# Logging configuration
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(BASE_DIR, "llm_validation.log")
}

# API keys from environment variables
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY")
}

# Check if necessary API keys are available
def validate_api_keys():
    """Check if necessary API keys are available."""
    missing_keys = []
    for key, value in API_KEYS.items():
        if not value:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"Warning: Missing API keys for: {', '.join(missing_keys)}")
        print("Please add them to your .env file.")
    
    return len(missing_keys) == 0

# Check for required directories
def check_directories():
    """Create required directories if they don't exist."""
    for directory in [DATA_DIR, GUIDELINES_PATH, PAPERS_PATH, OUTPUT_PATH]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Check for RECORD guidelines directory
    record_dir = os.path.join(GUIDELINES_PATH, "RECORD")
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
        print(f"Created directory: {record_dir}")
        print("Please add RECORD guideline PDFs to this directory.")