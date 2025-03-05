#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import time
import hashlib
from pathlib import Path
import traceback
import os
import logging

# Configure logging
logging.basicConfig(
    filename='llm_validation.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file manually
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score

@dataclass
class LLMConfig:
    """Configuration for LLM API"""
    api_key: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 2048

[Rest of the file content remains the same...]
