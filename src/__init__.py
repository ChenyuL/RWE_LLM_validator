"""
LLM Validation Framework for Research Papers.

This package provides a framework for validating biomedical research papers 
against reporting guidelines (like RECORD) using a multi-LLM approach.

The framework uses three different Large Language Models (LLMs):
1. Reasoner (LLM1): Processes guideline documents to generate prompts
2. Extractor (LLM2): Extracts information from research papers
3. Validator (LLM3): Validates the extracted information against guidelines

Package Structure:
- agents: Contains the different LLM agents used in the framework
- utils: Utility modules for PDF processing, logging, etc.
- config.py: Configuration settings for the framework
- framework.py: Main framework class that orchestrates the validation process
"""

__version__ = "0.1.0"
__author__ = "LLMEvaluation Team"

# Import key components for easier access
from src.framework import LLMValidationFramework
from src.agents.reasoner import Reasoner
from src.agents.extractor import Extractor
from src.agents.validator import Validator
from src.utils.pdf_processor import PDFProcessor

# Import configuration
from src.config import (
    API_KEYS,
    GUIDELINES_PATH,
    PAPERS_PATH,
    OUTPUT_PATH,
    validate_api_keys,
    check_directories
)

# Initialize package logging
from src.utils.logger import setup_logger
logger = setup_logger(name="llm_validation")