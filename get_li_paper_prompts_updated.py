#!/usr/bin/env python
# get_li_paper_prompts_updated.py

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List
import time

from src.agents.reasoner_rag import ReasonerRAG
from src.utils.pdf_utils import extract_text_from_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_api_keys() -> Dict[str, str]:
    """
    Load API keys from environment variables.
    
    Returns:
        Dictionary of API keys
    """
    api_keys = {}
    
    # OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        api_keys["openai"] = openai_api_key
    else:
        logger.warning("OpenAI API key not found in environment variables")
    
    # Anthropic API key
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        api_keys["anthropic"] = anthropic_api_key
    else:
        logger.warning("Anthropic API key not found in environment variables")
    
    # Voyage AI API key
    voyage_api_key = os.environ.get("VOYAGE_API_KEY")
    if voyage_api_key:
        api_keys["voyage"] = voyage_api_key
    else:
        logger.warning("Voyage AI API key not found in environment variables")
    
    return api_keys

def extract_guideline_text(pdf_path: str) -> List[str]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of text chunks from the PDF
    """
    logger.info(f"Extracting text from {pdf_path}")
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks of 10,000 characters
    chunks = []
    for i in range(0, len(text), 10000):
        chunk = text[i:i+10000]
        chunks.append(chunk)
    
    logger.info(f"Extracted {len(chunks)} text chunks from {pdf_path}")
    return chunks

def generate_prompts(guideline_path: str, output_dir: str, model: str = "gpt-4o") -> None:
    """
    Generate prompts for a guideline.
    
    Args:
        guideline_path: Path to the guideline PDF
        output_dir: Directory to save the prompts
        model: Model to use for LLM calls
    """
    logger.info(f"Generating prompts for {guideline_path} using {model}")
    
    # Load API keys
    api_keys = load_api_keys()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract guideline name from path
    guideline_name = os.path.basename(os.path.dirname(guideline_path))
    
    # Initialize reasoner
    reasoner = ReasonerRAG(api_keys, model)
    
    # Extract text from guideline PDF
    guideline_texts = extract_guideline_text(guideline_path)
    
    # Extract guideline items
    guideline_items = reasoner.extract_guideline_items(guideline_texts)
    
    # Generate prompts for each guideline item
    prompts = reasoner.generate_prompts(guideline_items)
    
    # Save prompts to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}_{model.replace('-', '_')}_reasoner_{guideline_name}_prompts.json")
    
    with open(output_path, 'w') as f:
        json.dump({
            "guideline": guideline_name,
            "model": model,
            "timestamp": timestamp,
            "items": guideline_items,
            "prompts": prompts
        }, f, indent=2)
    
    logger.info(f"Saved prompts to {output_path}")
    print(f"Generated prompts for {guideline_name} using {model}")
    print(f"Saved to {output_path}")

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Generate prompts for Li-Paper SOP")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use for LLM calls")
    parser.add_argument("--output-dir", type=str, default="output/prompts", help="Directory to save the prompts")
    args = parser.parse_args()
    
    # Path to Li-Paper SOP PDF
    li_paper_path = "data/Guidelines/Li-Paper/SOP-Li.pdf"
    
    # Generate prompts
    generate_prompts(li_paper_path, args.output_dir, args.model)

if __name__ == "__main__":
    main()
