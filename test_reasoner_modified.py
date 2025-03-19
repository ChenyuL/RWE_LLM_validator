#!/usr/bin/env python
# test_reasoner_modified.py

import os
import sys
import logging
from src.agents.reasoner_modified import Reasoner
from src.utils.pdf_processor import PDFProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Initialize the PDF processor
    pdf_processor = PDFProcessor()
    
    # Initialize the reasoner with API keys
    api_keys = {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "deepseek": os.environ.get("DEEPSEEK_API_KEY", "")
    }
    reasoner = Reasoner(api_keys=api_keys)
    
    # Path to the Li-Paper SOP PDF
    li_paper_path = "data/Guidelines/Li-Paper/SOP-Li.pdf"
    
    # Extract text from the PDF
    logger.info(f"Extracting text from {li_paper_path}")
    guideline_text = pdf_processor.extract_text(li_paper_path)
    guideline_texts = [guideline_text]  # Convert to list as expected by extract_guideline_items
    
    # Extract guideline items
    logger.info("Extracting guideline items")
    guideline_items = reasoner.extract_guideline_items(guideline_texts)
    
    # Print the number of items extracted
    logger.info(f"Extracted {len(guideline_items)} guideline items")
    
    # Print the first 5 items
    logger.info("First 5 items:")
    for i, item in enumerate(guideline_items[:5]):
        logger.info(f"Item {i+1}: {item}")
    
    # Generate prompts
    logger.info("Generating prompts")
    prompts = reasoner.generate_prompts(guideline_items)
    
    # Print the number of prompts generated
    logger.info(f"Generated {len(prompts)} prompts")
    
    # Print the first prompt
    if prompts:
        first_key = list(prompts.keys())[0]
        logger.info(f"First prompt ({first_key}):")
        logger.info(prompts[first_key][:200] + "...")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
