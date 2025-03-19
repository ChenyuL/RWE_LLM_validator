#!/usr/bin/env python
# get_li_paper_prompts.py

import os
import sys
import json
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
    
    # Generate prompts
    logger.info("Generating prompts")
    prompts = reasoner.generate_prompts(guideline_items)
    
    # Print the number of prompts generated
    logger.info(f"Generated {len(prompts)} prompts")
    
    # Save prompts to JSON file
    output_file = "li_paper_prompts.json"
    logger.info(f"Saving prompts to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    logger.info(f"Prompts saved to {output_file}")
    
    # Print all prompts
    logger.info("All prompts:")
    for item_id, prompt in prompts.items():
        print(f"\n{'='*80}\n")
        print(f"PROMPT FOR ITEM {item_id}:")
        print(f"\n{prompt}\n")
        print(f"{'='*80}\n")
    
    logger.info("All prompts displayed successfully")

if __name__ == "__main__":
    main()
