#!/usr/bin/env python
# get_li_paper_prompts_modified.py

import os
import sys
import json
import logging
import datetime
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
    
    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/prompts/{timestamp}_openai_reasoner_Li-Paper_prompts.json"
    
    # Save prompts to JSON file
    logger.info(f"Saving prompts to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    logger.info(f"Prompts saved to {output_file}")
    
    # Print the first few prompts as a sample
    logger.info("Sample prompts:")
    sample_count = min(3, len(prompts))
    sample_items = list(prompts.items())[:sample_count]
    for item_id, prompt in sample_items:
        print(f"\n{'='*80}\n")
        print(f"PROMPT FOR ITEM {item_id}:")
        print(f"\n{prompt[:200]}...\n")  # Print just the first 200 characters
        print(f"{'='*80}\n")
    
    logger.info(f"Sample of {sample_count} prompts displayed. Full prompts saved to {output_file}")
    
    # Return the output file path
    return output_file

if __name__ == "__main__":
    output_file = main()
    print(f"\nPrompts successfully generated and saved to: {output_file}")
