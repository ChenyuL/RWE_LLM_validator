#!/usr/bin/env python
# use_reasoner_directly.py

import os
import json
import datetime
from src.agents.reasoner_modified import Reasoner
from src.utils.pdf_processor import PDFProcessor

def main():
    print("Initializing PDF processor and reasoner...")
    
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
    
    print(f"Extracting text from {li_paper_path}...")
    guideline_text = pdf_processor.extract_text(li_paper_path)
    
    print("Extracting guideline items...")
    guideline_items = reasoner.extract_guideline_items([guideline_text])
    print(f"Extracted {len(guideline_items)} guideline items")
    
    print("Generating prompts...")
    prompts = reasoner.generate_prompts(guideline_items)
    print(f"Generated {len(prompts)} prompts")
    
    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/prompts/{timestamp}_direct_reasoner_Li-Paper_prompts.json"
    
    # Save prompts to JSON file
    print(f"Saving prompts to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Prompts successfully saved to {output_file}")
    return output_file

if __name__ == "__main__":
    output_file = main()
    print(f"\nDone! Prompts file: {output_file}")
