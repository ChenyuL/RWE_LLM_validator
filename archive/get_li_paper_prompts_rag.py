#!/usr/bin/env python
# get_li_paper_prompts_rag.py
# Script to generate prompts from Li-Paper SOP PDF using the RAG-enhanced reasoner

import os
import json
import logging
import argparse
import datetime
from pathlib import Path
import PyPDF2

from src.agents.reasoner_rag import ReasonerRAG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("get_li_paper_prompts_rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("get_li_paper_prompts_rag")

def get_api_keys_from_env():
    """
    Get API keys directly from the .env file instead of using environment variables.
    """
    api_keys = {}
    
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
        
        for line in env_content.split('\n'):
            if line.startswith('OPENAI_API_KEY='):
                api_keys["openai"] = line.split('=', 1)[1].strip()
            elif line.startswith('ANTHROPIC_API_KEY='):
                api_keys["anthropic"] = line.split('=', 1)[1].strip()
            elif line.startswith('VOYAGE_API_KEY='):
                api_keys["voyage"] = line.split('=', 1)[1].strip()
        
        logger.info(f"API keys loaded directly from .env file")
        
        return api_keys
    except Exception as e:
        logger.error(f"Error loading API keys from .env file: {e}")
        raise

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    try:
        logger.info(f"Extracting text from {pdf_path}")
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
            # Add page numbers to the text
            text += f"\nTotal pages: {len(reader.pages)}\n"
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def main():
    """
    Main function to generate prompts from Li-Paper SOP PDF.
    """
    parser = argparse.ArgumentParser(description='Generate prompts from Li-Paper SOP PDF using RAG-enhanced reasoner')
    parser.add_argument('--pdf', type=str, default='data/Guidelines/Li-Paper/SOP-Li.pdf', help='Path to Li-Paper SOP PDF')
    parser.add_argument('--output', type=str, help='Path to output file (default: auto-generated)')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use for LLM calls (default: gpt-4o)')
    
    args = parser.parse_args()
    
    # Get API keys
    api_keys = get_api_keys_from_env()
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(args.pdf)
    if not pdf_text:
        logger.error(f"Failed to extract text from {args.pdf}")
        return
    
    # Initialize RAG-enhanced reasoner
    reasoner = ReasonerRAG(api_keys, args.model)
    
    # Extract guideline items
    logger.info("Extracting guideline items from PDF text")
    guideline_items = reasoner.extract_guideline_items([pdf_text])
    
    # Generate prompts
    logger.info("Generating prompts for guideline items")
    prompts = reasoner.generate_prompts(guideline_items)
    
    # Save prompts to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"output/prompts/{timestamp}_rag_reasoner_Li-Paper_prompts.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    logger.info(f"Saved {len(prompts)} prompts to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"PROMPT GENERATION SUMMARY")
    print("="*80)
    print(f"PDF: {args.pdf}")
    print(f"Model: {args.model}")
    print(f"Extracted {len(guideline_items)} guideline items")
    print(f"Generated {len(prompts)} prompts")
    print(f"Saved prompts to {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
