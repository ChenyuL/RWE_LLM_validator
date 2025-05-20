#!/usr/bin/env python
# run_rag_batch_improved.py
# Script to run the improved RAG-based extractor and validator on multiple papers

import os
import json
import argparse
import subprocess
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_batch_improved.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_batch_improved")

def process_paper(paper_path, prompts_file, checklist, extractor_model, extractor_provider, validator_model, validator_provider):
    """
    Process a single paper using the improved RAG-based extractor and validator.
    """
    paper_id = os.path.splitext(os.path.basename(paper_path))[0]
    logger.info(f"Processing paper: {paper_id}")
    
    cmd = [
        "python", "rag_extractor_validator_improved.py",
        "--prompts", prompts_file,
        "--paper", paper_path,
        "--checklist", checklist,
        "--extractor-model", extractor_model,
        "--extractor-provider", extractor_provider,
        "--validator-model", validator_model,
        "--validator-provider", validator_provider
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully processed paper: {paper_id}")
        logger.debug(f"Output: {result.stdout}")
        return {
            "paper_id": paper_id,
            "success": True,
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing paper {paper_id}: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return {
            "paper_id": paper_id,
            "success": False,
            "error": str(e),
            "stderr": e.stderr
        }

def main():
    """
    Main function to run the improved RAG-based extractor and validator on multiple papers.
    """
    parser = argparse.ArgumentParser(description='Run improved RAG-based extractor and validator on multiple papers')
    parser.add_argument('--papers', type=str, required=True, help='Path to JSON file with list of paper IDs or directory containing papers')
    parser.add_argument('--prompts', type=str, required=True, help='Path to prompts file')
    parser.add_argument('--checklist', type=str, default='Li-Paper', help='Checklist type (default: Li-Paper)')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--papers-dir', type=str, default='data/Papers', help='Directory containing paper PDFs (default: data/Papers)')
    parser.add_argument('--extractor-model', type=str, default='gpt-4o', help='Model to use for extraction (default: gpt-4o)')
    parser.add_argument('--extractor-provider', type=str, default='openai', help='Provider to use for extraction (default: openai)')
    parser.add_argument('--validator-model', type=str, default='claude-3-5-sonnet-20241022', help='Model to use for validation (default: claude-3-5-sonnet-20241022)')
    parser.add_argument('--validator-provider', type=str, default='anthropic', help='Provider to use for validation (default: anthropic)')
    
    args = parser.parse_args()
    
    # Get list of papers to process
    paper_ids = []
    if os.path.isdir(args.papers):
        # If papers is a directory, get all PDF files
        papers_dir = args.papers
        paper_ids = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
        paper_paths = [os.path.join(papers_dir, f) for f in paper_ids]
    else:
        # If papers is a JSON file, load paper IDs
        with open(args.papers, 'r') as f:
            papers_data = json.load(f)
            
        if isinstance(papers_data, list):
            # If it's a list of paper IDs
            paper_ids = papers_data
        elif isinstance(papers_data, dict) and "papers" in papers_data:
            # If it's a dict with a "papers" key
            paper_ids = papers_data["papers"]
        else:
            # Try to extract paper IDs from the JSON structure
            paper_ids = []
            for key, value in papers_data.items():
                if isinstance(value, dict) and "paper_id" in value:
                    paper_ids.append(value["paper_id"])
                elif isinstance(value, str) and value.isdigit():
                    paper_ids.append(value)
                elif isinstance(key, str) and key.isdigit():
                    paper_ids.append(key)
        
        # Convert paper IDs to paths
        paper_paths = []
        for paper_id in paper_ids:
            # Check if paper_id already has .pdf extension
            if paper_id.endswith('.pdf'):
                paper_path = os.path.join(args.papers_dir, paper_id)
            else:
                paper_path = os.path.join(args.papers_dir, f"{paper_id}.pdf")
            
            if os.path.exists(paper_path):
                paper_paths.append(paper_path)
            else:
                logger.warning(f"Paper file not found: {paper_path}")
    
    logger.info(f"Found {len(paper_paths)} papers to process")
    
    # Process papers
    results = []
    
    if args.max_workers > 1:
        # Process papers in parallel
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for paper_path in paper_paths:
                future = executor.submit(
                    process_paper, paper_path, args.prompts, args.checklist,
                    args.extractor_model, args.extractor_provider,
                    args.validator_model, args.validator_provider
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed paper: {result['paper_id']}")
                except Exception as e:
                    logger.error(f"Error in future: {e}")
    else:
        # Process papers sequentially
        for paper_path in paper_paths:
            result = process_paper(
                paper_path, args.prompts, args.checklist,
                args.extractor_model, args.extractor_provider,
                args.validator_model, args.validator_provider
            )
            results.append(result)
    
    # Save results
    output_file = f"rag_improved_batch_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved batch results to {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print("\n" + "="*80)
    print(f"BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total papers: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("="*80)

if __name__ == "__main__":
    main()
