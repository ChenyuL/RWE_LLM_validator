#!/usr/bin/env python
# main.py
"""
Main entry point for the LLM Validation Framework.

This script provides a command-line interface for processing research papers
against reporting guidelines (like RECORD) using a multi-LLM validation approach.

Example usage:
    # Process all papers in the default directory against RECORD guidelines
    python main.py
    
    # Process specific papers with verbose logging
    python main.py --papers path/to/paper1.pdf path/to/paper2.pdf --verbose
    
    # Process all papers in a specific directory
    python main.py --papers_dir path/to/papers/
"""
import os
import sys
import argparse
import json
import logging
from typing import List, Dict, Any
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import framework components
from src.framework import LLMValidationFramework
from src.utils.logger import setup_logger
from src.config import API_KEYS, LOGGING, check_directories, validate_api_keys

def process_papers(framework: LLMValidationFramework, 
                  papers: List[str], 
                  guideline_type: str = "RECORD") -> Dict[str, Any]:
    """
    Process a list of papers against a specific guideline type.
    
    Args:
        framework: The LLM validation framework instance
        papers: List of paths to paper PDFs
        guideline_type: Type of guideline to validate against (default: "RECORD")
        
    Returns:
        Dictionary containing results for all papers
    """
    results = {}
    
    for paper_path in papers:
        paper_name = os.path.basename(paper_path)
        logger.info(f"Processing paper: {paper_name}")
        
        try:
            if guideline_type.upper() == "RECORD":
                paper_results = framework.process_record_paper(paper_path)
                results[paper_name] = paper_results
            else:
                logger.error(f"Guideline type not supported: {guideline_type}")
                raise ValueError(f"Unsupported guideline type: {guideline_type}")
                
            logger.info(f"Successfully processed paper: {paper_name}")
        except Exception as e:
            logger.error(f"Error processing paper {paper_name}: {str(e)}", exc_info=True)
            results[paper_name] = {"error": str(e)}
    
    return results

def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the validation results.
    
    Args:
        results: Results dictionary from processing papers
    """
    print("\n" + "="*80)
    print("VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    for paper_name, paper_results in results.items():
        print(f"\nPaper: {paper_name}")
        
        if "error" in paper_results:
            print(f"  Error: {paper_results['error']}")
            continue
        
        # Print validation summary metrics
        if "validation_summary" in paper_results:
            metrics = paper_results["validation_summary"]
            print("  Validation Metrics:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    print(f"    {metric_name}: {metric_value:.3f}")
                else:
                    print(f"    {metric_name}: {metric_value}")
        
        # Print compliance by item
        if "items" in paper_results:
            print("\n  Compliance by Item:")
            item_compliances = {}
            for item_id, item_data in paper_results["items"].items():
                compliance = item_data.get("compliance", "unknown")
                if compliance not in item_compliances:
                    item_compliances[compliance] = 0
                item_compliances[compliance] += 1
            
            for compliance_type, count in item_compliances.items():
                print(f"    {compliance_type}: {count} items")
            
            # Print items with disagreements
            items_with_disagreements = []
            for item_id, item_data in paper_results["items"].items():
                if item_data.get("disagreements", []):
                    items_with_disagreements.append(item_id)
            
            if items_with_disagreements:
                print("\n  Items with Disagreements:")
                for item_id in items_with_disagreements:
                    print(f"    - {item_id}")
    
    print("\n" + "="*80)

def main() -> int:
    """Main function to run the LLM validation framework."""
    parser = argparse.ArgumentParser(description="LLM Validation Framework for Research Papers")
    
    parser.add_argument("--papers", nargs="+", help="Paths to specific paper PDFs to process")
    parser.add_argument("--papers_dir", help="Directory containing paper PDFs to process")
    parser.add_argument("--guideline", default="RECORD", 
                      choices=["RECORD"], 
                      help="Guideline to validate against (default: RECORD)")
    parser.add_argument("--output", help="Path to save results (default: ./output)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else LOGGING["level"]
    global logger
    logger = setup_logger(
        log_file=LOGGING["file"],
        level=getattr(logging, log_level) if isinstance(log_level, str) else log_level,
        format_str=LOGGING["format"]
    )
    
    # Check for required directories
    check_directories()
    
    # Validate API keys
    if not validate_api_keys():
        logger.warning("Some API keys are missing. The framework may not function properly.")
    
    # Initialize framework
    try:
        framework = LLMValidationFramework(API_KEYS)
        logger.info("Framework initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize framework: {str(e)}", exc_info=True)
        return 1
    
    # Determine papers to process
    papers_to_process = []
    
    if args.papers:
        # Process specific papers
        for paper_path in args.papers:
            if os.path.exists(paper_path) and paper_path.lower().endswith('.pdf'):
                papers_to_process.append(paper_path)
            else:
                logger.warning(f"Invalid paper path: {paper_path}")
    elif args.papers_dir:
        # Process all PDFs in directory
        if os.path.isdir(args.papers_dir):
            pdf_files = glob.glob(os.path.join(args.papers_dir, "*.pdf"))
            papers_to_process.extend(pdf_files)
        else:
            logger.error(f"Papers directory not found: {args.papers_dir}")
            return 1
    
    if not papers_to_process:
        logger.error("No papers found to process")
        return 1
    
    logger.info(f"Found {len(papers_to_process)} papers to process")
    
    # Process the papers
    try:
        results = process_papers(framework, papers_to_process, args.guideline)
        
        # Determine output path
        output_path = args.output if args.output else os.path.join("output", f"{args.guideline.lower()}_validation_results.json")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results to JSON file
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary to console
        print_results_summary(results)
        
        return 0
    except Exception as e:
        logger.error(f"Error during paper processing: {str(e)}", exc_info=True)
        return 1
    else:
        # Default: look in the standard Papers directory
        papers_dir = os.path.join("data", "Papers")
        if os.path.isdir(papers_dir):
            pdf_files = glob.glob(os.path.join(papers_dir, "*.pdf"))
            papers_to_process.extend(pdf_files)
        else:
            logger.error(f"Default papers directory not found: {papers_dir}")
            return 1
        