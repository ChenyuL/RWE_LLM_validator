#!/usr/bin/env python
# process_record.py
"""
Script for processing research papers against RECORD guidelines.

This script is a simplified wrapper around the LLM Validation Framework
specifically focused on validating papers against the RECORD guidelines.

Example usage:
    # Process a specific paper
    python process_record.py path/to/paper.pdf
    
    # Process a paper and save results to a specific location
    python process_record.py path/to/paper.pdf --output path/to/output.json
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import framework components
from src.framework import LLMValidationFramework
from src.utils.logger import setup_logger
from src.config import API_KEYS, LOGGING, check_directories, validate_api_keys

def process_record_paper(paper_path: str, output_path: str = None) -> int:
    """
    Process a single paper against RECORD guidelines.
    
    Args:
        paper_path: Path to the paper PDF
        output_path: Path to save results (optional)
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Set up logging
    logger = setup_logger(
        log_file=LOGGING["file"],
        level=getattr(logging, LOGGING["level"]) if isinstance(LOGGING["level"], str) else LOGGING["level"],
        format_str=LOGGING["format"]
    )
    
    # Check for required directories
    check_directories()
    
    # Validate API keys
    if not validate_api_keys():
        logger.warning("Some API keys are missing. The framework may not function properly.")
    
    # Check if paper exists
    if not os.path.exists(paper_path):
        logger.error(f"Paper not found: {paper_path}")
        return 1
    
    # Check if paper is a PDF
    if not paper_path.lower().endswith('.pdf'):
        logger.error(f"File is not a PDF: {paper_path}")
        return 1
    
    # Initialize framework
    try:
        framework = LLMValidationFramework(API_KEYS)
        logger.info("Framework initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize framework: {str(e)}", exc_info=True)
        return 1
    
    # Process the paper
    try:
        paper_name = os.path.basename(paper_path)
        logger.info(f"Processing paper: {paper_name}")
        
        results = framework.process_record_paper(paper_path)
        
        # Determine output path if not provided
        if not output_path:
            paper_base = os.path.splitext(paper_name)[0]
            output_path = os.path.join("output", f"{paper_base}_record_validation.json")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results to JSON file
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print(f"RECORD VALIDATION RESULTS FOR {paper_name}")
        print("="*80)
        
        # Print validation summary metrics
        if "validation_summary" in results:
            metrics = results["validation_summary"]
            print("\nValidation Metrics:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    print(f"  {metric_name}: {metric_value:.3f}")
                else:
                    print(f"  {metric_name}: {metric_value}")
        
        # Count compliance by type
        if "items" in results:
            compliances = {"yes": 0, "partial": 0, "no": 0, "unknown": 0}
            
            for item_id, item_data in results["items"].items():
                compliance = item_data.get("compliance", "unknown")
                if compliance in compliances:
                    compliances[compliance] += 1
                else:
                    compliances["unknown"] += 1
            
            print("\nRECORD Compliance Summary:")
            print(f"  Total Items: {len(results['items'])}")
            print(f"  Compliant: {compliances['yes']} items")
            print(f"  Partially Compliant: {compliances['partial']} items")
            print(f"  Non-Compliant: {compliances['no']} items")
            print(f"  Unknown: {compliances['unknown']} items")
            
            compliance_rate = (compliances['yes'] + (compliances['partial'] * 0.5)) / len(results['items']) * 100
            print(f"  Overall Compliance Rate: {compliance_rate:.1f}%")
        
        print("\n" + "="*80)
        
        return 0
    except Exception as e:
        logger.error(f"Error processing paper {paper_path}: {str(e)}", exc_info=True)
        return 1

def main() -> int:
    """Main function to run the RECORD paper processing."""
    parser = argparse.ArgumentParser(description="Process a research paper against RECORD guidelines")
    
    parser.add_argument("paper", help="Path to the paper PDF to process")
    parser.add_argument("--output", help="Path to save results (default: ./output/<paper_name>_record_validation.json)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        LOGGING["level"] = "DEBUG"
    
    return process_record_paper(args.paper, args.output)

if __name__ == "__main__":
    sys.exit(main())