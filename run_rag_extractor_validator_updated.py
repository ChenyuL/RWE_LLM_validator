#!/usr/bin/env python
# run_rag_extractor_validator_updated.py
# Script to run the updated RAG-based extractor and validator

import os
import sys
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_rag_extractor_validator_updated.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_rag_extractor_validator_updated")

# Import the updated RAG extractor and validator
try:
    from rag_extractor_validator_updated import (
        load_prompts_from_file,
        process_paper_with_rag
    )
except ImportError:
    logger.error("Failed to import from rag_extractor_validator_updated.py")
    sys.exit(1)

def main():
    """
    Main function to run the updated RAG-based extractor and validator.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run updated RAG-based extractor and validator for Li-Paper SOP')
    parser.add_argument('--prompts', type=str, required=True, help='Path to prompts file')
    parser.add_argument('--paper', type=str, required=True, help='Path to paper file')
    parser.add_argument('--checklist', type=str, default='Li-Paper', help='Checklist type (default: Li-Paper)')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for processing items (default: 5)')
    parser.add_argument('--use-voyage', action='store_true', help='Use Voyage AI for embeddings')
    
    args = parser.parse_args()
    
    # Check if the prompts file exists
    if not os.path.exists(args.prompts):
        logger.error(f"Prompts file not found: {args.prompts}")
        sys.exit(1)
    
    # Check if the paper file exists
    if not os.path.exists(args.paper):
        logger.error(f"Paper file not found: {args.paper}")
        sys.exit(1)
    
    # Load prompts
    logger.info(f"Loading prompts from {args.prompts}")
    guideline_info = load_prompts_from_file(args.prompts, args.checklist)
    
    # Process paper
    logger.info(f"Processing paper: {args.paper}")
    result = process_paper_with_rag(args.paper, guideline_info, args.batch_size, args.use_voyage)
    
    if result:
        logger.info(f"Successfully processed paper: {os.path.basename(args.paper)}")
        
        # Print summary
        report = result["report"]
        metrics = report["validation_summary"]
        
        print("\n" + "="*80)
        print(f"VALIDATION RESULTS FOR {os.path.basename(args.paper)}")
        print("="*80)
        
        print("\nValidation Metrics:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.3f}")
            else:
                print(f"  {metric_name}: {metric_value}")
        
        print("\nModel Information:")
        print(f"  Extractor: {report['model_info']['extractor']}")
        print(f"  Validator: {report['model_info']['validator']}")
        
        print("\n" + "="*80)
    else:
        logger.error(f"Failed to process paper: {args.paper}")
        sys.exit(1)

if __name__ == "__main__":
    main()
