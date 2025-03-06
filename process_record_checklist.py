#!/usr/bin/env python
"""
Script for processing RECORD checklist and extracting guideline items.

This script uses the RecordReasoner to:
1. Extract all RECORD guideline items from the checklist PDF
2. Generate prompts for each item
3. Save the output to a JSON file for use by the Extractor

Usage:
    python process_record_checklist.py --input path/to/record_checklist.pdf [--output path/to/output.json]
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import framework components
from src.agents.record_reasoner import RecordReasoner
from src.utils.logger import setup_logger
from src.config import API_KEYS, LOGGING, check_directories, validate_api_keys, GUIDELINES_PATH

def process_record_checklist(checklist_path: str, output_path: str = None) -> int:
    """
    Process a RECORD checklist PDF and extract items.
    
    Args:
        checklist_path: Path to the RECORD checklist PDF
        output_path: Path to save the output (optional)
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Set up logging
    logger = setup_logger(
        log_file=LOGGING["file"],
        level=getattr(logging, LOGGING["level"]) if isinstance(LOGGING["level"], str) else LOGGING["level"],
        format_str=LOGGING["format"]
    )
    
    # Validate API keys
    if not validate_api_keys():
        logger.warning("Some API keys are missing. The framework may not function properly.")
    
    # Check if checklist exists
    if not os.path.exists(checklist_path):
        logger.error(f"Checklist file not found: {checklist_path}")
        return 1
    
    # Check if checklist is a PDF
    if not checklist_path.lower().endswith('.pdf'):
        logger.error(f"File is not a PDF: {checklist_path}")
        return 1
    
    # Initialize RecordReasoner
    try:
        reasoner = RecordReasoner(API_KEYS)
        logger.info("RecordReasoner initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RecordReasoner: {str(e)}", exc_info=True)
        return 1
    
    # Process the checklist
    try:
        logger.info(f"Processing RECORD checklist: {checklist_path}")
        
        # Extract items and generate prompts
        items_with_prompts = reasoner.process_record_checklist(checklist_path)
        
        # Determine output path if not provided
        if not output_path:
            output_path = os.path.join("output", "record_items_prompts.json")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to list for easier readability
        output_list = list(items_with_prompts.values())
        
        # Save results to JSON file
        with open(output_path, "w") as f:
            json.dump(output_list, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print(f"RECORD CHECKLIST PROCESSING RESULTS")
        print("="*80)
        
        total_items = len(output_list)
        strobe_items = sum(1 for item in output_list if ".0." in item["item_id"])
        record_items = sum(1 for item in output_list if not ".0." in item["item_id"])
        
        print(f"\nExtracted {total_items} total items:")
        print(f"  STROBE base items: {strobe_items}")
        print(f"  RECORD extension items: {record_items}")
        
        # Print by category
        categories = {}
        for item in output_list:
            category = item.get("category", "Unknown")
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        print("\nItems by category:")
        for category, count in categories.items():
            print(f"  {category}: {count}")
        
        print(f"\nPrompts generated: {len(output_list)}")
        print(f"Output saved to: {output_path}")
        print("\n" + "="*80)
        
        return 0
    except Exception as e:
        logger.error(f"Error processing checklist {checklist_path}: {str(e)}", exc_info=True)
        return 1

def find_record_checklist():
    """
    Find the RECORD checklist in the guidelines directory.
    
    Returns:
        Path to RECORD checklist, or None if not found
    """
    record_dir = os.path.join(GUIDELINES_PATH, "RECORD")
    if not os.path.exists(record_dir):
        return None
    
    # Look for files with "checklist" in the name
    for filename in os.listdir(record_dir):
        if filename.lower().endswith('.pdf') and 'checklist' in filename.lower():
            return os.path.join(record_dir, filename)
    
    # If no file with "checklist" in the name, return the first PDF
    for filename in os.listdir(record_dir):
        if filename.lower().endswith('.pdf'):
            return os.path.join(record_dir, filename)
    
    return None

def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(description="Process RECORD checklist to extract guideline items")
    
    parser.add_argument("--input", help="Path to the RECORD checklist PDF")
    parser.add_argument("--output", help="Path to save the output JSON (default: ./output/record_items_prompts.json)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        LOGGING["level"] = "DEBUG"
    
    # Check if input file is provided, otherwise try to find it
    checklist_path = args.input
    if not checklist_path:
        checklist_path = find_record_checklist()
        if not checklist_path:
            print("Error: No RECORD checklist specified and none found in the default location.")
            print(f"Please either specify a checklist with --input or add one to {os.path.join(GUIDELINES_PATH, 'RECORD')}")
            return 1
    
    return process_record_checklist(checklist_path, args.output)

if __name__ == "__main__":
    sys.exit(main())