#!/usr/bin/env python
"""
Enhanced Validation Runner

This script provides a clean interface to run the enhanced validation pipeline
with support for both base and RAG agents.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.core.pipeline import EnhancedValidationPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("validation_runner")

def load_api_keys():
    """Load API keys from environment variables."""
    load_dotenv()
    
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'voyage': os.getenv('VOYAGE_API_KEY'),
        'deepseek': os.getenv('DEEPSEEK_API_KEY')
    }
    
    # Check for required keys
    required_keys = ['openai', 'anthropic']
    missing_keys = [key for key in required_keys if not api_keys.get(key)]
    
    if missing_keys:
        logger.error(f"Missing required API keys: {missing_keys}")
        logger.error("Please set the following environment variables:")
        for key in missing_keys:
            logger.error(f"  {key.upper()}_API_KEY")
        sys.exit(1)
    
    return api_keys

def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults."""
    default_config = {
        "reasoner": {
            "type": "base",
            "model": "o3-mini-2025-01-31"
        },
        "extractor": {
            "type": "base",
            "model": "gpt-4o",
            "custom_prompt": ""
        },
        "validator": {
            "type": "base",
            "model": "claude-3-5-sonnet-20241022",
            "custom_prompt": ""
        },
        "rag_config": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 5,
            "similarity_threshold": 0.7,
            "embedding_model": "voyage-3"
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge with defaults
            for key, value in file_config.items():
                if key in default_config:
                    if isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("Using default configuration")
    
    return default_config

def run_reasoner_only(pipeline: EnhancedValidationPipeline, checklist_type: str, output_dir: str):
    """Run only the reasoner to generate prompts."""
    logger.info(f"Running reasoner for {checklist_type} checklist")
    
    try:
        guideline_info = pipeline.process_guideline(checklist_type)
        
        # Save prompts
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prompts_filename = f"{timestamp}_reasoner_{checklist_type}_prompts.json"
        prompts_path = os.path.join(output_dir, prompts_filename)
        
        with open(prompts_path, "w") as f:
            json.dump(guideline_info["prompts"], f, indent=2)
        
        logger.info(f"Generated {len(guideline_info['prompts'])} prompts")
        logger.info(f"Prompts saved to: {prompts_path}")
        
        return prompts_path
        
    except Exception as e:
        logger.error(f"Error in reasoner mode: {e}")
        return None

def run_extractor_mode(pipeline: EnhancedValidationPipeline, paper_path: str, 
                      prompts_file: str, checklist_type: str):
    """Run extractor and validator using existing prompts."""
    logger.info(f"Running extractor mode for {os.path.basename(paper_path)}")
    
    try:
        # Load prompts
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        
        # Create mock guideline info
        guideline_info = {
            "guideline_type": checklist_type,
            "items": [],
            "prompts": prompts
        }
        
        # Create mock items based on prompts
        for item_id in prompts.keys():
            guideline_info["items"].append({
                "id": item_id,
                "description": f"Checklist item {item_id}",
                "category": "General"
            })
        
        # Process paper
        paper_info = pipeline.process_paper(paper_path, guideline_info)
        
        # Validate extraction
        validation_results = pipeline.validate_extraction(paper_info, guideline_info)
        
        # Generate final report
        final_report = pipeline.generate_report(paper_info, guideline_info, validation_results)
        
        # Save results
        pipeline._save_results(paper_path, guideline_info, paper_info, validation_results, final_report)
        
        logger.info("Extractor mode completed successfully")
        return final_report
        
    except Exception as e:
        logger.error(f"Error in extractor mode: {e}")
        return None

def run_full_pipeline(pipeline: EnhancedValidationPipeline, paper_path: str, checklist_type: str):
    """Run the complete validation pipeline."""
    logger.info(f"Running full pipeline for {os.path.basename(paper_path)} against {checklist_type}")
    
    try:
        final_report = pipeline.run_full_pipeline(paper_path, checklist_type)
        logger.info("Full pipeline completed successfully")
        return final_report
        
    except Exception as e:
        logger.error(f"Error in full pipeline: {e}")
        return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced LLM Validation Pipeline")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["full", "reasoner", "extractor"],
        default="full",
        help="Pipeline mode: full (complete pipeline), reasoner (generate prompts only), extractor (use existing prompts)"
    )
    
    # Input files
    parser.add_argument(
        "--paper",
        type=str,
        help="Path to the paper PDF file"
    )
    
    parser.add_argument(
        "--checklist",
        type=str,
        default="RECORD",
        help="Checklist type (e.g., RECORD, STROBE, Li-Paper)"
    )
    
    parser.add_argument(
        "--prompts",
        type=str,
        help="Path to prompts file (required for extractor mode)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ["full", "extractor"] and not args.paper:
        logger.error("Paper path is required for full and extractor modes")
        parser.print_help()
        sys.exit(1)
    
    if args.mode == "extractor" and not args.prompts:
        logger.error("Prompts file is required for extractor mode")
        parser.print_help()
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load API keys and configuration
    api_keys = load_api_keys()
    config = load_config(args.config)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Checklist: {args.checklist}")
    if args.paper:
        logger.info(f"  Paper: {args.paper}")
    if args.prompts:
        logger.info(f"  Prompts: {args.prompts}")
    logger.info(f"  Output: {args.output}")
    
    # Initialize pipeline
    try:
        pipeline = EnhancedValidationPipeline(api_keys, config)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Run based on mode
    if args.mode == "reasoner":
        result = run_reasoner_only(pipeline, args.checklist, args.output)
        if result:
            print(f"\nReasoner completed successfully!")
            print(f"Prompts saved to: {result}")
            print(f"You can now run extractor mode with: python {__file__} --mode extractor --paper <paper_path> --prompts {result} --checklist {args.checklist}")
        else:
            sys.exit(1)
    
    elif args.mode == "extractor":
        result = run_extractor_mode(pipeline, args.paper, args.prompts, args.checklist)
        if result:
            print(f"\nExtractor completed successfully!")
            print(f"Results saved to output directory")
        else:
            sys.exit(1)
    
    else:  # full mode
        result = run_full_pipeline(pipeline, args.paper, args.checklist)
        if result:
            print(f"\nFull pipeline completed successfully!")
            print(f"Results saved to output directory")
            
            # Print summary
            if 'validation_summary' in result:
                metrics = result['validation_summary']
                print(f"\nValidation Summary:")
                print(f"  Total items: {metrics.get('total_items', 0)}")
                print(f"  Agreement rate: {metrics.get('agreement_rate', 0):.1f}%")
                print(f"  Items for review: {metrics.get('items_for_review', 0)}")
        else:
            sys.exit(1)

if __name__ == "__main__":
    import datetime
    main()
