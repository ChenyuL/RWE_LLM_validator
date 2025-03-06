#!/usr/bin/env python
# test_record_validation.py

import os
import json
import logging
from src.framework import LLMValidationFramework
from src.config import API_KEYS, GUIDELINES_PATH, PAPERS_PATH, OUTPUT_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("record_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("record_test")

def run_test():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    logger.info("Initializing LLM Validation Framework")
    framework = LLMValidationFramework(API_KEYS)
    
    # Step 1: Process RECORD guidelines
    logger.info("Processing RECORD guidelines")
    guideline_info = framework.process_guideline("RECORD")
    
    # Save guideline info for inspection
    with open(os.path.join(OUTPUT_PATH, "record_guideline_info.json"), "w") as f:
        # Convert complex types to strings for JSON serialization
        simplified_info = {
            "guideline_type": guideline_info["guideline_type"],
            "items_count": len(guideline_info["items"]),
            "prompts_count": len(guideline_info["prompts"]),
            "sample_items": guideline_info["items"][:3] if len(guideline_info["items"]) > 3 else guideline_info["items"]
        }
        json.dump(simplified_info, f, indent=2)
    
    logger.info(f"Extracted {len(guideline_info['items'])} guideline items")
    logger.info(f"Generated {len(guideline_info['prompts'])} prompts")
    
    # Step 2: Process a sample paper against the guidelines
    # List available papers
    paper_files = [f for f in os.listdir(PAPERS_PATH) if f.endswith('.pdf')]
    
    if not paper_files:
        logger.error("No PDF papers found in papers directory")
        return
    
    # Select first paper for testing
    paper_path = os.path.join(PAPERS_PATH, paper_files[0])
    logger.info(f"Processing paper: {paper_path}")
    
    try:
        # Process the paper
        paper_info = framework.process_paper(paper_path, guideline_info)
        
        # Step 3: Validate the extraction
        logger.info("Validating extracted information")
        validation_results = framework.validate_extraction(paper_info, guideline_info)
        
        # Step 4: Generate final report
        logger.info("Generating final report")
        final_report = framework.generate_report(paper_info, guideline_info, validation_results)
        
        # Save the report
        report_path = os.path.join(OUTPUT_PATH, f"{os.path.basename(paper_path).replace('.pdf', '')}_report.json")
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print(f"VALIDATION RESULTS FOR {os.path.basename(paper_path)}")
        print("="*80)
        
        # Print validation summary
        if "validation_summary" in final_report:
            metrics = final_report["validation_summary"]
            print("\nValidation Metrics:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    print(f"  {metric_name}: {metric_value:.3f}")
                else:
                    print(f"  {metric_name}: {metric_value}")
        
        # Count compliance by type
        if "items" in final_report:
            compliances = {"yes": 0, "partial": 0, "no": 0, "unknown": 0}
            
            for item_id, item_data in final_report["items"].items():
                compliance = item_data.get("compliance", "unknown")
                if compliance in compliances:
                    compliances[compliance] += 1
                else:
                    compliances["unknown"] += 1
            
            print("\nCompliance Summary:")
            print(f"  Total Items: {len(final_report['items'])}")
            print(f"  Compliant: {compliances['yes']} items")
            print(f"  Partially Compliant: {compliances['partial']} items")
            print(f"  Non-Compliant: {compliances['no']} items")
            print(f"  Unknown: {compliances['unknown']} items")
            
            # Calculate compliance rate if we have items
            if len(final_report['items']) > 0:
                compliance_rate = (compliances['yes'] + (compliances['partial'] * 0.5)) / len(final_report['items']) * 100
                print(f"  Overall Compliance Rate: {compliance_rate:.1f}%")
        
        print("\n" + "="*80)
        
        logger.info(f"Report saved to {report_path}")
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing paper: {str(e)}", exc_info=True)

if __name__ == "__main__":
    run_test()