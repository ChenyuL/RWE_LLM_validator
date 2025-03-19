#!/usr/bin/env python
import os
import sys
import json
import subprocess
import datetime
import glob
import argparse

# Define paths
PAPERS_PATH = "data/Papers"
OUTPUT_PATH = "output"
PROMPT_FILE = "output/20250318_133652_openai_reasoner_Li-Paper_prompts.json"

def run_test_on_paper(paper_file, prompt_file):
    """Run the test on a single paper using the extractor mode"""
    paper_path = os.path.join(PAPERS_PATH, paper_file)
    
    if not os.path.exists(paper_path):
        print(f"Error: Paper file not found: {paper_path}")
        return None
    
    if not os.path.exists(prompt_file):
        print(f"Error: Prompt file not found: {prompt_file}")
        return None
    
    # Command to run the test
    cmd = [
        "python", "test_record_validation_openai_claude_fixed.py",
        "--mode", "extractor",
        "--prompts", prompt_file,
        "--paper", paper_path,
        "--checklist", "Li-Paper"
    ]
    
    print(f"\nRunning test on {paper_file}...")
    subprocess.run(cmd)
    
    # Return the timestamp of the most recent report file for this paper
    paper_id = os.path.splitext(paper_file)[0]
    report_pattern = f"*_openai_claude_report_{paper_id}_Li-Paper.json"
    report_files = glob.glob(os.path.join(OUTPUT_PATH, report_pattern))
    
    if not report_files:
        print(f"Warning: No report file found for {paper_file}")
        return None
    
    # Get the most recent report file
    latest_report = max(report_files, key=os.path.getctime)
    print(f"Report generated: {os.path.basename(latest_report)}")
    
    return latest_report

def analyze_report(report_file):
    """Analyze the report file to check if agree_with_extractor_percent is 100.0%"""
    if not report_file or not os.path.exists(report_file):
        return None
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    paper_name = report.get("paper", "Unknown")
    validation_summary = report.get("validation_summary", {})
    
    # Get the agreement percentage
    agreement_percent = validation_summary.get("agree with extractor_percent", 0.0)
    
    result = {
        "paper": paper_name,
        "agreement_percent": agreement_percent,
        "report_file": os.path.basename(report_file),
        "total_items": validation_summary.get("total_items", 0),
        "agree_with_extractor": validation_summary.get("agree_with_extractor", 0),
        "disagree_with_extractor": validation_summary.get("disagree_with_extractor", 0),
        "unknown": validation_summary.get("unknown", 0)
    }
    
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run test on a single paper')
    parser.add_argument('paper_file', help='Name of the paper file (e.g., 12345678.pdf)')
    parser.add_argument('--prompt-file', default=PROMPT_FILE, help='Path to the prompt file')
    args = parser.parse_args()
    
    # Run test on the paper
    report_file = run_test_on_paper(args.paper_file, args.prompt_file)
    
    if report_file:
        # Analyze the report
        result = analyze_report(report_file)
        
        if result:
            # Print the result
            print("\n" + "="*80)
            print(f"VALIDATION RESULT FOR {args.paper_file}")
            print("="*80)
            print(f"Agreement percentage: {result['agreement_percent']}%")
            print(f"Total items: {result['total_items']}")
            print(f"Agree with extractor: {result['agree_with_extractor']}")
            print(f"Disagree with extractor: {result['disagree_with_extractor']}")
            print(f"Unknown: {result['unknown']}")
            print(f"Report file: {result['report_file']}")
            
            # Check if the paper needs review
            if result['agreement_percent'] < 100.0:
                print("\nThis paper needs manual review.")
            else:
                print("\nThis paper has 100% agreement and does not need manual review.")

if __name__ == "__main__":
    main()
