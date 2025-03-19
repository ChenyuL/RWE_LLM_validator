#!/usr/bin/env python
import os
import random
import json
import subprocess
import datetime
import glob
import re
import argparse

# Define paths
PAPERS_PATH = "data/Papers"
OUTPUT_PATH = "output"
PROMPT_FILE = "output/20250318_133652_openai_reasoner_Li-Paper_prompts.json"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Sample papers and run validation tests using Claude as extractor and OpenAI as validator')
parser.add_argument('--num-samples', type=int, default=30, help='Number of papers to sample (default: 30)')
parser.add_argument('--test-limit', type=int, default=None, help='Limit the number of papers to test (for debugging)')
parser.add_argument('--dry-run', action='store_true', help='Only sample papers without running tests')
args = parser.parse_args()

def sample_papers(num_samples=30):
    """Randomly sample papers from the Papers directory"""
    # Get all PDF files
    pdf_files = [f for f in os.listdir(PAPERS_PATH) if f.endswith('.pdf')]
    
    # Check if we have enough papers
    if len(pdf_files) < num_samples:
        print(f"Warning: Only {len(pdf_files)} papers available, using all of them")
        return pdf_files
    
    # Randomly sample papers
    sampled_papers = random.sample(pdf_files, num_samples)
    
    print(f"Randomly sampled {len(sampled_papers)} papers:")
    for i, paper in enumerate(sampled_papers, 1):
        print(f"{i}. {paper}")
    
    return sampled_papers

def run_test_on_paper(paper_file, prompt_file):
    """Run the test on a single paper using the extractor mode"""
    paper_path = os.path.join(PAPERS_PATH, paper_file)
    
    # Command to run the test
    cmd = [
        "python", "test_record_validation_claude_openai_fixed.py",  # Using the new script with Claude as extractor
        "--mode", "extractor",
        "--prompts", prompt_file,
        "--paper", paper_path,
        "--checklist", "Li-Paper",
        "--config", "model_config.json"  # Use the model configuration file
    ]
    
    print(f"\nRunning test on {paper_file}...")
    subprocess.run(cmd)
    
    # Return the timestamp of the most recent report file for this paper
    paper_id = os.path.splitext(paper_file)[0]
    report_pattern = f"*_claude_openai_report_{paper_id}_Li-Paper.json"  # Looking for claude_openai_report files
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
    
    # Get model information
    model_info = report.get("model_info", {})
    extractor_model = model_info.get("extractor", "claude")
    validator_model = model_info.get("validator", "openai")
    
    result = {
        "paper": paper_name,
        "agreement_percent": agreement_percent,
        "report_file": os.path.basename(report_file),
        "extractor_model": extractor_model,
        "validator_model": validator_model,
        "total_items": validation_summary.get("total_items", 0),
        "agree_with_extractor": validation_summary.get("agree_with_extractor", 0),
        "disagree_with_extractor": validation_summary.get("disagree_with_extractor", 0),
        "unknown": validation_summary.get("unknown", 0)
    }
    
    return result

def main():
    # Create timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sample papers
    sampled_papers = sample_papers(args.num_samples)
    
    # Save the list of sampled papers
    sampled_papers_file = os.path.join(OUTPUT_PATH, f"{timestamp}_sampled_papers_claude_openai.json")
    with open(sampled_papers_file, 'w') as f:
        json.dump(sampled_papers, f, indent=2)
    print(f"Sampled papers list saved to: {sampled_papers_file}")
    
    # If dry run, exit here
    if args.dry_run:
        print("Dry run completed. Exiting without running tests.")
        return
    
    # Limit the number of papers to test if specified
    if args.test_limit and args.test_limit < len(sampled_papers):
        print(f"Limiting tests to {args.test_limit} papers (out of {len(sampled_papers)} sampled)")
        test_papers = sampled_papers[:args.test_limit]
    else:
        test_papers = sampled_papers
    
    # Run tests and collect results
    results = []
    for paper in test_papers:
        report_file = run_test_on_paper(paper, PROMPT_FILE)
        result = analyze_report(report_file)
        if result:
            results.append(result)
    
    # Identify papers that need review (agreement < 100%)
    papers_for_review = [r for r in results if r["agreement_percent"] < 100.0]
    
    # Generate summary report
    summary = {
        "timestamp": timestamp,
        "total_papers_tested": len(results),
        "papers_with_100_percent_agreement": len(results) - len(papers_for_review),
        "papers_needing_review": len(papers_for_review),
        "papers_for_review": papers_for_review,
        "model_configuration": {
            "extractor": "claude-3-5-sonnet-20241022",
            "validator": "gpt-4o"
        }
    }
    
    # Save summary report
    summary_file = os.path.join(OUTPUT_PATH, f"{timestamp}_validation_summary_claude_openai.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print(f"VALIDATION SUMMARY (Claude-3.5-Sonnet as Extractor, GPT-4o as Validator)")
    print("="*80)
    print(f"Total papers tested: {summary['total_papers_tested']}")
    print(f"Papers with 100% agreement: {summary['papers_with_100_percent_agreement']}")
    print(f"Papers needing review: {summary['papers_needing_review']}")
    
    if papers_for_review:
        print("\nPapers that need manual review:")
        for i, paper in enumerate(papers_for_review, 1):
            print(f"{i}. {paper['paper']} - Agreement: {paper['agreement_percent']}% - Report: {paper['report_file']}")
            print(f"   Extractor: {paper['extractor_model']}, Validator: {paper['validator_model']}")
    
    print(f"\nSummary report saved to: {summary_file}")

if __name__ == "__main__":
    main()
