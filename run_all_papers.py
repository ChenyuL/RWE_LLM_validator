#!/usr/bin/env python
"""
Script to run validation on all papers in the data/Papers directory.
"""

import os
import sys
import argparse
import subprocess
import json
import datetime
import shutil
from pathlib import Path

# Define paths
PAPERS_DIR = os.path.join("data", "Papers")
OUTPUT_DIR = "output"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "paper_results")

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def get_all_papers():
    """Get all PDF files in the papers directory."""
    papers = []
    for file in os.listdir(PAPERS_DIR):
        if file.lower().endswith('.pdf'):
            papers.append(os.path.join(PAPERS_DIR, file))
    return papers

def run_validation(paper_path, mode="full", prompts_file=None):
    """Run validation on a single paper."""
    print(f"\n{'='*80}")
    print(f"Processing paper: {os.path.basename(paper_path)}")
    print(f"{'='*80}")
    
    # Build the command
    cmd = ["python", "test_record_validation_openai_claude_fixed.py"]
    
    if mode != "full":
        cmd.extend(["--mode", mode])
    
    if prompts_file:
        cmd.extend(["--prompts", prompts_file])
    
    cmd.extend(["--paper", paper_path])
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing paper {os.path.basename(paper_path)}: {e}")
        return False

def organize_results(paper_path):
    """Organize results for a paper into its own directory."""
    paper_basename = os.path.basename(paper_path)
    pubmed_id = os.path.splitext(paper_basename)[0]
    if '.' in pubmed_id:
        pubmed_id = pubmed_id.split('.')[0]
    
    # Create a directory for this paper's results
    paper_results_dir = os.path.join(RESULTS_DIR, pubmed_id)
    os.makedirs(paper_results_dir, exist_ok=True)
    
    # Find the most recent results for this paper
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    result_files = []
    
    for file in os.listdir(OUTPUT_DIR):
        if pubmed_id in file and timestamp in file:
            result_files.append(os.path.join(OUTPUT_DIR, file))
    
    # Move the results to the paper's directory
    for file in result_files:
        shutil.copy2(file, paper_results_dir)
    
    print(f"Results for {pubmed_id} organized in {paper_results_dir}")
    
    # Return the path to the full record checklist
    checklist_files = [f for f in result_files if "full_record_checklist" in f]
    if checklist_files:
        return os.path.join(paper_results_dir, os.path.basename(checklist_files[0]))
    return None

def generate_summary(results):
    """Generate a summary of all papers processed."""
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_papers": len(results),
        "successful": sum(1 for r in results.values() if r["success"]),
        "papers": results
    }
    
    summary_file = os.path.join(RESULTS_DIR, f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total papers processed: {summary['total_papers']}")
    print(f"Successfully processed: {summary['successful']}")
    print(f"Failed: {summary['total_papers'] - summary['successful']}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Run validation on all papers in the data/Papers directory.")
    parser.add_argument("--mode", choices=["full", "reasoner", "extractor"], default="full",
                      help="Mode to run: full (default), reasoner (LLM1 only), or extractor (LLM2+LLM3 using existing prompts)")
    parser.add_argument("--prompts", type=str, help="Path to prompts file (required for extractor mode)")
    parser.add_argument("--paper", type=str, help="Path to specific paper to process (optional)")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Get papers to process
    if args.paper:
        papers = [args.paper]
    else:
        papers = get_all_papers()
    
    if not papers:
        print("No papers found in the data/Papers directory.")
        return
    
    # Check if extractor mode is selected but no prompts file is provided
    if args.mode == "extractor" and not args.prompts:
        print("Error: Prompts file is required for extractor mode.")
        parser.print_help()
        return
    
    # Process each paper
    results = {}
    for paper in papers:
        paper_basename = os.path.basename(paper)
        pubmed_id = os.path.splitext(paper_basename)[0]
        if '.' in pubmed_id:
            pubmed_id = pubmed_id.split('.')[0]
        
        success = run_validation(paper, args.mode, args.prompts)
        
        if success:
            checklist_path = organize_results(paper)
            results[pubmed_id] = {
                "success": True,
                "paper_path": paper,
                "checklist_path": checklist_path
            }
        else:
            results[pubmed_id] = {
                "success": False,
                "paper_path": paper,
                "error": "Validation failed"
            }
    
    # Generate summary
    generate_summary(results)
    
    # Run cleanup script
    print("\nRunning cleanup script...")
    try:
        subprocess.run(["./cleanup_output.sh"], check=True)
        print("Cleanup completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running cleanup script: {e}")
    except FileNotFoundError:
        print("Cleanup script not found. Make sure cleanup_output.sh is executable.")

if __name__ == "__main__":
    main()
