#!/usr/bin/env python
import os
import random
import json
import glob
import datetime
import re

# Define paths
PAPERS_PATH = "data/Papers"
OUTPUT_PATH = "output"

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

def find_report_for_paper(paper_file):
    """Find the most recent report file for a paper"""
    paper_id = os.path.splitext(paper_file)[0]
    report_pattern = f"*_openai_claude_report_{paper_id}_Li-Paper.json"
    report_files = glob.glob(os.path.join(OUTPUT_PATH, report_pattern))
    
    if not report_files:
        print(f"No report file found for {paper_file}")
        return None
    
    # Get the most recent report file
    latest_report = max(report_files, key=os.path.getctime)
    print(f"Found report for {paper_file}: {os.path.basename(latest_report)}")
    
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
    # Create timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sample papers
    sampled_papers = sample_papers(30)
    
    # Save the list of sampled papers
    sampled_papers_file = os.path.join(OUTPUT_PATH, f"{timestamp}_sampled_papers.json")
    with open(sampled_papers_file, 'w') as f:
        json.dump(sampled_papers, f, indent=2)
    print(f"Sampled papers list saved to: {sampled_papers_file}")
    
    # Find and analyze reports for sampled papers
    results = []
    for paper in sampled_papers:
        report_file = find_report_for_paper(paper)
        if report_file:
            result = analyze_report(report_file)
            if result:
                results.append(result)
    
    # Identify papers that need review (agreement < 100%)
    papers_for_review = [r for r in results if r["agreement_percent"] < 100.0]
    
    # Generate summary report
    summary = {
        "timestamp": timestamp,
        "total_papers_sampled": len(sampled_papers),
        "papers_with_reports": len(results),
        "papers_with_100_percent_agreement": len(results) - len(papers_for_review),
        "papers_needing_review": len(papers_for_review),
        "papers_for_review": papers_for_review
    }
    
    # Save summary report
    summary_file = os.path.join(OUTPUT_PATH, f"{timestamp}_validation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print(f"VALIDATION SUMMARY")
    print("="*80)
    print(f"Total papers sampled: {summary['total_papers_sampled']}")
    print(f"Papers with existing reports: {summary['papers_with_reports']}")
    print(f"Papers with 100% agreement: {summary['papers_with_100_percent_agreement']}")
    print(f"Papers needing review: {summary['papers_needing_review']}")
    
    if papers_for_review:
        print("\nPapers that need manual review:")
        for i, paper in enumerate(papers_for_review, 1):
            print(f"{i}. {paper['paper']} - Agreement: {paper['agreement_percent']}%")
            print(f"   Total items: {paper['total_items']}, Agree: {paper['agree_with_extractor']}, Disagree: {paper['disagree_with_extractor']}, Unknown: {paper['unknown']}")
            print(f"   Report: {paper['report_file']}")
    
    print(f"\nSummary report saved to: {summary_file}")

if __name__ == "__main__":
    main()
