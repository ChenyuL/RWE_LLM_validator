#!/usr/bin/env python
import os
import json
import subprocess
import datetime
import glob
import time
import argparse

# Define paths
OUTPUT_PATH = "output"

def find_report_for_paper(paper_file):
    """Find the most recent report file for a paper"""
    paper_id = os.path.splitext(paper_file)[0]
    report_pattern = f"*_openai_claude_report_{paper_id}_RECORD.json"
    report_files = glob.glob(os.path.join(OUTPUT_PATH, report_pattern))
    
    if not report_files:
        return None
    
    # Get the most recent report file
    latest_report = max(report_files, key=os.path.getctime)
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

def process_papers(paper_list, delay=60):
    """Process papers that don't have report files yet"""
    # Create timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a log file
    log_file = os.path.join(OUTPUT_PATH, f"{timestamp}_record_processing_log.txt")
    
    # Check which papers already have reports
    existing_results = []
    papers_to_process = []
    
    print("Checking for existing RECORD reports...")
    for paper in paper_list:
        report_file = find_report_for_paper(paper)
        if report_file:
            print(f"Found existing RECORD report for {paper}: {os.path.basename(report_file)}")
            result = analyze_report(report_file)
            if result:
                existing_results.append(result)
        else:
            papers_to_process.append(paper)
    
    print(f"\nFound {len(existing_results)} papers with existing RECORD reports")
    print(f"Need to process {len(papers_to_process)} papers\n")
    
    # Process papers that don't have reports yet
    new_results = []
    papers_with_issues = []
    
    with open(log_file, 'w') as log:
        log.write(f"Processing papers with RECORD checklist\n")
        log.write(f"Timestamp: {timestamp}\n\n")
        log.write(f"Found {len(existing_results)} papers with existing RECORD reports\n")
        log.write(f"Need to process {len(papers_to_process)} papers\n\n")
        
        # Process each paper
        for i, paper in enumerate(papers_to_process):
            log.write(f"Processing paper {i+1}/{len(papers_to_process)}: {paper}\n")
            print(f"Processing paper {i+1}/{len(papers_to_process)}: {paper}")
            
            # Run the test_single_paper_record.py script
            cmd = ["python", "test_single_paper_record.py", paper]
            
            try:
                # Run the command and capture output
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Write the output to the log
                log.write(f"Command output:\n{result.stdout}\n")
                if result.stderr:
                    log.write(f"Command error:\n{result.stderr}\n")
                
                # Find the report file
                report_file = find_report_for_paper(paper)
                if report_file:
                    result = analyze_report(report_file)
                    if result:
                        new_results.append(result)
                
            except Exception as e:
                log.write(f"Error processing paper {paper}: {str(e)}\n")
                papers_with_issues.append(paper)
            
            log.write("\n" + "-"*80 + "\n\n")
            
            # Add a delay between papers to avoid API rate limits
            if i < len(papers_to_process) - 1:
                log.write(f"Waiting {delay} seconds before processing next paper...\n\n")
                time.sleep(delay)
    
    # Combine existing and new results
    all_results = existing_results + new_results
    
    # Identify papers that need review (agreement < 100%)
    papers_for_review = [r for r in all_results if r["agreement_percent"] < 100.0]
    
    # Generate summary report
    summary = {
        "timestamp": timestamp,
        "total_papers_sampled": len(paper_list),
        "papers_with_reports": len(all_results),
        "papers_with_100_percent_agreement": len(all_results) - len(papers_for_review),
        "papers_needing_review": len(papers_for_review),
        "papers_with_issues": len(papers_with_issues),
        "papers_for_review": papers_for_review,
        "papers_with_issues": papers_with_issues
    }
    
    # Save summary report
    summary_file = os.path.join(OUTPUT_PATH, f"{timestamp}_record_validation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save papers for review
    if papers_for_review:
        review_file = os.path.join(OUTPUT_PATH, f"{timestamp}_record_papers_for_review.json")
        with open(review_file, 'w') as f:
            json.dump(papers_for_review, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print(f"RECORD VALIDATION SUMMARY")
    print("="*80)
    print(f"Total papers sampled: {summary['total_papers_sampled']}")
    print(f"Papers with reports: {summary['papers_with_reports']}")
    print(f"Papers with 100% agreement: {summary['papers_with_100_percent_agreement']}")
    print(f"Papers needing review: {summary['papers_needing_review']}")
    print(f"Papers with processing issues: {summary['papers_with_issues']}")
    
    if papers_for_review:
        print("\nPapers that need manual review:")
        for i, paper in enumerate(papers_for_review, 1):
            print(f"{i}. {paper['paper']} - Agreement: {paper['agreement_percent']}%")
            print(f"   Total items: {paper['total_items']}, Agree: {paper['agree_with_extractor']}, Disagree: {paper['disagree_with_extractor']}, Unknown: {paper['unknown']}")
            print(f"   Report: {paper['report_file']}")
    
    if papers_with_issues:
        print("\nPapers with processing issues:")
        for i, paper in enumerate(papers_with_issues, 1):
            print(f"{i}. {paper}")
    
    print(f"\nLog file: {log_file}")
    print(f"Summary file: {summary_file}")
    
    # Generate a detailed markdown report
    generate_markdown_report(all_results, papers_with_issues, timestamp)
    
    return summary

def generate_markdown_report(results, papers_with_issues, timestamp):
    """Generate a detailed markdown report"""
    # Sort results by agreement percentage
    sorted_results = sorted(results, key=lambda x: x["agreement_percent"])
    
    # Identify papers that need review (agreement < 100%)
    papers_for_review = [r for r in sorted_results if r["agreement_percent"] < 100.0]
    papers_with_full_agreement = [r for r in sorted_results if r["agreement_percent"] == 100.0]
    
    # Create the markdown report
    report_content = f"# RECORD Validation Report\n\n"
    report_content += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report_content += f"## Summary\n"
    report_content += f"- Total papers analyzed: {len(results)}\n"
    report_content += f"- Papers with 100% agreement: {len(papers_with_full_agreement)}\n"
    report_content += f"- Papers needing review: {len(papers_for_review)}\n"
    report_content += f"- Papers with processing issues: {len(papers_with_issues)}\n\n"
    
    if papers_for_review:
        report_content += f"## Papers Needing Manual Review\n\n"
        for i, paper in enumerate(papers_for_review, 1):
            report_content += f"### {i}. {paper['paper']}\n"
            report_content += f"- Agreement rate: {paper['agreement_percent']}%\n"
            report_content += f"- Total items: {paper['total_items']}\n"
            report_content += f"- Agree with extractor: {paper['agree_with_extractor']}\n"
            report_content += f"- Disagree with extractor: {paper['disagree_with_extractor']}\n"
            report_content += f"- Unknown: {paper['unknown']}\n"
            report_content += f"- Report file: {paper['report_file']}\n\n"
    
    if papers_with_full_agreement:
        report_content += f"## Papers with 100% Agreement\n\n"
        for i, paper in enumerate(papers_with_full_agreement, 1):
            report_content += f"{i}. **{paper['paper']}** - Report file: {paper['report_file']}\n"
        report_content += "\n"
    
    if papers_with_issues:
        report_content += f"## Papers with Processing Issues\n\n"
        for i, paper in enumerate(papers_with_issues, 1):
            report_content += f"{i}. {paper}\n"
        report_content += "\n"
    
    # Save the markdown report
    report_file = os.path.join(OUTPUT_PATH, f"{timestamp}_record_validation_report.md")
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"Detailed report saved to: {report_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process papers with RECORD checklist')
    parser.add_argument('--papers-file', required=True, help='JSON file containing the list of papers to process')
    parser.add_argument('--delay', type=int, default=60, help='Delay in seconds between papers (default: 60)')
    args = parser.parse_args()
    
    # Load the papers file
    with open(args.papers_file, 'r') as f:
        paper_list = json.load(f)
    
    print(f"Loaded {len(paper_list)} papers from {args.papers_file}")
    
    # Process the papers
    process_papers(paper_list, args.delay)

if __name__ == "__main__":
    main()
