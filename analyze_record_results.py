#!/usr/bin/env python
import os
import json
import glob
import datetime

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

def analyze_all_record_reports():
    """Analyze all RECORD validation reports"""
    # Find all RECORD report files
    report_pattern = f"*_openai_claude_report_*_RECORD.json"
    report_files = glob.glob(os.path.join(OUTPUT_PATH, report_pattern))
    
    if not report_files:
        print("No RECORD validation reports found.")
        return None
    
    # Analyze each report
    results = []
    for report_file in report_files:
        result = analyze_report(report_file)
        if result:
            results.append(result)
    
    # Sort results by agreement percentage
    sorted_results = sorted(results, key=lambda x: x["agreement_percent"])
    
    # Identify papers that need review (agreement < 100%)
    papers_for_review = [r for r in sorted_results if r["agreement_percent"] < 100.0]
    papers_with_full_agreement = [r for r in sorted_results if r["agreement_percent"] == 100.0]
    
    # Generate summary
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_papers_analyzed": len(results),
        "papers_with_100_percent_agreement": len(papers_with_full_agreement),
        "papers_needing_review": len(papers_for_review),
        "papers_for_review": papers_for_review
    }
    
    # Save summary report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(OUTPUT_PATH, f"{timestamp}_record_analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save papers for review
    if papers_for_review:
        review_file = os.path.join(OUTPUT_PATH, f"{timestamp}_record_papers_for_review.json")
        with open(review_file, 'w') as f:
            json.dump(papers_for_review, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print(f"RECORD VALIDATION ANALYSIS")
    print("="*80)
    print(f"Total papers analyzed: {summary['total_papers_analyzed']}")
    print(f"Papers with 100% agreement: {summary['papers_with_100_percent_agreement']}")
    print(f"Papers needing review: {summary['papers_needing_review']}")
    
    if papers_for_review:
        print("\nPapers that need manual review:")
        for i, paper in enumerate(papers_for_review, 1):
            print(f"{i}. {paper['paper']} - Agreement: {paper['agreement_percent']}%")
            print(f"   Total items: {paper['total_items']}, Agree: {paper['agree_with_extractor']}, Disagree: {paper['disagree_with_extractor']}, Unknown: {paper['unknown']}")
            print(f"   Report: {paper['report_file']}")
    
    # Generate a detailed markdown report
    generate_markdown_report(sorted_results, papers_for_review, papers_with_full_agreement, timestamp)
    
    print(f"\nSummary file: {summary_file}")
    
    return summary

def generate_markdown_report(results, papers_for_review, papers_with_full_agreement, timestamp):
    """Generate a detailed markdown report"""
    # Create the markdown report
    report_content = f"# RECORD Validation Analysis Report\n\n"
    report_content += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report_content += f"## Summary\n"
    report_content += f"- Total papers analyzed: {len(results)}\n"
    report_content += f"- Papers with 100% agreement: {len(papers_with_full_agreement)}\n"
    report_content += f"- Papers needing review: {len(papers_for_review)}\n\n"
    
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
    
    # Save the markdown report
    report_file = os.path.join(OUTPUT_PATH, f"{timestamp}_record_analysis_report.md")
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"Detailed report saved to: {report_file}")

if __name__ == "__main__":
    analyze_all_record_reports()
