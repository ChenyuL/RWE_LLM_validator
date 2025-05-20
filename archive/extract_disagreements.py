#!/usr/bin/env python
import os
import json
import glob
import datetime

# Define paths
OUTPUT_PATH = "output"

def find_disagreements_in_report(report_file):
    """Find disagreements between extractor and validator in a report file"""
    if not os.path.exists(report_file):
        return None
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    paper_name = report.get("paper", "Unknown")
    checklist = report.get("checklist", "Unknown")
    items = report.get("items", {})
    
    disagreements = []
    
    for item_id, item_data in items.items():
        compliance = item_data.get("compliance", "")
        if compliance == "do not agree with extractor":
            disagreement = {
                "item_id": item_id,
                "description": item_data.get("description", ""),
                "extractor_answer": item_data.get("correct_answer", ""),
                "validator_reasoning": item_data.get("reasoning", ""),
                "evidence": item_data.get("evidence", [])
            }
            disagreements.append(disagreement)
    
    if disagreements:
        return {
            "paper": paper_name,
            "checklist": checklist,
            "report_file": os.path.basename(report_file),
            "disagreements": disagreements
        }
    
    return None

def extract_all_disagreements():
    """Extract disagreements from all RECORD validation reports"""
    # Find all RECORD report files
    report_pattern = f"*_openai_claude_report_*_RECORD.json"
    report_files = glob.glob(os.path.join(OUTPUT_PATH, report_pattern))
    
    if not report_files:
        print("No RECORD validation reports found.")
        return None
    
    # Analyze each report
    papers_with_disagreements = []
    for report_file in report_files:
        disagreements = find_disagreements_in_report(report_file)
        if disagreements:
            papers_with_disagreements.append(disagreements)
    
    # Generate summary
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save disagreements to JSON file
    if papers_with_disagreements:
        disagreements_file = os.path.join(OUTPUT_PATH, f"{timestamp}_record_disagreements.json")
        with open(disagreements_file, 'w') as f:
            json.dump(papers_with_disagreements, f, indent=2)
        
        # Generate markdown report
        generate_markdown_report(papers_with_disagreements, timestamp)
        
        print(f"\nFound disagreements in {len(papers_with_disagreements)} papers.")
        print(f"Disagreements saved to: {disagreements_file}")
    else:
        print("\nNo disagreements found in any of the papers.")
    
    return papers_with_disagreements

def generate_markdown_report(papers_with_disagreements, timestamp):
    """Generate a detailed markdown report of disagreements"""
    # Create the markdown report
    report_content = f"# RECORD Validation Disagreements Report\n\n"
    report_content += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report_content += f"## Summary\n"
    report_content += f"- Total papers with disagreements: {len(papers_with_disagreements)}\n\n"
    
    for paper in papers_with_disagreements:
        report_content += f"## Paper: {paper['paper']}\n"
        report_content += f"- Checklist: {paper['checklist']}\n"
        report_content += f"- Report file: {paper['report_file']}\n\n"
        
        report_content += f"### Disagreements\n\n"
        for disagreement in paper['disagreements']:
            report_content += f"#### Item {disagreement['item_id']}: {disagreement['description']}\n\n"
            report_content += f"**Extractor's Answer:**\n{disagreement['extractor_answer']}\n\n"
            report_content += f"**Validator's Reasoning:**\n{disagreement['validator_reasoning']}\n\n"
            
            if disagreement['evidence']:
                report_content += f"**Evidence:**\n"
                for evidence in disagreement['evidence']:
                    if 'quote' in evidence and 'location' in evidence:
                        report_content += f"- \"{evidence['quote']}\" ({evidence['location']})\n"
                report_content += "\n"
            
            report_content += "---\n\n"
    
    # Save the markdown report
    report_file = os.path.join(OUTPUT_PATH, f"{timestamp}_record_disagreements_report.md")
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"Detailed disagreements report saved to: {report_file}")

if __name__ == "__main__":
    extract_all_disagreements()
