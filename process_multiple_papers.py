#!/usr/bin/env python
import os
import json
import subprocess
import datetime
import argparse
import time

# Define paths
OUTPUT_PATH = "output"

def process_papers(paper_list, start_index=0, max_papers=None, delay=60):
    """Process multiple papers in sequence"""
    papers_processed = 0
    papers_with_issues = []
    papers_needing_review = []
    
    # Create timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a log file
    log_file = os.path.join(OUTPUT_PATH, f"{timestamp}_processing_log.txt")
    
    with open(log_file, 'w') as log:
        log.write(f"Processing papers starting at index {start_index}\n")
        log.write(f"Timestamp: {timestamp}\n\n")
        
        # Process each paper
        for i, paper in enumerate(paper_list[start_index:], start=start_index):
            # Check if we've reached the maximum number of papers
            if max_papers and (i - start_index) >= max_papers:
                log.write(f"Reached maximum number of papers ({max_papers}). Stopping.\n")
                break
            
            log.write(f"Processing paper {i+1}/{len(paper_list)}: {paper}\n")
            print(f"Processing paper {i+1}/{len(paper_list)}: {paper}")
            
            # Run the test_single_paper.py script
            cmd = ["python", "test_single_paper.py", paper]
            
            try:
                # Run the command and capture output
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Write the output to the log
                log.write(f"Command output:\n{result.stdout}\n")
                if result.stderr:
                    log.write(f"Command error:\n{result.stderr}\n")
                
                # Check if the paper needs review
                if "This paper needs manual review" in result.stdout:
                    papers_needing_review.append(paper)
                    log.write(f"Paper {paper} needs manual review.\n")
                
                papers_processed += 1
                
            except Exception as e:
                log.write(f"Error processing paper {paper}: {str(e)}\n")
                papers_with_issues.append(paper)
            
            log.write("\n" + "-"*80 + "\n\n")
            
            # Add a delay between papers to avoid API rate limits
            if i < len(paper_list) - 1:
                log.write(f"Waiting {delay} seconds before processing next paper...\n\n")
                time.sleep(delay)
    
    # Generate summary
    summary = {
        "timestamp": timestamp,
        "papers_processed": papers_processed,
        "papers_with_issues": papers_with_issues,
        "papers_needing_review": papers_needing_review
    }
    
    # Save summary
    summary_file = os.path.join(OUTPUT_PATH, f"{timestamp}_processing_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print(f"PROCESSING SUMMARY")
    print("="*80)
    print(f"Papers processed: {papers_processed}")
    print(f"Papers with issues: {len(papers_with_issues)}")
    print(f"Papers needing review: {len(papers_needing_review)}")
    
    if papers_needing_review:
        print("\nPapers that need manual review:")
        for i, paper in enumerate(papers_needing_review, 1):
            print(f"{i}. {paper}")
    
    print(f"\nLog file: {log_file}")
    print(f"Summary file: {summary_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process multiple papers in sequence')
    parser.add_argument('--papers-file', required=True, help='JSON file containing the list of papers to process')
    parser.add_argument('--start-index', type=int, default=0, help='Index of the first paper to process (default: 0)')
    parser.add_argument('--max-papers', type=int, default=None, help='Maximum number of papers to process (default: all)')
    parser.add_argument('--delay', type=int, default=60, help='Delay in seconds between papers (default: 60)')
    args = parser.parse_args()
    
    # Load the papers file
    with open(args.papers_file, 'r') as f:
        paper_list = json.load(f)
    
    print(f"Loaded {len(paper_list)} papers from {args.papers_file}")
    
    # Process the papers
    process_papers(paper_list, args.start_index, args.max_papers, args.delay)

if __name__ == "__main__":
    main()
