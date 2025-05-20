#!/usr/bin/env python
import os
import json
import glob
import datetime
import time
import argparse
import subprocess

# Define paths
OUTPUT_PATH = "output"

def count_processed_papers():
    """Count the number of papers that have been processed with RECORD checklist"""
    report_pattern = f"*_openai_claude_report_*_RECORD.json"
    report_files = glob.glob(os.path.join(OUTPUT_PATH, report_pattern))
    
    # Extract unique paper IDs
    paper_ids = set()
    for report_file in report_files:
        filename = os.path.basename(report_file)
        parts = filename.split('_')
        if len(parts) >= 5:
            paper_id = parts[4]  # Extract paper ID from filename
            paper_ids.add(paper_id)
    
    return len(paper_ids)

def monitor_validation_progress(interval=300, max_duration=None):
    """Monitor the progress of RECORD validation"""
    print(f"Starting RECORD validation monitoring...")
    print(f"Checking for new results every {interval} seconds")
    
    start_time = datetime.datetime.now()
    last_count = count_processed_papers()
    print(f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} - Initial count: {last_count} papers processed")
    
    try:
        while True:
            # Sleep for the specified interval
            time.sleep(interval)
            
            # Check if max duration has been reached
            if max_duration is not None:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                if elapsed > max_duration:
                    print(f"Maximum monitoring duration of {max_duration} seconds reached.")
                    break
            
            # Count processed papers
            current_count = count_processed_papers()
            
            # If new papers have been processed, run analysis
            if current_count > last_count:
                now = datetime.datetime.now()
                print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} - New results detected! {current_count - last_count} new papers processed.")
                print(f"Total papers processed: {current_count}")
                
                # Run analysis scripts
                print("Running analysis scripts...")
                try:
                    subprocess.run(["python", "analyze_record_results.py"], check=True)
                    subprocess.run(["python", "extract_disagreements.py"], check=True)
                    print("Analysis complete.")
                except subprocess.CalledProcessError as e:
                    print(f"Error running analysis scripts: {e}")
                
                last_count = current_count
            else:
                now = datetime.datetime.now()
                print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} - No new results. Total papers processed: {current_count}")
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    # Final analysis
    print("\nRunning final analysis...")
    try:
        subprocess.run(["python", "analyze_record_results.py"], check=True)
        subprocess.run(["python", "extract_disagreements.py"], check=True)
        print("Final analysis complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error running final analysis scripts: {e}")
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nMonitoring ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"Final count: {count_processed_papers()} papers processed")

def main():
    parser = argparse.ArgumentParser(description='Monitor RECORD validation progress')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds (default: 300)')
    parser.add_argument('--duration', type=int, help='Maximum monitoring duration in seconds (default: unlimited)')
    args = parser.parse_args()
    
    monitor_validation_progress(interval=args.interval, max_duration=args.duration)

if __name__ == "__main__":
    main()
