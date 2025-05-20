#!/usr/bin/env python
# extract_rag_model_answers.py
# Script to extract model answers from RAG reports into a CSV file

import os
import json
import pandas as pd
import re
import argparse
from pathlib import Path

def extract_model_info(filename):
    """
    Extract model information from the filename.
    """
    if "claude_extractor_openai_validator" in filename:
        extractor = "claude"
        validator = "openai"
    else:
        extractor = "openai"
        validator = "claude"
    
    return extractor, validator

def extract_paper_id(filename):
    """
    Extract paper ID from filename.
    """
    match = re.search(r'report_(\d+)_Li-Paper\.json$', filename)
    if match:
        return match.group(1)
    return None

def load_prompts(prompts_file):
    """
    Load prompts to get item descriptions.
    """
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    item_descriptions = {}
    for item_id, prompt in prompts.items():
        description_match = re.search(r'DESCRIPTION: (.*?)(\n|\r)', prompt)
        if description_match:
            item_descriptions[item_id] = description_match.group(1).strip()
        else:
            item_descriptions[item_id] = f"Li-Paper SOP item {item_id}"
    
    return item_descriptions

def main():
    parser = argparse.ArgumentParser(description='Extract model answers from RAG reports into a CSV file')
    parser.add_argument('--reports_dir', type=str, default='output/reports_rag', help='Directory containing RAG reports')
    parser.add_argument('--prompts_file', type=str, default='/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250319_220659_openai_reasoner_Li-Paper_prompts.json', help='Path to prompts file')
    parser.add_argument('--output_file', type=str, default='rag_model_answers_comparison.xlsx', help='Output Excel file')
    parser.add_argument('--csv_output_file', type=str, default='rag_model_answers_comparison.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Load prompts to get item descriptions
    print(f"Loading prompts from {args.prompts_file}")
    item_descriptions = load_prompts(args.prompts_file)
    print(f"Loaded {len(item_descriptions)} item descriptions")
    
    # Get all RAG report files
    reports_dir = args.reports_dir
    print(f"Scanning directory: {reports_dir}")
    files = os.listdir(reports_dir)
    report_files = [f for f in files if 'rag' in f.lower() and 'report' in f and f.endswith('_Li-Paper.json')]
    print(f"Found {len(report_files)} RAG report files")
    
    # Dictionary to store results by paper ID, item ID, and model
    results = {}
    
    # Process each report file
    for filename in report_files:
        file_path = os.path.join(reports_dir, filename)
        paper_id = extract_paper_id(filename)
        extractor, validator = extract_model_info(filename)
        
        if not paper_id:
            print(f"Skipping {filename}: Could not determine paper ID")
            continue
        
        # Load report data
        try:
            with open(file_path, 'r') as f:
                report = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        
        # Initialize paper entry in results dictionary if not exists
        if paper_id not in results:
            results[paper_id] = {}
        
        # Extract correct_answer for each item
        items = report.get('items', {})
        for item_id, item_data in items.items():
            correct_answer = item_data.get('correct_answer', '')
            
            # Initialize item entry in paper's dictionary if not exists
            if item_id not in results[paper_id]:
                results[paper_id][item_id] = {}
            
            # Store correct_answer by extractor type
            model_key = f"{extractor}_extractor"
            results[paper_id][item_id][model_key] = correct_answer
    
    # Convert results to DataFrame for Excel export
    rows = []
    for paper_id, paper_data in results.items():
        for item_id, item_data in paper_data.items():
            openai_answer = item_data.get('openai_extractor', '')
            claude_answer = item_data.get('claude_extractor', '')
            
            # Get item description
            item_name = item_descriptions.get(item_id, f"Li-Paper SOP item {item_id}")
            
            # Create a row for the DataFrame
            row = {
                'paper_id': paper_id,
                'item_id': item_id,
                'item_name': item_name,
                'openai_extractor_answer': openai_answer,
                'claude_extractor_answer': claude_answer,
                'manual_validation': ''  # Empty column for manual validation
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by paper_id and item_id
    try:
        # Try to convert paper_id and item_id to numeric for proper sorting
        df['paper_id'] = pd.to_numeric(df['paper_id'])
        df['item_id'] = pd.to_numeric(df['item_id'])
    except ValueError:
        # If conversion fails, sort as strings
        pass
    
    df = df.sort_values(['paper_id', 'item_id'])
    
    # Save to Excel
    df.to_excel(args.output_file, index=False)
    print(f"Saved model answers to {args.output_file}")
    
    # Also save to CSV with tab delimiter for easier processing
    df.to_csv(args.csv_output_file, index=False, sep='\t')
    print(f"Saved model answers to {args.csv_output_file} (tab-delimited)")
    
    print("Done!")

if __name__ == "__main__":
    main()
