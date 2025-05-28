#!/usr/bin/env python
# add_human_comparison.py

import os
import pandas as pd
import numpy as np
import argparse
from typing import Dict, Any, List
import re

def get_string_output(output):
    """
    Safely get string representation of output.
    
    Args:
        output: The output to convert to string
        
    Returns:
        String representation of output
    """
    if isinstance(output, dict):
        # If output is a dictionary, convert it to a string representation
        return str(output)
    elif output is None:
        return ""
    else:
        return str(output)

def outputs_agree(output1, output2):
    """
    Compare model outputs and determine if they agree.
    
    Args:
        output1: First output
        output2: Second output
        
    Returns:
        True if outputs agree, False otherwise
    """
    # Convert outputs to strings
    str_output1 = get_string_output(output1)
    str_output2 = get_string_output(output2)
    
    # Simple string comparison for now
    if not str_output1 or not str_output2:
        return False
    
    # Check for exact match
    if str_output1.lower() == str_output2.lower():
        return True
    
    # Check for 'unknown' or similar values
    unknown_patterns = ['unknown', 'not enough information', 'cannot determine', 'unclear']
    if any(pattern in str_output1.lower() for pattern in unknown_patterns) and \
       any(pattern in str_output2.lower() for pattern in unknown_patterns):
        return True
    
    # Check for yes/no agreement
    yes_patterns = ['yes', 'complies', 'compliant', 'fulfilled', 'reported']
    no_patterns = ['no', 'does not comply', 'non-compliant', 'not fulfilled', 'not reported']
    
    output1_yes = any(pattern in str_output1.lower() for pattern in yes_patterns)
    output1_no = any(pattern in str_output1.lower() for pattern in no_patterns)
    output2_yes = any(pattern in str_output2.lower() for pattern in yes_patterns)
    output2_no = any(pattern in str_output2.lower() for pattern in no_patterns)
    
    if (output1_yes and output2_yes) or (output1_no and output2_no):
        return True
    
    return False

def add_human_comparison(model_csv_path, human_excel_path, output_csv_path):
    """
    Add human comparison to model comparison CSV.
    
    Args:
        model_csv_path: Path to model comparison CSV
        human_excel_path: Path to human comparison Excel
        output_csv_path: Path to output CSV
    """
    print(f"Reading model comparison from {model_csv_path}")
    model_df = pd.read_csv(model_csv_path)
    
    print(f"Reading human comparison from {human_excel_path}")
    human_df = pd.read_excel(human_excel_path)
    
    # Check if human_df has the expected columns
    expected_columns = ['paper_id', 'item_id', 'human_output']
    missing_columns = [col for col in expected_columns if col not in human_df.columns]
    
    if missing_columns:
        print(f"Warning: Human comparison Excel is missing expected columns: {missing_columns}")
        print("Available columns:", human_df.columns.tolist())
        
        # Try to identify columns that might contain the required information
        paper_id_cols = [col for col in human_df.columns if 'paper' in col.lower() or 'id' in col.lower()]
        item_id_cols = [col for col in human_df.columns if 'item' in col.lower() or 'id' in col.lower()]
        human_output_cols = [col for col in human_df.columns if 'manual' in col.lower() or 'validation' in col.lower() or 'human' in col.lower()]
        
        # Map columns if possible
        column_mapping = {}
        if paper_id_cols and 'paper_id' not in human_df.columns:
            column_mapping[paper_id_cols[0]] = 'paper_id'
            print(f"Mapping {paper_id_cols[0]} to paper_id")
        
        if item_id_cols and 'item_id' not in human_df.columns:
            # Filter out paper_id columns that might also match item_id pattern
            item_id_cols = [col for col in item_id_cols if col not in paper_id_cols]
            if item_id_cols:
                column_mapping[item_id_cols[0]] = 'item_id'
                print(f"Mapping {item_id_cols[0]} to item_id")
        
        if human_output_cols and 'human_output' not in human_df.columns:
            column_mapping[human_output_cols[0]] = 'human_output'
            print(f"Mapping {human_output_cols[0]} to human_output")
        
        # Rename columns if mapping is available
        if column_mapping:
            human_df = human_df.rename(columns=column_mapping)
        else:
            print("Could not automatically map columns. Please check the Excel file structure.")
            return
    
    # Ensure paper_id and item_id are strings
    model_df['paper_id'] = model_df['paper_id'].astype(str)
    model_df['item_id'] = model_df['item_id'].astype(str)
    human_df['paper_id'] = human_df['paper_id'].astype(str)
    human_df['item_id'] = human_df['item_id'].astype(str)
    
    # Merge dataframes
    print("Merging model and human comparison data")
    merged_df = pd.merge(model_df, human_df, on=['paper_id', 'item_id'], how='left')
    
    # Calculate agreement between Claude and human
    print("Calculating agreement between Claude and human")
    merged_df['claude_human_agreement'] = merged_df.apply(
        lambda row: outputs_agree(row['claude_openai_output'], row['human_output']), axis=1
    )
    
    # Calculate agreement between OpenAI and human
    print("Calculating agreement between OpenAI and human")
    merged_df['openai_human_agreement'] = merged_df.apply(
        lambda row: outputs_agree(row['openai_claude_output'], row['human_output']), axis=1
    )
    
    # Calculate overall agreement rates
    claude_human_agreement_rate = merged_df['claude_human_agreement'].mean() * 100
    openai_human_agreement_rate = merged_df['openai_human_agreement'].mean() * 100
    
    print(f"Claude-Human Agreement Rate: {claude_human_agreement_rate:.2f}%")
    print(f"OpenAI-Human Agreement Rate: {openai_human_agreement_rate:.2f}%")
    
    # Save to CSV
    print(f"Saving results to {output_csv_path}")
    merged_df.to_csv(output_csv_path, index=False)
    
    # Create a summary CSV
    summary_df = pd.DataFrame([
        ['Claude-Human Agreement Rate (%)', claude_human_agreement_rate],
        ['OpenAI-Human Agreement Rate (%)', openai_human_agreement_rate]
    ], columns=['Metric', 'Value'])
    
    summary_path = os.path.splitext(output_csv_path)[0] + '_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    return merged_df, claude_human_agreement_rate, openai_human_agreement_rate

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Add human comparison to model comparison CSV")
    parser.add_argument("--model-csv", type=str, default="output/report_Li-22/model_answers_comparison.csv", help="Path to model comparison CSV")
    parser.add_argument("--human-excel", type=str, default="data/validation/manul_pulling_human_comparison.xlsx", help="Path to human comparison Excel")
    parser.add_argument("--output-csv", type=str, default="output/report_Li-22/model_human_comparison.csv", help="Path to output CSV")
    args = parser.parse_args()
    
    add_human_comparison(args.model_csv, args.human_excel, args.output_csv)

if __name__ == "__main__":
    main()
