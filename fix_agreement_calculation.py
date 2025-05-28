#!/usr/bin/env python

import os
import json
import pandas as pd
import glob
from collections import defaultdict

def analyze_reports_directory(reports_dir):
    # Load the model_answers_comparison.csv file
    comparison_file = os.path.join(reports_dir, 'model_answers_comparison.csv')
    if not os.path.exists(comparison_file):
        print(f"Error: Comparison file {comparison_file} not found")
        return
    
    # Load the CSV file
    df = pd.read_csv(comparison_file)
    
    # Calculate agreement statistics based on the 'agreement' column
    total_items = len(df)
    agree_count = df[df['agreement'] == 'Yes'].shape[0]
    disagree_count = df[df['agreement'] == 'No'].shape[0]
    
    # Calculate percentages
    agreement_rate = (agree_count / total_items) * 100 if total_items > 0 else 0
    disagreement_rate = (disagree_count / total_items) * 100 if total_items > 0 else 0
    
    print("\n=== Agreement Analysis Based on model_answers_comparison.csv ===")
    print(f"Total items: {total_items}")
    print(f"Items with agreement: {agree_count}")
    print(f"Items with disagreement: {disagree_count}")
    print(f"Agreement rate: {agreement_rate:.2f}%")
    print(f"Disagreement rate: {disagreement_rate:.2f}%")
    
    # Now analyze individual report files
    report_files = glob.glob(os.path.join(reports_dir, '*report*.json'))
    print(f"\nFound {len(report_files)} report files")
    
    # Collect validation summaries from all reports
    all_summaries = []
    for report_file in report_files:
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
                if 'validation_summary' in data:
                    summary = data['validation_summary']
                    paper_id = data.get('paper', '').replace('.pdf', '')
                    summary['paper_id'] = paper_id
                    summary['file'] = os.path.basename(report_file)
                    all_summaries.append(summary)
        except Exception as e:
            print(f"Error reading {report_file}: {e}")
    
    print(f"Loaded validation summaries from {len(all_summaries)} reports")
    
    # Calculate aggregate statistics from individual reports
    if all_summaries:
        total_items_reports = sum(summary.get('total_items', 0) for summary in all_summaries)
        total_agree = sum(summary.get('agree_with_extractor', 0) for summary in all_summaries)
        total_disagree = sum(summary.get('disagree_with_extractor', 0) for summary in all_summaries)
        total_unknown = sum(summary.get('unknown', 0) for summary in all_summaries)
        
        # Calculate percentages
        agree_percent = (total_agree / total_items_reports) * 100 if total_items_reports > 0 else 0
        disagree_percent = (total_disagree / total_items_reports) * 100 if total_items_reports > 0 else 0
        unknown_percent = (total_unknown / total_items_reports) * 100 if total_items_reports > 0 else 0
        
        print("\n=== Aggregate Statistics from Individual Reports ===")
        print(f"Total items: {total_items_reports}")
        print(f"Items with agreement: {total_agree}")
        print(f"Items with disagreement: {total_disagree}")
        print(f"Unknown items: {total_unknown}")
        print(f"Agreement rate: {agree_percent:.2f}%")
        print(f"Disagreement rate: {disagree_percent:.2f}%")
        print(f"Unknown rate: {unknown_percent:.2f}%")
        
        # Compare with model_answers_comparison.csv
        print("\n=== Comparison ===")
        print(f"Difference in total items: {total_items - total_items_reports}")
        print(f"Difference in agreement rate: {agreement_rate - agree_percent:.2f}%")
    
    return {
        'csv_analysis': {
            'total_items': total_items,
            'agree_count': agree_count,
            'disagree_count': disagree_count,
            'agreement_rate': agreement_rate,
            'disagreement_rate': disagreement_rate
        },
        'report_analysis': {
            'total_items': total_items_reports,
            'agree_count': total_agree,
            'disagree_count': total_disagree,
            'unknown_count': total_unknown,
            'agreement_rate': agree_percent,
            'disagreement_rate': disagree_percent,
            'unknown_rate': unknown_percent
        } if all_summaries else None
    }

if __name__ == "__main__":
    reports_dir = "/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/reports_direct_20250325"
    analyze_reports_directory(reports_dir)
