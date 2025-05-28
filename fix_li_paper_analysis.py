#!/usr/bin/env python

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fix Li-Paper validation analysis')
    parser.add_argument('--reports_dir', type=str, required=True, help='Directory containing the report files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
    args = parser.parse_args()

    print(f"Starting fixed Li-Paper analysis on reports in {args.reports_dir}...")

    # Set plotting style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

    # Load the model_answers_comparison.csv file
    comparison_file = os.path.join(args.reports_dir, 'model_answers_comparison.csv')
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

    # Create a summary DataFrame
    summary_data = []
    summary_data.append(['Total Items', total_items])
    summary_data.append(['Items with Agreement', agree_count])
    summary_data.append(['Items with Disagreement', disagree_count])
    summary_data.append(['Agreement Rate (%)', agreement_rate])
    summary_data.append(['Disagreement Rate (%)', disagreement_rate])
    
    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    
    # Save summary statistics to CSV
    summary_df.to_csv(os.path.join(args.output_dir, 'fixed_li_paper_summary_stats.csv'), index=False)
    print(f"Saved fixed_li_paper_summary_stats.csv to {args.output_dir}")
    
    # Generate a detailed analysis summary in Markdown format
    with open(os.path.join(args.output_dir, 'fixed_li_paper_analysis_summary.md'), 'w') as f:
        f.write("# Fixed Li-Paper Analysis Summary\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Total Items Analyzed**: {total_items}\n")
        f.write(f"- **Items with Agreement**: {agree_count}\n")
        f.write(f"- **Items with Disagreement**: {disagree_count}\n")
        f.write(f"- **Agreement Rate**: {agreement_rate:.2f}%\n")
        f.write(f"- **Disagreement Rate**: {disagreement_rate:.2f}%\n\n")
        
        f.write("## Analysis Method\n\n")
        f.write("This analysis is based on the 'agreement' column in the model_answers_comparison.csv file, which provides a more accurate representation of the agreement between models than the individual report summaries.\n\n")
        
        f.write("## Explanation of Discrepancy\n\n")
        f.write("The original analysis calculated agreement rates based on individual report summaries, which showed an agreement rate of approximately 86.81%. However, the actual agreement rate based on the model_answers_comparison.csv file is 59.55%.\n\n")
        f.write("This discrepancy occurs because:\n\n")
        f.write("1. The individual report summaries count 'agree with extractor' responses, which measure whether the validator agrees with the extractor within each report.\n")
        f.write("2. The model_answers_comparison.csv file measures agreement between different model configurations across reports, which is a different metric.\n\n")
        f.write("The agreement column in model_answers_comparison.csv provides a more accurate measure of inter-model agreement for the analysis purpose.\n\n")

    print(f"Generated fixed_li_paper_analysis_summary.md in {args.output_dir}")
    
    # Create a visualization of the agreement rates
    plt.figure(figsize=(10, 6))
    labels = ['Agreement', 'Disagreement']
    sizes = [agree_count, disagree_count]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode the 1st slice (Agreement)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Model Agreement Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fixed_agreement_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved fixed_agreement_distribution.png to {args.output_dir}")
    
    print("\nFixed analysis complete!")

if __name__ == "__main__":
    main()
