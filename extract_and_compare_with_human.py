#!/usr/bin/env python

import os
import json
import pandas as pd
import re
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract model answers and compare with human validation')
    parser.add_argument('--reports_dir', type=str, required=True, help='Directory containing the report files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
    parser.add_argument('--human_comparison', type=str, required=True, help='Path to human comparison Excel file')
    args = parser.parse_args()

    # Set plotting style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

    print(f"Starting extraction and comparison from reports in {args.reports_dir}...")

    # Function to extract data from a report file
    def extract_report_data(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract paper ID
            paper_id = data.get('paper', '').replace('.pdf', '')
            
            # Extract model information
            model_info = data.get('model_info', {})
            
            # Extract items data
            items = data.get('items', {})
            
            # Determine configuration based on model info
            config = 'unknown'
            extractor = model_info.get('extractor', '')
            validator = model_info.get('validator', '')
            
            if extractor and validator:
                if ('openai' in extractor.lower() or 'gpt' in extractor.lower()):
                    config = 'openai_extractor'
                elif ('claude' in extractor.lower() or 'anthropic' in extractor.lower()):
                    config = 'claude_extractor'
            
            return {
                'paper_id': paper_id,
                'config': config,
                'model_info': model_info,
                'items': items,
                'file_path': file_path
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    # Load all report data
    all_reports = []
    report_files = [f for f in os.listdir(args.reports_dir) if f.endswith('.json') and ('report' in f.lower() or 'Report' in f)]
    print(f"Found {len(report_files)} report files")

    # Process each report file
    for file in report_files:
        file_path = os.path.join(args.reports_dir, file)
        report_data = extract_report_data(file_path)
        if report_data:
            all_reports.append(report_data)

    print(f"Loaded {len(all_reports)} reports")

    # Dictionary to store results by paper ID and item ID
    results = {}

    # Process all report data
    for report in all_reports:
        paper_id = report['paper_id']
        config = report['config']
        items = report['items']
        
        # Initialize paper entry in results dictionary if not exists
        if paper_id not in results:
            results[paper_id] = {}
        
        # Extract final_correct_answer for each item
        for item_id, item_data in items.items():
            final_answer = item_data.get('final_correct_answer', '')
            extractor_answer = item_data.get('extractor_correct_answer', '')
            validator_answer = item_data.get('validator_correct_answer', '')
            description = item_data.get('description', '')
            
            # Initialize item entry in paper's dictionary if not exists
            if item_id not in results[paper_id]:
                results[paper_id][item_id] = {
                    'description': description,
                    'openai_extractor': '',
                    'claude_extractor': '',
                    'final_answer': ''
                }
            
            # Store answers by configuration
            if config == 'openai_extractor':
                results[paper_id][item_id]['openai_extractor'] = extractor_answer
            elif config == 'claude_extractor':
                results[paper_id][item_id]['claude_extractor'] = extractor_answer
            
            # Store final answer if not already set
            if not results[paper_id][item_id]['final_answer'] and final_answer:
                results[paper_id][item_id]['final_answer'] = final_answer

    # Convert results to DataFrame for comparison
    rows = []
    for paper_id, paper_data in results.items():
        for item_id, item_data in paper_data.items():
            openai_answer = item_data.get('openai_extractor', '')
            claude_answer = item_data.get('claude_extractor', '')
            final_answer = item_data.get('final_answer', '')
            description = item_data.get('description', '')
            
            # Create a row for the DataFrame
            row = {
                'paper_id': paper_id,
                'item_id': item_id,
                'description': description,
                'openai_answer': openai_answer,
                'claude_answer': claude_answer,
                'final_answer': final_answer,
                'agreement': 'Yes' if openai_answer == claude_answer else 'No'
            }
            rows.append(row)

    # Create DataFrame
    model_df = pd.DataFrame(rows)

    # Sort by paper_id and item_id
    try:
        # Try to convert paper_id and item_id to numeric for proper sorting
        model_df['paper_id'] = pd.to_numeric(model_df['paper_id'])
        model_df['item_id'] = pd.to_numeric(model_df['item_id'])
    except ValueError:
        # If conversion fails, sort as strings
        pass

    model_df = model_df.sort_values(['paper_id', 'item_id'])

    # Save model comparison to CSV
    model_csv_path = os.path.join(args.output_dir, 'model_answers_comparison_new.csv')
    model_df.to_csv(model_csv_path, index=False)
    print(f"Saved model answers to {model_csv_path}")

    # Load human comparison data
    print(f"Loading human comparison data from {args.human_comparison}")
    try:
        human_df = pd.read_excel(args.human_comparison)
        print(f"Loaded human comparison data with shape: {human_df.shape}")
        
        # Print column names for debugging
        print("Columns in human comparison data:")
        for col in human_df.columns:
            print(f"  - {col}")
        
        # Merge model and human data
        # Assuming human_df has columns 'paper_id', 'item_id', and 'human_answer'
        # Adjust column names as needed based on the actual human data structure
        
        # Create a copy of human_df with standardized column names
        human_df_copy = human_df.copy()
        
        # Print all columns for debugging
        print("Columns in human comparison data:")
        for col in human_df_copy.columns:
            print(f"  - {col}")
        
        # Rename the manual validation column to human_answer
        if 'manual_validation' in human_df_copy.columns:
            human_df_copy.rename(columns={'manual_validation': 'human_answer'}, inplace=True)
            print("Renamed 'manual_validation' to 'human_answer'")
        else:
            print("WARNING: 'manual_validation' column not found. Using available columns.")
            # Try to find a suitable column for human answers
            potential_columns = ['manual_validation', 'Human Answer', 'Human Validation', 'Manual Validation']
            for col in potential_columns:
                if col in human_df_copy.columns:
                    human_df_copy.rename(columns={col: 'human_answer'}, inplace=True)
                    print(f"Renamed '{col}' to 'human_answer'")
                    break
            else:
                # If no suitable column found, create a dummy column
                print("No suitable column found for human answers. Creating a dummy column.")
                human_df_copy['human_answer'] = 'Unknown'
        
        # Ensure paper_id and item_id are in the right format for merging
        try:
            human_df_copy['paper_id'] = human_df_copy['paper_id'].astype(str)
            human_df_copy['item_id'] = human_df_copy['item_id'].astype(str)
            model_df['paper_id'] = model_df['paper_id'].astype(str)
            model_df['item_id'] = model_df['item_id'].astype(str)
        except Exception as e:
            print(f"Error converting data types: {e}")
        
        # Merge the dataframes
        print("Merging model and human data...")
        merged_df = pd.merge(
            model_df, 
            human_df_copy[['paper_id', 'item_id', 'human_answer']], 
            on=['paper_id', 'item_id'], 
            how='left'
        )
        print(f"Merged data shape: {merged_df.shape}")
        
        # Calculate agreement with human
        merged_df['agree_with_human_openai'] = merged_df.apply(
            lambda row: 'Yes' if str(row['openai_answer']).lower() == str(row['human_answer']).lower() else 'No', 
            axis=1
        )
        
        merged_df['agree_with_human_claude'] = merged_df.apply(
            lambda row: 'Yes' if str(row['claude_answer']).lower() == str(row['human_answer']).lower() else 'No', 
            axis=1
        )
        
        merged_df['agree_with_human_final'] = merged_df.apply(
            lambda row: 'Yes' if str(row['final_answer']).lower() == str(row['human_answer']).lower() else 'No', 
            axis=1
        )
        
        # Save merged data to CSV
        merged_csv_path = os.path.join(args.output_dir, 'model_human_comparison.csv')
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Saved merged model-human comparison to {merged_csv_path}")
        
        # Calculate agreement statistics
        total_items = len(merged_df)
        openai_human_agree = (merged_df['agree_with_human_openai'] == 'Yes').sum()
        claude_human_agree = (merged_df['agree_with_human_claude'] == 'Yes').sum()
        final_human_agree = (merged_df['agree_with_human_final'] == 'Yes').sum()
        
        openai_human_rate = (openai_human_agree / total_items) * 100 if total_items > 0 else 0
        claude_human_rate = (claude_human_agree / total_items) * 100 if total_items > 0 else 0
        final_human_rate = (final_human_agree / total_items) * 100 if total_items > 0 else 0
        
        print("\n=== Human-Model Agreement Statistics ===")
        print(f"Total items: {total_items}")
        print(f"OpenAI-Human agreement: {openai_human_agree} items ({openai_human_rate:.2f}%)")
        print(f"Claude-Human agreement: {claude_human_agree} items ({claude_human_rate:.2f}%)")
        print(f"Final-Human agreement: {final_human_agree} items ({final_human_rate:.2f}%)")
        
        # Create visualizations
        
        # 1. Human-Model Agreement Rates
        plt.figure(figsize=(10, 6))
        agreement_data = {
            'Model': ['OpenAI', 'Claude', 'Final Answer'],
            'Agreement Rate (%)': [openai_human_rate, claude_human_rate, final_human_rate]
        }
        agreement_df = pd.DataFrame(agreement_data)
        
        ax = sns.barplot(x='Model', y='Agreement Rate (%)', data=agreement_df, palette='viridis')
        
        # Add value labels on top of bars
        for i, v in enumerate(agreement_df['Agreement Rate (%)']):
            ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
        
        plt.title('Human-Model Agreement Rates', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'human_model_agreement_rates.png'), dpi=300, bbox_inches='tight')
        print(f"Saved human_model_agreement_rates.png to {args.output_dir}")
        
        # 2. Agreement by paper
        paper_agreement = merged_df.groupby('paper_id').apply(
            lambda x: {
                'openai_rate': (x['agree_with_human_openai'] == 'Yes').mean() * 100,
                'claude_rate': (x['agree_with_human_claude'] == 'Yes').mean() * 100,
                'final_rate': (x['agree_with_human_final'] == 'Yes').mean() * 100,
                'count': len(x)
            }
        ).reset_index()
        
        # Convert dictionary to separate columns
        paper_agreement['openai_rate'] = paper_agreement[0].apply(lambda x: x['openai_rate'])
        paper_agreement['claude_rate'] = paper_agreement[0].apply(lambda x: x['claude_rate'])
        paper_agreement['final_rate'] = paper_agreement[0].apply(lambda x: x['final_rate'])
        paper_agreement['count'] = paper_agreement[0].apply(lambda x: x['count'])
        paper_agreement.drop(columns=[0], inplace=True)
        
        # Sort by final agreement rate
        paper_agreement_sorted = paper_agreement.sort_values('final_rate', ascending=False)
        
        # Save paper agreement to CSV
        paper_agreement_sorted.to_csv(os.path.join(args.output_dir, 'paper_human_agreement.csv'), index=False)
        print(f"Saved paper_human_agreement.csv to {args.output_dir}")
        
        # 3. Top 10 papers by agreement rate
        plt.figure(figsize=(14, 8))
        top_papers = paper_agreement_sorted.head(10)
        
        # Reshape data for grouped bar plot
        top_papers_melted = pd.melt(
            top_papers, 
            id_vars=['paper_id', 'count'], 
            value_vars=['openai_rate', 'claude_rate', 'final_rate'],
            var_name='Model', value_name='Agreement Rate (%)'
        )
        
        # Map model names to more readable labels
        top_papers_melted['Model'] = top_papers_melted['Model'].map({
            'openai_rate': 'OpenAI', 
            'claude_rate': 'Claude', 
            'final_rate': 'Final'
        })
        
        ax = sns.barplot(x='paper_id', y='Agreement Rate (%)', hue='Model', data=top_papers_melted, palette='viridis')
        
        plt.title('Top 10 Papers by Human-Model Agreement Rate', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Paper ID', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 105)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'top_papers_human_agreement.png'), dpi=300, bbox_inches='tight')
        print(f"Saved top_papers_human_agreement.png to {args.output_dir}")
        
        # 4. Bottom 10 papers by agreement rate
        plt.figure(figsize=(14, 8))
        bottom_papers = paper_agreement_sorted.tail(10).sort_values('final_rate')
        
        # Reshape data for grouped bar plot
        bottom_papers_melted = pd.melt(
            bottom_papers, 
            id_vars=['paper_id', 'count'], 
            value_vars=['openai_rate', 'claude_rate', 'final_rate'],
            var_name='Model', value_name='Agreement Rate (%)'
        )
        
        # Map model names to more readable labels
        bottom_papers_melted['Model'] = bottom_papers_melted['Model'].map({
            'openai_rate': 'OpenAI', 
            'claude_rate': 'Claude', 
            'final_rate': 'Final'
        })
        
        ax = sns.barplot(x='paper_id', y='Agreement Rate (%)', hue='Model', data=bottom_papers_melted, palette='viridis')
        
        plt.title('Bottom 10 Papers by Human-Model Agreement Rate', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Paper ID', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 105)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'bottom_papers_human_agreement.png'), dpi=300, bbox_inches='tight')
        print(f"Saved bottom_papers_human_agreement.png to {args.output_dir}")
        
        # 5. Agreement by item
        item_agreement = merged_df.groupby('item_id').apply(
            lambda x: {
                'openai_rate': (x['agree_with_human_openai'] == 'Yes').mean() * 100,
                'claude_rate': (x['agree_with_human_claude'] == 'Yes').mean() * 100,
                'final_rate': (x['agree_with_human_final'] == 'Yes').mean() * 100,
                'count': len(x),
                'description': x['description'].iloc[0] if not x['description'].empty else f"Item {x.name}"
            }
        ).reset_index()
        
        # Convert dictionary to separate columns
        item_agreement['openai_rate'] = item_agreement[0].apply(lambda x: x['openai_rate'])
        item_agreement['claude_rate'] = item_agreement[0].apply(lambda x: x['claude_rate'])
        item_agreement['final_rate'] = item_agreement[0].apply(lambda x: x['final_rate'])
        item_agreement['count'] = item_agreement[0].apply(lambda x: x['count'])
        item_agreement['description'] = item_agreement[0].apply(lambda x: x['description'])
        item_agreement.drop(columns=[0], inplace=True)
        
        # Sort by final agreement rate
        item_agreement_sorted = item_agreement.sort_values('final_rate')
        
        # Save item agreement to CSV
        item_agreement_sorted.to_csv(os.path.join(args.output_dir, 'item_human_agreement.csv'), index=False)
        print(f"Saved item_human_agreement.csv to {args.output_dir}")
        
        # 6. Most disagreed items
        plt.figure(figsize=(14, 8))
        most_disagreed_items = item_agreement_sorted.head(10)
        
        # Reshape data for grouped bar plot
        most_disagreed_items_melted = pd.melt(
            most_disagreed_items, 
            id_vars=['item_id', 'count', 'description'], 
            value_vars=['openai_rate', 'claude_rate', 'final_rate'],
            var_name='Model', value_name='Agreement Rate (%)'
        )
        
        # Map model names to more readable labels
        most_disagreed_items_melted['Model'] = most_disagreed_items_melted['Model'].map({
            'openai_rate': 'OpenAI', 
            'claude_rate': 'Claude', 
            'final_rate': 'Final'
        })
        
        ax = sns.barplot(x='item_id', y='Agreement Rate (%)', hue='Model', data=most_disagreed_items_melted, palette='viridis')
        
        plt.title('Most Disagreed Items (Human-Model)', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Item ID', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 105)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'most_disagreed_items_human.png'), dpi=300, bbox_inches='tight')
        print(f"Saved most_disagreed_items_human.png to {args.output_dir}")
        
        # Generate a detailed analysis summary in Markdown format
        with open(os.path.join(args.output_dir, 'human_model_comparison_summary.md'), 'w') as f:
            f.write("# Human-Model Comparison Summary\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"- **Total Items Analyzed**: {total_items}\n")
            f.write(f"- **OpenAI-Human Agreement**: {openai_human_agree} items ({openai_human_rate:.2f}%)\n")
            f.write(f"- **Claude-Human Agreement**: {claude_human_agree} items ({claude_human_rate:.2f}%)\n")
            f.write(f"- **Final-Human Agreement**: {final_human_agree} items ({final_human_rate:.2f}%)\n\n")
            
            f.write("## Top 5 Papers by Human-Model Agreement Rate\n\n")
            f.write("| Paper ID | OpenAI Rate (%) | Claude Rate (%) | Final Rate (%) | Count |\n")
            f.write("|----------|----------------|----------------|----------------|-------|\n")
            for _, row in paper_agreement_sorted.head(5).iterrows():
                f.write(f"| {row['paper_id']} | {row['openai_rate']:.2f} | {row['claude_rate']:.2f} | {row['final_rate']:.2f} | {row['count']} |\n")
            f.write("\n")
            
            f.write("## Bottom 5 Papers by Human-Model Agreement Rate\n\n")
            f.write("| Paper ID | OpenAI Rate (%) | Claude Rate (%) | Final Rate (%) | Count |\n")
            f.write("|----------|----------------|----------------|----------------|-------|\n")
            for _, row in paper_agreement_sorted.tail(5).iterrows():
                f.write(f"| {row['paper_id']} | {row['openai_rate']:.2f} | {row['claude_rate']:.2f} | {row['final_rate']:.2f} | {row['count']} |\n")
            f.write("\n")
            
            f.write("## Most Disagreed Items (Human-Model)\n\n")
            f.write("| Item ID | Description | OpenAI Rate (%) | Claude Rate (%) | Final Rate (%) | Count |\n")
            f.write("|---------|-------------|----------------|----------------|----------------|-------|\n")
            for _, row in item_agreement_sorted.head(5).iterrows():
                # Truncate description if too long
                description = row['description']
                if len(description) > 50:
                    description = description[:47] + "..."
                f.write(f"| {row['item_id']} | {description} | {row['openai_rate']:.2f} | {row['claude_rate']:.2f} | {row['final_rate']:.2f} | {row['count']} |\n")
            f.write("\n")
            
            f.write("## Visualizations\n\n")
            f.write("The following visualizations have been generated:\n\n")
            f.write("1. `human_model_agreement_rates.png`: Bar chart showing agreement rates between human and different models\n")
            f.write("2. `top_papers_human_agreement.png`: Bar chart showing top 10 papers by human-model agreement rate\n")
            f.write("3. `bottom_papers_human_agreement.png`: Bar chart showing bottom 10 papers by human-model agreement rate\n")
            f.write("4. `most_disagreed_items_human.png`: Bar chart showing the most disagreed items between human and models\n")
        
        print(f"Generated human_model_comparison_summary.md in {args.output_dir}")
        
    except Exception as e:
        print(f"Error processing human comparison data: {e}")
        import traceback
        traceback.print_exc()

    print("\nExtraction and comparison complete!")

if __name__ == "__main__":
    main()
