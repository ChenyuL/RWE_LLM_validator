#!/usr/bin/env python

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import re
import sys
import argparse
from collections import defaultdict

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze Li-Paper validation results')
parser.add_argument('--reports_dir', type=str, required=True, help='Directory containing the report files')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
parser.add_argument('--human_comparison', type=str, help='Path to human comparison Excel file')
args = parser.parse_args()

# Set default human comparison file if not provided
if not args.human_comparison:
    args.human_comparison = "/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/report_Li-22/model_human_comparison_finalized.xlsx"

# Set plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print(f"Starting Li-Paper analysis on reports in {args.reports_dir}...")

# Directory containing the paper results
base_dir = args.reports_dir
output_dir = args.output_dir

# Function to extract data from a report file
def extract_report_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract paper ID
        paper_id = data.get('paper', '').replace('.pdf', '')
        
        # Extract validation summary
        summary = data.get('validation_summary', {})
        
        # Extract model information if available
        model_info = data.get('model_info', {})
        if not model_info and 'model_info' in summary:
            model_info = summary.get('model_info', {})
        
        # Extract items data
        items = data.get('items', {})
        
        # Determine configuration based on filename and model info
        config = 'unknown'
        
        # First try to determine from filename
        if 'openai_claude' in file_path.lower():
            config = 'openai_claude'
        elif 'claude_openai' in file_path.lower():
            config = 'claude_openai'
        
        # If still unknown, try to determine from model info
        if config == 'unknown' and model_info:
            extractor = model_info.get('extractor', '')
            validator = model_info.get('validator', '')
            
            if extractor and validator:
                if ('openai' in extractor.lower() or 'gpt' in extractor.lower()) and ('claude' in validator.lower() or 'anthropic' in validator.lower()):
                    config = 'openai_claude'
                elif ('claude' in extractor.lower() or 'anthropic' in extractor.lower()) and ('openai' in validator.lower() or 'gpt' in validator.lower()):
                    config = 'claude_openai'
                elif 'rag' in file_path.lower():
                    config = 'rag'
        
        # If still unknown, try to determine from the first item's extractor and validator
        if config == 'unknown' and items:
            first_item_key = list(items.keys())[0]
            first_item = items[first_item_key]
            
            extractor_reasoning = first_item.get('extractor_reasoning', '')
            validator_reasoning = first_item.get('validator_reasoning', '')
            
            if extractor_reasoning and validator_reasoning:
                if ('openai' in extractor_reasoning.lower() or 'gpt' in extractor_reasoning.lower()) and \
                   ('claude' in validator_reasoning.lower() or 'anthropic' in validator_reasoning.lower()):
                    config = 'openai_claude'
                elif ('claude' in extractor_reasoning.lower() or 'anthropic' in extractor_reasoning.lower()) and \
                     ('openai' in validator_reasoning.lower() or 'gpt' in validator_reasoning.lower()):
                    config = 'claude_openai'
        
        return {
            'paper_id': paper_id,
            'config': config,
            'summary': summary,
            'model_info': model_info,
            'items': items,
            'file_path': file_path
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load all report data
all_reports = []

# Check if the directory exists
if not os.path.exists(base_dir):
    print(f"Error: Directory {base_dir} does not exist")
    exit(1)

# List all files in the directory
print(f"Scanning directory: {base_dir}")
files = os.listdir(base_dir)
report_files = [f for f in files if f.endswith('.json') and ('report' in f.lower() or 'Report' in f)]
print(f"Found {len(report_files)} report files")

# Process each report file
for file in report_files:
    file_path = os.path.join(base_dir, file)
    report_data = extract_report_data(file_path)
    if report_data:
        all_reports.append(report_data)

print(f"Loaded {len(all_reports)} reports")

# Create a DataFrame with summary information
summary_data = []

for report in all_reports:
    summary = report['summary']
    summary_data.append({
        'paper_id': report['paper_id'],
        'config': report['config'],
        'total_items': summary.get('total_items', 0),
        'agree_with_extractor': summary.get('agree_with_extractor', 0),
        'disagree_with_extractor': summary.get('disagree_with_extractor', 0),
        'unknown': summary.get('unknown', 0),
        'agreement_rate': summary.get('agreement_rate', 0),
        'extractor': report['model_info'].get('extractor', ''),
        'validator': report['model_info'].get('validator', '')
    })

summary_df = pd.DataFrame(summary_data)
print("Summary DataFrame created with shape:", summary_df.shape)

# Analysis 1: Agreement Rates by Configuration
print("\nAnalysis 1: Agreement Rates by Configuration")
avg_agreement = summary_df.groupby('config')['agreement_rate'].agg(['mean', 'std', 'count']).reset_index()
avg_agreement.columns = ['Configuration', 'Mean Agreement Rate (%)', 'Std Dev', 'Count']
print(avg_agreement)

# Visualize agreement rates by configuration
plt.figure(figsize=(12, 6))

# Bar plot
ax = sns.barplot(x='Configuration', y='Mean Agreement Rate (%)', data=avg_agreement, palette='viridis', hue='Configuration', legend=False)

# Add error bars
for i, row in avg_agreement.iterrows():
    ax.errorbar(i, row['Mean Agreement Rate (%)'], yerr=row['Std Dev'], color='black', capsize=10, linewidth=2)

# Add value labels on top of bars
for i, v in enumerate(avg_agreement['Mean Agreement Rate (%)']):
    ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')

plt.title('Mean Agreement Rate by Configuration', fontsize=16)
plt.ylabel('Agreement Rate (%)', fontsize=14)
plt.xlabel('Configuration', fontsize=14)
plt.ylim(0, 105)  # Set y-axis limit to accommodate error bars and labels
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'agreement_by_config.png'), dpi=300, bbox_inches='tight')
print(f"Saved agreement_by_config.png to {output_dir}")

# Analysis 2: Distribution of Agreement Rates
print("\nAnalysis 2: Distribution of Agreement Rates")
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='config', y='agreement_rate', data=summary_df, palette='viridis', inner='box', hue='config', legend=False)
ax = sns.swarmplot(x='config', y='agreement_rate', data=summary_df, color='white', edgecolor='black', size=8, alpha=0.7)

plt.title('Distribution of Agreement Rates by Configuration', fontsize=16)
plt.ylabel('Agreement Rate (%)', fontsize=14)
plt.xlabel('Configuration', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'agreement_distribution.png'), dpi=300, bbox_inches='tight')
print(f"Saved agreement_distribution.png to {output_dir}")

# Analysis 3: Statistical Comparison of Configurations
print("\nAnalysis 3: Statistical Comparison of Configurations")
# Only perform statistical comparison if we have both configurations
if 'openai_claude' in summary_df['config'].values and 'claude_openai' in summary_df['config'].values:
    openai_claude = summary_df[summary_df['config'] == 'openai_claude']['agreement_rate']
    claude_openai = summary_df[summary_df['config'] == 'claude_openai']['agreement_rate']

    # Perform Mann-Whitney U test (non-parametric test for independent samples)
    u_stat, p_value = stats.mannwhitneyu(openai_claude, claude_openai, alternative='two-sided')

    print(f"Mann-Whitney U test results:")
    print(f"U statistic: {u_stat}")
    print(f"p-value: {p_value}")
    print(f"Significant difference at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
else:
    print("Cannot perform statistical comparison: missing one or both configurations")
    u_stat, p_value = None, None

# Analysis 4: Agreement on Individual Checklist Items
print("\nAnalysis 4: Agreement on Individual Checklist Items")
item_data = []

for report in all_reports:
    for item_id, item in report['items'].items():
        item_data.append({
            'paper_id': report['paper_id'],
            'config': report['config'],
            'item_id': item_id,
            'compliance': item.get('compliance', ''),
            'correct_answer': item.get('correct_answer', ''),
            'description': item.get('description', '')
        })

item_df = pd.DataFrame(item_data)
print("Item DataFrame created with shape:", item_df.shape)

# Calculate agreement rate by checklist item
item_agreement = item_df.groupby(['item_id', 'config'])['compliance'].apply(
    lambda x: (x == 'agree with extractor').mean() * 100
).reset_index()
item_agreement.columns = ['Item ID', 'Configuration', 'Agreement Rate (%)']

# Pivot the data for easier comparison
item_agreement_pivot = item_agreement.pivot(index='Item ID', columns='Configuration', values='Agreement Rate (%)')
item_agreement_pivot.reset_index(inplace=True)

# Sort by item ID (handle mixed numeric and string IDs)
try:
    # Try to convert to numeric if all IDs are numeric
    item_agreement_pivot['Item ID'] = pd.to_numeric(item_agreement_pivot['Item ID'])
    item_agreement_pivot.sort_values('Item ID', inplace=True)
except ValueError:
    # If there are non-numeric IDs, sort as strings
    print("Warning: Some item IDs are not numeric. Sorting as strings.")
    item_agreement_pivot.sort_values('Item ID', inplace=True)

print("Item agreement pivot table created")

# Visualize agreement rates by checklist item
plt.figure(figsize=(15, 10))

# Get the available configurations
available_configs = [col for col in item_agreement_pivot.columns if col != 'Item ID']

# Reshape data for plotting
plot_data = pd.melt(item_agreement_pivot, id_vars=['Item ID'], 
                    value_vars=available_configs,
                    var_name='Configuration', value_name='Agreement Rate (%)')

# Create the grouped bar chart
ax = sns.barplot(x='Item ID', y='Agreement Rate (%)', hue='Configuration', data=plot_data, palette='viridis')

plt.title('Agreement Rate by Checklist Item and Configuration', fontsize=16)
plt.ylabel('Agreement Rate (%)', fontsize=14)
plt.xlabel('Checklist Item ID', fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 105)
plt.legend(title='Configuration', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'agreement_by_item.png'), dpi=300, bbox_inches='tight')
print(f"Saved agreement_by_item.png to {output_dir}")

# Analysis 5: Model Output Comparison
print("\nAnalysis 5: Model Output Comparison")

# Organize papers by their IDs
papers_by_id = defaultdict(list)
for report in all_reports:
    papers_by_id[report['paper_id']].append(report)

# Function to safely get string representation of output
def get_string_output(output):
    if isinstance(output, dict):
        # If output is a dictionary, convert it to a string representation
        return str(output)
    elif output is None:
        return ""
    else:
        return str(output)

# Function to compare model outputs and determine if they agree
def outputs_agree(output1, output2):
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

# Calculate model output agreement
output_agreement_data = []
model_comparison_data = []

# Find papers that have both openai_claude and claude_openai configurations
papers_with_both_configs = []
for paper_id, reports in papers_by_id.items():
    configs = set(report['config'] for report in reports)
    if 'openai_claude' in configs and 'claude_openai' in configs:
        papers_with_both_configs.append(paper_id)

print(f"Found {len(papers_with_both_configs)} papers with both configurations")

# For each paper with both configurations, compare the model outputs
for paper_id in papers_with_both_configs:
    reports = papers_by_id[paper_id]
    
    # Find the reports for each configuration
    openai_claude_report = next((r for r in reports if r['config'] == 'openai_claude'), None)
    claude_openai_report = next((r for r in reports if r['config'] == 'claude_openai'), None)
    
    if not openai_claude_report or not claude_openai_report:
        continue
    
    # Get the items from each report
    openai_claude_items = openai_claude_report['items']
    claude_openai_items = claude_openai_report['items']
    
    # Find the common item IDs
    common_item_ids = set(openai_claude_items.keys()) & set(claude_openai_items.keys())
    
    # Compare the model outputs for each common item
    for item_id in common_item_ids:
        # Get the correct answers from the extractor and validator
        openai_claude_output = openai_claude_items[item_id].get('extractor_correct_answer', '')
        claude_openai_output = claude_openai_items[item_id].get('extractor_correct_answer', '')
        
        try:
            agreement = outputs_agree(openai_claude_output, claude_openai_output)
            
            output_agreement_data.append({
                'paper_id': paper_id,
                'item_id': item_id,
                'openai_claude_output': get_string_output(openai_claude_output),
                'claude_openai_output': get_string_output(claude_openai_output),
                'models_agree': agreement
            })
            
            # Add to model comparison data for CSV export
            model_comparison_data.append({
                'paper_id': paper_id,
                'item_id': item_id,
                'openai_claude_output': get_string_output(openai_claude_output),
                'claude_openai_output': get_string_output(claude_openai_output),
                'agreement': 'Yes' if agreement else 'No',
                'item_description': openai_claude_items[item_id].get('description', '')
            })
        except Exception as e:
            print(f"Error processing agreement for paper {paper_id}, item {item_id}: {e}")
            print(f"OpenAI-Claude output: {type(openai_claude_output)}, {openai_claude_output}")
            print(f"Claude-OpenAI output: {type(claude_openai_output)}, {claude_openai_output}")

# Create a DataFrame with the output agreement data
if output_agreement_data:
    output_agreement_df = pd.DataFrame(output_agreement_data)
    print("Output agreement DataFrame created with shape:", output_agreement_df.shape)

    # Calculate overall model output agreement rate
    overall_output_agreement = output_agreement_df['models_agree'].mean() * 100
    print(f"Overall model output agreement rate: {overall_output_agreement:.2f}%")

    # Calculate model output agreement by paper
    paper_output_agreement = output_agreement_df.groupby('paper_id')['models_agree'].agg(['mean', 'count']).reset_index()
    paper_output_agreement.columns = ['Paper ID', 'Model Output Agreement Rate (%)', 'Count']
    paper_output_agreement['Model Output Agreement Rate (%)'] *= 100
    paper_output_agreement.sort_values('Model Output Agreement Rate (%)', ascending=False, inplace=True)
    
    print("Top 5 papers by model output agreement rate:")
    print(paper_output_agreement.head())

    # Visualize model output agreement by paper
    plt.figure(figsize=(15, 8))
    
    # Only include papers with at least 5 items
    papers_with_enough_items = paper_output_agreement[paper_output_agreement['Count'] >= 5]
    
    if not papers_with_enough_items.empty:
        ax = sns.barplot(x='Paper ID', y='Model Output Agreement Rate (%)', 
                        data=papers_with_enough_items, 
                        hue='Paper ID', palette='viridis', legend=False)

        plt.title('Model Output Agreement Rate by Paper', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Paper ID', fontsize=14)
        plt.xticks(rotation=90)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_output_agreement_by_paper.png'), dpi=300, bbox_inches='tight')
        print(f"Saved model_output_agreement_by_paper.png to {output_dir}")
    else:
        print("Not enough papers with sufficient items to create model_output_agreement_by_paper.png")

    # Calculate model output agreement by checklist item
    item_output_agreement = output_agreement_df.groupby('item_id')['models_agree'].agg(['mean', 'count']).reset_index()
    item_output_agreement.columns = ['Item ID', 'Model Output Agreement Rate (%)', 'Count']
    item_output_agreement['Model Output Agreement Rate (%)'] *= 100

    # Sort by item ID (handle mixed numeric and string IDs)
    try:
        # Try to convert to numeric if all IDs are numeric
        item_output_agreement['Item ID'] = pd.to_numeric(item_output_agreement['Item ID'])
        item_output_agreement.sort_values('Item ID', inplace=True)
    except ValueError:
        # If there are non-numeric IDs, sort as strings
        print("Warning: Some item IDs are not numeric. Sorting as strings.")
        item_output_agreement.sort_values('Item ID', inplace=True)

    print("Item output agreement table created")

    # Visualize model output agreement by checklist item
    plt.figure(figsize=(15, 8))
    
    # Only include items with at least 5 papers
    items_with_enough_papers = item_output_agreement[item_output_agreement['Count'] >= 5]
    
    if not items_with_enough_papers.empty:
        ax = sns.barplot(x='Item ID', y='Model Output Agreement Rate (%)', 
                        data=items_with_enough_papers, 
                        hue='Item ID', palette='viridis', legend=False)

        plt.title('Model Output Agreement Rate by Checklist Item', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Checklist Item ID', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_output_agreement_by_item.png'), dpi=300, bbox_inches='tight')
        print(f"Saved model_output_agreement_by_item.png to {output_dir}")
    else:
        print("Not enough items with sufficient papers to create model_output_agreement_by_item.png")

    # Analysis 6: Correlation Between Validator Agreement and Model Output Agreement
    print("\nAnalysis 6: Correlation Between Validator Agreement and Model Output Agreement")
    paper_agreement = summary_df.groupby(['paper_id', 'config'])['agreement_rate'].mean().reset_index()
    paper_agreement_pivot = paper_agreement.pivot(index='paper_id', columns='config', values='agreement_rate').reset_index()
    
    # Check if both configurations are available
    if 'openai_claude' in paper_agreement_pivot.columns and 'claude_openai' in paper_agreement_pivot.columns:
        paper_agreement_pivot.columns = ['paper_id', 'claude_openai_agreement', 'openai_claude_agreement']

        # Merge with model output agreement data
        merged_agreement = pd.merge(paper_agreement_pivot, 
                                    paper_output_agreement[['Paper ID', 'Model Output Agreement Rate (%)']], 
                                    left_on='paper_id', 
                                    right_on='Paper ID', 
                                    how='inner')
        
        print("Merged agreement DataFrame created with shape:", merged_agreement.shape)

        if len(merged_agreement) >= 5:  # Only calculate correlations if we have enough data points
            # Calculate correlation coefficients
            corr_openai_claude = merged_agreement['openai_claude_agreement'].corr(merged_agreement['Model Output Agreement Rate (%)'])
            corr_claude_openai = merged_agreement['claude_openai_agreement'].corr(merged_agreement['Model Output Agreement Rate (%)'])

            print(f"Correlation between OpenAI-Claude validator agreement and model output agreement: {corr_openai_claude:.4f}")
            print(f"Correlation between Claude-OpenAI validator agreement and model output agreement: {corr_claude_openai:.4f}")

            # Create scatter plots to visualize correlations
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            # Plot for OpenAI-Claude configuration
            sns.regplot(x='openai_claude_agreement', y='Model Output Agreement Rate (%)', 
                        data=merged_agreement, ax=axes[0], scatter_kws={'alpha':0.7, 's':100}, line_kws={'color':'red'})
            axes[0].set_title(f'OpenAI-Claude Validator Agreement vs. Model Output Agreement\nCorrelation: {corr_openai_claude:.4f}', fontsize=14)
            axes[0].set_xlabel('OpenAI-Claude Validator Agreement Rate (%)', fontsize=12)
            axes[0].set_ylabel('Model Output Agreement Rate (%)', fontsize=12)
            axes[0].grid(True, linestyle='--', alpha=0.7)

            # Plot for Claude-OpenAI configuration
            sns.regplot(x='claude_openai_agreement', y='Model Output Agreement Rate (%)', 
                        data=merged_agreement, ax=axes[1], scatter_kws={'alpha':0.7, 's':100}, line_kws={'color':'red'})
            axes[1].set_title(f'Claude-OpenAI Validator Agreement vs. Model Output Agreement\nCorrelation: {corr_claude_openai:.4f}', fontsize=14)
            axes[1].set_xlabel('Claude-OpenAI Validator Agreement Rate (%)', fontsize=12)
            axes[1].set_ylabel('Model Output Agreement Rate (%)', fontsize=12)
            axes[1].grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_plots.png'), dpi=300, bbox_inches='tight')
            # Also save to the root directory for easy access
            plt.savefig('correlation_plots.png', dpi=300, bbox_inches='tight')
            print(f"Saved correlation_plots.png to {output_dir} and root directory")
        else:
            print("Not enough data points to calculate correlations and create correlation plots")
            corr_openai_claude, corr_claude_openai = None, None
    else:
        print("Cannot perform correlation analysis: missing one or both configurations in pivot table")
        corr_openai_claude, corr_claude_openai = None, None
else:
    print("Cannot perform model output agreement analysis: no papers with both configurations")
    overall_output_agreement = None
    corr_openai_claude, corr_claude_openai = None, None

# Create a summary table
summary_stats_data = []

# Add basic metrics
summary_stats_data.append(['Number of Papers Analyzed', len(summary_df['paper_id'].unique())])
summary_stats_data.append(['Number of Reports Analyzed', len(all_reports)])

# Add configuration-specific metrics
for config in summary_df['config'].unique():
    config_data = summary_df[summary_df['config'] == config]
    mean_agreement = config_data['agreement_rate'].mean()
    summary_stats_data.append([f'{config} Mean Agreement Rate (%)', mean_agreement])

# Add model output agreement metrics if available
if 'overall_output_agreement' in locals() and overall_output_agreement is not None:
    summary_stats_data.append(['Model Output Agreement Rate (%)', overall_output_agreement])
    summary_stats_data.append(['Number of Papers with Both Configurations', len(papers_with_both_configs)])
    
if 'corr_openai_claude' in locals() and corr_openai_claude is not None:
    summary_stats_data.append(['Correlation: OpenAI-Claude vs. Model Output', corr_openai_claude])
    
if 'corr_claude_openai' in locals() and corr_claude_openai is not None:
    summary_stats_data.append(['Correlation: Claude-OpenAI vs. Model Output', corr_claude_openai])
    
if u_stat is not None and p_value is not None:
    summary_stats_data.append(['Significant Difference Between Configurations', 'Yes' if p_value < 0.05 else 'No'])

summary_stats = pd.DataFrame(summary_stats_data, columns=['Metric', 'Value'])

print("\nSummary Statistics:")
print(summary_stats)

# Save summary statistics to CSV
summary_stats.to_csv(os.path.join(output_dir, 'li_paper_summary_stats.csv'), index=False)
print(f"Saved li_paper_summary_stats.csv to {output_dir}")

# Save model comparison data to CSV
if model_comparison_data:
    model_comparison_df = pd.DataFrame(model_comparison_data)
    model_comparison_df.to_csv(os.path.join(output_dir, 'model_answers_comparison.csv'), index=False)
    # Also save to the root directory for easy access
    model_comparison_df.to_csv('model_answers_comparison.csv', index=False)
    print(f"Saved model_answers_comparison.csv to {output_dir} and root directory")

# Analysis 7: Human-Model Agreement Analysis
print("\nAnalysis 7: Human-Model Agreement Analysis")

# Check if human comparison file exists
if os.path.exists(args.human_comparison):
    print(f"Loading human comparison data from {args.human_comparison}")
    try:
        # Load human comparison data
        human_data = pd.read_excel(args.human_comparison)
        print(f"Loaded {len(human_data)} rows of human comparison data")
        
        # Create a dictionary to store model answers
        model_answers = {}
        
        # Extract model answers from all reports
        for report in all_reports:
            paper_id = report['paper_id']
            config = report['config']
            
            if config not in model_answers:
                model_answers[config] = {}
                
            if paper_id not in model_answers[config]:
                model_answers[config][paper_id] = {}
                
            for item_id, item in report['items'].items():
                final_answer = item.get('final_correct_answer', '')
                if final_answer:
                    model_answers[config][paper_id][item_id] = final_answer
        
        # Create a DataFrame to store human-model agreement data
        human_model_agreement_data = []
        
        # Process each row in human data
        for _, row in human_data.iterrows():
            paper_id = str(row.get('paper_id', ''))
            item_id = str(row.get('item_id', ''))
            human_answer = str(row.get('human_answer', '')).lower()
            
            # Use the existing agreement column if available
            if 'agreement' in row:
                # Convert to string and check if it's "Yes"
                agreement_value = str(row['agreement']).strip()
                agreement = agreement_value.lower() == "yes"
                
                # Print for debugging
                if paper_id == "20473188" and item_id == "1":
                    print(f"DEBUG: Paper {paper_id}, Item {item_id}, Agreement value: '{agreement_value}', Parsed as: {agreement}")
            else:
                # If agreement column doesn't exist, calculate it
                agreement = False
                
                # Print for debugging
                if paper_id == "20473188" and item_id == "1":
                    print(f"DEBUG: Paper {paper_id}, Item {item_id}, No agreement column found")
                
            if not paper_id or not item_id or not human_answer:
                continue
                
            # Check each model configuration
            for config in model_answers:
                if paper_id in model_answers[config] and item_id in model_answers[config][paper_id]:
                    model_answer = str(model_answers[config][paper_id][item_id]).lower()
                    
                    # Add to agreement data
                    human_model_agreement_data.append({
                        'paper_id': paper_id,
                        'item_id': item_id,
                        'config': config,
                        'human_answer': human_answer,
                        'model_answer': model_answer,
                        'agreement': agreement
                    })
        
        # Create DataFrame from agreement data
        if human_model_agreement_data:
            human_model_df = pd.DataFrame(human_model_agreement_data)
            print(f"Created human-model agreement DataFrame with {len(human_model_df)} rows")
            
            # Calculate overall agreement rate
            overall_human_model_agreement = human_model_df['agreement'].mean() * 100
            print(f"Overall human-model agreement rate: {overall_human_model_agreement:.2f}%")
            
            # Calculate agreement by configuration
            config_agreement = human_model_df.groupby('config')['agreement'].agg(['mean', 'count']).reset_index()
            config_agreement.columns = ['Configuration', 'Agreement Rate', 'Count']
            config_agreement['Agreement Rate'] *= 100
            print("\nHuman-Model Agreement by Configuration:")
            print(config_agreement)
            
            # Visualize agreement by configuration
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Configuration', y='Agreement Rate', data=config_agreement, palette='viridis')
            
            # Add value labels on top of bars
            for i, v in enumerate(config_agreement['Agreement Rate']):
                ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
                
            plt.title('Human-Model Agreement Rate by Configuration', fontsize=16)
            plt.ylabel('Agreement Rate (%)', fontsize=14)
            plt.xlabel('Configuration', fontsize=14)
            plt.ylim(0, 105)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'human_model_agreement_by_config.png'), dpi=300, bbox_inches='tight')
            print(f"Saved human_model_agreement_by_config.png to {output_dir}")
            
            # Calculate agreement by item
            item_agreement = human_model_df.groupby(['item_id', 'config'])['agreement'].agg(['mean', 'count']).reset_index()
            item_agreement.columns = ['Item ID', 'Configuration', 'Agreement Rate', 'Count']
            item_agreement['Agreement Rate'] *= 100
            
            # Pivot for easier visualization
            item_agreement_pivot = item_agreement.pivot(index='Item ID', columns='Configuration', values='Agreement Rate').reset_index()
            
            # Sort by item ID
            try:
                item_agreement_pivot['Item ID'] = pd.to_numeric(item_agreement_pivot['Item ID'])
                item_agreement_pivot.sort_values('Item ID', inplace=True)
            except ValueError:
                item_agreement_pivot.sort_values('Item ID', inplace=True)
                
            # Visualize agreement by item
            plt.figure(figsize=(15, 8))
            
            # Get available configurations
            available_configs = [col for col in item_agreement_pivot.columns if col != 'Item ID']
            
            # Reshape for plotting
            plot_data = pd.melt(item_agreement_pivot, id_vars=['Item ID'], 
                                value_vars=available_configs,
                                var_name='Configuration', value_name='Agreement Rate')
                                
            # Create grouped bar chart
            ax = sns.barplot(x='Item ID', y='Agreement Rate', hue='Configuration', data=plot_data, palette='viridis')
            
            plt.title('Human-Model Agreement Rate by Checklist Item', fontsize=16)
            plt.ylabel('Agreement Rate (%)', fontsize=14)
            plt.xlabel('Checklist Item ID', fontsize=14)
            plt.xticks(rotation=45)
            plt.ylim(0, 105)
            plt.legend(title='Configuration', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'human_model_agreement_by_item.png'), dpi=300, bbox_inches='tight')
            print(f"Saved human_model_agreement_by_item.png to {output_dir}")
            
            # Save human-model agreement data to CSV
            human_model_df.to_csv(os.path.join(output_dir, 'human_model_agreement.csv'), index=False)
            print(f"Saved human_model_agreement.csv to {output_dir}")
            
            # Calculate agreement by paper
            paper_agreement = human_model_df.groupby(['paper_id', 'config'])['agreement'].agg(['mean', 'count']).reset_index()
            paper_agreement.columns = ['Paper ID', 'Configuration', 'Agreement Rate', 'Count']
            paper_agreement['Agreement Rate'] *= 100
            
            # Save paper agreement to CSV
            paper_agreement.to_csv(os.path.join(output_dir, 'paper_human_model_agreement.csv'), index=False)
            print(f"Saved paper_human_model_agreement.csv to {output_dir}")
            
            # Print agreement rates for all papers
            print("\nHuman-Model Agreement by Paper:")
            print(f"Total papers with human comparison data: {len(paper_agreement['Paper ID'].unique())}")
            print(f"Total rows in human comparison data: {len(human_data)}")
            print(f"Total 'Yes' agreements in human comparison data: {human_data['agreement'].str.lower().eq('yes').sum()}")
            print(f"Agreement rate in raw data: {human_data['agreement'].str.lower().eq('yes').mean() * 100:.2f}%")
            
            for config in paper_agreement['Configuration'].unique():
                print(f"\nConfiguration: {config}")
                config_papers = paper_agreement[paper_agreement['Configuration'] == config].sort_values('Agreement Rate', ascending=False)
                print(f"Number of papers for {config}: {len(config_papers)}")
                print(f"Average agreement rate for {config}: {config_papers['Agreement Rate'].mean():.2f}%")
                for _, row in config_papers.iterrows():
                    print(f"Paper {row['Paper ID']}: {row['Agreement Rate']:.2f}% ({int(row['Count'])} items)")
            
            # Add human-model agreement to summary statistics
            summary_stats_data.append(['Overall Human-Model Agreement Rate (%)', overall_human_model_agreement])
            
            # Create a detailed CSV with all human-model comparisons
            detailed_comparison = []
            for _, row in human_model_df.iterrows():
                detailed_comparison.append({
                    'paper_id': row['paper_id'],
                    'item_id': row['item_id'],
                    'config': row['config'],
                    'human_answer': row['human_answer'],
                    'model_answer': row['model_answer'],
                    'agreement': 'Yes' if row['agreement'] else 'No'
                })
            
            detailed_df = pd.DataFrame(detailed_comparison)
            detailed_df.to_csv(os.path.join(output_dir, 'detailed_human_model_comparison.csv'), index=False)
            print(f"Saved detailed_human_model_comparison.csv to {output_dir}")
            
            # Add to markdown summary
            with open(os.path.join(output_dir, 'human_model_agreement_summary.md'), 'w') as f:
                f.write("# Human-Model Agreement Analysis\n\n")
                f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Overall Agreement\n\n")
                f.write(f"- **Overall Human-Model Agreement Rate**: {overall_human_model_agreement:.2f}%\n\n")
                
                f.write("## Agreement by Configuration\n\n")
                f.write("| Configuration | Agreement Rate (%) | Count |\n")
                f.write("|---------------|-------------------|-------|\n")
                for _, row in config_agreement.iterrows():
                    f.write(f"| {row['Configuration']} | {row['Agreement Rate']:.2f} | {row['Count']} |\n")
                f.write("\n")
                
                f.write("## Top 5 Items with Highest Agreement\n\n")
                top_items = item_agreement.sort_values('Agreement Rate', ascending=False).head(5)
                f.write("| Item ID | Configuration | Agreement Rate (%) | Count |\n")
                f.write("|---------|---------------|-------------------|-------|\n")
                for _, row in top_items.iterrows():
                    f.write(f"| {row['Item ID']} | {row['Configuration']} | {row['Agreement Rate']:.2f} | {row['Count']} |\n")
                f.write("\n")
                
                f.write("## Bottom 5 Items with Lowest Agreement\n\n")
                bottom_items = item_agreement.sort_values('Agreement Rate').head(5)
                f.write("| Item ID | Configuration | Agreement Rate (%) | Count |\n")
                f.write("|---------|---------------|-------------------|-------|\n")
                for _, row in bottom_items.iterrows():
                    f.write(f"| {row['Item ID']} | {row['Configuration']} | {row['Agreement Rate']:.2f} | {row['Count']} |\n")
                
                f.write("\n## Agreement by Paper\n\n")
                f.write("| Paper ID | Configuration | Agreement Rate (%) | Count |\n")
                f.write("|----------|---------------|-------------------|-------|\n")
                for _, row in paper_agreement.iterrows():
                    f.write(f"| {row['Paper ID']} | {row['Configuration']} | {row['Agreement Rate']:.2f} | {row['Count']} |\n")
                
            print(f"Generated human_model_agreement_summary.md in {output_dir}")
        else:
            print("No human-model agreement data could be generated")
    except Exception as e:
        print(f"Error processing human comparison data: {e}")
else:
    print(f"Human comparison file not found: {args.human_comparison}")

# Generate a detailed analysis summary in Markdown format
with open(os.path.join(output_dir, 'li_paper_analysis_summary.md'), 'w') as f:
    f.write("# Li-Paper Analysis Summary\n\n")
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## Overview\n\n")
    f.write(f"- **Number of Papers Analyzed**: {len(summary_df['paper_id'].unique())}\n")
    f.write(f"- **Total Report Files Processed**: {len(all_reports)}\n\n")
    
    f.write("## Agreement Rates by Configuration\n\n")
    f.write("| Configuration | Mean Agreement Rate (%) | Std Dev | Count |\n")
    f.write("|---------------|-------------------------|---------|-------|\n")
    for _, row in avg_agreement.iterrows():
        f.write(f"| {row['Configuration']} | {row['Mean Agreement Rate (%)']} | {row['Std Dev']} | {row['Count']} |\n")
    f.write("\n")
    
    if u_stat is not None and p_value is not None:
        f.write("## Statistical Comparison\n\n")
        f.write(f"- **Mann-Whitney U Test**: U = {u_stat}, p-value = {p_value}\n")
        f.write(f"- **Significant Difference at α=0.05**: {'Yes' if p_value < 0.05 else 'No'}\n\n")
    
    if 'overall_output_agreement' in locals() and overall_output_agreement is not None:
        f.write("## Model Output Agreement\n\n")
        f.write(f"- **Overall Model Output Agreement Rate**: {overall_output_agreement:.2f}%\n")
        f.write(f"- **Number of Papers with Both Configurations**: {len(papers_with_both_configs)}\n")
        
        if 'corr_openai_claude' in locals() and corr_openai_claude is not None:
            f.write(f"- **Correlation with OpenAI-Claude Validator Agreement**: {corr_openai_claude:.4f}\n")
            
        if 'corr_claude_openai' in locals() and corr_claude_openai is not None:
            f.write(f"- **Correlation with Claude-OpenAI Validator Agreement**: {corr_claude_openai:.4f}\n\n")
        
        if 'paper_output_agreement' in locals() and not paper_output_agreement.empty:
            f.write("## Top 5 Papers by Model Output Agreement\n\n")
            f.write("| Paper ID | Model Output Agreement Rate (%) |\n")
            f.write("|----------|--------------------------------|\n")
            for _, row in paper_output_agreement.head().iterrows():
                f.write(f"| {row['Paper ID']} | {row['Model Output Agreement Rate (%)']} |\n")
            f.write("\n")
            
            f.write("## Bottom 5 Papers by Model Output Agreement\n\n")
            f.write("| Paper ID | Model Output Agreement Rate (%) |\n")
            f.write("|----------|--------------------------------|\n")
            for _, row in paper_output_agreement.tail().iterrows():
                f.write(f"| {row['Paper ID']} | {row['Model Output Agreement Rate (%)']} |\n")
            f.write("\n")
    
    f.write("## Visualizations\n\n")
    f.write("The following visualizations have been generated:\n\n")
    f.write(f"1. `agreement_by_config.png`: Bar chart showing mean agreement rates by configuration\n")
    f.write(f"2. `agreement_distribution.png`: Violin plot showing the distribution of agreement rates\n")
    f.write(f"3. `agreement_by_item.png`: Bar chart showing agreement rates by checklist item\n")
    
    if 'overall_output_agreement' in locals() and overall_output_agreement is not None:
        f.write(f"4. `model_output_agreement_by_paper.png`: Bar chart showing model output agreement by paper\n")
        f.write(f"5. `model_output_agreement_by_item.png`: Bar chart showing model output agreement by checklist item\n")
        
        if 'corr_openai_claude' in locals() and corr_openai_claude is not None and 'corr_claude_openai' in locals() and corr_claude_openai is not None:
            f.write(f"6. `correlation_plots.png`: Scatter plots showing correlations between validator agreement and model output agreement\n")
    
    # Add human-model agreement visualizations if available
    if 'overall_human_model_agreement' in locals() and overall_human_model_agreement is not None:
        f.write(f"7. `human_model_agreement_by_config.png`: Bar chart showing human-model agreement by configuration\n")
        f.write(f"8. `human_model_agreement_by_item.png`: Bar chart showing human-model agreement by checklist item\n")
        f.write(f"\nA detailed human-model agreement analysis is available in `human_model_agreement_summary.md`.\n")

print(f"Generated li_paper_analysis_summary.md in {output_dir}")
