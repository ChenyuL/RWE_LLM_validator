import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import re

# Set plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Starting Li-Paper analysis...")

# Directory containing the paper results
base_dir = 'output/paper_results'

# Get all directories with Li-Paper suffix
li_paper_dirs = [d for d in os.listdir(base_dir) if d.endswith('_Li-Paper') and os.path.isdir(os.path.join(base_dir, d))]
print(f"Found {len(li_paper_dirs)} Li-Paper directories")
print(li_paper_dirs[:5], "...")

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
        
        # Determine configuration based on filename
        if 'openai_claude' in file_path:
            config = 'openai_claude'
        elif 'claude_openai' in file_path:
            config = 'claude_openai'
        else:
            config = 'unknown'
        
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

for paper_dir in li_paper_dirs:
    dir_path = os.path.join(base_dir, paper_dir)
    for file in os.listdir(dir_path):
        if ('openai_claude_report' in file or 'claude_openai_report' in file) and file.endswith('.json'):
            file_path = os.path.join(dir_path, file)
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
plt.savefig('agreement_by_config.png', dpi=300, bbox_inches='tight')
print("Saved agreement_by_config.png")

# Analysis 2: Distribution of Agreement Rates
print("\nAnalysis 2: Distribution of Agreement Rates")
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='config', y='agreement_rate', data=summary_df, palette='viridis', inner='box', hue='config', legend=False)
ax = sns.swarmplot(x='config', y='agreement_rate', data=summary_df, color='white', edgecolor='black', size=8, alpha=0.7)

plt.title('Distribution of Agreement Rates by Configuration', fontsize=16)
plt.ylabel('Agreement Rate (%)', fontsize=14)
plt.xlabel('Configuration', fontsize=14)
plt.tight_layout()
plt.savefig('agreement_distribution.png', dpi=300, bbox_inches='tight')
print("Saved agreement_distribution.png")

# Analysis 3: Statistical Comparison of Configurations
print("\nAnalysis 3: Statistical Comparison of Configurations")
openai_claude = summary_df[summary_df['config'] == 'openai_claude']['agreement_rate']
claude_openai = summary_df[summary_df['config'] == 'claude_openai']['agreement_rate']

# Perform Mann-Whitney U test (non-parametric test for independent samples)
u_stat, p_value = stats.mannwhitneyu(openai_claude, claude_openai, alternative='two-sided')

print(f"Mann-Whitney U test results:")
print(f"U statistic: {u_stat}")
print(f"p-value: {p_value}")
print(f"Significant difference at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

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

# Reshape data for plotting
plot_data = pd.melt(item_agreement_pivot, id_vars=['Item ID'], 
                    value_vars=['openai_claude', 'claude_openai'],
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
plt.savefig('agreement_by_item.png', dpi=300, bbox_inches='tight')
print("Saved agreement_by_item.png")

# Analysis 5: Model Output Comparison
print("\nAnalysis 5: Model Output Comparison")
model_outputs = {}

for report in all_reports:
    paper_id = report['paper_id']
    config = report['config']
    
    if paper_id not in model_outputs:
        model_outputs[paper_id] = {}
    
    for item_id, item in report['items'].items():
        if item_id not in model_outputs[paper_id]:
            model_outputs[paper_id][item_id] = {}
        
        model_outputs[paper_id][item_id][config] = item.get('correct_answer', '')

# Function to compare model outputs and determine if they agree
def outputs_agree(output1, output2):
    # Simple string comparison for now
    # Could be enhanced with semantic similarity or other NLP techniques
    if not output1 or not output2:
        return False
    
    # Check for exact match
    if output1 == output2:
        return True
    
    # Check for 'unknown' or similar values
    unknown_patterns = ['unknown', 'not enough information', 'cannot determine']
    if any(pattern in output1.lower() for pattern in unknown_patterns) and \
       any(pattern in output2.lower() for pattern in unknown_patterns):
        return True
    
    # Check for yes/no agreement
    yes_patterns = ['yes', 'complies', 'compliant', 'fulfilled']
    no_patterns = ['no', 'does not comply', 'non-compliant', 'not fulfilled']
    
    output1_yes = any(pattern in output1.lower() for pattern in yes_patterns)
    output1_no = any(pattern in output1.lower() for pattern in no_patterns)
    output2_yes = any(pattern in output2.lower() for pattern in yes_patterns)
    output2_no = any(pattern in output2.lower() for pattern in no_patterns)
    
    if (output1_yes and output2_yes) or (output1_no and output2_no):
        return True
    
    return False

# Calculate model output agreement
output_agreement_data = []

for paper_id, items in model_outputs.items():
    for item_id, configs in items.items():
        if 'openai_claude' in configs and 'claude_openai' in configs:
            openai_claude_output = configs['openai_claude']
            claude_openai_output = configs['claude_openai']
            
            agreement = outputs_agree(openai_claude_output, claude_openai_output)
            
            output_agreement_data.append({
                'paper_id': paper_id,
                'item_id': item_id,
                'openai_claude_output': openai_claude_output,
                'claude_openai_output': claude_openai_output,
                'models_agree': agreement
            })

output_agreement_df = pd.DataFrame(output_agreement_data)
print("Output agreement DataFrame created with shape:", output_agreement_df.shape)

# Calculate overall model output agreement rate
overall_output_agreement = output_agreement_df['models_agree'].mean() * 100
print(f"Overall model output agreement rate: {overall_output_agreement:.2f}%")

# Calculate model output agreement by paper
paper_output_agreement = output_agreement_df.groupby('paper_id')['models_agree'].mean() * 100
paper_output_agreement = paper_output_agreement.reset_index()
paper_output_agreement.columns = ['Paper ID', 'Model Output Agreement Rate (%)']
paper_output_agreement.sort_values('Model Output Agreement Rate (%)', ascending=False, inplace=True)
print("Top 5 papers by model output agreement rate:")
print(paper_output_agreement.head())

# Visualize model output agreement by paper
plt.figure(figsize=(15, 8))
ax = sns.barplot(x='Paper ID', y='Model Output Agreement Rate (%)', data=paper_output_agreement, palette='viridis', hue='Paper ID', legend=False)

plt.title('Model Output Agreement Rate by Paper', fontsize=16)
plt.ylabel('Agreement Rate (%)', fontsize=14)
plt.xlabel('Paper ID', fontsize=14)
plt.xticks(rotation=90)
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_output_agreement_by_paper.png', dpi=300, bbox_inches='tight')
print("Saved model_output_agreement_by_paper.png")

# Calculate model output agreement by checklist item
item_output_agreement = output_agreement_df.groupby('item_id')['models_agree'].mean() * 100
item_output_agreement = item_output_agreement.reset_index()
item_output_agreement.columns = ['Item ID', 'Model Output Agreement Rate (%)']

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
ax = sns.barplot(x='Item ID', y='Model Output Agreement Rate (%)', data=item_output_agreement, palette='viridis', hue='Item ID', legend=False)

plt.title('Model Output Agreement Rate by Checklist Item', fontsize=16)
plt.ylabel('Agreement Rate (%)', fontsize=14)
plt.xlabel('Checklist Item ID', fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_output_agreement_by_item.png', dpi=300, bbox_inches='tight')
print("Saved model_output_agreement_by_item.png")

# Analysis 6: Correlation Between Validator Agreement and Model Output Agreement
print("\nAnalysis 6: Correlation Between Validator Agreement and Model Output Agreement")
paper_agreement = summary_df.groupby(['paper_id', 'config'])['agreement_rate'].mean().reset_index()
paper_agreement_pivot = paper_agreement.pivot(index='paper_id', columns='config', values='agreement_rate').reset_index()
paper_agreement_pivot.columns = ['paper_id', 'claude_openai_agreement', 'openai_claude_agreement']

# Merge with model output agreement data
merged_agreement = pd.merge(paper_agreement_pivot, paper_output_agreement, left_on='paper_id', right_on='Paper ID', how='inner')
print("Merged agreement DataFrame created with shape:", merged_agreement.shape)

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
plt.savefig('correlation_plots.png', dpi=300, bbox_inches='tight')
print("Saved correlation_plots.png")

# Create a summary table
summary_stats = pd.DataFrame({
    'Metric': [
        'Number of Papers Analyzed',
        'OpenAI-Claude Mean Agreement Rate (%)',
        'Claude-OpenAI Mean Agreement Rate (%)',
        'Model Output Agreement Rate (%)',
        'Correlation: OpenAI-Claude vs. Model Output',
        'Correlation: Claude-OpenAI vs. Model Output',
        'Significant Difference Between Configurations'
    ],
    'Value': [
        len(paper_agreement_pivot),
        avg_agreement[avg_agreement['Configuration'] == 'openai_claude']['Mean Agreement Rate (%)'].values[0],
        avg_agreement[avg_agreement['Configuration'] == 'claude_openai']['Mean Agreement Rate (%)'].values[0],
        overall_output_agreement,
        corr_openai_claude,
        corr_claude_openai,
        'Yes' if p_value < 0.05 else 'No'
    ]
})

print("\nSummary Statistics:")
print(summary_stats)

# Save summary statistics to CSV
summary_stats.to_csv('li_paper_summary_stats.csv', index=False)
print("Saved li_paper_summary_stats.csv")

print("\nAnalysis complete. All visualizations and statistics have been generated.")
