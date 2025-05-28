#!/usr/bin/env python

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze updated human-model comparison data')
    parser.add_argument('--comparison_file', type=str, required=True, help='Path to the updated model_human_comparison.csv file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
    args = parser.parse_args()

    # Set plotting style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

    print(f"Analyzing updated human-model comparison data from {args.comparison_file}...")

    # Load the updated comparison data
    try:
        df = pd.read_csv(args.comparison_file)
        print(f"Loaded comparison data with shape: {df.shape}")
        
        # Print column names for debugging
        print("Columns in comparison data:")
        for col in df.columns:
            print(f"  - {col}")
        
        # Check if the necessary columns exist
        required_columns = ['paper_id', 'item_id', 'openai_answer', 'claude_answer', 'final_answer', 'human_answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"WARNING: Missing required columns: {missing_columns}")
            return
        
        # Calculate agreement with human
        # These columns might already exist, but we'll recalculate them to be sure
        df['agree_with_human_openai'] = df.apply(
            lambda row: 'Yes' if str(row['openai_answer']).lower() == str(row['human_answer']).lower() else 'No', 
            axis=1
        )
        
        df['agree_with_human_claude'] = df.apply(
            lambda row: 'Yes' if str(row['claude_answer']).lower() == str(row['human_answer']).lower() else 'No', 
            axis=1
        )
        
        df['agree_with_human_final'] = df.apply(
            lambda row: 'Yes' if str(row['final_answer']).lower() == str(row['human_answer']).lower() else 'No', 
            axis=1
        )
        
        # Calculate agreement statistics
        total_items = len(df)
        openai_human_agree = (df['agree_with_human_openai'] == 'Yes').sum()
        claude_human_agree = (df['agree_with_human_claude'] == 'Yes').sum()
        final_human_agree = (df['agree_with_human_final'] == 'Yes').sum()
        
        openai_human_rate = (openai_human_agree / total_items) * 100 if total_items > 0 else 0
        claude_human_rate = (claude_human_agree / total_items) * 100 if total_items > 0 else 0
        final_human_rate = (final_human_agree / total_items) * 100 if total_items > 0 else 0
        
        print("\n=== Updated Human-Model Agreement Statistics ===")
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
        
        ax = sns.barplot(x='Model', y='Agreement Rate (%)', data=agreement_df, palette='viridis', hue='Model', legend=False)
        
        # Add value labels on top of bars
        for i, v in enumerate(agreement_df['Agreement Rate (%)']):
            ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
        
        plt.title('Human-Model Agreement Rates (Updated)', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'updated_human_model_agreement_rates.png'), dpi=300, bbox_inches='tight')
        print(f"Saved updated_human_model_agreement_rates.png to {args.output_dir}")
        
        # 2. Agreement by paper
        paper_agreement = df.groupby('paper_id').apply(
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
        paper_agreement_sorted.to_csv(os.path.join(args.output_dir, 'updated_paper_human_agreement.csv'), index=False)
        print(f"Saved updated_paper_human_agreement.csv to {args.output_dir}")
        
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
        
        plt.title('Top 10 Papers by Human-Model Agreement Rate (Updated)', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Paper ID', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 105)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'updated_top_papers_human_agreement.png'), dpi=300, bbox_inches='tight')
        print(f"Saved updated_top_papers_human_agreement.png to {args.output_dir}")
        
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
        
        plt.title('Bottom 10 Papers by Human-Model Agreement Rate (Updated)', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Paper ID', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 105)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'updated_bottom_papers_human_agreement.png'), dpi=300, bbox_inches='tight')
        print(f"Saved updated_bottom_papers_human_agreement.png to {args.output_dir}")
        
        # 5. Agreement by item
        item_agreement = df.groupby('item_id').apply(
            lambda x: {
                'openai_rate': (x['agree_with_human_openai'] == 'Yes').mean() * 100,
                'claude_rate': (x['agree_with_human_claude'] == 'Yes').mean() * 100,
                'final_rate': (x['agree_with_human_final'] == 'Yes').mean() * 100,
                'count': len(x),
                'description': x['description'].iloc[0] if 'description' in x.columns and not x['description'].empty else f"Item {x.name}"
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
        item_agreement_sorted.to_csv(os.path.join(args.output_dir, 'updated_item_human_agreement.csv'), index=False)
        print(f"Saved updated_item_human_agreement.csv to {args.output_dir}")
        
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
        
        plt.title('Most Disagreed Items (Human-Model) (Updated)', fontsize=16)
        plt.ylabel('Agreement Rate (%)', fontsize=14)
        plt.xlabel('Item ID', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 105)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'updated_most_disagreed_items_human.png'), dpi=300, bbox_inches='tight')
        print(f"Saved updated_most_disagreed_items_human.png to {args.output_dir}")
        
        # Generate a detailed analysis summary in Markdown format
        with open(os.path.join(args.output_dir, 'updated_human_model_comparison_summary.md'), 'w') as f:
            f.write("# Updated Human-Model Comparison Summary\n\n")
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
            f.write("1. `updated_human_model_agreement_rates.png`: Bar chart showing agreement rates between human and different models\n")
            f.write("2. `updated_top_papers_human_agreement.png`: Bar chart showing top 10 papers by human-model agreement rate\n")
            f.write("3. `updated_bottom_papers_human_agreement.png`: Bar chart showing bottom 10 papers by human-model agreement rate\n")
            f.write("4. `updated_most_disagreed_items_human.png`: Bar chart showing the most disagreed items between human and models\n")
        
        print(f"Generated updated_human_model_comparison_summary.md in {args.output_dir}")
        
    except Exception as e:
        print(f"Error processing updated comparison data: {e}")
        import traceback
        traceback.print_exc()

    print("\nUpdated analysis complete!")

if __name__ == "__main__":
    main()
