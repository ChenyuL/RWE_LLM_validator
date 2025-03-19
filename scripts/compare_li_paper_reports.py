import os
import json
import pandas as pd

# Directory containing the report files
reports_dir = 'output/reports'

# Initialize a list to store comparison data
comparison_data = []

# Function to extract correct_answer fields from a report
def extract_correct_answers(report_path):
    with open(report_path, 'r') as file:
        report = json.load(file)
    return {item_id: item['correct_answer'] for item_id, item in report['items'].items()}

# Iterate over files in the reports directory
print("Processing files in:", reports_dir)
for filename in os.listdir(reports_dir):
    print("Processing file:", filename)
    if '_Li-Paper.json' in filename:
        # Determine the paper ID and configuration type
        parts = filename.split('_')
        paper_id = parts[4]
        config_type = parts[2]

        # Extract correct answers
        correct_answers = extract_correct_answers(os.path.join(reports_dir, filename))

        # Store the correct answers in the comparison data
        for item_id, correct_answer in correct_answers.items():
            comparison_data.append({
                'paper_id': paper_id,
                'checklist_item': item_id,
                'config_type': config_type,
                'correct_answer': correct_answer
            })

# Convert the comparison data to a DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Pivot the DataFrame to have separate columns for each configuration's correct answers
comparison_pivot = comparison_df.pivot_table(index=['paper_id', 'checklist_item'], columns='config_type', values=[col for col in comparison_df.columns if 'correct_answer' in col], aggfunc='first').reset_index()

print("Comparison data:", comparison_data)
# Save the comparison DataFrame to a CSV file
print("Saving comparison data to CSV...")
comparison_pivot.to_csv('output/li_paper_comparison.csv', index=False)
