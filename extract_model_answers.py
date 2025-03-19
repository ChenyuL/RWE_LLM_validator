import os
import json
import pandas as pd
import re

# Directory containing the report files
reports_dir = 'output/reports'

# Output file path
output_file = 'model_answers_comparison.xlsx'

# Function to extract paper ID from filename
def extract_paper_id(filename):
    # Extract paper ID using regex
    match = re.search(r'_(\d+)_Li-Paper\.json$', filename)
    if match:
        return match.group(1)
    return None

# Function to load report data
def load_report_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to determine if a report is from OpenAI or Claude extractor
def get_extractor_type(filename):
    if 'openai_claude_report' in filename:
        return 'openai'
    elif 'claude_openai_report' in filename:
        return 'claude'
    return None

# Dictionary to store results by paper ID and item ID
results = {}

# Process all report files
print(f"Scanning directory: {reports_dir}")
files = os.listdir(reports_dir)
report_files = [f for f in files if ('_report_' in f) and f.endswith('_Li-Paper.json')]
print(f"Found {len(report_files)} report files")

# Process each report file
for filename in report_files:
    file_path = os.path.join(reports_dir, filename)
    paper_id = extract_paper_id(filename)
    extractor_type = get_extractor_type(filename)
    
    if not paper_id or not extractor_type:
        print(f"Skipping {filename}: Could not determine paper ID or extractor type")
        continue
    
    # Load report data
    report_data = load_report_data(file_path)
    if not report_data:
        continue
    
    # Initialize paper entry in results dictionary if not exists
    if paper_id not in results:
        results[paper_id] = {}
    
    # Extract correct_answer for each item
    items = report_data.get('items', {})
    for item_id, item_data in items.items():
        correct_answer = item_data.get('correct_answer', '')
        
        # Initialize item entry in paper's dictionary if not exists
        if item_id not in results[paper_id]:
            results[paper_id][item_id] = {}
        
        # Store correct_answer by extractor type
        results[paper_id][item_id][extractor_type] = correct_answer

# Convert results to DataFrame for Excel export
rows = []
for paper_id, paper_data in results.items():
    for item_id, item_data in paper_data.items():
        openai_answer = item_data.get('openai', '')
        claude_answer = item_data.get('claude', '')
        
        # Create a row for the DataFrame
        row = {
            'paper_id': paper_id,
            'item_id': item_id,
            'openai_answer': openai_answer,
            'claude_answer': claude_answer,
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
df.to_excel(output_file, index=False)
print(f"Saved model answers to {output_file}")

# Also save to CSV with tab delimiter for easier processing
csv_output_file = 'model_answers_comparison.csv'
df.to_csv(csv_output_file, index=False, sep='\t')
print(f"Saved model answers to {csv_output_file} (tab-delimited)")

# Try to load the manual validation data for reference
manual_validation_path = "/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/data/validation/Merged_Included_Article_2024.xlsx"
try:
    manual_df = pd.read_excel(manual_validation_path)
    print(f"Manual validation data loaded from {manual_validation_path}")
    print(f"Manual validation data shape: {manual_df.shape}")
    print("Column names in manual validation data:")
    for col in manual_df.columns:
        print(f"  - {col}")
except Exception as e:
    print(f"Could not load manual validation data: {e}")

print("Done!")
