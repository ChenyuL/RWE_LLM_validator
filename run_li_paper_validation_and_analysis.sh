#!/bin/bash

# Handle keyboard interrupts gracefully
trap 'echo -e "\n\nScript interrupted by user. Exiting gracefully..."; exit 1' INT
# run_li_paper_validation_and_analysis.sh
# Script to run the entire Li-Paper validation and analysis process

# Set the working directory
cd /Users/chenyuli/LLMEvaluation/RWE_LLM_validator

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/paper_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

# Create a symlink to the latest results
ln -sf "paper_results_${TIMESTAMP}" "output/paper_results_latest"

# Step 1: Process each paper in the fixed_sampled_papers.json list
echo "Step 1: Processing papers from fixed_sampled_papers.json"

# Read the list of papers using Python instead of jq
PAPERS=$(python -c "import json; print('\n'.join(json.load(open('output/paper_results/fixed_sampled_papers.json'))))")

# Prompt file path
PROMPT_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250319_123039_openai_reasoner_Li-Paper_prompts.json"

# Copy the fixed_sampled_papers.json to the new output directory
cp output/paper_results/fixed_sampled_papers.json "$OUTPUT_DIR/"

# Step 2: Run OpenAI as extractor and Claude as validator
echo "Step 2: Running OpenAI as extractor and Claude as validator"
for PAPER in $PAPERS; do
    echo "Processing paper: $PAPER"
    PAPER_PATH="data/Papers/$PAPER"
    
    # Check if the paper file exists
    if [ -f "$PAPER_PATH" ]; then
        echo "Running OpenAI-Claude validation for $PAPER"
        # Extract paper ID from filename
        PAPER_ID=$(basename "$PAPER" .pdf)
        
        # Run the validation with error handling and retry capability
        echo "Running OpenAI-Claude validation for $PAPER_ID..."
        if python run_validation_with_retry.py --script test_record_validation_openai_claude_fixed.py --mode extractor --prompts "$PROMPT_FILE" --paper "$PAPER_PATH" --checklist "Li-Paper" --retries 3 --delay 30; then
            echo "OpenAI-Claude validation completed successfully for $PAPER_ID"
        else
            echo "WARNING: OpenAI-Claude validation failed for $PAPER_ID, but continuing with next paper"
        fi
        
        # Move the output files to the new directory
        find output -name "*_${PAPER_ID}_Li-Paper.json" -newer "$OUTPUT_DIR/fixed_sampled_papers.json" -exec mv {} "$OUTPUT_DIR/" \;
    else
        echo "Warning: Paper file not found: $PAPER_PATH"
    fi
done

# Step 3: Run Claude as extractor and OpenAI as validator
echo "Step 3: Running Claude as extractor and OpenAI as validator"
for PAPER in $PAPERS; do
    echo "Processing paper: $PAPER"
    PAPER_PATH="data/Papers/$PAPER"
    
    # Check if the paper file exists
    if [ -f "$PAPER_PATH" ]; then
        echo "Running Claude-OpenAI validation for $PAPER"
        # Extract paper ID from filename
        PAPER_ID=$(basename "$PAPER" .pdf)
        
        # Run the validation with error handling and retry capability
        echo "Running Claude-OpenAI validation for $PAPER_ID..."
        if python run_validation_with_retry.py --script test_record_validation_claude_openai_fixed.py --mode extractor --prompts "$PROMPT_FILE" --paper "$PAPER_PATH" --checklist "Li-Paper" --retries 3 --delay 30; then
            echo "Claude-OpenAI validation completed successfully for $PAPER_ID"
        else
            echo "WARNING: Claude-OpenAI validation failed for $PAPER_ID, but continuing with next paper"
        fi
        
        # Move the output files to the new directory
        find output -name "*_${PAPER_ID}_Li-Paper.json" -newer "$OUTPUT_DIR/fixed_sampled_papers.json" -exec mv {} "$OUTPUT_DIR/" \;
    else
        echo "Warning: Paper file not found: $PAPER_PATH"
    fi
done

# Step 4: Run the analysis script to generate visualizations and output results
echo "Step 4: Running analysis script"

# Use the modified analysis script that works with the reports directory
echo "Running modified Li-Paper analysis script"

# Copy all report files to the reports directory
echo "Copying report files to reports directory"
./copy_results_to_reports.sh

# Run the analysis
python run_li_paper_analysis_modified.py

# Step 5: Clean up temporary files
echo "Step 5: Cleaning up temporary files"
# Remove batch extraction and validation files, keeping only the final reports
find "$OUTPUT_DIR" -name "*batch_*_extraction_*.json" -delete
find "$OUTPUT_DIR" -name "*batch_*_validation_*.json" -delete

echo "Process completed successfully!"
echo "Results can be found in: $OUTPUT_DIR"
echo "Visualizations have been saved as PNG files in the current directory."
