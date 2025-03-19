#!/bin/bash

# Handle keyboard interrupts gracefully
trap 'echo -e "\n\nScript interrupted by user. Exiting gracefully..."; exit 1' INT
# run_openai_claude_validation.sh
# Script to run the OpenAI-Claude validation process for Li-Paper

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
echo "Processing papers from fixed_sampled_papers.json using OpenAI as extractor and Claude as validator"

# Read the list of papers using Python instead of jq
PAPERS=$(python -c "import json; print('\n'.join(json.load(open('output/paper_results/fixed_sampled_papers.json'))))")

# Prompt file path
PROMPT_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250319_123039_openai_reasoner_Li-Paper_prompts.json"

# Copy the fixed_sampled_papers.json to the new output directory
cp output/paper_results/fixed_sampled_papers.json "$OUTPUT_DIR/"

# Run OpenAI as extractor and Claude as validator
for PAPER in $PAPERS; do
    echo "Processing paper: $PAPER"
    PAPER_PATH="data/Papers/$PAPER"
    
    # Check if the paper file exists
    if [ -f "$PAPER_PATH" ]; then
        echo "Running OpenAI-Claude validation for $PAPER"
        # Extract paper ID from filename
        PAPER_ID=$(basename "$PAPER" .pdf)
        
        # Run the validation with error handling and retry capability
        echo "Running validation for $PAPER_ID..."
        if python run_validation_with_retry.py --script test_record_validation_openai_claude_fixed.py --mode extractor --prompts "$PROMPT_FILE" --paper "$PAPER_PATH" --checklist "Li-Paper" --retries 3 --delay 30; then
            echo "Validation completed successfully for $PAPER_ID"
        else
            echo "WARNING: Validation failed for $PAPER_ID, but continuing with next paper"
        fi
        
        # Move the output files to the new directory
        find output -name "*_${PAPER_ID}_Li-Paper.json" -newer "$OUTPUT_DIR/fixed_sampled_papers.json" -exec mv {} "$OUTPUT_DIR/" \;
        
        # Add a short delay between papers to avoid rate limiting
        echo "Waiting 5 seconds before processing the next paper..."
        sleep 5
    else
        echo "Warning: Paper file not found: $PAPER_PATH"
    fi
done

echo "OpenAI-Claude validation process completed!"
echo "Results can be found in: $OUTPUT_DIR"
