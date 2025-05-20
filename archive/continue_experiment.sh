#!/bin/bash

# Handle keyboard interrupts gracefully
trap 'echo -e "\n\nScript interrupted by user. Exiting gracefully..."; exit 1' INT

# Default checklist name
CHECKLIST_NAME="Li-Paper"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --checklist)
      CHECKLIST_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--checklist CHECKLIST_NAME]"
      exit 1
      ;;
  esac
done

echo "Using checklist: $CHECKLIST_NAME"

# Set the working directory
cd /Users/chenyuli/LLMEvaluation/RWE_LLM_validator

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/experiment_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

# Create a symlink to the latest results
ln -sf "experiment_results_${TIMESTAMP}" "output/experiment_results_latest"

# Prompt file path
PROMPT_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250319_220659_openai_reasoner_${CHECKLIST_NAME}_prompts.json"
PAPERS_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/paper_results/fixed_sampled_papers.json"

# Copy the fixed_sampled_papers.json to the new output directory
cp "$PAPERS_FILE" "$OUTPUT_DIR/"

# Copy the existing batch files to the new output directory
echo "Copying existing batch files to the new output directory..."
find output -name "*_35870161_Li-Paper.json" -exec cp {} "$OUTPUT_DIR/" \;

# Read the list of papers using Python, skipping the first paper (35870161.pdf) as it was partially processed
PAPERS=$(python -c "import json; papers = json.load(open('$PAPERS_FILE')); print('\n'.join(papers[1:]))")
echo "Skipping the first paper (35870161.pdf) as it was partially processed"
echo "Processing the remaining 29 papers from the original list"

echo "====================================================="
echo "STEP 1: Running OpenAI-extractor with Claude-validator"
echo "====================================================="
echo "Skipping the first paper (35870161.pdf) as it was partially processed."

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
        if python run_validation_with_retry.py --script test_record_validation_openai_claude_modified.py --mode extractor --prompts "$PROMPT_FILE" --paper "$PAPER_PATH" --checklist "$CHECKLIST_NAME" --retries 3 --delay 30; then
            echo "Validation completed successfully for $PAPER_ID"
        else
            echo "WARNING: Validation failed for $PAPER_ID, but continuing with next paper"
        fi
        
        # Move the output files to the new directory
        find output -name "*_${PAPER_ID}_${CHECKLIST_NAME}.json" -newer "$OUTPUT_DIR/fixed_sampled_papers.json" -exec cp {} "$OUTPUT_DIR/" \;
        
# Add a shorter delay between papers
echo "Waiting 2 seconds before processing the next paper..."
sleep 2
    else
        echo "Warning: Paper file not found: $PAPER_PATH"
    fi
done

echo "OpenAI-Claude validation process completed!"

echo "====================================================="
echo "STEP 2: Running Claude-extractor with OpenAI-validator"
echo "====================================================="

# Read the list of papers again, using all papers from the original list
PAPERS=$(python -c "import json; papers = json.load(open('$PAPERS_FILE')); print('\n'.join(papers))")
echo "Processing all 30 papers from the original list"

# Run Claude as extractor and OpenAI as validator
for PAPER in $PAPERS; do
    echo "Processing paper: $PAPER"
    PAPER_PATH="data/Papers/$PAPER"
    
    # Check if the paper file exists
    if [ -f "$PAPER_PATH" ]; then
        echo "Running Claude-OpenAI validation for $PAPER"
        # Extract paper ID from filename
        PAPER_ID=$(basename "$PAPER" .pdf)
        
        # Run the validation with error handling and retry capability
        echo "Running validation for $PAPER_ID..."
        if python run_validation_with_retry.py --script test_record_validation_claude_openai_modified.py --mode extractor --prompts "$PROMPT_FILE" --paper "$PAPER_PATH" --checklist "$CHECKLIST_NAME" --retries 3 --delay 30; then
            echo "Validation completed successfully for $PAPER_ID"
        else
            echo "WARNING: Validation failed for $PAPER_ID, but continuing with next paper"
        fi
        
        # Move the output files to the new directory
        find output -name "*_${PAPER_ID}_${CHECKLIST_NAME}.json" -newer "$OUTPUT_DIR/fixed_sampled_papers.json" -exec cp {} "$OUTPUT_DIR/" \;
        
# Add a shorter delay between papers
echo "Waiting 2 seconds before processing the next paper..."
sleep 2
    else
        echo "Warning: Paper file not found: $PAPER_PATH"
    fi
done

echo "Claude-OpenAI validation process completed!"

echo "====================================================="
echo "STEP 3: Running analysis"
echo "====================================================="

# Run the analysis script
./run_analysis_only.sh

echo "====================================================="
echo "Experiment completed!"
echo "Results can be found in: $OUTPUT_DIR"
echo "Analysis results are in the current directory."
echo "====================================================="
