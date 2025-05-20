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

# Create output directory with date and checklist name
DATE=$(date +"%Y%m%d")
OUTPUT_DIR="output/experiment_results_${DATE}_${CHECKLIST_NAME}"
mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

# Create a symlink to the latest results
ln -sf "experiment_results_${DATE}_${CHECKLIST_NAME}" "output/experiment_results_latest"

# Prompt file path
PROMPT_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250319_220659_openai_reasoner_${CHECKLIST_NAME}_prompts.json"
PAPERS_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/paper_results/fixed_sampled_papers.json"

# Copy the fixed_sampled_papers.json to the new output directory
cp "$PAPERS_FILE" "$OUTPUT_DIR/"

# Get the list of all papers
ALL_PAPERS=$(python -c "import json; print('\n'.join(json.load(open('$PAPERS_FILE'))))")

# Create a temporary file to store the list of papers that have been processed with OpenAI-Claude
TEMP_OPENAI_CLAUDE_PROCESSED=$(mktemp)

# Find all papers that have been fully processed with OpenAI-Claude
echo "Scanning for papers already processed with OpenAI-Claude..."
for PAPER in $ALL_PAPERS; do
    PAPER_ID=$(basename "$PAPER" .pdf)
    
    # Check if there are validation results for this paper (only in the top-level output directory)
    if find output -maxdepth 1 -name "*validator_${PAPER_ID}_${CHECKLIST_NAME}.json" | grep -q .; then
        echo "Found OpenAI-Claude validation results for $PAPER_ID"
        echo "$PAPER" >> "$TEMP_OPENAI_CLAUDE_PROCESSED"
        
        # Copy the existing files to the new output directory (only from the top-level output directory)
        find output -maxdepth 1 -name "*_${PAPER_ID}_${CHECKLIST_NAME}.json" -exec cp {} "$OUTPUT_DIR/" \;
    else
        # Check if there are extraction results for all batches (only in the top-level output directory)
        BATCH_COUNT=$(find output -maxdepth 1 -name "*batch_*_extraction_${PAPER_ID}_${CHECKLIST_NAME}.json" | wc -l)
        if [ "$BATCH_COUNT" -ge 4 ]; then
            echo "Found complete extraction results for $PAPER_ID but no validation"
            # We'll need to validate this paper
        fi
    fi
done

# Create a temporary file to store the list of papers that have been processed with Claude-OpenAI
TEMP_CLAUDE_OPENAI_PROCESSED=$(mktemp)

# Find all papers that have been fully processed with Claude-OpenAI
echo "Scanning for papers already processed with Claude-OpenAI..."
for PAPER in $ALL_PAPERS; do
    PAPER_ID=$(basename "$PAPER" .pdf)
    
    # Check if there are validation results for this paper with Claude as extractor (only in the top-level output directory)
    if find output -maxdepth 1 -name "*claude-*_extractor_${PAPER_ID}_${CHECKLIST_NAME}.json" | grep -q .; then
        if find output -maxdepth 1 -name "*openai-*_validator_${PAPER_ID}_${CHECKLIST_NAME}.json" | grep -q .; then
            echo "Found Claude-OpenAI validation results for $PAPER_ID"
            echo "$PAPER" >> "$TEMP_CLAUDE_OPENAI_PROCESSED"
            
            # Copy the existing files to the new output directory (only from the top-level output directory)
            find output -maxdepth 1 -name "*claude-*_extractor_${PAPER_ID}_${CHECKLIST_NAME}.json" -exec cp {} "$OUTPUT_DIR/" \;
            find output -maxdepth 1 -name "*openai-*_validator_${PAPER_ID}_${CHECKLIST_NAME}.json" -exec cp {} "$OUTPUT_DIR/" \;
        fi
    fi
done

# Get the list of papers that still need to be processed with OpenAI-Claude
OPENAI_CLAUDE_PAPERS=$(comm -23 <(echo "$ALL_PAPERS" | sort) <(sort "$TEMP_OPENAI_CLAUDE_PROCESSED"))

# Get the list of papers that still need to be processed with Claude-OpenAI
CLAUDE_OPENAI_PAPERS=$(comm -23 <(echo "$ALL_PAPERS" | sort) <(sort "$TEMP_CLAUDE_OPENAI_PROCESSED"))

# Clean up temporary files
rm "$TEMP_OPENAI_CLAUDE_PROCESSED" "$TEMP_CLAUDE_OPENAI_PROCESSED"

echo "====================================================="
echo "Running OpenAI-extractor with Claude-validator and Claude-extractor with OpenAI-validator in parallel"
echo "====================================================="

# Function to process a paper with OpenAI-Claude
process_openai_claude() {
    PAPER=$1
    PAPER_PATH="data/Papers/$PAPER"
    
    # Check if the paper file exists
    if [ -f "$PAPER_PATH" ]; then
        echo "Running OpenAI-Claude validation for $PAPER"
        # Extract paper ID from filename
        PAPER_ID=$(basename "$PAPER" .pdf)
        
        # Run the validation with error handling and retry capability
        echo "Running validation for $PAPER_ID..."
        if python run_validation_with_retry.py --script test_record_validation_openai_claude_modified.py --mode extractor --prompts "$PROMPT_FILE" --paper "$PAPER_PATH" --checklist "$CHECKLIST_NAME" --retries 3 --delay 30; then
            echo "OpenAI-Claude validation completed successfully for $PAPER_ID"
        else
            echo "WARNING: OpenAI-Claude validation failed for $PAPER_ID"
        fi
        
        # Move the output files to the new directory (only from the top-level output directory)
        find output -maxdepth 1 -name "*_${PAPER_ID}_${CHECKLIST_NAME}.json" -newer "$OUTPUT_DIR/fixed_sampled_papers.json" -exec cp {} "$OUTPUT_DIR/" \;
    else
        echo "Warning: Paper file not found: $PAPER_PATH"
    fi
}

# Function to process a paper with Claude-OpenAI
process_claude_openai() {
    PAPER=$1
    PAPER_PATH="data/Papers/$PAPER"
    
    # Check if the paper file exists
    if [ -f "$PAPER_PATH" ]; then
        echo "Running Claude-OpenAI validation for $PAPER"
        # Extract paper ID from filename
        PAPER_ID=$(basename "$PAPER" .pdf)
        
        # Run the validation with error handling and retry capability
        echo "Running validation for $PAPER_ID..."
        if python run_validation_with_retry.py --script test_record_validation_claude_openai_modified.py --mode extractor --prompts "$PROMPT_FILE" --paper "$PAPER_PATH" --checklist "$CHECKLIST_NAME" --retries 3 --delay 30; then
            echo "Claude-OpenAI validation completed successfully for $PAPER_ID"
        else
            echo "WARNING: Claude-OpenAI validation failed for $PAPER_ID"
        fi
        
        # Move the output files to the new directory (only from the top-level output directory)
        find output -maxdepth 1 -name "*_${PAPER_ID}_${CHECKLIST_NAME}.json" -newer "$OUTPUT_DIR/fixed_sampled_papers.json" -exec cp {} "$OUTPUT_DIR/" \;
    else
        echo "Warning: Paper file not found: $PAPER_PATH"
    fi
}

# Process papers in parallel
for PAPER in $OPENAI_CLAUDE_PAPERS; do
    process_openai_claude "$PAPER" &
    # Add a short delay to avoid starting too many processes at once
    sleep 1
done

for PAPER in $CLAUDE_OPENAI_PAPERS; do
    process_claude_openai "$PAPER" &
    # Add a short delay to avoid starting too many processes at once
    sleep 1
done

# Wait for all background processes to complete
echo "Waiting for all validation processes to complete..."
wait

echo "All validation processes completed!"

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
