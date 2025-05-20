#!/bin/bash

# Handle keyboard interrupts gracefully
trap 'echo -e "\n\nScript interrupted by user. Exiting gracefully..."; exit 1' INT

# Script to copy validation results to the reports directory
# This ensures that the analysis script can find the results

# Set the working directory
cd /Users/chenyuli/LLMEvaluation/RWE_LLM_validator

# Create reports directory if it doesn't exist
REPORTS_DIR="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/reports"
mkdir -p "$REPORTS_DIR"

echo "Copying validation results to reports directory: $REPORTS_DIR"

# Find all validation report files in the output directory
# This includes files with patterns like:
# - *_openai_claude_report_*_Li-Paper.json
# - *_claude_openai_report_*_Li-Paper.json

# Copy OpenAI-Claude reports
find output -name "*_openai_claude_report_*_Li-Paper.json" -exec cp {} "$REPORTS_DIR/" \;
echo "Copied OpenAI-Claude reports"

# Copy Claude-OpenAI reports
find output -name "*_claude_openai_report_*_Li-Paper.json" -exec cp {} "$REPORTS_DIR/" \;
echo "Copied Claude-OpenAI reports"

# Copy any other relevant files
find output -name "*_report_*_Li-Paper.json" -exec cp {} "$REPORTS_DIR/" \;
echo "Copied other report files"

# Count the number of files copied
NUM_FILES=$(find "$REPORTS_DIR" -name "*_Li-Paper.json" | wc -l)
echo "Total files in reports directory: $NUM_FILES"

echo "Copy completed successfully!"
echo "You can now run the analysis script with: ./run_analysis_only.sh"
