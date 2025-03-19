#!/bin/bash

# Handle keyboard interrupts gracefully
trap 'echo -e "\n\nScript interrupted by user. Exiting gracefully..."; exit 1' INT
# run_analysis_only.sh
# Script to run only the analysis part of the Li-Paper validation process

# Set the working directory
cd /Users/chenyuli/LLMEvaluation/RWE_LLM_validator

# Use the modified analysis script that works with the reports directory
echo "Running modified Li-Paper analysis script"
python run_li_paper_analysis_modified.py

echo "Analysis completed!"
echo "Visualizations have been saved as PNG files in the current directory."
echo "Summary statistics have been saved to li_paper_summary_stats.csv"
echo "A detailed analysis report has been generated in li_paper_analysis_summary.md"
