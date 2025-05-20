#!/bin/bash

# Handle keyboard interrupts gracefully
trap 'echo -e "\n\nScript interrupted by user. Exiting gracefully..."; exit 1' INT
# run_analysis_fixed.sh
# A script to run analysis on any reports directory using the fixed analysis script

# Function to print usage
print_usage() {
    echo "Usage: $0 --reports-dir <directory> [--output-dir <directory>]"
    echo "Options:"
    echo "  --reports-dir <directory>  Directory containing the report files to analyze"
    echo "  --output-dir <directory>   Directory to save the output files (defaults to reports directory)"
    echo "  --help                     Display this help message"
}

# Default values
REPORTS_DIR=""
OUTPUT_DIR=""

# Parse command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --reports-dir)
            REPORTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if reports directory is provided
if [ -z "$REPORTS_DIR" ]; then
    echo "Error: Reports directory is required"
    print_usage
    exit 1
fi

# If output directory is not provided, use the reports directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$REPORTS_DIR"
fi

# Check if reports directory exists
if [ ! -d "$REPORTS_DIR" ]; then
    echo "Error: Reports directory does not exist: $REPORTS_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set the working directory
cd /Users/chenyuli/LLMEvaluation/RWE_LLM_validator

# Run the fixed analysis script
echo "Running analysis on reports in $REPORTS_DIR"
python run_li_paper_analysis_fixed.py --reports_dir "$REPORTS_DIR" --output_dir "$OUTPUT_DIR"

echo "Analysis completed!"
echo "Visualizations and analysis files have been saved to $OUTPUT_DIR"
