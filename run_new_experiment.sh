#!/bin/bash
# run_new_experiment.sh
# Script to run a new experiment with OpenAI and Claude validators

# Handle keyboard interrupts gracefully
trap 'echo -e "\n\nScript interrupted by user. Exiting gracefully..."; exit 1' INT

# Set variables
PROMPT_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250319_220659_openai_reasoner_Li-Paper_prompts.json"
PAPERS_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/paper_results/fixed_sampled_papers.json"
PAPERS_DIR="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/data/Papers"
OPENAI_CLAUDE_SCRIPT="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/test_record_validation_openai_claude_fixed.py"
CLAUDE_OPENAI_SCRIPT="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/test_record_validation_claude_openai_fixed.py"
ANALYSIS_SCRIPT="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/run_analysis_only.sh"

# Check if files exist
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

if [ ! -f "$PAPERS_FILE" ]; then
    echo "Error: Papers file not found: $PAPERS_FILE"
    exit 1
fi

if [ ! -f "$OPENAI_CLAUDE_SCRIPT" ]; then
    echo "Error: OpenAI-Claude validation script not found: $OPENAI_CLAUDE_SCRIPT"
    exit 1
fi

if [ ! -f "$CLAUDE_OPENAI_SCRIPT" ]; then
    echo "Error: Claude-OpenAI validation script not found: $CLAUDE_OPENAI_SCRIPT"
    exit 1
fi

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "Error: Analysis script not found: $ANALYSIS_SCRIPT"
    exit 1
fi

# Create timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting experiment at $TIMESTAMP"

# Create output directory for this experiment
OUTPUT_DIR="output/${TIMESTAMP}_experiment"
mkdir -p "$OUTPUT_DIR"

# Read the papers list using Python
PAPERS=$(python -c "import json; f = open('$PAPERS_FILE'); papers = json.load(f); f.close(); print('\n'.join(papers))")

# Process each paper with OpenAI extractor and Claude validator
echo "Running OpenAI extractor with Claude validator for each paper..."
for PAPER in $PAPERS; do
    echo "Processing paper: $PAPER"
    PAPER_PATH="$PAPERS_DIR/$PAPER"
    
    if [ ! -f "$PAPER_PATH" ]; then
        echo "Warning: Paper file not found: $PAPER_PATH"
        continue
    fi
    
    python "$OPENAI_CLAUDE_SCRIPT" --mode extractor --prompts "$PROMPT_FILE" --paper "$PAPER_PATH" --checklist "Li-Paper"
    
    # Sleep for a short time to avoid rate limiting
    sleep 2
done

# Process each paper with Claude extractor and OpenAI validator
echo "Running Claude extractor with OpenAI validator for each paper..."
for PAPER in $PAPERS; do
    echo "Processing paper: $PAPER"
    PAPER_PATH="$PAPERS_DIR/$PAPER"
    
    if [ ! -f "$PAPER_PATH" ]; then
        echo "Warning: Paper file not found: $PAPER_PATH"
        continue
    fi
    
    python "$CLAUDE_OPENAI_SCRIPT" --mode extractor --prompts "$PROMPT_FILE" --paper "$PAPER_PATH" --checklist "Li-Paper"
    
    # Sleep for a short time to avoid rate limiting
    sleep 2
done

# Run analysis
echo "Running analysis..."
bash "$ANALYSIS_SCRIPT"

echo "Experiment completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
