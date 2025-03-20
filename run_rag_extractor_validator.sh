#!/bin/bash
# run_rag_extractor_validator.sh
# Script to run the RAG-based extractor and validator on a sample paper

# Check if PyPDF2 is installed
if ! python -c "import PyPDF2" &> /dev/null; then
    echo "Installing PyPDF2..."
    pip install PyPDF2
fi

# Check if numpy is installed
if ! python -c "import numpy" &> /dev/null; then
    echo "Installing numpy..."
    pip install numpy
fi

# Check if tqdm is installed
if ! python -c "import tqdm" &> /dev/null; then
    echo "Installing tqdm..."
    pip install tqdm
fi

# Check if openai is installed
if ! python -c "import openai" &> /dev/null; then
    echo "Installing openai..."
    pip install openai
fi

# Check if anthropic is installed
if ! python -c "import anthropic" &> /dev/null; then
    echo "Installing anthropic..."
    pip install anthropic
fi

# Create output directories
mkdir -p output/embeddings

# Set variables
PROMPTS_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250319_220659_openai_reasoner_Li-Paper_prompts.json"
PAPER_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/data/Papers/34831722.pdf"
CHECKLIST="Li-Paper"
BATCH_SIZE=5

# Run the RAG-based extractor and validator
echo "Running RAG-based extractor and validator..."
python rag_extractor_validator.py --prompts "$PROMPTS_FILE" --paper "$PAPER_FILE" --checklist "$CHECKLIST" --batch-size "$BATCH_SIZE"

echo "Done!"
