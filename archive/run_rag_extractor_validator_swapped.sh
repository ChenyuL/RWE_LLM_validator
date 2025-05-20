#!/bin/bash
# run_rag_extractor_validator_swapped.sh
# Script to run the swapped RAG-based extractor and validator on a sample paper

# Make the Python script executable
chmod +x rag_extractor_validator_swapped.py

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
PROMPTS_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250322_144753_openai_reasoner_Li-Paper_prompts.json"
PAPER_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/data/Papers/34831722.pdf"
CHECKLIST="Li-Paper"
BATCH_SIZE=30

# Run the swapped RAG-based extractor and validator
echo "Running swapped RAG-based extractor (Claude) and validator (OpenAI)..."
python rag_extractor_validator_swapped.py --prompts "$PROMPTS_FILE" --paper "$PAPER_FILE" --checklist "$CHECKLIST" --batch-size "$BATCH_SIZE"

echo "Done!"
