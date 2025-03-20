#!/bin/bash
# run_rag_batch.sh
# Script to run the RAG-based batch processing on the sampled papers

# Make the Python scripts executable
chmod +x rag_extractor_validator.py
chmod +x run_rag_batch.py

# Check if required packages are installed
if ! python -c "import PyPDF2" &> /dev/null; then
    echo "Installing PyPDF2..."
    pip install PyPDF2
fi

if ! python -c "import numpy" &> /dev/null; then
    echo "Installing numpy..."
    pip install numpy
fi

if ! python -c "import tqdm" &> /dev/null; then
    echo "Installing tqdm..."
    pip install tqdm
fi

if ! python -c "import openai" &> /dev/null; then
    echo "Installing openai..."
    pip install openai
fi

if ! python -c "import anthropic" &> /dev/null; then
    echo "Installing anthropic..."
    pip install anthropic
fi

# Create output directories
mkdir -p output/embeddings

# Set variables
PROMPTS_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250319_220659_openai_reasoner_Li-Paper_prompts.json"
PAPERS_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/paper_results/fixed_sampled_papers.json"
PAPERS_DIR="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/data/Papers"
CHECKLIST="Li-Paper"
BATCH_SIZE=5
MAX_WORKERS=2 # Set to higher value for parallel processing

# Run the RAG-based batch processing
echo "Running RAG-based batch processing..."
python run_rag_batch.py \
    --papers "$PAPERS_FILE" \
    --prompts "$PROMPTS_FILE" \
    --checklist "$CHECKLIST" \
    --batch-size "$BATCH_SIZE" \
    --max-workers "$MAX_WORKERS" \
    --papers-dir "$PAPERS_DIR"

echo "Done!"
