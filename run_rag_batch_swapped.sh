#!/bin/bash
# run_rag_batch_swapped.sh
# Script to run the swapped RAG-based batch processing on the sampled papers

# Make the Python scripts executable
chmod +x rag_extractor_validator_swapped.py
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
MAX_WORKERS=20  # Set to higher value for parallel processing

# Create a modified version of run_rag_batch.py for the swapped models
cat > run_rag_batch_swapped.py << 'EOF'
#!/usr/bin/env python
# run_rag_batch_swapped.py
# Script to run the swapped RAG-based extractor and validator on multiple papers

import os
import json
import argparse
import subprocess
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_batch_swapped.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_batch_swapped")

def process_paper(paper_path, prompts_file, checklist, batch_size):
    """
    Process a single paper using the swapped RAG-based extractor and validator.
    """
    paper_id = os.path.splitext(os.path.basename(paper_path))[0]
    logger.info(f"Processing paper: {paper_id}")
    
    cmd = [
        "python", "rag_extractor_validator_swapped.py",
        "--prompts", prompts_file,
        "--paper", paper_path,
        "--checklist", checklist,
        "--batch-size", str(batch_size)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully processed paper: {paper_id}")
        logger.debug(f"Output: {result.stdout}")
        return {
            "paper_id": paper_id,
            "success": True,
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing paper {paper_id}: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return {
            "paper_id": paper_id,
            "success": False,
            "error": str(e),
            "stderr": e.stderr
        }

def main():
    """
    Main function to run the swapped RAG-based extractor and validator on multiple papers.
    """
    parser = argparse.ArgumentParser(description='Run swapped RAG-based extractor and validator on multiple papers')
    parser.add_argument('--papers', type=str, required=True, help='Path to JSON file with list of paper IDs or directory containing papers')
    parser.add_argument('--prompts', type=str, required=True, help='Path to prompts file')
    parser.add_argument('--checklist', type=str, default='Li-Paper', help='Checklist type (default: Li-Paper)')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for processing items (default: 5)')
    parser.add_argument('--max-workers', type=int, default=1, help='Maximum number of parallel workers (default: 1)')
    parser.add_argument('--papers-dir', type=str, default='data/Papers', help='Directory containing paper PDFs (default: data/Papers)')
    
    args = parser.parse_args()
    
    # Get list of papers to process
    paper_ids = []
    if os.path.isdir(args.papers):
        # If papers is a directory, get all PDF files
        papers_dir = args.papers
        paper_ids = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
        paper_paths = [os.path.join(papers_dir, f) for f in paper_ids]
    else:
        # If papers is a JSON file, load paper IDs
        with open(args.papers, 'r') as f:
            papers_data = json.load(f)
            
        if isinstance(papers_data, list):
            # If it's a list of paper IDs
            paper_ids = papers_data
        elif isinstance(papers_data, dict) and "papers" in papers_data:
            # If it's a dict with a "papers" key
            paper_ids = papers_data["papers"]
        else:
            # Try to extract paper IDs from the JSON structure
            paper_ids = []
            for key, value in papers_data.items():
                if isinstance(value, dict) and "paper_id" in value:
                    paper_ids.append(value["paper_id"])
                elif isinstance(value, str) and value.isdigit():
                    paper_ids.append(value)
                elif isinstance(key, str) and key.isdigit():
                    paper_ids.append(key)
        
        # Convert paper IDs to paths
        paper_paths = []
        for paper_id in paper_ids:
            # Check if paper_id already has .pdf extension
            if paper_id.endswith('.pdf'):
                paper_path = os.path.join(args.papers_dir, paper_id)
            else:
                paper_path = os.path.join(args.papers_dir, f"{paper_id}.pdf")
            
            if os.path.exists(paper_path):
                paper_paths.append(paper_path)
            else:
                logger.warning(f"Paper file not found: {paper_path}")
    
    logger.info(f"Found {len(paper_paths)} papers to process")
    
    # Process papers
    results = []
    
    if args.max_workers > 1:
        # Process papers in parallel
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for paper_path in paper_paths:
                future = executor.submit(
                    process_paper, paper_path, args.prompts, args.checklist, args.batch_size
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed paper: {result['paper_id']}")
                except Exception as e:
                    logger.error(f"Error in future: {e}")
    else:
        # Process papers sequentially
        for paper_path in paper_paths:
            result = process_paper(paper_path, args.prompts, args.checklist, args.batch_size)
            results.append(result)
    
    # Save results
    output_file = f"rag_swapped_batch_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved batch results to {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print("\n" + "="*80)
    print(f"BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total papers: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("="*80)

if __name__ == "__main__":
    main()
EOF

# Make the Python script executable
chmod +x run_rag_batch_swapped.py

# Run the swapped RAG-based batch processing
echo "Running swapped RAG-based batch processing..."
python run_rag_batch_swapped.py \
    --papers "$PAPERS_FILE" \
    --prompts "$PROMPTS_FILE" \
    --checklist "$CHECKLIST" \
    --batch-size "$BATCH_SIZE" \
    --max-workers "$MAX_WORKERS" \
    --papers-dir "$PAPERS_DIR"

echo "Done!"
