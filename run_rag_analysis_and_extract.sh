#!/bin/bash
# run_rag_analysis_and_extract.sh
# Script to run RAG analysis and extract model answers

# Make the scripts executable
chmod +x copy_rag_reports_and_analyze.sh
chmod +x extract_rag_model_answers.py

# Run the copy_rag_reports_and_analyze.sh script
echo "Step 1: Copying RAG reports and running analysis..."
./copy_rag_reports_and_analyze.sh

# Run the extract_rag_model_answers.py script
echo "Step 2: Extracting model answers from RAG reports..."
python extract_rag_model_answers.py

# Copy the RAG analysis summary to the main directory
echo "Step 3: Copying RAG analysis summary to main directory..."
cp rag_li_paper_analysis_summary.md li_paper_analysis_summary.md

# Copy the RAG visualization images to the main directory
echo "Step 4: Copying RAG visualization images to main directory..."
cp rag_*.png .

echo "All done!"
echo "Analysis results are in li_paper_analysis_summary.md and visualizations are in PNG files."
echo "Model answers comparison is in rag_model_answers_comparison.xlsx and rag_model_answers_comparison.csv."
