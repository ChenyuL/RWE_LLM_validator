#!/bin/bash

# Create a directory for the most recent run
mkdir -p output/latest_run

# Move the most recent final results to the latest_run directory
mv output/20250307_212656_claude-sonnet_validator_34923518.json output/latest_run/
mv output/20250307_212656_full_record_checklist_34923518.json output/latest_run/
mv output/20250307_212656_openai-gpt4o_extractor_34923518.json output/latest_run/
mv output/20250307_212656_openai_claude_report_34923518.json output/latest_run/
mv output/20250307_212656_openai_reasoner_34923518.json output/latest_run/
mv output/20250307_212656_openai_reasoner_34923518_process_log.txt output/latest_run/

# Create a directory for guideline info files
mkdir -p output/guideline_info
mv output/record_guideline_info_*.json output/guideline_info/

# Remove all batch files
rm -f output/*_batch_*_extraction_*.json
rm -f output/*_batch_*_validation_*.json

# Remove old run results (keeping only the latest)
rm -f output/2025030[1-6]_*.json
rm -f output/20250307_0*_*.json
rm -f output/20250307_0*_*_*.json
rm -f output/20250307_0*_*_*_*.txt

# Create an archive directory for other files
mkdir -p output/archive
mv output/20250306_*_process_log.txt output/archive/
mv output/Benoit.2019.ReductionInNephrotoxic_report_modified.json output/archive/

echo "Cleanup completed. Latest results are in output/latest_run/"
echo "Guideline info files are in output/guideline_info/"
