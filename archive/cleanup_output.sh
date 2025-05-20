#!/bin/bash

# Create necessary directories
mkdir -p output/paper_results output/prompts output/reports

# Function to extract paper_id and checklist_name from filename
extract_info() {
    local filename="$1"
    local paper_id=$(echo "$filename" | grep -o '[0-9]\{8\}' | head -1)
    local checklist_name=$(echo "$filename" | grep -o 'Li-Paper\|RECORD\|STROBE\|CHEERS' | head -1)
    
    if [ -n "$paper_id" ] && [ -n "$checklist_name" ]; then
        echo "${paper_id}_${checklist_name}"
    else
        echo ""
    fi
}

# Process files for each paper and checklist
echo "Processing paper files..."
for paper_checklist in $(find output -name "*batch_7_validation_*.json" | grep -v "paper_results" | 
                         awk -F'_batch_7_validation_' '{print $2}' | 
                         awk -F'.json' '{print $1}' | 
                         sort | uniq); do
    
    # Extract paper_id and checklist_name
    paper_id=$(echo "$paper_checklist" | awk -F'_' '{print $1}')
    checklist_name=$(echo "$paper_checklist" | awk -F'_' '{print $2}')
    
    if [ -z "$checklist_name" ]; then
        # If checklist_name is empty, try to extract it differently
        checklist_name=$(echo "$paper_checklist" | grep -o 'Li-Paper\|RECORD\|STROBE\|CHEERS')
    fi
    
    if [ -n "$paper_id" ] && [ -n "$checklist_name" ]; then
        dir_name="output/paper_results/${paper_id}_${checklist_name}"
        mkdir -p "$dir_name"
        
        # Find the latest timestamp for this paper_id and checklist_name
        latest_timestamp=$(find output -name "*_${paper_id}_${checklist_name}.json" | 
                          grep -v "paper_results" | 
                          awk -F'/' '{print $NF}' | 
                          awk -F'_' '{print $1"_"$2}' | 
                          sort -r | 
                          head -1)
        
        if [ -n "$latest_timestamp" ]; then
            echo "Processing files for ${paper_id}_${checklist_name} with timestamp $latest_timestamp"
            
            # Copy reasoner file
            reasoner_file=$(find output -name "${latest_timestamp}_openai_reasoner_${paper_id}_${checklist_name}.json" | grep -v "paper_results")
            if [ -n "$reasoner_file" ]; then
                echo "  Copying $reasoner_file to $dir_name/"
                cp "$reasoner_file" "$dir_name/"
            fi
            
            # Copy extractor file
            extractor_file=$(find output -name "${latest_timestamp}_openai-gpt4o_extractor_${paper_id}_${checklist_name}.json" | grep -v "paper_results")
            if [ -n "$extractor_file" ]; then
                echo "  Copying $extractor_file to $dir_name/"
                cp "$extractor_file" "$dir_name/"
            fi
            
            # Copy validator file
            validator_file=$(find output -name "${latest_timestamp}_claude-sonnet_validator_${paper_id}_${checklist_name}.json" | grep -v "paper_results")
            if [ -n "$validator_file" ]; then
                echo "  Copying $validator_file to $dir_name/"
                cp "$validator_file" "$dir_name/"
            fi
            
            # Copy report file
            report_file=$(find output -name "${latest_timestamp}_openai_claude_report_${paper_id}_${checklist_name}.json" | grep -v "paper_results" | grep -v "reports")
            if [ -n "$report_file" ]; then
                echo "  Copying $report_file to $dir_name/"
                cp "$report_file" "$dir_name/"
                echo "  Copying $report_file to output/reports/"
                cp "$report_file" "output/reports/"
            fi
            
            # Also look for claude_openai_report files (for the new test)
            report_file=$(find output -name "${latest_timestamp}_claude_openai_report_${paper_id}_${checklist_name}.json" | grep -v "paper_results" | grep -v "reports")
            if [ -n "$report_file" ]; then
                echo "  Copying $report_file to $dir_name/"
                cp "$report_file" "$dir_name/"
                echo "  Copying $report_file to output/reports/"
                cp "$report_file" "output/reports/"
            fi
            
            # Copy checklist file
            checklist_file=$(find output -name "${latest_timestamp}_full_${checklist_name}_checklist_${paper_id}.json" | grep -v "paper_results")
            if [ -n "$checklist_file" ]; then
                echo "  Copying $checklist_file to $dir_name/"
                cp "$checklist_file" "$dir_name/"
            else
                # Try alternative naming pattern
                checklist_file=$(find output -name "${latest_timestamp}_full_${checklist_name}_checklist_${paper_id}*.json" | grep -v "paper_results")
                if [ -n "$checklist_file" ]; then
                    echo "  Copying $checklist_file to $dir_name/"
                    cp "$checklist_file" "$dir_name/"
                fi
            fi
            
            # Copy batch_7_validation file
            batch7_file=$(find output -name "*batch_7_validation_${paper_id}_${checklist_name}.json" | 
                         grep -v "paper_results" | 
                         sort -r | 
                         head -1)
            if [ -n "$batch7_file" ]; then
                echo "  Copying $batch7_file to $dir_name/"
                cp "$batch7_file" "$dir_name/"
            fi
        else
            echo "No files found for ${paper_id}_${checklist_name}"
        fi
    fi
done

# Copy prompt files
echo "Copying prompt files to output/prompts/"
find output -name "*prompts.json" | grep -v "paper_results" | grep -v "output/prompts" | 
while read prompt_file; do
    echo "Copying $prompt_file to output/prompts/"
    cp "$prompt_file" "output/prompts/"
done

# Copy all report files to output/reports
echo "Copying all report files to output/reports/"
find output -name "*_report_*.json" | grep -v "paper_results" | grep -v "output/reports" | 
while read report_file; do
    echo "Copying $report_file to output/reports/"
    cp "$report_file" "output/reports/"
done

echo "Done!"
