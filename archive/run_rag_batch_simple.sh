#!/bin/bash
# run_rag_batch_simple.sh
# A simplified version of run_rag_batch_improved.sh that focuses on argument parsing

# Default values
DEFAULT_PROMPTS_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250322_144753_openai_reasoner_Li-Paper_prompts.json"
DEFAULT_PAPERS_FILE="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/paper_results/fixed_sampled_papers.json"
DEFAULT_PAPERS_DIR="/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/data/Papers"
DEFAULT_CHECKLIST="Li-Paper"
DEFAULT_MAX_WORKERS=10
DEFAULT_EXTRACTOR_MODEL="gpt-4o"
DEFAULT_EXTRACTOR_PROVIDER="openai"
DEFAULT_VALIDATOR_MODEL="claude-3-5-sonnet-20241022"
DEFAULT_VALIDATOR_PROVIDER="anthropic"

# Initialize variables with default values
PROMPTS_FILE=$DEFAULT_PROMPTS_FILE
PAPERS_FILE=$DEFAULT_PAPERS_FILE
PAPERS_DIR=$DEFAULT_PAPERS_DIR
CHECKLIST=$DEFAULT_CHECKLIST
MAX_WORKERS=$DEFAULT_MAX_WORKERS
EXTRACTOR_MODEL=$DEFAULT_EXTRACTOR_MODEL
EXTRACTOR_PROVIDER=$DEFAULT_EXTRACTOR_PROVIDER
VALIDATOR_MODEL=$DEFAULT_VALIDATOR_MODEL
VALIDATOR_PROVIDER=$DEFAULT_VALIDATOR_PROVIDER

# Function to print usage
print_usage() {
    echo "Usage: $0 --prompts <file> --papers <file> [--papers-dir <dir>] [--checklist <type>] [--max-workers <num>] [--extractor-model <model>] [--extractor-provider <provider>] [--validator-model <model>] [--validator-provider <provider>]"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    # Print the current argument being processed
    echo "Processing argument: '$1'"
    
    case "$1" in
        --prompts)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --prompts requires a value"
                print_usage
                exit 1
            fi
            PROMPTS_FILE="$2"
            echo "  Setting PROMPTS_FILE to: $PROMPTS_FILE"
            shift 2
            ;;
        --papers)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --papers requires a value"
                print_usage
                exit 1
            fi
            PAPERS_FILE="$2"
            echo "  Setting PAPERS_FILE to: $PAPERS_FILE"
            shift 2
            ;;
        --papers-dir)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --papers-dir requires a value"
                print_usage
                exit 1
            fi
            PAPERS_DIR="$2"
            echo "  Setting PAPERS_DIR to: $PAPERS_DIR"
            shift 2
            ;;
        --checklist)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --checklist requires a value"
                print_usage
                exit 1
            fi
            CHECKLIST="$2"
            echo "  Setting CHECKLIST to: $CHECKLIST"
            shift 2
            ;;
        --max-workers)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --max-workers requires a value"
                print_usage
                exit 1
            fi
            MAX_WORKERS="$2"
            echo "  Setting MAX_WORKERS to: $MAX_WORKERS"
            shift 2
            ;;
        --extractor-model)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --extractor-model requires a value"
                print_usage
                exit 1
            fi
            EXTRACTOR_MODEL="$2"
            echo "  Setting EXTRACTOR_MODEL to: $EXTRACTOR_MODEL"
            shift 2
            ;;
        --extractor-provider)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --extractor-provider requires a value"
                print_usage
                exit 1
            fi
            EXTRACTOR_PROVIDER="$2"
            echo "  Setting EXTRACTOR_PROVIDER to: $EXTRACTOR_PROVIDER"
            shift 2
            ;;
        --validator-model)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --validator-model requires a value"
                print_usage
                exit 1
            fi
            VALIDATOR_MODEL="$2"
            echo "  Setting VALIDATOR_MODEL to: $VALIDATOR_MODEL"
            shift 2
            ;;
        --validator-provider)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --validator-provider requires a value"
                print_usage
                exit 1
            fi
            VALIDATOR_PROVIDER="$2"
            echo "  Setting VALIDATOR_PROVIDER to: $VALIDATOR_PROVIDER"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

echo ""
echo "Final parameter values:"
echo "PROMPTS_FILE: $PROMPTS_FILE"
echo "PAPERS_FILE: $PAPERS_FILE"
echo "PAPERS_DIR: $PAPERS_DIR"
echo "CHECKLIST: $CHECKLIST"
echo "MAX_WORKERS: $MAX_WORKERS"
echo "EXTRACTOR_MODEL: $EXTRACTOR_MODEL"
echo "EXTRACTOR_PROVIDER: $EXTRACTOR_PROVIDER"
echo "VALIDATOR_MODEL: $VALIDATOR_MODEL"
echo "VALIDATOR_PROVIDER: $VALIDATOR_PROVIDER"

echo ""
echo "This is a diagnostic script. No actual processing will be performed."
echo "If all parameters look correct, you can use run_rag_batch_improved.sh with the same arguments."
