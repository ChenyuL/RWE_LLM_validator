# Experiment Scripts for RECORD Validation

This directory contains scripts for running experiments with OpenAI and Claude validators on research papers.

## Scripts

### 1. run_test_experiment.sh

This script runs a test experiment on a small subset of papers (2 papers) to verify that the setup works correctly.

```bash
./run_test_experiment.sh
```

### 2. run_new_experiment.sh

This script runs the full experiment on all 30 papers specified in the `fixed_sampled_papers.json` file.

```bash
./run_new_experiment.sh
```

## Workflow

The experiment workflow consists of the following steps:

1. For each paper:
   - Process the paper with OpenAI extractor and Claude validator
   - Process the paper with Claude extractor and OpenAI validator

2. Run analysis on the results

## Output

The results are saved in the `output` directory with a timestamp. Each experiment creates its own subdirectory.

## Configuration

The scripts use the following files:

- Prompt file: `/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/prompts/20250319_220659_openai_reasoner_Li-Paper_prompts.json`
- Papers list: `/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/output/paper_results/fixed_sampled_papers.json`
- Papers directory: `/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/data/Papers`
- OpenAI-Claude validation script: `/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/test_record_validation_openai_claude_fixed.py`
- Claude-OpenAI validation script: `/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/test_record_validation_claude_openai_fixed.py`
- Analysis script: `/Users/chenyuli/LLMEvaluation/RWE_LLM_validator/run_analysis_only.sh`

## Troubleshooting

If you encounter any issues, check the following:

1. Make sure all the required files exist
2. Check the log files in the output directory
3. Verify that the API keys are correctly set in the `.env` file
