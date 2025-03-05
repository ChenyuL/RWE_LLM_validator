# LLM Validation Framework for Biomedical Research

A self-validating framework for LLM-generated methodology checklists in biomedical research. This framework leverages dual LLM processing with human oversight to enhance the accuracy and reliability of automated checklist generation.

## Features

- **Dual-LLM Validation**: Uses two LLMs to validate each other's outputs
- **FHIR-Compatible Output**: Generates structured evidence output following FHIR standards
- **Comprehensive Metrics**: Calculates kappa score, accuracy, precision, recall, and F1 score
- **PDF Processing**: Built-in support for processing PDF research papers
- **Robust Error Handling**: Comprehensive logging and error management

## Components

1. **Reasoner**
   - Primary: deepseek-reasoner
   - Secondary: o3-mini-2025
   - Handles prompt generation and initial analysis

2. **Validator**
   - Primary: claude-3.5
   - Secondary: gpt-4
   - Validates outputs and calculates agreement metrics

3. **GuidelineProcessor**
   - Processes PDF documents
   - Extracts structured information
   - Manages document chunking

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key
```

## Usage

Basic usage example:

```python
from llm_validation_framework import LLMValidationFramework

# Initialize framework with API keys
api_keys = {
    "openai": "your-openai-key",
    "anthropic": "your-anthropic-key",
    "deepseek": "your-deepseek-key"
}

framework = LLMValidationFramework(api_keys)

# Process a document
results = framework.process_document("path/to/document.pdf")

# Results will be in FHIR-compatible format
print(f"Agreement Level: {results['certainty'][0]['rating']['value']}")
```

## Output Format

The framework generates FHIR-compatible output in JSON format:

```json
{
    "resourceType": "Evidence",
    "status": "active",
    "statisticalAnalysis": {
        "description": "LLM Validation Results",
        "modelCharacteristics": {
            "type": {
                "text": "Inter-rater reliability"
            },
            "value": {
                "kappa_score": 0.85,
                "accuracy": 0.90,
                "precision": 0.88,
                "recall": 0.87,
                "f1": 0.87
            }
        }
    },
    "certainty": [
        {
            "rating": {
                "text": "Agreement Level",
                "value": "Almost Perfect Agreement"
            }
        }
    ]
}
```

## Testing

Run the test script to validate the framework:

```bash
python test_framework.py
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
