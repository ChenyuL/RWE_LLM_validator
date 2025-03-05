#!/usr/bin/env python3
import json
from pathlib import Path
from llm_validation_framework import LLMValidationFramework, GuidelineProcessor

class MockEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

def save_json(data, filename):
    """Save data to JSON file with proper formatting"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python process_record.py <path_to_biomedical_paper>")
        sys.exit(1)
        
    paper_path = sys.argv[1]
    
    # Initialize framework with mock embeddings for testing
    framework = LLMValidatithonFramework(embeddings=MockEmbeddings())
    
    print(f'Processing biomedical paper: {paper_path}')
    
    # Process the paper and get results
    results = framework.process_document(paper_path)
    
    # Save outputs
    output_base = f'output/{Path(paper_path).stem}'
    
    # Save analysis results
    analysis_output = {
        'paper_metadata': results['paper_metadata'],
        'analysis_results': results['reasoning']
    }
    save_json(analysis_output, f'{output_base}_analysis.json')
    
    # Save validation results
    validation_output = {
        'paper_metadata': results['paper_metadata'],
        'validation_metrics': results['statisticalAnalysis']['modelCharacteristics']['value'],
        'agreement_level': results['certainty'][0]['rating']['value']
    }
    save_json(validation_output, f'{output_base}_validation.json')
    
    print('\nOutputs saved:')
    print(f'1. Analysis results: {output_base}_analysis.json')
    print(f'2. Validation results: {output_base}_validation.json')
    
    # Print summary metrics
    print('\nValidation Metrics:')
    metrics = results['statisticalAnalysis']['modelCharacteristics']['value']
    for key, value in metrics.items():
        print(f'{key}: {value:.3f}')
    
    print('\nAgreement Level:', results['certainty'][0]['rating']['value'])

if __name__ == '__main__':
    main()
