#!/usr/bin/env python3
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from llm_validation_framework import LLMValidationFramework

# Load environment variables
load_dotenv()

def load_api_keys() -> dict:
    """Load API keys from environment variables"""
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY")
    }

def save_results(results: dict, output_path: str):
    """Save validation results to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    # Initialize framework
    api_keys = load_api_keys()
    framework = LLMValidationFramework(api_keys)
    
    # Example document paths
    test_docs = [
        "path/to/test/document1.pdf",
        "path/to/test/document2.pdf"
    ]
    
    # Process each document
    for doc_path in test_docs:
        try:
            print(f"\nProcessing document: {doc_path}")
            
            # Validate document
            results = framework.process_document(doc_path)
            
            # Save results
            output_path = f"results_{Path(doc_path).stem}.json"
            save_results(results, output_path)
            
            # Print summary
            print("\nValidation Summary:")
            print(f"Overall Agreement Level: {results['certainty'][0]['rating']['value']}")
            print("\nMetrics:")
            metrics = results['statisticalAnalysis']['modelCharacteristics']['value']
            for metric, value in metrics.items():
                print(f"{metric}: {value:.3f}")
                
        except Exception as e:
            print(f"Error processing {doc_path}: {str(e)}")

if __name__ == "__main__":
    main()
