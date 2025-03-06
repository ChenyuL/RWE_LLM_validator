#!/usr/bin/env python
# test_record_validation_deepseek.py

import os
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.config import API_KEYS, GUIDELINES_PATH, PAPERS_PATH, OUTPUT_PATH, LLM_CONFIGS
from src.framework import LLMValidationFramework
from src.agents.reasoner_modified import Reasoner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("record_test_deepseek.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("record_test_deepseek")

# Create a custom handler to capture logs for LLM1 process
class LLM1LogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
        
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
        
    def get_logs(self):
        return "\n".join(self.logs)

# Create the custom handler
llm1_log_handler = LLM1LogHandler()
llm1_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
llm1_log_handler.setLevel(logging.INFO)

# Add the handler to the logger
logger.addHandler(llm1_log_handler)

class DeepSeekExtractor:
    """
    A simplified extractor that uses DeepSeek API instead of OpenAI or Anthropic.
    """
    
    def __init__(self, api_key, model="deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def extract_information(self, paper_text, prompt, item_id):
        """
        Extract information from paper text based on a prompt.
        """
        import requests
        
        self.logger.info(f"Extracting information for guideline item: {item_id}")
        
        # Simplified extraction - just use the first 8000 characters of the paper
        paper_excerpt = paper_text[:8000]
        
        # Create a prompt for DeepSeek
        deepseek_prompt = f"""
        {prompt}
        
        PAPER TEXT:
        {paper_excerpt}
        
        Please provide your extraction in the following JSON format:
        {{
            "paper_title": "Title of the paper",
            "record_item_id": "{item_id}",
            "extracted_content": {{
                "compliance": "yes", "partial", "no", or "unknown",
                "evidence": [
                    {{
                        "quote": "direct quote from paper",
                        "location": "section/page information if available"
                    }}
                ],
                "confidence": 0.0-1.0 (your confidence in this assessment),
                "reasoning": "explanation of your assessment"
            }}
        }}
        """
        
        # Call DeepSeek API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": deepseek_prompt}],
            "temperature": 0.2,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                
                # Try to parse the result as JSON
                try:
                    extraction = json.loads(result)
                    return extraction
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    import re
                    json_match = re.search(r'(\{.*\})', result, re.DOTALL)
                    if json_match:
                        try:
                            extracted_json = json_match.group(1)
                            extraction = json.loads(extracted_json)
                            return extraction
                        except json.JSONDecodeError:
                            pass
                
                # If parsing fails, return a basic structure
                return {
                    "paper_title": "Unknown",
                    "record_item_id": item_id,
                    "extracted_content": {
                        "compliance": "unknown",
                        "evidence": [],
                        "confidence": 0.0,
                        "reasoning": f"Failed to parse result: {result[:500]}..."
                    }
                }
            else:
                self.logger.error(f"DeepSeek API call failed with status code: {response.status_code}, response: {response.text}")
                return {
                    "paper_title": "Unknown",
                    "record_item_id": item_id,
                    "extracted_content": {
                        "compliance": "unknown",
                        "evidence": [],
                        "confidence": 0.0,
                        "reasoning": f"API call failed: {response.text}"
                    }
                }
        except Exception as e:
            self.logger.error(f"Error calling DeepSeek API: {e}")
            return {
                "paper_title": "Unknown",
                "record_item_id": item_id,
                "extracted_content": {
                    "compliance": "unknown",
                    "evidence": [],
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                }
            }

class DeepSeekValidator:
    """
    A simplified validator that uses DeepSeek API instead of OpenAI or Anthropic.
    """
    
    def __init__(self, api_key, model="deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate(self, extraction, guideline_item, item_id):
        """
        Validate an extraction against a guideline item.
        """
        import requests
        
        self.logger.info(f"Validating extraction for guideline item: {item_id}")
        
        # Extract key information
        paper_title = extraction.get("paper_title", "Unknown")
        extracted_content = extraction.get("extracted_content", {})
        compliance = extracted_content.get("compliance", "unknown")
        evidence = extracted_content.get("evidence", [])
        confidence = extracted_content.get("confidence", 0.0)
        reasoning = extracted_content.get("reasoning", "")
        
        # Extract guideline information
        description = guideline_item.get("description", "")
        
        # Build prompt for validation
        deepseek_prompt = f"""
        You are an expert validator for biomedical research reporting guidelines.
        
        GUIDELINE ITEM: {item_id}
        DESCRIPTION: {description}
        
        EXTRACTION RESULTS:
        - Paper Title: {paper_title}
        - Compliance: {compliance}
        - Confidence: {confidence}
        - Evidence: {json.dumps(evidence, indent=2)}
        - Reasoning: {reasoning}
        
        VALIDATION TASK:
        1. Evaluate whether the compliance assessment is correct based on the evidence provided.
        2. Assess whether the evidence is sufficient and relevant to the guideline item.
        3. Provide a final assessment of compliance.
        
        Please provide your validation in the following JSON format:
        {{
            "paper_title": "{paper_title}",
            "record_item_id": "{item_id}",
            "validate_result": "yes", "partial", "no", or "unknown",
            "Reason": "your assessment of the extraction"
        }}
        """
        
        # Call DeepSeek API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": deepseek_prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                
                # Try to parse the result as JSON
                try:
                    validation = json.loads(result)
                    
                    # Save validation result
                    self.validation_results[item_id] = validation
                    
                    return validation
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    import re
                    json_match = re.search(r'(\{.*\})', result, re.DOTALL)
                    if json_match:
                        try:
                            extracted_json = json_match.group(1)
                            validation = json.loads(extracted_json)
                            
                            # Save validation result
                            self.validation_results[item_id] = validation
                            
                            return validation
                        except json.JSONDecodeError:
                            pass
                
                # If parsing fails, return a basic structure
                validation = {
                    "paper_title": paper_title,
                    "record_item_id": item_id,
                    "validate_result": compliance,
                    "Reason": f"Failed to parse result: {result[:500]}..."
                }
                
                # Save validation result
                self.validation_results[item_id] = validation
                
                return validation
            else:
                self.logger.error(f"DeepSeek API call failed with status code: {response.status_code}, response: {response.text}")
                validation = {
                    "paper_title": paper_title,
                    "record_item_id": item_id,
                    "validate_result": "unknown",
                    "Reason": f"API call failed: {response.text}"
                }
                
                # Save validation result
                self.validation_results[item_id] = validation
                
                return validation
        except Exception as e:
            self.logger.error(f"Error calling DeepSeek API: {e}")
            validation = {
                "paper_title": paper_title,
                "record_item_id": item_id,
                "validate_result": "unknown",
                "Reason": f"Error: {str(e)}"
            }
            
            # Save validation result
            self.validation_results[item_id] = validation
            
            return validation
    
    def calculate_metrics(self, validation_results):
        """
        Calculate overall metrics for the validation results.
        """
        self.logger.info("Calculating overall validation metrics")
        
        # Count compliance categories
        counts = {"yes": 0, "partial": 0, "no": 0, "unknown": 0}
        
        for item_id, result in validation_results.items():
            compliance = result.get("validate_result", "unknown")
            counts[compliance] = counts.get(compliance, 0) + 1
        
        # Calculate percentages
        total_items = len(validation_results)
        percentages = {}
        
        if total_items > 0:
            for category, count in counts.items():
                percentages[f"{category}_percent"] = (count / total_items) * 100
        
        # Calculate overall compliance rate (full and partial)
        if total_items > 0:
            compliance_rate = ((counts["yes"] + (counts["partial"] * 0.5)) / total_items) * 100
        else:
            compliance_rate = 0.0
        
        # Compile metrics
        metrics = {
            "total_items": total_items,
            "fully_compliant": counts["yes"],
            "partially_compliant": counts["partial"],
            "non_compliant": counts["no"],
            "unknown": counts["unknown"],
            "compliance_rate": compliance_rate,
            "average_confidence": 0.0,  # Not calculated for this simplified version
            "items_for_review": 0,  # Not calculated for this simplified version
            "review_percentage": 0.0  # Not calculated for this simplified version
        }
        
        # Add percentages
        metrics.update(percentages)
        
        return metrics

class DeepSeekLLMValidationFramework:
    """
    A modified version of the LLMValidationFramework that uses DeepSeek API for all agents.
    """
    
    def __init__(self, api_keys, config=None):
        """
        Initialize the framework with API keys for LLM providers.
        """
        self.logger = logging.getLogger(__name__)
        self.api_keys = api_keys
        self.config = config or {}
        
        # Initialize PDF processor
        from src.utils.pdf_processor import PDFProcessor
        self.pdf_processor = PDFProcessor()
        
        # Initialize agents with DeepSeek
        self.reasoner = Reasoner(api_keys, self.config.get("reasoner", {}))
        self.extractor = DeepSeekExtractor(api_keys["deepseek"])
        self.validator = DeepSeekValidator(api_keys["deepseek"])
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    def process_guideline(self, guideline_type):
        """
        Process a specific guideline (e.g., RECORD) to generate prompts.
        """
        self.logger.info(f"Processing {guideline_type} guideline")
        
        # Get all guideline PDFs
        guideline_path = os.path.join(GUIDELINES_PATH, guideline_type)
        if not os.path.exists(guideline_path):
            raise FileNotFoundError(f"Guideline path not found: {guideline_path}")
            
        guideline_files = [f for f in os.listdir(guideline_path) if f.endswith('.pdf')]
        if not guideline_files:
            raise FileNotFoundError(f"No PDF files found in {guideline_path}")
        
        # Process all guideline PDFs
        guideline_texts = []
        for file in guideline_files:
            file_path = os.path.join(guideline_path, file)
            text = self.pdf_processor.extract_text(file_path)
            guideline_texts.append(text)
        
        # Use reasoner to process guidelines and generate prompts
        guideline_items = self.reasoner.extract_guideline_items(guideline_texts)
        prompts = self.reasoner.generate_prompts(guideline_items)
        
        return {
            "guideline_type": guideline_type,
            "items": guideline_items,
            "prompts": prompts
        }
    
    def process_paper(self, paper_path, guideline_prompts):
        """
        Process a research paper using the prompts generated from guidelines.
        """
        self.logger.info(f"Processing paper: {os.path.basename(paper_path)}")
        
        # Extract text from paper
        paper_text = self.pdf_processor.extract_text(paper_path)
        
        # Use extractor to extract information from paper
        # Process each guideline item separately as requested
        extracted_info = {}
        for item_id, prompt in guideline_prompts["prompts"].items():
            extracted_info[item_id] = self.extractor.extract_information(paper_text, prompt, item_id)
        
        return {
            "paper_path": paper_path,
            "extracted_info": extracted_info
        }
    
    def validate_extraction(self, paper_info, guideline_info):
        """
        Validate the extracted information against the guideline.
        """
        self.logger.info("Validating extracted information")
        
        validation_results = {}
        for item_id, extraction in paper_info["extracted_info"].items():
            # Get corresponding guideline item
            guideline_item = next((item for item in guideline_info["items"] if item["id"] == item_id), None)
            if guideline_item is None:
                self.logger.warning(f"No guideline item found for ID: {item_id}")
                continue
                
            # Validate the extraction against the guideline item
            validation_results[item_id] = self.validator.validate(
                extraction, 
                guideline_item,
                item_id
            )
        
        # Calculate overall metrics
        metrics = self.validator.calculate_metrics(validation_results)
        
        return {
            "validation_results": validation_results,
            "metrics": metrics
        }
    
    def generate_report(self, paper_info, guideline_info, validation_results):
        """
        Generate a final report based on validation results.
        """
        paper_name = os.path.basename(paper_info["paper_path"])
        
        report = {
            "paper": paper_name,
            "guideline": guideline_info["guideline_type"],
            "validation_summary": validation_results["metrics"],
            "items": {}
        }
        
        # Compile detailed item-by-item results
        for item_id, validation in validation_results["validation_results"].items():
            guideline_item = next((item for item in guideline_info["items"] if item["id"] == item_id), None)
            
            report["items"][item_id] = {
                "description": guideline_item["description"] if guideline_item else "Unknown",
                "compliance": validation.get("validate_result", "unknown"),
                "confidence": 0.0,  # Not provided in this simplified version
                "evidence": [],  # Not provided in this simplified version
                "disagreements": []  # Not provided in this simplified version
            }
        
        return report
    
    def _save_results(self, paper_path, guideline_info, paper_info, validation_results, final_report):
        """
        Save all results to output files.
        """
        import datetime
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        paper_name = os.path.splitext(os.path.basename(paper_path))[0]
        
        # Save guideline prompts (LLM1-reasoner)
        reasoner_filename = f"{timestamp}_deepseek-reasoner_reasoner_{paper_name}_prompts.json"
        with open(os.path.join(OUTPUT_PATH, reasoner_filename), "w") as f:
            json.dump(guideline_info["prompts"], f, indent=2)
        
        # Save LLM1 process log
        log_filename = f"{timestamp}_deepseek-reasoner_reasoner_{paper_name}_process_log.txt"
        with open(os.path.join(OUTPUT_PATH, log_filename), "w") as f:
            f.write(llm1_log_handler.get_logs())
        
        # Save extracted information (LLM2-extractor)
        extractor_filename = f"{timestamp}_deepseek-chat_extractor_{paper_name}_extraction.json"
        with open(os.path.join(OUTPUT_PATH, extractor_filename), "w") as f:
            json.dump(paper_info["extracted_info"], f, indent=2)
        
        # Save validation results (LLM3-validator)
        validator_filename = f"{timestamp}_deepseek-chat_validator_{paper_name}_validation.json"
        with open(os.path.join(OUTPUT_PATH, validator_filename), "w") as f:
            json.dump(validation_results, f, indent=2)
        
        # Save final report
        report_filename = f"{timestamp}_deepseek_report_{paper_name}.json"
        with open(os.path.join(OUTPUT_PATH, report_filename), "w") as f:
            json.dump(final_report, f, indent=2)
        
        self.logger.info(f"All results saved to {OUTPUT_PATH}")
        self.logger.info(f"Reasoner output: {reasoner_filename}")
        self.logger.info(f"Process log: {log_filename}")
        self.logger.info(f"Extractor output: {extractor_filename}")
        self.logger.info(f"Validator output: {validator_filename}")
        self.logger.info(f"Final report: {report_filename}")

def load_prompts_from_file(prompts_file):
    """
    Load prompts from a previously saved file.
    
    Args:
        prompts_file: Path to the prompts file
        
    Returns:
        Dictionary containing the prompts and a mock guideline_info structure
    """
    logger.info(f"Loading prompts from {prompts_file}")
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # Create a mock guideline_info structure with the loaded prompts
    guideline_info = {
        "guideline_type": "RECORD",
        "items": [],  # This will be populated with dummy items based on prompt IDs
        "prompts": prompts
    }
    
    # Create dummy guideline items based on the prompt IDs
    for item_id, prompt_data in prompts.items():
        guideline_info["items"].append({
            "id": item_id,
            "description": prompt_data.get("content", ""),
            "category": prompt_data.get("category", ""),
            "notes": ""
        })
    
    logger.info(f"Loaded {len(prompts)} prompts")
    return guideline_info

def run_test(prompts_file=None):
    """
    Run the validation test.
    
    Args:
        prompts_file: Optional path to a file containing pre-generated prompts
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    logger.info("Initializing DeepSeek LLM Validation Framework")
    framework = DeepSeekLLMValidationFramework(API_KEYS)
    
    # Step 1: Process RECORD guidelines or load existing prompts
    if prompts_file:
        # Load prompts from file (LLM1 output)
        guideline_info = load_prompts_from_file(prompts_file)
    else:
        # Process guidelines with LLM1
        logger.info("Processing RECORD guidelines")
        guideline_info = framework.process_guideline("RECORD")
        
        # Save guideline info for inspection
        with open(os.path.join(OUTPUT_PATH, "record_guideline_info_deepseek.json"), "w") as f:
            # Convert complex types to strings for JSON serialization
            simplified_info = {
                "guideline_type": guideline_info["guideline_type"],
                "items_count": len(guideline_info["items"]),
                "prompts_count": len(guideline_info["prompts"]),
                "sample_items": guideline_info["items"][:3] if len(guideline_info["items"]) > 3 else guideline_info["items"]
            }
            json.dump(simplified_info, f, indent=2)
        
        logger.info(f"Extracted {len(guideline_info['items'])} guideline items")
        logger.info(f"Generated {len(guideline_info['prompts'])} prompts")
    
    # Step 2: Process a sample paper against the guidelines
    # List available papers
    paper_files = [f for f in os.listdir(PAPERS_PATH) if f.endswith('.pdf')]
    
    if not paper_files:
        logger.error("No PDF papers found in papers directory")
        return
    
    # Select first paper for testing
    paper_path = os.path.join(PAPERS_PATH, paper_files[0])
    logger.info(f"Processing paper: {paper_path}")
    
    try:
        # Process the paper
        paper_info = framework.process_paper(paper_path, guideline_info)
        
        # Step 3: Validate the extraction
        logger.info("Validating extracted information")
        validation_results = framework.validate_extraction(paper_info, guideline_info)
        
        # Step 4: Generate final report
        logger.info("Generating final report")
        final_report = framework.generate_report(paper_info, guideline_info, validation_results)
        
        # Save the results
        framework._save_results(paper_path, guideline_info, paper_info, validation_results, final_report)
        
        # Print summary
        print("\n" + "="*80)
        print(f"VALIDATION RESULTS FOR {os.path.basename(paper_path)}")
        print("="*80)
        
        # Print validation summary
        if "validation_summary" in final_report:
            metrics = final_report["validation_summary"]
            print("\nValidation Metrics:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    print(f"  {metric_name}: {metric_value:.3f}")
                else:
                    print(f"  {metric_name}: {metric_value}")
        
        # Count compliance by type
        if "items" in final_report:
            compliances = {"yes": 0, "partial": 0, "no": 0, "unknown": 0}
            
            for item_id, item_data in final_report["items"].items():
                compliance = item_data.get("compliance", "unknown")
                if compliance in compliances:
                    compliances[compliance] += 1
                else:
                    compliances["unknown"] += 1
            
            print("\nCompliance Summary:")
            print(f"  Total Items: {len(final_report['items'])}")
            print(f"  Compliant: {compliances['yes']} items")
            print(f"  Partially Compliant: {compliances['partial']} items")
            print(f"  Non-Compliant: {compliances['no']} items")
            print(f"  Unknown: {compliances['unknown']} items")
            
            # Calculate compliance rate if we have items
            if len(final_report['items']) > 0:
                compliance_rate = (compliances['yes'] + (compliances['partial'] * 0.5)) / len(final_report['items']) * 100
                print(f"  Overall Compliance Rate: {compliance_rate:.1f}%")
        
        print("\n" + "="*80)
        
        logger.info(f"Report saved to {os.path.join(OUTPUT_PATH, os.path.basename(paper_path).replace('.pdf', '_report_deepseek.json'))}")
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing paper: {str(e)}", exc_info=True)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run RECORD validation with DeepSeek LLMs')
    parser.add_argument('--mode', choices=['full', 'reasoner', 'extractor'], default='full',
                      help='Mode to run: full (default), reasoner (LLM1 only), or extractor (LLM2+LLM3 using existing prompts)')
    parser.add_argument('--prompts', type=str, help='Path to prompts file (required for extractor mode)')
    parser.add_argument('--paper', type=str, help='Path to specific paper to process (optional)')
    
    args = parser.parse_args()
    
    if args.mode == 'reasoner':
        # Run only the reasoner part (LLM1)
        logger.info("Running in REASONER mode (LLM1 only)")
        
        # Create framework and process guidelines
        framework = DeepSeekLLMValidationFramework(API_KEYS)
        guideline_info = framework.process_guideline("RECORD")
        
        # Save prompts with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prompts_filename = f"{timestamp}_deepseek-reasoner_reasoner_RECORD_prompts.json"
        
        with open(os.path.join(OUTPUT_PATH, prompts_filename), "w") as f:
            json.dump(guideline_info["prompts"], f, indent=2)
        
        # Save LLM1 process log
        log_filename = f"{timestamp}_deepseek-reasoner_reasoner_RECORD_process_log.txt"
        with open(os.path.join(OUTPUT_PATH, log_filename), "w") as f:
            f.write(llm1_log_handler.get_logs())
        
        logger.info(f"Extracted {len(guideline_info['items'])} guideline items")
        logger.info(f"Generated {len(guideline_info['prompts'])} prompts")
        logger.info(f"Saved prompts to {prompts_filename}")
        logger.info(f"Saved process log to {log_filename}")
        
        print(f"\nReasoner (LLM1) completed successfully.")
        print(f"Generated prompts saved to: {os.path.join(OUTPUT_PATH, prompts_filename)}")
        print(f"Process log saved to: {os.path.join(OUTPUT_PATH, log_filename)}")
        print(f"You can now run the extractor with: python {__file__} --mode extractor --prompts {os.path.join(OUTPUT_PATH, prompts_filename)}")
        
    elif args.mode == 'extractor':
        # Run only the extractor and validator parts (LLM2 + LLM3)
        logger.info("Running in EXTRACTOR mode (LLM2 + LLM3)")
        
        if not args.prompts:
            logger.error("Prompts file is required for extractor mode")
            parser.print_help()
            sys.exit(1)
        
        # Run the test with the specified prompts file
        run_test(prompts_file=args.prompts)
        
    else:
        # Run the full pipeline
        logger.info("Running in FULL mode (LLM1 + LLM2 + LLM3)")
        run_test()
