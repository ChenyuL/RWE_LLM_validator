#!/usr/bin/env python
# test_record_validation_claude_openai_fixed.py

import os
import json
import logging
import sys
import datetime
import time
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.config import GUIDELINES_PATH, PAPERS_PATH, OUTPUT_PATH, LLM_CONFIGS
from src.framework import LLMValidationFramework
from src.agents.reasoner_modified import Reasoner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("record_test_claude_openai_fixed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("record_test_claude_openai_fixed")

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

# Get API keys directly from .env file
def get_api_keys_from_env():
    """
    Get API keys directly from the .env file instead of using environment variables.
    """
    api_keys = {}
    
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
        
        for line in env_content.split('\n'):
            if line.startswith('OPENAI_API_KEY='):
                api_keys["openai"] = line.split('=', 1)[1].strip()
            elif line.startswith('ANTHROPIC_API_KEY='):
                api_keys["anthropic"] = line.split('=', 1)[1].strip()
            elif line.startswith('DEEPSEEK_API_KEY='):
                api_keys["deepseek"] = line.split('=', 1)[1].strip()
        
        logger.info(f"API keys loaded directly from .env file")
        logger.info(f"OpenAI API Key (first 10 chars): {api_keys['openai'][:10]}...")
        logger.info(f"Anthropic API Key (first 10 chars): {api_keys['anthropic'][:10]}...")
        logger.info(f"DeepSeek API Key (first 10 chars): {api_keys['deepseek'][:10]}...")
        
        return api_keys
    except Exception as e:
        logger.error(f"Error loading API keys from .env file: {e}")
        raise

# Get API keys
API_KEYS = get_api_keys_from_env()

class ClaudeExtractor:
    """
    An extractor that uses Claude API.
    """
    
    def __init__(self, api_key, model="claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def extract_information(self, paper_text, prompt, item_id):
        """
        Extract information from paper text based on a prompt.
        """
        from anthropic import Anthropic
        
        self.logger.info(f"Extracting information for checklist item: {item_id}")
        
        # Simplified extraction - just use the first 8000 characters of the paper
        paper_excerpt = paper_text[:8000]
        
        # Extract paper identifier from the paper filename
        paper_identifier = "Unknown"
        if hasattr(self, 'paper_path') and self.paper_path:
            paper_basename = os.path.basename(self.paper_path)
            paper_identifier = os.path.splitext(paper_basename)[0]
            if '.' in paper_identifier:
                paper_identifier = paper_identifier.split('.')[0]
        
        # Get the checklist name from the class or use a default
        checklist_name = getattr(self, 'checklist_name', 'RECORD')
        
        # Create a prompt for Claude
        claude_prompt = f"""
        {prompt}
        
        PAPER TEXT:
        {paper_excerpt}
        
        Please provide your extraction in the following JSON format:
        {{
            "paper_identifier": "{paper_identifier}",
            "{checklist_name}_{paper_identifier}": "{item_id}",
            "extracted_content": {{
                "compliance": "yes", "partial", "no", or "unknown",
                "evidence": [
                    {{
                        "quote": "direct quote from paper",
                        "location": "section/page information if available"
                    }}
                ],
                "reasoning": "explanation of your assessment",
                "correct_answer": "your answer to the specific checklist item question (e.g., for item 1.0.a 'Indicate the study's design with a commonly used term in the title or the abstract', your answer should address whether and where the study design is indicated, not just what the study design is)"
            }}
        }}
        """
        
        # Call Claude API
        client = Anthropic(api_key=self.api_key)
        
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": claude_prompt}]
            )
            
            result = response.content[0].text
            
            # Try to parse the result as JSON
            try:
                extraction = json.loads(result)
                # Add model information to the extraction
                extraction["model_info"] = {
                    "extractor": f"claude-{self.model}",
                    "role": "extractor"
                }
                return extraction
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'(\{.*\})', result, re.DOTALL)
                if json_match:
                    try:
                        extracted_json = json_match.group(1)
                        extraction = json.loads(extracted_json)
                        # Add model information to the extraction
                        extraction["model_info"] = {
                            "extractor": f"claude-{self.model}",
                            "role": "extractor"
                        }
                        return extraction
                    except json.JSONDecodeError:
                        pass
            
            # Get the checklist name from the class or use a default
            checklist_name = getattr(self, 'checklist_name', 'RECORD')
            
            # If parsing fails, return a basic structure
            return {
                "paper_identifier": paper_identifier,
                f"{checklist_name}_{paper_identifier}": item_id,
                "extracted_content": {
                    "compliance": "unknown",
                    "evidence": [],
                    "reasoning": f"Failed to parse result: {result[:500]}..."
                },
                "model_info": {
                    "extractor": f"claude-{self.model}",
                    "role": "extractor"
                }
            }
        except Exception as e:
            self.logger.error(f"Error calling Claude API: {e}")
            
            # Get the checklist name from the class or use a default
            checklist_name = getattr(self, 'checklist_name', 'RECORD')
            
            return {
                "paper_identifier": paper_identifier,
                f"{checklist_name}_{paper_identifier}": item_id,
                "extracted_content": {
                    "compliance": "unknown",
                    "evidence": [],
                    "reasoning": f"Error: {str(e)}"
                },
                "model_info": {
                    "extractor": f"claude-{self.model}",
                    "role": "extractor"
                }
            }

class OpenAIValidator:
    """
    A validator that uses OpenAI API.
    """
    
    def __init__(self, api_key, model="gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate(self, extraction, guideline_item, item_id):
        """
        Validate an extraction against a guideline item.
        """
        from openai import OpenAI
        
        self.logger.info(f"Validating extraction for checklist item: {item_id}")
        
        # Extract key information
        paper_identifier = extraction.get("paper_identifier", extraction.get("paper_title", "Unknown"))
        extracted_content = extraction.get("extracted_content", {})
        compliance = extracted_content.get("compliance", "unknown")
        evidence = extracted_content.get("evidence", [])
        confidence = extracted_content.get("confidence", 0.0)
        reasoning = extracted_content.get("reasoning", "")
        
        # Extract checklist information
        description = guideline_item.get("description", "")
        
        # Get the checklist name from the class or use a default
        checklist_name = getattr(self, 'checklist_name', 'RECORD')
        
        # Build prompt for validation
        openai_prompt = f"""
        You are an expert validator for biomedical research reporting checklists.
        
        CHECKLIST ITEM: {item_id}
        DESCRIPTION: {description}
        
        EXTRACTION RESULTS:
        - Paper Identifier: {paper_identifier}
        - Compliance: {compliance}
        - Evidence: {json.dumps(evidence, indent=2)}
        - Reasoning: {reasoning}
        
        VALIDATION TASK:
        1. Evaluate whether the compliance assessment is correct based on the evidence provided.
        2. Assess whether the evidence is sufficient and relevant to the checklist item.
        3. Provide a final assessment of whether you agree with the extractor's assessment.
        4. Provide a correct answer that will be used in the final checklist.
        
        Please provide your validation in the following JSON format:
        {{
            "paper_identifier": "{paper_identifier}",
            "{checklist_name}_{paper_identifier}": "{item_id}",
            "validate_result": "agree with extractor", "do not agree with extractor", or "unknown",
            "Reason": "your assessment of the extraction",
            "correct_answer": "your answer to the specific checklist item question (e.g., for item 1.0.a 'Indicate the study's design with a commonly used term in the title or the abstract', your answer should address whether and where the study design is indicated, not just what the study design is)"
        }}
        """
        
        # Call OpenAI API
        client = OpenAI(api_key=self.api_key)
        
        try:
            # Check if the model is an o3 model, which requires max_completion_tokens instead of max_tokens
            if "o3" in self.model:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": openai_prompt}],
                    temperature=0.1,
                    max_completion_tokens=2000
                )
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": openai_prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
            
            result = response.choices[0].message.content
            
            # Try to parse the result as JSON
            try:
                validation = json.loads(result)
                
                # Add model information to the validation
                validation["model_info"] = {
                    "validator": f"openai-{self.model}",
                    "role": "validator"
                }
                
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
                        
                        # Add model information to the validation
                        validation["model_info"] = {
                            "validator": f"openai-{self.model}",
                            "role": "validator"
                        }
                        
                        # Save validation result
                        self.validation_results[item_id] = validation
                        
                        return validation
                    except json.JSONDecodeError:
                        pass
            
            # Get the checklist name from the class or use a default
            checklist_name = getattr(self, 'checklist_name', 'RECORD')
            
            # If parsing fails, return a basic structure
            validation = {
                "paper_identifier": paper_identifier,
                f"{checklist_name}_{paper_identifier}": item_id,
                "validate_result": compliance,
                "Reason": f"Failed to parse result: {result[:500]}...",
                "model_info": {
                    "validator": f"openai-{self.model}",
                    "role": "validator"
                }
            }
            
            # Save validation result
            self.validation_results[item_id] = validation
            
            return validation
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            
            # Get the checklist name from the class or use a default
            checklist_name = getattr(self, 'checklist_name', 'RECORD')
            
            validation = {
                "paper_identifier": paper_identifier,
                f"{checklist_name}_{paper_identifier}": item_id,
                "validate_result": "unknown",
                "Reason": f"Error: {str(e)}",
                "model_info": {
                    "validator": f"openai-{self.model}",
                    "role": "validator"
                }
            }
            
            # Save validation result
            self.validation_results[item_id] = validation
            
            return validation
    
    def calculate_metrics(self, validation_results):
        """
        Calculate overall metrics for the validation results.
        """
        self.logger.info("Calculating overall validation metrics")
        
        # Count validation result categories
        counts = {
            "agree with extractor": 0, 
            "do not agree with extractor": 0, 
            "unknown": 0
        }
        
        # Also track the original compliance values from the extractor
        extractor_compliance = {"yes": 0, "partial": 0, "no": 0, "unknown": 0}
        
        for item_id, result in validation_results.items():
            # Get the validation result
            validate_result = result.get("validate_result", "unknown")
            counts[validate_result] = counts.get(validate_result, 0) + 1
            
            # Try to get the original compliance from the extraction
            # This might be in a nested structure
            if "extracted_content" in result:
                compliance = result["extracted_content"].get("compliance", "unknown")
                extractor_compliance[compliance] = extractor_compliance.get(compliance, 0) + 1
        
        # Calculate percentages
        total_items = len(validation_results)
        percentages = {}
        
        if total_items > 0:
            for category, count in counts.items():
                percentages[f"{category}_percent"] = (count / total_items) * 100
        
        # Calculate agreement rate
        if total_items > 0:
            agreement_rate = (counts["agree with extractor"] / total_items) * 100
        else:
            agreement_rate = 0.0
        
        # Compile metrics
        metrics = {
            "total_items": total_items,
            "agree_with_extractor": counts["agree with extractor"],
            "disagree_with_extractor": counts["do not agree with extractor"],
            "unknown": counts["unknown"],
            "agreement_rate": agreement_rate,
            "items_for_review": counts["do not agree with extractor"],  # Items where validator disagrees with extractor
            "review_percentage": (counts["do not agree with extractor"] / total_items * 100) if total_items > 0 else 0.0
        }
        
        # Add percentages
        metrics.update(percentages)
        
        # Add model information
        metrics["model_info"] = {
            "extractor": f"claude-{self.model}",
            "validator": f"openai-{self.model}"
        }
        
        return metrics

class ClaudeOpenAIValidationFramework:
    """
    A modified version of the LLMValidationFramework that uses Claude for LLM1 and LLM2, and OpenAI for LLM3.
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
        
        # Initialize agents with custom reasoner that uses o3
        # Create a custom config that specifies oe as the model
        reasoner_config = self.config.get("reasoner", {})
        reasoner_config["openai_model"] = "o3-mini-2025-01-31" ### edited 
        
        # Use the modified reasoner that includes STROBE items
        self.reasoner = Reasoner(api_keys, reasoner_config)
        
        # Initialize extractor with the specified model
        extractor_config = self.config.get("extractor", {})
        extractor_model = extractor_config.get("model", "claude-3-5-sonnet-20241022")
        self.extractor = ClaudeExtractor(api_keys["anthropic"], model=extractor_model)
        
        # Initialize validator with the specified model
        validator_config = self.config.get("validator", {})
        validator_model = validator_config.get("model", "gpt-4o")
        self.validator = OpenAIValidator(api_keys["openai"], model=validator_model)
        
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
    
    def process_paper(self, paper_path, guideline_prompts, batch_size=5):
        """
        Process a research paper using the prompts generated from guidelines.
        Process items in batches to reduce token usage.
        
        Args:
            paper_path: Path to the paper to process
            guideline_prompts: Dictionary containing the prompts
            batch_size: Number of items to process in each batch
            
        Returns:
            Dictionary containing the paper path and extracted information
        """
        self.logger.info(f"Processing paper: {os.path.basename(paper_path)}")
        
        # Extract text from paper
        paper_text = self.pdf_processor.extract_text(paper_path)
        
        # Use extractor to extract information from paper
        # Process each guideline item separately as requested
        extracted_info = {}
        
        # Get all item IDs
        item_ids = list(guideline_prompts["prompts"].keys())
        
        # Set the checklist name in the extractor based on the guideline type
        guideline_type = guideline_prompts.get("guideline_type", "RECORD")
        self.extractor.checklist_name = guideline_type
        
        # Process items in batches
        for i in range(0, len(item_ids), batch_size):
            batch_item_ids = item_ids[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(item_ids) + batch_size - 1)//batch_size}: {batch_item_ids}")
            
            # Set the paper_path in the extractor so it can extract the paper identifier
            self.extractor.paper_path = paper_path
            
            for item_id in batch_item_ids:
                prompt = guideline_prompts["prompts"][item_id]
                extracted_info[item_id] = self.extractor.extract_information(paper_text, prompt, item_id)
                
            # Save intermediate results after each batch
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            paper_basename = os.path.basename(paper_path)
            paper_identifier = os.path.splitext(paper_basename)[0]
            if '.' in paper_identifier:
                paper_identifier = paper_identifier.split('.')[0]
                
            batch_filename = f"{timestamp}_batch_{i//batch_size + 1}_extraction_{paper_identifier}_{guideline_type}.json"
            with open(os.path.join(OUTPUT_PATH, batch_filename), "w") as f:
                json.dump(extracted_info, f, indent=2)
            
            self.logger.info(f"Saved batch {i//batch_size + 1} results to {batch_filename}")
        
        return {
            "paper_path": paper_path,
            "extracted_info": extracted_info
        }
    
    def validate_extraction(self, paper_info, guideline_info, batch_size=5):
        """
        Validate the extracted information against the guideline.
        Process items in batches to reduce token usage.
        
        Args:
            paper_info: Dictionary containing the paper path and extracted information
            guideline_info: Dictionary containing the guideline items
            batch_size: Number of items to process in each batch
            
        Returns:
            Dictionary containing the validation results and metrics
        """
        self.logger.info("Validating extracted information")
        
        validation_results = {}
        
        # Get all item IDs
        item_ids = list(paper_info["extracted_info"].keys())
        
        # Set the checklist name in the validator based on the guideline type
        guideline_type = guideline_info.get("guideline_type", "RECORD")
        self.validator.checklist_name = guideline_type
        
        # Process items in batches
        for i in range(0, len(item_ids), batch_size):
            batch_item_ids = item_ids[i:i+batch_size]
            self.logger.info(f"Validating batch {i//batch_size + 1}/{(len(item_ids) + batch_size - 1)//batch_size}: {batch_item_ids}")
            
            for item_id in batch_item_ids:
                extraction = paper_info["extracted_info"][item_id]
                
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
            
            # Save intermediate results after each batch
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            paper_basename = os.path.basename(paper_info["paper_path"])
            paper_identifier = os.path.splitext(paper_basename)[0]
            if '.' in paper_identifier:
                paper_identifier = paper_identifier.split('.')[0]
                
            batch_filename = f"{timestamp}_batch_{i//batch_size + 1}_validation_{paper_identifier}_{guideline_type}.json"
            with open(os.path.join(OUTPUT_PATH, batch_filename), "w") as f:
                json.dump(validation_results, f, indent=2)
            
            self.logger.info(f"Saved batch {i//batch_size + 1} validation results to {batch_filename}")
        
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
        
        # Get model information from the first validation result (if available)
        extractor_model = "claude"
        validator_model = "openai"
        
        # Try to get more specific model information from the validation results
        if validation_results["validation_results"] and len(validation_results["validation_results"]) > 0:
            first_item_id = next(iter(validation_results["validation_results"]))
            first_validation = validation_results["validation_results"][first_item_id]
            
            # Get model info from the validation result
            if "model_info" in first_validation:
                validator_model = first_validation["model_info"].get("validator", "openai")
            
            # Get model info from the extraction
            first_extraction = paper_info["extracted_info"].get(first_item_id, {})
            if "model_info" in first_extraction:
                extractor_model = first_extraction["model_info"].get("extractor", "claude")
        
        report = {
            "paper": paper_name,
            "checklist": guideline_info["guideline_type"],
            "validation_summary": validation_results["metrics"],
            "model_info": {
                "extractor": extractor_model,
                "validator": validator_model
            },
            "items": {}
        }
        
        # Compile detailed item-by-item results
        for item_id, validation in validation_results["validation_results"].items():
            guideline_item = next((item for item in guideline_info["items"] if item["id"] == item_id), None)
            extraction = paper_info["extracted_info"].get(item_id, {})
            extracted_content = extraction.get("extracted_content", {})
            
            # Get the evidence from the extraction
            evidence = extracted_content.get("evidence", [])
            
            # Get the correct answer from the validation
            correct_answer = validation.get("correct_answer", validation.get("validate_result", "unknown"))
            
            # Get model information from the extraction and validation
            extraction_model_info = extraction.get("model_info", {"extractor": extractor_model, "role": "extractor"})
            validation_model_info = validation.get("model_info", {"validator": validator_model, "role": "validator"})
            
            report["items"][item_id] = {
                "description": guideline_item["description"] if guideline_item else "Unknown",
                "compliance": validation.get("validate_result", "unknown"),
                "evidence": evidence,
                "correct_answer": correct_answer,
                "reasoning": validation.get("Reason", ""),
                "disagreements": [],  # Not provided in this simplified version
                "model_info": {
                    "extractor": extraction_model_info.get("extractor", extractor_model),
                    "validator": validation_model_info.get("validator", validator_model)
                }
            }
        
        return report
    
    def _save_results(self, paper_path, guideline_info, paper_info, validation_results, final_report):
        """
        Save all results to output files.
        """
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract paper identifier from the paper filename
        paper_basename = os.path.basename(paper_path)
        paper_identifier = os.path.splitext(paper_basename)[0]
        
        # If the filename contains a dot (e.g., "34923518.2022.something.pdf"), 
        # extract just the paper identifier part
        if '.' in paper_identifier:
            paper_identifier = paper_identifier.split('.')[0]
        
        # Get the checklist name
        checklist_name = guideline_info.get("guideline_type", "RECORD")
        
        # Get model information from the final report
        extractor_model = final_report.get("model_info", {}).get("extractor", "claude")
        validator_model = final_report.get("model_info", {}).get("validator", "openai")
        
        # Save guideline prompts (LLM1-reasoner)
        reasoner_filename = f"{timestamp}_openai_reasoner_{paper_identifier}_{checklist_name}.json"
        with open(os.path.join(OUTPUT_PATH, reasoner_filename), "w") as f:
            json.dump(guideline_info["prompts"], f, indent=2)
        
        # Save LLM1 process log
        log_filename = f"{timestamp}_openai_reasoner_{paper_identifier}_{checklist_name}_process_log.txt"
        with open(os.path.join(OUTPUT_PATH, log_filename), "w") as f:
            f.write(llm1_log_handler.get_logs())
        
        # Save extracted information (LLM2-extractor)
        extractor_filename = f"{timestamp}_{extractor_model}_extractor_{paper_identifier}_{checklist_name}.json"
        with open(os.path.join(OUTPUT_PATH, extractor_filename), "w") as f:
            json.dump(paper_info["extracted_info"], f, indent=2)
        
        # Save validation results (LLM3-validator)
        validator_filename = f"{timestamp}_{validator_model}_validator_{paper_identifier}_{checklist_name}.json"
        with open(os.path.join(OUTPUT_PATH, validator_filename), "w") as f:
            json.dump(validation_results, f, indent=2)
        
        # Save final report
        report_filename = f"{timestamp}_claude_openai_report_{paper_identifier}_{checklist_name}.json"
        with open(os.path.join(OUTPUT_PATH, report_filename), "w") as f:
            json.dump(final_report, f, indent=2)
        
        # Generate and save the full checklist
        full_record = self._generate_full_checklist(final_report)
        record_filename = f"{timestamp}_full_{guideline_info['guideline_type']}_checklist_{paper_identifier}.json"
        with open(os.path.join(OUTPUT_PATH, record_filename), "w") as f:
            json.dump(full_record, f, indent=2)
        
        self.logger.info(f"All results saved to {OUTPUT_PATH}")
        self.logger.info(f"Reasoner output: {reasoner_filename}")
        self.logger.info(f"Process log: {log_filename}")
        self.logger.info(f"Extractor output: {extractor_filename}")
        self.logger.info(f"Validator output: {validator_filename}")
        self.logger.info(f"Final report: {report_filename}")
        self.logger.info(f"Full {guideline_info['guideline_type']} checklist: {record_filename}")
    
    def _generate_full_checklist(self, final_report):
        """
        Generate a full checklist from the correct answers in the final report.
        
        Args:
            final_report: The final report containing all items with their correct answers
            
        Returns:
            A dictionary containing the full checklist
        """
        full_record = {
            "paper": final_report.get("paper", ""),
            "checklist_type": final_report.get("checklist", ""),
            "checklist": {},
            "model_info": final_report.get("model_info", {})  # Include model information
        }
        
        # Extract the correct answers from each item
        for item_id, item_data in final_report.get("items", {}).items():
            correct_answer = item_data.get("correct_answer", "unknown")
            description = item_data.get("description", "")
            model_info = item_data.get("model_info", {})  # Get model info for this item
            
            full_record["checklist"][item_id] = {
                "description": description,
                "answer": correct_answer,
                "model_info": model_info  # Include model information for each item
            }
        
        return full_record

def load_prompts_from_file(prompts_file, guideline_type="RECORD"):
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
        "guideline_type": guideline_type,  # Use the provided guideline_type
        "items": [],  # This will be populated with dummy items based on prompt IDs
        "prompts": prompts
    }
    
    # Create dummy guideline items based on the prompt IDs
    for item_id, prompt_text in prompts.items():
        # Extract description from the prompt text
        description = ""
        if isinstance(prompt_text, str):
            # Try to extract the description from the prompt text
            description_match = re.search(r'DESCRIPTION: (.*?)(\n|\r)', prompt_text)
            if description_match:
                description = description_match.group(1).strip()
            
            # Try to extract the category from the prompt text
            category_match = re.search(r'CATEGORY: (.*?)(\n|\r)', prompt_text)
            category = category_match.group(1).strip() if category_match else ""
        else:
            # If prompt_text is not a string, try to get content and category
            description = prompt_text.get("content", "")
            category = prompt_text.get("category", "")
        
        guideline_info["items"].append({
            "id": item_id,
            "description": description,
            "category": category,
            "notes": ""
        })
    
    logger.info(f"Loaded {len(prompts)} prompts")
    return guideline_info

def run_test(prompts_file=None, config=None, guideline_type="RECORD"):
    """
    Run the validation test.
    
    Args:
        prompts_file: Optional path to a file containing pre-generated prompts
        config: Optional configuration dictionary with model choices
        guideline_type: Type of guideline to use (default: RECORD)
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    logger.info("Initializing Claude+OpenAI LLM Validation Framework")
    framework = ClaudeOpenAIValidationFramework(API_KEYS, config)
    
    # Step 1: Process guidelines or load existing prompts
    if prompts_file:
        # Load prompts from file (LLM1 output)
        guideline_info = load_prompts_from_file(prompts_file, guideline_type)
        # Update guideline type if it's different from RECORD
        if guideline_type != "RECORD":
            guideline_info["guideline_type"] = guideline_type
    else:
        # Process guidelines with LLM1
        logger.info(f"Processing {guideline_type} guidelines")
        guideline_info = framework.process_guideline(guideline_type)
        
        # Save guideline info for inspection
        with open(os.path.join(OUTPUT_PATH, "record_guideline_info_claude_openai.json"), "w") as f:
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
    
    # Step 2: Process the specified paper against the guidelines
    # Use the paper specified by the user, or default to 34923518.pdf
    if args.paper:
        paper_path = args.paper
        if not os.path.isabs(paper_path):
            # If a relative path is provided, make it relative to the current working directory
            paper_path = os.path.join(os.getcwd(), paper_path)
    else:
        paper_path = os.path.join(PAPERS_PATH, "34923518.pdf")
    
    if not os.path.exists(paper_path):
        logger.error(f"Specified paper not found: {paper_path}")
        return
    
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
        
        # Print model information
        if "model_info" in final_report:
            model_info = final_report["model_info"]
            print("\nModel Information:")
            print(f"  Extractor: {model_info.get('extractor', 'claude')}")
            print(f"  Validator: {model_info.get('validator', 'openai')}")
        
        # Count validation results by type
        if "items" in final_report:
            validation_counts = {
                "agree with extractor": 0, 
                "do not agree with extractor": 0, 
                "unknown": 0
            }
            
            for item_id, item_data in final_report["items"].items():
                result = item_data.get("compliance", "unknown")
                validation_counts[result] = validation_counts.get(result, 0) + 1
            
            print("\nValidation Results Summary:")
            print(f"  Total Items: {len(final_report['items'])}")
            print(f"  Agree with Extractor: {validation_counts['agree with extractor']} items")
            print(f"  Disagree with Extractor: {validation_counts['do not agree with extractor']} items")
            print(f"  Unknown: {validation_counts['unknown']} items")
            
            # Calculate agreement rate if we have items
            if len(final_report['items']) > 0:
                agreement_rate = (validation_counts['agree with extractor'] / len(final_report['items'])) * 100
                print(f"  Overall Agreement Rate: {agreement_rate:.1f}%")
        
        print("\n" + "="*80)
        
        logger.info(f"Report saved to {os.path.join(OUTPUT_PATH, os.path.basename(paper_path).replace('.pdf', '_report_claude_openai.json'))}")
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing paper: {str(e)}", exc_info=True)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run RECORD validation with Claude and OpenAI LLMs')
    parser.add_argument('--mode', choices=['full', 'reasoner', 'extractor'], default='full',
                      help='Mode to run: full (default), reasoner (LLM1 only), or extractor (LLM2+LLM3 using existing prompts)')
    parser.add_argument('--prompts', type=str, help='Path to prompts file (required for extractor mode)')
    parser.add_argument('--paper', type=str, help='Path to specific paper to process (optional)')
    parser.add_argument('--checklist', type=str, default='RECORD', help='Checklist type to use (default: RECORD)')
    parser.add_argument('--config', type=str, help='Path to configuration file with model choices')
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
    
    if args.mode == 'reasoner':
        # Run only the reasoner part (LLM1)
        logger.info("Running in REASONER mode (LLM1 only)")
        
        # Create framework and process guidelines
        framework = ClaudeOpenAIValidationFramework(API_KEYS, config)
        guideline_info = framework.process_guideline(args.checklist)
        
        # Save prompts with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prompts_filename = f"{timestamp}_openai_reasoner_{args.checklist}_prompts.json"
        
        with open(os.path.join(OUTPUT_PATH, prompts_filename), "w") as f:
            json.dump(guideline_info["prompts"], f, indent=2)
        
        # Save LLM1 process log
        log_filename = f"{timestamp}_openai_reasoner_{args.checklist}_process_log.txt"
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
        run_test(prompts_file=args.prompts, config=config, guideline_type=args.checklist)
        
    else:
        # Run the full pipeline
        logger.info("Running in FULL mode (LLM1 + LLM2 + LLM3)")
        run_test(config=config, guideline_type=args.checklist)
