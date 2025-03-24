"""
LLM Validation Framework for Research Papers.

This module contains the LLMValidationFramework class, which orchestrates the entire
validation process using multiple LLMs for analyzing research papers against
reporting guidelines like RECORD.
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from src.agents.reasoner import Reasoner
from src.agents.extractor import Extractor
from src.agents.validator import Validator
from src.utils.pdf_processor import PDFProcessor
from src.config import GUIDELINES_PATH, PAPERS_PATH, OUTPUT_PATH

class LLMValidationFramework:
    """
    A framework for validating research papers against reporting guidelines using LLMs.
    
    The framework uses three different LLMs:
    1. Reasoner (LLM1): Processes guideline documents to generate prompts
    2. Extractor (LLM2): Extracts information from research papers
    3. Validator (LLM3): Validates the extracted information against guidelines
    """
    
    def __init__(self, api_keys: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the framework with API keys for LLM providers.
        
        Args:
            api_keys: Dictionary containing API keys for OpenAI, Anthropic, etc.
            config: Optional configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.api_keys = api_keys
        self.config = config or {}
        self.pdf_processor = PDFProcessor()
        
        # Initialize agents
        self.reasoner = Reasoner(api_keys, self.config.get("reasoner", {}))
        self.extractor = Extractor(api_keys, self.config.get("extractor", {}))
        self.validator = Validator(api_keys, self.config.get("validator", {}))
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    def process_guideline(self, guideline_type: str) -> Dict[str, Any]:
        """
        Process a specific guideline (e.g., RECORD) to generate prompts.
        
        Args:
            guideline_type: The type of guideline (e.g., 'RECORD')
            
        Returns:
            Dictionary containing the processed guideline items and prompts
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
    
    def process_paper(self, paper_path: str, guideline_prompts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a research paper using the prompts generated from guidelines.
        
        Args:
            paper_path: Path to the research paper PDF
            guideline_prompts: Prompts generated from the guideline
            
        Returns:
            Dictionary containing extracted information from the paper
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
    
    def validate_extraction(self, paper_info: Dict[str, Any], guideline_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the extracted information against the guideline.
        
        Args:
            paper_info: Information extracted from the paper
            guideline_info: Information about the guideline
            
        Returns:
            Dictionary containing validation results
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
    
    def process_record_paper(self, paper_path: str) -> Dict[str, Any]:
        """
        Process a paper against the RECORD guidelines.
        
        Args:
            paper_path: Path to the research paper PDF
            
        Returns:
            Dictionary containing validation results
        """
        # Step 1: Process RECORD guidelines
        guideline_info = self.process_guideline("RECORD")
        
        # Step 2: Process the paper
        paper_info = self.process_paper(paper_path, guideline_info)
        
        # Step 3: Validate extraction
        validation_results = self.validate_extraction(paper_info, guideline_info)
        
        # Step 4: Generate final report
        final_report = self.generate_report(paper_info, guideline_info, validation_results)
        
        # Save results
        self._save_results(paper_path, guideline_info, paper_info, validation_results, final_report)
        
        return final_report
    
    def generate_report(self, paper_info: Dict[str, Any], guideline_info: Dict[str, Any], 
                       validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a final report based on validation results.
        
        Args:
            paper_info: Information extracted from the paper
            guideline_info: Information about the guideline
            validation_results: Results of validation
            
        Returns:
            Dictionary containing the final report
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
                "compliance": validation.get("compliance", "unknown"),
                "confidence": validation.get("confidence", 0.0),
                "evidence": validation.get("evidence", []),
                "disagreements": validation.get("disagreements", [])
            }
        
        return report
        
    def _save_results(self, paper_path: str, guideline_info: Dict[str, Any], 
                     paper_info: Dict[str, Any], validation_results: Dict[str, Any],
                     final_report: Dict[str, Any]) -> None:
        """
        Save all results to output files.
        
        Args:
            paper_path: Path to the research paper PDF
            guideline_info: Information about the guideline
            paper_info: Information extracted from the paper
            validation_results: Results of validation
            final_report: Final report generated
        """
        paper_name = os.path.splitext(os.path.basename(paper_path))[0]
        
        # Save guideline prompts
        with open(os.path.join(OUTPUT_PATH, f"{paper_name}_prompts.json"), "w") as f:
            json.dump(guideline_info["prompts"], f, indent=2)
        
        # Save extracted information
        with open(os.path.join(OUTPUT_PATH, f"{paper_name}_extraction.json"), "w") as f:
            json.dump(paper_info["extracted_info"], f, indent=2)
        
        # Save validation results
        with open(os.path.join(OUTPUT_PATH, f"{paper_name}_validation.json"), "w") as f:
            json.dump(validation_results, f, indent=2)
        
        # Save final report
        with open(os.path.join(OUTPUT_PATH, f"{paper_name}_report.json"), "w") as f:
            json.dump(final_report, f, indent=2)
        
        self.logger.info(f"All results saved to {OUTPUT_PATH}")