#!/usr/bin/env python
"""
Enhanced LLM Validation Pipeline

This module provides a unified pipeline that supports both base and RAG agents
for validating research papers against reporting checklists.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.agents.reasoner_modified import Reasoner as BaseReasoner
from src.agents.reasoner_rag import ReasonerRAG
from src.agents.extractor import Extractor as BaseExtractor
from src.agents.validator import Validator as BaseValidator
from src.utils.pdf_processor import PDFProcessor
from src.config import GUIDELINES_PATH, PAPERS_PATH, OUTPUT_PATH

# Import RAG implementations
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

class EnhancedValidationPipeline:
    """
    Enhanced validation pipeline supporting both base and RAG agents.
    """
    
    def __init__(self, api_keys: Dict[str, str], config: Dict[str, Any]):
        """
        Initialize the enhanced validation pipeline.
        
        Args:
            api_keys: Dictionary containing API keys for different providers
            config: Configuration dictionary with agent settings
        """
        self.logger = logging.getLogger(__name__)
        self.api_keys = api_keys
        self.config = config
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()
        
        # Initialize agents based on configuration
        self._initialize_agents()
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    def _initialize_agents(self):
        """Initialize agents based on configuration."""
        # Initialize Reasoner
        reasoner_config = self.config.get('reasoner', {})
        reasoner_type = reasoner_config.get('type', 'base')
        
        if reasoner_type == 'rag':
            self.reasoner = ReasonerRAG(self.api_keys, reasoner_config.get('model', 'o3-mini-2025-01-31'))
        else:
            self.reasoner = BaseReasoner(self.api_keys, reasoner_config)
        
        # Initialize Extractor
        extractor_config = self.config.get('extractor', {})
        extractor_type = extractor_config.get('type', 'base')
        
        if extractor_type == 'rag':
            # Use RAG extractor implementation
            self.extractor = self._create_rag_extractor(extractor_config)
        else:
            self.extractor = BaseExtractor(self.api_keys, extractor_config)
        
        # Initialize Validator
        validator_config = self.config.get('validator', {})
        validator_type = validator_config.get('type', 'base')
        
        if validator_type == 'rag':
            # Use RAG validator implementation
            self.validator = self._create_rag_validator(validator_config)
        else:
            self.validator = BaseValidator(self.api_keys, validator_config)
        
        self.logger.info(f"Initialized pipeline with {reasoner_type} reasoner, {extractor_type} extractor, {validator_type} validator")
    
    def _create_rag_extractor(self, config: Dict[str, Any]):
        """Create RAG extractor based on configuration."""
        from archive.rag_extractor_validator_improved import RAGExtractor
        
        model = config.get('model', 'gpt-4o')
        provider = self._get_provider_for_model(model)
        
        return RAGExtractor(
            api_key=self.api_keys[provider],
            model=model,
            provider=provider
        )
    
    def _create_rag_validator(self, config: Dict[str, Any]):
        """Create RAG validator based on configuration."""
        from archive.rag_extractor_validator_improved import RAGValidator
        
        model = config.get('model', 'claude-3-5-sonnet-20241022')
        provider = self._get_provider_for_model(model)
        
        return RAGValidator(
            api_key=self.api_keys[provider],
            model=model,
            provider=provider
        )
    
    def _get_provider_for_model(self, model: str) -> str:
        """Determine the provider for a given model."""
        if any(x in model.lower() for x in ['gpt', 'o1', 'o3']):
            return 'openai'
        elif 'claude' in model.lower():
            return 'anthropic'
        elif 'deepseek' in model.lower():
            return 'deepseek'
        else:
            return 'openai'  # Default fallback
    
    def process_guideline(self, guideline_type: str) -> Dict[str, Any]:
        """
        Process a specific guideline to generate prompts.
        
        Args:
            guideline_type: The type of guideline (e.g., 'RECORD', 'STROBE', 'Li-Paper')
            
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
        try:
            guideline_items = self.reasoner.extract_guideline_items(guideline_texts)
            prompts = self.reasoner.generate_prompts(guideline_items)
            
            return {
                "guideline_type": guideline_type,
                "items": guideline_items,
                "prompts": prompts
            }
        except Exception as e:
            self.logger.error(f"Error processing guideline {guideline_type}: {e}")
            # Return empty structure to prevent pipeline failure
            return {
                "guideline_type": guideline_type,
                "items": [],
                "prompts": {}
            }
    
    def process_paper(self, paper_path: str, guideline_prompts: Dict[str, Any], 
                     batch_size: int = 5) -> Dict[str, Any]:
        """
        Process a research paper using the prompts generated from guidelines.
        
        Args:
            paper_path: Path to the research paper PDF
            guideline_prompts: Prompts generated from the guideline
            batch_size: Number of items to process in each batch
            
        Returns:
            Dictionary containing extracted information from the paper
        """
        self.logger.info(f"Processing paper: {os.path.basename(paper_path)}")
        
        # Extract text from paper
        paper_text = self.pdf_processor.extract_text(paper_path)
        
        # Get paper identifier
        paper_basename = os.path.basename(paper_path)
        paper_identifier = os.path.splitext(paper_basename)[0]
        if '.' in paper_identifier:
            paper_identifier = paper_identifier.split('.')[0]
        
        # Process based on extractor type
        extracted_info = {}
        
        if hasattr(self.extractor, 'extract_information'):
            # Base extractor
            for item_id, prompt in guideline_prompts["prompts"].items():
                extraction_result = self.extractor.extract_information(paper_text, prompt, item_id)
                # Wrap the result in extracted_content for consistency with the pipeline
                extracted_info[item_id] = {
                    "extracted_content": extraction_result
                }
        else:
            # RAG extractor - needs additional parameters
            # First create chunks and embeddings if using RAG
            if self.config.get('extractor', {}).get('type') == 'rag':
                chunks = self._create_chunks(paper_text)
                embeddings = self._generate_embeddings(chunks)
                
                for item_id, prompt in guideline_prompts["prompts"].items():
                    extracted_info[item_id] = self.extractor.extract_information(
                        paper_text, prompt, item_id, paper_identifier, embeddings, chunks
                    )
            else:
                # Fallback to base extraction
                for item_id, prompt in guideline_prompts["prompts"].items():
                    extracted_info[item_id] = self.extractor.extract_information(paper_text, prompt, item_id)
        
        return {
            "paper_path": paper_path,
            "extracted_info": extracted_info
        }
    
    def validate_extraction(self, paper_info: Dict[str, Any], guideline_info: Dict[str, Any],
                           batch_size: int = 5) -> Dict[str, Any]:
        """
        Validate the extracted information against the guideline.
        
        Args:
            paper_info: Information extracted from the paper
            guideline_info: Information about the guideline
            batch_size: Number of items to process in each batch
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Validating extracted information")
        
        validation_results = {}
        
        # Get paper text for RAG validation if needed
        paper_text = None
        if self.config.get('validator', {}).get('type') == 'rag':
            paper_text = self.pdf_processor.extract_text(paper_info["paper_path"])
        
        for item_id, extraction in paper_info["extracted_info"].items():
            # Get corresponding guideline item
            guideline_item = next((item for item in guideline_info["items"] if item["id"] == item_id), None)
            if guideline_item is None:
                self.logger.warning(f"No guideline item found for ID: {item_id}")
                continue
            
            # Validate based on validator type
            if hasattr(self.validator, 'validate') and len(self.validator.validate.__code__.co_varnames) == 4:
                # Base validator
                validation_results[item_id] = self.validator.validate(
                    extraction, guideline_item, item_id
                )
            else:
                # RAG validator - needs additional parameters
                if self.config.get('validator', {}).get('type') == 'rag' and paper_text:
                    chunks = self._create_chunks(paper_text)
                    embeddings = self._generate_embeddings(chunks)
                    
                    paper_identifier = os.path.splitext(os.path.basename(paper_info["paper_path"]))[0]
                    if '.' in paper_identifier:
                        paper_identifier = paper_identifier.split('.')[0]
                    
                    validation_results[item_id] = self.validator.validate(
                        extraction, guideline_item, item_id, paper_identifier, embeddings, chunks
                    )
                else:
                    # Fallback to base validation
                    validation_results[item_id] = self.validator.validate(
                        extraction, guideline_item, item_id
                    )
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(validation_results)
        
        return {
            "validation_results": validation_results,
            "metrics": metrics
        }
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create text chunks for RAG processing."""
        rag_config = self.config.get('rag_config', {})
        chunk_size = rag_config.get('chunk_size', 1000)
        chunk_overlap = rag_config.get('chunk_overlap', 200)
        
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) < 100:  # Skip very small chunks
                continue
            chunks.append(chunk)
        
        return chunks
    
    def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        # Use Voyage AI if available, otherwise OpenAI
        if 'voyage' in self.api_keys and self.api_keys['voyage']:
            return self._generate_voyage_embeddings(chunks)
        else:
            return self._generate_openai_embeddings(chunks)
    
    def _generate_voyage_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings using Voyage AI."""
        import voyageai
        
        client = voyageai.Client(api_key=self.api_keys['voyage'])
        embeddings = []
        
        for i in range(0, len(chunks), 10):  # Process in batches
            batch_chunks = chunks[i:i+10]
            try:
                result = client.embed(
                    batch_chunks,
                    model="voyage-3",
                    input_type="document"
                )
                embeddings.extend(result.embeddings)
            except Exception as e:
                self.logger.error(f"Error generating Voyage embeddings: {e}")
                # Add placeholder embeddings
                for _ in range(len(batch_chunks)):
                    embeddings.append([0] * 1024)
        
        return embeddings
    
    def _generate_openai_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_keys['openai'])
        embeddings = []
        
        for chunk in chunks:
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk,
                    dimensions=1536
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                self.logger.error(f"Error generating OpenAI embedding: {e}")
                embeddings.append([0] * 1536)
        
        return embeddings
    
    def _calculate_metrics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics for validation results."""
        counts = {
            "agree with extractor": 0,
            "do not agree with extractor": 0,
            "unknown": 0
        }
        
        for result in validation_results.values():
            validate_result = result.get("validate_result", "unknown")
            counts[validate_result] = counts.get(validate_result, 0) + 1
        
        total_items = len(validation_results)
        agreement_rate = (counts["agree with extractor"] / total_items * 100) if total_items > 0 else 0.0
        
        return {
            "total_items": total_items,
            "agree_with_extractor": counts["agree with extractor"],
            "disagree_with_extractor": counts["do not agree with extractor"],
            "unknown": counts["unknown"],
            "agreement_rate": agreement_rate,
            "items_for_review": counts["do not agree with extractor"],
            "review_percentage": (counts["do not agree with extractor"] / total_items * 100) if total_items > 0 else 0.0
        }
    
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
        
        # Get model information
        model_info = {
            "reasoner": f"{self.config.get('reasoner', {}).get('type', 'base')}-{self.config.get('reasoner', {}).get('model', 'unknown')}",
            "extractor": f"{self.config.get('extractor', {}).get('type', 'base')}-{self.config.get('extractor', {}).get('model', 'unknown')}",
            "validator": f"{self.config.get('validator', {}).get('type', 'base')}-{self.config.get('validator', {}).get('model', 'unknown')}"
        }
        
        # Add embedding model if RAG is used
        if any(self.config.get(agent, {}).get('type') == 'rag' for agent in ['reasoner', 'extractor', 'validator']):
            model_info["embeddings"] = "voyage-3" if 'voyage' in self.api_keys else "openai-text-embedding-3-small"
        
        report = {
            "paper": paper_name,
            "checklist": guideline_info["guideline_type"],
            "validation_summary": validation_results["metrics"],
            "model_info": model_info,
            "items": {}
        }
        
        # Compile detailed item-by-item results
        for item_id, validation in validation_results["validation_results"].items():
            guideline_item = next((item for item in guideline_info["items"] if item["id"] == item_id), None)
            extraction = paper_info["extracted_info"].get(item_id, {})
            extracted_content = extraction.get("extracted_content", {})
            
            # Get evidence from extraction
            evidence = extracted_content.get("evidence", [])
            
            # Get correct answer from validation
            correct_answer = validation.get("correct_answer", validation.get("validate_result", "unknown"))
            
            report["items"][item_id] = {
                "description": guideline_item["description"] if guideline_item else "Unknown",
                "compliance": validation.get("validate_result", "unknown"),
                "evidence": evidence,
                "correct_answer": correct_answer,
                "reasoning": validation.get("Reason", ""),
                "disagreements": []  # Could be enhanced with more detailed disagreement analysis
            }
        
        return report
    
    def run_full_pipeline(self, paper_path: str, guideline_type: str) -> Dict[str, Any]:
        """
        Run the complete validation pipeline.
        
        Args:
            paper_path: Path to the research paper PDF
            guideline_type: Type of guideline to validate against
            
        Returns:
            Dictionary containing the final report
        """
        self.logger.info(f"Running full pipeline for {os.path.basename(paper_path)} against {guideline_type}")
        
        # Step 1: Process guidelines
        guideline_info = self.process_guideline(guideline_type)
        
        # Step 2: Process paper
        paper_info = self.process_paper(paper_path, guideline_info)
        
        # Step 3: Validate extraction
        validation_results = self.validate_extraction(paper_info, guideline_info)
        
        # Step 4: Generate final report
        final_report = self.generate_report(paper_info, guideline_info, validation_results)
        
        # Step 5: Save results
        self._save_results(paper_path, guideline_info, paper_info, validation_results, final_report)
        
        return final_report
    
    def _save_results(self, paper_path: str, guideline_info: Dict[str, Any],
                     paper_info: Dict[str, Any], validation_results: Dict[str, Any],
                     final_report: Dict[str, Any]) -> None:
        """Save all results to output files."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        paper_basename = os.path.basename(paper_path)
        paper_identifier = os.path.splitext(paper_basename)[0]
        
        if '.' in paper_identifier:
            paper_identifier = paper_identifier.split('.')[0]
        
        checklist_name = guideline_info.get("guideline_type", "RECORD")
        
        # Create results directory for this paper
        results_dir = os.path.join(OUTPUT_PATH, "paper_results", f"{paper_identifier}_{checklist_name}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save reasoner output
        reasoner_filename = f"{timestamp}_reasoner_{paper_identifier}_{checklist_name}.json"
        with open(os.path.join(results_dir, reasoner_filename), "w") as f:
            json.dump(guideline_info["prompts"], f, indent=2)
        
        # Save extractor output
        extractor_filename = f"{timestamp}_extractor_{paper_identifier}_{checklist_name}.json"
        with open(os.path.join(results_dir, extractor_filename), "w") as f:
            json.dump(paper_info["extracted_info"], f, indent=2)
        
        # Save validator output
        validator_filename = f"{timestamp}_validator_{paper_identifier}_{checklist_name}.json"
        with open(os.path.join(results_dir, validator_filename), "w") as f:
            json.dump(validation_results, f, indent=2)
        
        # Save final report
        report_filename = f"{timestamp}_report_{paper_identifier}_{checklist_name}.json"
        with open(os.path.join(results_dir, report_filename), "w") as f:
            json.dump(final_report, f, indent=2)
        
        # Generate and save the full checklist
        full_checklist = self._generate_full_checklist(final_report)
        checklist_filename = f"{timestamp}_full_{checklist_name}_checklist_{paper_identifier}.json"
        with open(os.path.join(results_dir, checklist_filename), "w") as f:
            json.dump(full_checklist, f, indent=2)
        
        self.logger.info(f"All results saved to {results_dir}")
    
    def _generate_full_checklist(self, final_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a full checklist from the final report."""
        full_checklist = {
            "paper": final_report.get("paper", ""),
            "checklist_type": final_report.get("checklist", ""),
            "checklist": {}
        }
        
        for item_id, item_data in final_report.get("items", {}).items():
            correct_answer = item_data.get("correct_answer", "unknown")
            description = item_data.get("description", "")
            
            full_checklist["checklist"][item_id] = {
                "description": description,
                "answer": correct_answer
            }
        
        return full_checklist
