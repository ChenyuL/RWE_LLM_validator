"""
Validator Agent (LLM3) for the LLM Validation Framework.

This module contains the Validator class, which is responsible for:
1. Validating information extracted by the Extractor
2. Cross-checking extractions against guideline requirements
3. Calculating agreement metrics between different LLM assessments
4. Identifying disagreements requiring human review
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import re

class Validator:
    """
    LLM3 - The Validator agent validates extractions from research papers.
    
    The Validator is responsible for:
    1. Validating extractions against RECORD guideline items
    2. Cross-checking multiple LLM outputs for consistency
    3. Calculating agreement metrics
    4. Identifying issues requiring human review
    5. Generating a final assessment for each guideline item
    """
    
    def __init__(self, api_keys: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Validator agent.
        
        Args:
            api_keys: Dictionary containing API keys for LLM providers
            config: Optional configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.api_keys = api_keys
        self.config = config or {}
        
        # Store metrics for API calls
        self.call_metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_time_ms": 0
        }
        
        # Store validation results
        self.validation_results = {}
        
        # Initialize LLM client based on available API keys
        # Use a different model from the Reasoner and Extractor if possible
        if "openai" in api_keys and api_keys["openai"]:
            self._initialize_openai()
        elif "anthropic" in api_keys and api_keys["anthropic"]:
            self._initialize_anthropic()
        else:
            self.logger.warning("No valid API keys provided for Validator LLM, using a mock validator")
            self._initialize_mock()
            
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            self.client_type = "openai"
            self.client = OpenAI(api_key=self.api_keys["openai"])
            
            # Use model from config or default to gpt-4 (can use gpt-3.5-turbo for cost savings)
            self.model = self.config.get("openai_model", "gpt-4")
            self.logger.info(f"Initialized OpenAI client for Validator using model: {self.model}")
        except ImportError:
            self.logger.error("Failed to import OpenAI module")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {e}")
            raise
    
    def _initialize_anthropic(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
            
            self.client_type = "anthropic"
            self.client = Anthropic(api_key=self.api_keys["anthropic"])
            
            # Use a different model than the other agents if possible
            self.model = self.config.get("anthropic_model", "claude-3-haiku-20240307")
            self.logger.info(f"Initialized Anthropic client for Validator using model: {self.model}")
        except ImportError:
            self.logger.error("Failed to import Anthropic module")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing Anthropic client: {e}")
            raise
            
    def _initialize_mock(self):
        """Initialize a mock validator for testing."""
        self.client_type = "mock"
        self.model = "mock-validator"
        self.logger.info("Initialized mock validator")
    
    def validate(self, extraction: Dict[str, Any], guideline_item: Dict[str, Any], 
                item_id: str) -> Dict[str, Any]:
        """
        Validate an extraction against a guideline item.
        
        Args:
            extraction: Information extracted from a research paper
            guideline_item: Guideline item to validate against
            item_id: ID of the guideline item
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info(f"Validating extraction for guideline item: {item_id}")
        
        # Extract key information from the extraction
        compliance = extraction.get("compliance", "unknown")
        evidence = extraction.get("evidence", [])
        confidence = extraction.get("confidence", 0.0)
        reasoning = extraction.get("reasoning", "")
        
        # Extract guideline information
        description = guideline_item.get("description", "")
        
        if hasattr(self, 'client_type') and self.client_type != "mock":
            # Build prompt for validation
            prompt = f"""
            You are an expert validator for biomedical research reporting guidelines.
            
            GUIDELINE ITEM: {item_id}
            DESCRIPTION: {description}
            
            EXTRACTION RESULTS:
            - Compliance: {compliance}
            - Confidence: {confidence}
            - Evidence: {json.dumps(evidence, indent=2)}
            - Reasoning: {reasoning}
            
            VALIDATION TASK:
            1. Evaluate whether the compliance assessment is correct based on the evidence provided.
            2. Assess whether the evidence is sufficient and relevant to the guideline item.
            3. Identify any disagreements or issues that require human review.
            4. Provide a final assessment of compliance.
            
            Please provide your validation in the following JSON format:
            {{
                "compliance": "yes", "partial", "no", or "unknown",
                "confidence": 0.0-1.0 (your confidence in this validation),
                "validated_reasoning": "your assessment of the extraction",
                "evidence_quality": "strong", "moderate", "weak", or "insufficient",
                "disagreements": [
                    {{
                        "issue": "description of the issue",
                        "severity": "high", "medium", or "low"
                    }}
                ]
            }}
            """
            
            # Call LLM for validation
            result = self._call_llm(prompt)
            
            # Parse the result
            validation = self._parse_validation_result(result)
        else:
            # Use mock validation for testing
            validation = {
                "compliance": compliance,
                "confidence": confidence,
                "validated_reasoning": "Mock validation: accepting extractor's assessment",
                "evidence_quality": "moderate" if evidence else "insufficient",
                "disagreements": []
            }
        
        # Add original evidence and item_id
        validation["evidence"] = evidence
        validation["item_id"] = item_id
        
        # Save validation result
        self.validation_results[item_id] = validation
        
        return validation
        
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with appropriate parameters based on client type.
        
        Args:
            prompt: Prompt to send to LLM
            
        Returns:
            Response from LLM
        """
        start_time = time.time()
        result = ""
        
        try:
            if self.client_type == "openai":
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                result = response.choices[0].message.content
                
                # Update metrics
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
            elif self.client_type == "anthropic":
                # Call Anthropic API
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                result = response.content[0].text
                
                # Approximate token counts for Anthropic
                prompt_tokens = len(prompt) // 4  # Rough approximation
                completion_tokens = len(result) // 4  # Rough approximation
            
            # Calculate time taken
            time_taken_ms = (time.time() - start_time) * 1000
            
            # Update call metrics
            self.call_metrics["total_calls"] += 1
            self.call_metrics["total_tokens"] += (prompt_tokens + completion_tokens)
            self.call_metrics["total_time_ms"] += time_taken_ms
            
            self.logger.info(f"LLM call completed in {time_taken_ms:.2f}ms, "
                           f"used {prompt_tokens} prompt tokens and {completion_tokens} completion tokens")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            self.call_metrics["total_calls"] += 1
            
            # Return a basic error response
            return json.dumps({
                "compliance": "unknown",
                "confidence": 0.0,
                "validated_reasoning": f"Error calling LLM: {str(e)}",
                "evidence_quality": "insufficient",
                "disagreements": [{
                    "issue": f"Error calling LLM: {str(e)}",
                    "severity": "high"
                }]
            })
    
    def _parse_validation_result(self, result: str) -> Dict[str, Any]:
        """
        Parse validation result from LLM.
        
        Args:
            result: Response from LLM
            
        Returns:
            Parsed validation result
        """
        try:
            # Try to parse the entire response as JSON
            validation = json.loads(result)
            return validation
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'(\{.*\})', result, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json_match.group(1)
                    validation = json.loads(extracted_json)
                    return validation
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, create a basic validation
            self.logger.warning("Failed to parse validation result as JSON")
            return {
                "compliance": "unknown",
                "confidence": 0.0,
                "validated_reasoning": f"Failed to parse result: {result[:500]}...",
                "evidence_quality": "insufficient",
                "disagreements": [{
                    "issue": "Could not parse validation result",
                    "severity": "high"
                }],
                "recommended_for_human_review": True
            }
    
    def calculate_metrics(self, validation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall metrics for the validation results.
        
        Args:
            validation_results: Dictionary mapping item IDs to validation results
            
        Returns:
            Dictionary containing overall metrics
        """
        self.logger.info("Calculating overall validation metrics")
        
        # Count compliance categories
        counts = {"yes": 0, "partial": 0, "no": 0, "unknown": 0}
        confidence_sum = 0
        review_count = 0
        
        for item_id, result in validation_results.items():
            compliance = result.get("compliance", "unknown")
            counts[compliance] = counts.get(compliance, 0) + 1
            
            confidence_sum += result.get("confidence", 0.0)
            
            if result.get("recommended_for_human_review", False):
                review_count += 1
        
        # Calculate percentages
        total_items = len(validation_results)
        percentages = {}
        
        if total_items > 0:
            for category, count in counts.items():
                percentages[f"{category}_percent"] = (count / total_items) * 100
            
            avg_confidence = confidence_sum / total_items
            review_percent = (review_count / total_items) * 100
        else:
            avg_confidence = 0.0
            review_percent = 0.0
        
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
            "average_confidence": avg_confidence,
            "items_for_review": review_count,
            "review_percentage": review_percent
        }
        
        # Add percentages
        metrics.update(percentages)
        
        return metrics