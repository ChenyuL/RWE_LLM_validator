"""
Reasoner Agent (LLM1) for the LLM Validation Framework.

This module contains the Reasoner class, which is responsible for:
1. Processing guideline documents (e.g., RECORD)
2. Extracting individual guideline items
3. Generating prompts for the Extractor agent
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
import re

class Reasoner:
    """
    LLM1 - The Reasoner agent processes guideline documents and generates prompts.
    
    The Reasoner is responsible for:
    1. Extracting individual guideline items from guideline documents
    2. Understanding the requirements of each guideline item
    3. Generating targeted prompts for the Extractor to use when analyzing papers
    """
    
    def __init__(self, api_keys: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Reasoner agent.
        
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
        
        # Initialize LLM client based on available API keys
        if "openai" in api_keys and api_keys["openai"]:
            self._initialize_openai()
        elif "anthropic" in api_keys and api_keys["anthropic"]:
            self._initialize_anthropic()
        else:
            raise ValueError("No valid API keys provided for Reasoner LLM")
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            self.client_type = "openai"
            # Create a new client instance with explicit API key
            self.client = OpenAI(api_key=self.api_keys["openai"])
            
            # Use model from config if provided
            self.model = self.config.get("openai_model", "gpt-4o")
            self.logger.info(f"Initialized OpenAI client for Reasoner using model: {self.model}")
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
            
            # Use model from config or default to Claude model
            self.model = self.config.get("anthropic_model", "claude-3-5-sonnet-20241022")
            self.logger.info(f"Initialized Anthropic client for Reasoner using model: {self.model}")
        except ImportError:
            self.logger.error("Failed to import Anthropic module")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing Anthropic client: {e}")
            raise

    def extract_guideline_items(self, guideline_texts: List[str], batch_size: int = 3) -> List[Dict[str, Any]]:
        """
        Extract individual guideline items from guideline texts.
        Process texts in batches to reduce token usage.
        
        Args:
            guideline_texts: List of texts extracted from guideline PDFs
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of dictionaries containing information about each guideline item
        """
        self.logger.info(f"Extracting guideline items from texts using batch size of {batch_size}")
        
        all_guideline_items = []
        
        # Process texts in batches
        for i in range(0, len(guideline_texts), batch_size):
            batch_texts = guideline_texts[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(guideline_texts) + batch_size - 1)//batch_size}")
            
            # Combine batch texts
            combined_text = "\n\n".join(batch_texts)
            
            # Prompt for extracting guideline items
            prompt = """
            You are an expert in biomedical research methodology and reporting guidelines.
            
            Please extract all individual items from the following reporting guideline.
            For each item, provide:
            1. Item ID or number (e.g., "1a", "2b", etc.)
            2. The complete item description/requirement
            3. The category or section it belongs to
            4. Any additional guidance or explanatory notes provided for this item
            
            Format your response as a JSON array of objects, with each object representing 
            one guideline item with the following properties:
            - "id": the item identifier
            - "description": the complete item description
            - "category": the section or category
            - "notes": any additional guidance
            
            Do not paraphrase or summarize the guidelines. Extract them verbatim from the text.
            
            REPORTING GUIDELINE TEXT:
            {text}
            """.format(text=combined_text[:50000])  # Reduced text length limit for each batch
            
            # Call LLM to extract guideline items
            result = self._call_llm(prompt)
            
            # Try to parse the result as JSON
            batch_items = self._parse_json_result(result)
            
            # If parsing failed, try to extract JSON from text
            if not batch_items:
                batch_items = self._extract_json_from_text(result)
            
            # If extraction failed, try a different approach
            if not batch_items:
                self.logger.warning(f"Failed to extract guideline items as JSON for batch {i//batch_size + 1}. Trying with structured text approach.")
                batch_items = self._extract_items_structured(batch_texts)
            
            self.logger.info(f"Extracted {len(batch_items)} guideline items from batch {i//batch_size + 1}")
            
            # Add batch items to all items, avoiding duplicates
            existing_ids = {item["id"] for item in all_guideline_items}
            for item in batch_items:
                if item["id"] not in existing_ids:
                    all_guideline_items.append(item)
                    existing_ids.add(item["id"])
        
        self.logger.info(f"Extracted a total of {len(all_guideline_items)} unique guideline items")
        
        return all_guideline_items
    
    def _extract_items_structured(self, guideline_texts: List[str], batch_size: int = 3) -> List[Dict[str, Any]]:
        """
        Extract guideline items using a structured text approach if JSON parsing fails.
        Process texts in batches to reduce token usage.
        
        Args:
            guideline_texts: List of texts extracted from guideline PDFs
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of dictionaries containing information about each guideline item
        """
        self.logger.info(f"Extracting guideline items using structured approach with batch size of {batch_size}")
        
        all_items = []
        existing_ids = set()
        
        # Process texts in batches
        for i in range(0, len(guideline_texts), batch_size):
            batch_texts = guideline_texts[i:i+batch_size]
            self.logger.info(f"Processing structured extraction batch {i//batch_size + 1}/{(len(guideline_texts) + batch_size - 1)//batch_size}")
            
            # Combine batch texts
            combined_text = "\n\n".join(batch_texts)
            
            # Prompt for structured extraction
            prompt = """
            You are an expert in biomedical research methodology and reporting guidelines.
            
            Please extract all individual items from the following reporting guideline.
            Format each item in the following way:
            
            ITEM ID: [item identifier]
            DESCRIPTION: [complete description,recommendation, or requirement, where to find in the paper]
            CATEGORY: [category/section]
            NOTES: [any additional guidance]
            
            Present each item separated by three dashes (---).
            Do not paraphrase or summarize the guidelines. Extract them verbatim from the text.
            
            REPORTING GUIDELINE TEXT:
            {text}
            """.format(text=combined_text[:50000])  # Reduced text length limit for each batch
            
            # Call LLM
            result = self._call_llm(prompt)
            
            # Parse structured text
            batch_items = []
            for item_text in result.split("---"):
                item = {}
                
                # Extract item ID
                id_match = re.search(r"ITEM ID:\s*(.+)", item_text)
                if id_match:
                    item["id"] = id_match.group(1).strip()
                
                # Extract description
                desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?=CATEGORY:|NOTES:|$)", item_text, re.DOTALL)
                if desc_match:
                    item["description"] = desc_match.group(1).strip()
                
                # Extract category
                cat_match = re.search(r"CATEGORY:\s*(.+?)(?=ITEM ID:|DESCRIPTION:|NOTES:|$)", item_text, re.DOTALL)
                if cat_match:
                    item["category"] = cat_match.group(1).strip()
                
                # Extract notes
                notes_match = re.search(r"NOTES:\s*(.+?)(?=ITEM ID:|DESCRIPTION:|CATEGORY:|$)", item_text, re.DOTALL)
                if notes_match:
                    item["notes"] = notes_match.group(1).strip()
                
                # Only add if we have at least an ID and description
                if "id" in item and "description" in item:
                    batch_items.append(item)
            
            self.logger.info(f"Extracted {len(batch_items)} guideline items from structured batch {i//batch_size + 1}")
            
            # Add batch items to all items, avoiding duplicates
            for item in batch_items:
                if item["id"] not in existing_ids:
                    all_items.append(item)
                    existing_ids.add(item["id"])
        
        self.logger.info(f"Extracted a total of {len(all_items)} unique guideline items using structured approach")
        
        return all_items
    
    def generate_prompts(self, guideline_items: List[Dict[str, Any]], batch_size: int = 50) -> Dict[str, str]:
        """
        Generate prompts for the Extractor based on guideline items.
        Process items in batches to reduce token usage.
        
        Args:
            guideline_items: List of dictionaries containing information about each guideline item
            batch_size: Number of items to process in each batch
            
        Returns:
            Dictionary mapping item IDs to prompts
        """
        self.logger.info(f"Generating prompts for guideline items using batch size of {batch_size}")
        
        prompts = {}
        
        # Process in batches if there are many items
        for i in range(0, len(guideline_items), batch_size):
            batch_items = guideline_items[i:i+batch_size]
            self.logger.info(f"Processing prompt generation batch {i//batch_size + 1}/{(len(guideline_items) + batch_size - 1)//batch_size}")
            
            # Generate prompts for each item in the batch
            for item in batch_items:
                item_id = item.get("id", "unknown")
                description = item.get("description", "")
                category = item.get("category", "")
                notes = item.get("notes", "")
                
                # Generate a prompt for this specific guideline item
                prompt = f"""
                REPORTING GUIDELINE ITEM: {item_id}
                
                DESCRIPTION: {description}
                
                CATEGORY: {category}
                
                ADDITIONAL NOTES: {notes}
                
                TASK: Please examine the research paper and extract specific information related to this 
                reporting guideline item. Identify whether the paper complies with this item, providing 
                direct quotes and page/section references where the information can be found.
                
                The response should include:
                1. Whether the paper complies with this guideline item (yes, partial, or no)
                2. Evidence from the paper (direct quotes with locations)
                3. Your confidence in this assessment (0.0-1.0)
                4. Reasoning for your assessment
                
                If the information is missing, clearly state that it is not reported in the paper.
                """
                
                prompts[item_id] = prompt.strip()
        
        self.logger.info(f"Generated {len(prompts)} prompts")
        return prompts
    
    def analyze_guideline_compliance(self, validation_results: Dict[str, Any], 
                                    guideline_items: List[Dict[str, Any]], 
                                    batch_size: int = 20) -> Dict[str, Any]:
        """
        Analyze compliance with guidelines based on validation results.
        Process items in batches to reduce token usage.
        
        Args:
            validation_results: Results from validation
            guideline_items: List of guideline items
            batch_size: Number of items to process in each batch
            
        Returns:
            Dictionary with analysis of compliance
        """
        self.logger.info(f"Analyzing guideline compliance using batch size of {batch_size}")
        
        # Prepare input for LLM
        items_summary = []
        for item in guideline_items:
            item_id = item.get("id", "unknown")
            if item_id in validation_results:
                result = validation_results[item_id]
                items_summary.append({
                    "id": item_id,
                    "description": item.get("description", ""),
                    "compliance": result.get("compliance", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "evidence": result.get("evidence", [])
                })
        
        # Process in batches if there are many items
        if len(items_summary) > batch_size:
            self.logger.info(f"Processing compliance analysis in batches due to large number of items ({len(items_summary)})")
            
            # Calculate basic metrics across all items
            compliances = {"yes": 0, "partial": 0, "no": 0, "unknown": 0}
            for item in items_summary:
                compliance = item.get("compliance", "unknown")
                compliances[compliance] = compliances.get(compliance, 0) + 1
            
            total_items = len(items_summary)
            compliance_rate = 0
            if total_items > 0:
                compliance_rate = (compliances.get("yes", 0) + 
                                 0.5 * compliances.get("partial", 0)) / total_items * 100
            
            # Process batches to identify strengths and weaknesses
            all_strengths = []
            all_weaknesses = []
            all_recommendations = []
            
            for i in range(0, len(items_summary), batch_size):
                batch_items = items_summary[i:i+batch_size]
                self.logger.info(f"Processing compliance analysis batch {i//batch_size + 1}/{(len(items_summary) + batch_size - 1)//batch_size}")
                
                batch_prompt = f"""
                You are an expert in biomedical research methodology and reporting guidelines.
                
                Please analyze the following validation results for a research paper 
                against reporting guidelines. Focus only on identifying:
                
                1. Key areas of strength (well-reported aspects)
                2. Key areas for improvement (poorly reported aspects)
                3. Specific recommendations for authors
                
                VALIDATION RESULTS (BATCH {i//batch_size + 1}/{(len(items_summary) + batch_size - 1)//batch_size}):
                {json.dumps(batch_items, indent=2)}
                
                Please provide your analysis in JSON format with the following structure:
                {{
                    "key_strengths": [list of strengths],
                    "key_weaknesses": [list of weaknesses],
                    "recommendations": [list of recommendations]
                }}
                """
                
                # Call LLM for batch analysis
                batch_result = self._call_llm(batch_prompt)
                
                # Parse the result
                try:
                    batch_analysis = json.loads(batch_result)
                    
                    # Add unique strengths, weaknesses, and recommendations
                    for strength in batch_analysis.get("key_strengths", []):
                        if strength not in all_strengths:
                            all_strengths.append(strength)
                    
                    for weakness in batch_analysis.get("key_weaknesses", []):
                        if weakness not in all_weaknesses:
                            all_weaknesses.append(weakness)
                    
                    for recommendation in batch_analysis.get("recommendations", []):
                        if recommendation not in all_recommendations:
                            all_recommendations.append(recommendation)
                            
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    batch_analysis = self._extract_json_from_text(batch_result)
                    
                    if batch_analysis:
                        # Add unique strengths, weaknesses, and recommendations
                        for strength in batch_analysis.get("key_strengths", []):
                            if strength not in all_strengths:
                                all_strengths.append(strength)
                        
                        for weakness in batch_analysis.get("key_weaknesses", []):
                            if weakness not in all_weaknesses:
                                all_weaknesses.append(weakness)
                        
                        for recommendation in batch_analysis.get("recommendations", []):
                            if recommendation not in all_recommendations:
                                all_recommendations.append(recommendation)
            
            # Combine results from all batches
            return {
                "overall_compliance_rate": compliance_rate,
                "fully_compliant_count": compliances.get("yes", 0),
                "partially_compliant_count": compliances.get("partial", 0),
                "non_compliant_count": compliances.get("no", 0),
                "key_strengths": all_strengths[:10],  # Limit to top 10
                "key_weaknesses": all_weaknesses[:10],  # Limit to top 10
                "recommendations": all_recommendations[:10]  # Limit to top 10
            }
        
        # For smaller sets, process all at once
        prompt = f"""
        You are an expert in biomedical research methodology and reporting guidelines.
        
        Please analyze the following validation results for a research paper 
        against reporting guidelines. Provide:
        
        1. Overall compliance rate (percentage of items fully complied with)
        2. Key areas of strength (well-reported aspects)
        3. Key areas for improvement (poorly reported aspects)
        4. Recommendations for authors to improve compliance
        
        VALIDATION RESULTS:
        {json.dumps(items_summary, indent=2)}
        
        Please provide a comprehensive analysis in JSON format with the following structure:
        {{
            "overall_compliance_rate": percentage,
            "fully_compliant_count": number,
            "partially_compliant_count": number,
            "non_compliant_count": number,
            "key_strengths": [list of strengths],
            "key_weaknesses": [list of weaknesses],
            "recommendations": [list of recommendations]
        }}
        """
        
        # Call LLM for analysis
        result = self._call_llm(prompt)
        
        # Parse the result
        try:
            analysis = json.loads(result)
            self.logger.info("Successfully generated compliance analysis")
            return analysis
        except json.JSONDecodeError:
            self.logger.error("Failed to parse compliance analysis as JSON")
            # Try to extract JSON from text
            analysis = self._extract_json_from_text(result)
            
            if analysis:
                return analysis
            else:
                # Return a simple analysis if JSON parsing fails
                self.logger.warning("Returning simplified compliance analysis")
                
                # Calculate basic metrics
                compliances = {"yes": 0, "partial": 0, "no": 0, "unknown": 0}
                for item_id, result in validation_results.items():
                    compliance = result.get("compliance", "unknown")
                    compliances[compliance] = compliances.get(compliance, 0) + 1
                
                total_items = len(validation_results)
                compliance_rate = 0
                if total_items > 0:
                    compliance_rate = (compliances.get("yes", 0) + 
                                     0.5 * compliances.get("partial", 0)) / total_items * 100
                
                return {
                    "overall_compliance_rate": compliance_rate,
                    "fully_compliant_count": compliances.get("yes", 0),
                    "partially_compliant_count": compliances.get("partial", 0),
                    "non_compliant_count": compliances.get("no", 0),
                    "key_strengths": ["Analysis could not be generated"],
                    "key_weaknesses": ["Analysis could not be generated"],
                    "recommendations": ["Please review the detailed validation results"]
                }
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with appropriate parameters.
        
        Args:
            prompt: Prompt to send to the LLM
            
        Returns:
            LLM response as a string
        """
        start_time = time.time()
        result = ""
        
        try:
            if self.client_type == "openai":
                # Call OpenAI API
                # Check if the model is an o3 model, which requires max_completion_tokens instead of max_tokens
                # and may not support temperature
                if "o3" in self.model:
                    try:
                        # First try without temperature
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=4000
                        )
                    except Exception as e:
                        if "temperature" in str(e):
                            # If that fails, try with temperature
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.1,
                                max_completion_tokens=4000
                            )
                        else:
                            # If it's a different error, re-raise it
                            raise
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=4000
                    )
                
                result = response.choices[0].message.content
                
                # Update metrics
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
            elif self.client_type == "anthropic":
                # Call Anthropic API
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
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
            raise
    
    def _parse_json_result(self, result: str) -> List[Dict[str, Any]]:
        """
        Parse a JSON result from the LLM.
        
        Args:
            result: LLM response text
            
        Returns:
            Parsed JSON as a list of dictionaries, or empty list if parsing fails
        """
        try:
            # Try to parse the entire response as JSON
            parsed = json.loads(result)
            
            # Check if it's a list of dictionaries as expected
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                return parsed
            # If it's a dict with a key containing the list
            elif isinstance(parsed, dict):
                for key, value in parsed.items():
                    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                        return value
            
            self.logger.warning("Parsed JSON does not contain a list of guideline items")
            return []
        
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM response as JSON")
            return []
    
    def _extract_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract JSON from text when the entire response isn't valid JSON.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON as a list of dictionaries, or empty list if extraction fails
        """
        # Look for JSON array pattern
        json_match = re.search(r'(\[\s*\{.*\}\s*\])', text, re.DOTALL)
        if json_match:
            try:
                extracted_json = json_match.group(1)
                parsed = json.loads(extracted_json)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Look for JSON object pattern that might contain our array
        json_obj_match = re.search(r'(\{\s*".*"\s*:.*\})', text, re.DOTALL)
        if json_obj_match:
            try:
                extracted_json = json_obj_match.group(1)
                parsed = json.loads(extracted_json)
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        if isinstance(value, list):
                            return value
            except json.JSONDecodeError:
                pass
        
        self.logger.warning("Failed to extract JSON from text")
        return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics for the Reasoner.
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            "agent": "reasoner",
            "llm_provider": self.client_type,
            "model": self.model,
            "metrics": self.call_metrics
        }
