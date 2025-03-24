#!/usr/bin/env pythonP
# reasoner_rag.py

import os
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import voyageai

from src.agents.reasoner import Reasoner as BaseReasoner

class ReasonerRAG(BaseReasoner):
    """
    RAG-enhanced Reasoner class that uses Voyage AI embeddings and processes items one by one.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None, model: str = "o3-mini-2025-01-31"):
        """
        Initialize the ReasonerRAG with API keys and model.
        
        Args:
            api_keys: Dictionary of API keys for different providers
            model: Model to use for LLM calls
        """
        # Create a config dictionary to pass to the parent class
        config = {"openai_model": model} if "openai" in api_keys else {"anthropic_model": model}
        super().__init__(api_keys, config)
        
        # Store the model name for reference
        self.model = model
        
        # Initialize Voyage AI client if API key is provided
        if api_keys and "voyage" in api_keys and api_keys["voyage"]:
            self.voyage_client = voyageai.Client(api_key=api_keys["voyage"])
            self.logger.info("Initialized Voyage AI client")
        else:
            self.voyage_client = None
            self.logger.warning("Voyage AI client not initialized (API key not provided)")
    
    def extract_guideline_items(self, guideline_texts: List[str], batch_size: int = 3) -> List[Dict[str, Any]]:
        """
        Extract guideline items from the checklist texts.
        This modified version extracts both RECORD-specific items and STROBE items.
        
        Args:
            guideline_texts: List of guideline texts
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of guideline items
        """
        self.logger.info(f"Extracting checklist items from texts using batch size of {batch_size}")
        
        # Check the checklist type
        combined_text = "\n\n".join(guideline_texts)
        is_li_paper = "Li Paper SOP" in combined_text or "Li-Paper" in combined_text or "SOP-Li" in combined_text
        is_strobe = "STROBE" in combined_text and not is_li_paper  # Only consider STROBE if not Li-Paper
        
        # Log the checklist type
        if is_li_paper:
            self.logger.info("Detected Li-Paper SOP checklist")
        elif is_strobe:
            self.logger.info("Detected STROBE checklist")
        
        # For Li-Paper SOP, we'll handle it completely in this class
        if is_li_paper:
            # Skip the base method and extract items directly
            all_items = self._extract_additional_items(guideline_texts)
            # Ensure we have all 35 Li-Paper SOP items
            all_items = self._ensure_complete_li_paper_items(all_items)
        else:
            # First, extract the items using the base method
            base_items = super().extract_guideline_items(guideline_texts, batch_size)
            
            # Now, add the missing STROBE and RECORD items
            additional_items = self._extract_additional_items(guideline_texts)
            
            # Combine the items, avoiding duplicates
            all_items = base_items.copy()
            existing_ids = {item["id"] for item in all_items}
            
            for item in additional_items:
                if item["id"] not in existing_ids:
                    all_items.append(item)
                    existing_ids.add(item["id"])
        
        self.logger.info(f"Extracted {len(all_items)} guideline items (including STROBE and RECORD items)")
        return all_items
    
    def _extract_additional_items(self, guideline_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract checklist items directly from the provided guideline texts.
        This method specifically looks for STROBE items when STROBE checklist is provided.
        For Li-Paper SOP, it uses a specialized prompt to extract items from the PDF.
        
        Args:
            guideline_texts: List of guideline texts
            
        Returns:
            List of checklist items
        """
        self.logger.info("Extracting checklist items directly from the provided texts")
        
        # Use LLM to extract items from the guideline texts
        combined_text = "\n\n".join(guideline_texts)
        
        # Check the checklist type
        is_li_paper = "Li Paper SOP" in combined_text or "Li-Paper" in combined_text or "SOP-Li" in combined_text
        is_strobe = "STROBE" in combined_text and not is_li_paper  # Only consider STROBE if not Li-Paper
        is_record = "RECORD" in combined_text and not is_li_paper  # Only consider RECORD if not Li-Paper
        
        # For Li-Paper SOP, use a structured approach with LLM
        if is_li_paper:
            self.logger.info("Using structured LLM extraction for Li-Paper SOP checklist")
            
            # First try the structured approach that specifically extracts all 35 items
            items = self._extract_li_paper_items_structured(combined_text)
            
            # If that fails, fall back to the generic approach
            if not items:
                self.logger.warning("Structured extraction failed. Falling back to generic approach.")
                
                # Use a generic prompt that doesn't assume any specific knowledge about the checklist
                prompt = """
                You are an expert in biomedical research methodology and reporting checklists.
                
                Your task is to extract ALL items from the Li-Paper SOP checklist.
                For each item, provide:
                1. Item number/ID
                2. The complete item description including the field name, recommendation, and where to find the information
                3. The category or section it belongs to (if available)
                
                Format your response as a JSON array of objects, with each object representing 
                one checklist item with the following properties:
                - "id": the item number/ID
                - "description": the complete description including field name, recommendation, and where to find
                - "category": the section or category it belongs to (if available, otherwise use "General")
                - "notes": "Li-Paper SOP"
                
                IMPORTANT: Make sure to extract ALL items from the checklist. Each item should include:
                - The field name (e.g., any specific term or concept being checked)
                - The recommendation or requirement
                - Where to find the information in the paper
                
                REPORTING CHECKLIST TEXT:
                {text}
                """.format(text=combined_text[:50000])
                
                # Call LLM to extract checklist items
                self.logger.info("Extracting Li-Paper SOP items using generic LLM approach")
                result = self._call_llm(prompt)
                
                # Try to parse the result as JSON
                items = self._parse_json_result(result)
                
                # If parsing failed, try to extract JSON from text
                if not items:
                    items = self._extract_json_from_text(result)
                
                # If extraction failed, try a more structured approach
                if not items:
                    self.logger.warning("Failed to extract Li-Paper SOP items as JSON. Trying with structured text approach.")
                    items = self._extract_items_structured([combined_text])
            
            self.logger.info(f"Extracted {len(items)} Li-Paper SOP items")
            return items
        
        # For other checklists, use the LLM approach
        if is_strobe:
            checklist_type = "STROBE"
            expected_items = 22  # STROBE has 22 items
        elif is_record:
            checklist_type = "RECORD"
            expected_items = 13  # RECORD has 13 items
        else:
            checklist_type = "reporting guideline"
            expected_items = None  # Unknown number of items
        
        # Create a generic prompt that works for any checklist type
        prompt = """
        You are an expert in biomedical research methodology and reporting checklists.
        
        Your task is to extract ALL items from the following {checklist_type} checklist.
        For each item, provide:
        1. Item ID or number (as it appears in the checklist)
        2. The complete item description/requirement
        3. The category or section it belongs to (if available)
        
        Format your response as a JSON array of objects, with each object representing 
        one checklist item with the following properties:
        - "id": the item identifier (exactly as it appears in the checklist)
        - "description": the complete item description
        - "category": the section or category it belongs to (if available, otherwise use "General")
        - "notes": "{checklist_type}"
        
        Do not paraphrase or summarize the checklist. Extract items verbatim from the text.
        
        IMPORTANT: Make sure to extract ALL items from the checklist. If you can't find all items in the text, 
        use the information you have to create entries for all items.
        
        REPORTING CHECKLIST TEXT:
        {text}
        """.format(
            checklist_type=checklist_type,
            text=combined_text[:50000]  # Limit text length
        )
        
        # Call LLM to extract checklist items
        self.logger.info(f"Extracting items from {checklist_type} checklist")
        result = self._call_llm(prompt)
        
        # Try to parse the result as JSON
        items = self._parse_json_result(result)
        
        # If parsing failed, try to extract JSON from text
        if not items:
            items = self._extract_json_from_text(result)
        
        # If extraction failed, try a different approach
        if not items:
            self.logger.warning("Failed to extract checklist items as JSON. Trying with structured text approach.")
            items = self._extract_items_structured([combined_text])
        
        # Log the number of items extracted
        if expected_items and len(items) < expected_items:
            self.logger.warning(f"LLM extracted only {len(items)} {checklist_type} items (expected {expected_items}). This may indicate an issue with extraction.")
        
        # Ensure all items have the correct note
        for item in items:
            item["notes"] = checklist_type
        
        self.logger.info(f"Extracted {len(items)} checklist items from the provided texts")
        return items
    
    def generate_prompts(self, guideline_items: List[Dict[str, Any]], batch_size: int = 50) -> Dict[str, str]:
        """
        Generate prompts for each guideline item.
        Process items in batches to reduce token usage.
        
        Args:
            guideline_items: List of guideline items
            batch_size: Number of items to process in each batch
            
        Returns:
            Dictionary of prompts, keyed by item ID
        """
        prompts = {}
        
        # Process all items individually
        self.logger.info(f"Generating prompts for {len(guideline_items)} guideline items")
        
        for item in tqdm(guideline_items, desc="Generating prompts"):
            item_id = item["id"]
            prompts[item_id] = self._generate_prompt_for_item(item)
        
        self.logger.info(f"Generated {len(prompts)} prompts")
        return prompts
    
    def _parse_json_result(self, result: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON result from the LLM.
        
        Args:
            result: The LLM response
            
        Returns:
            List of checklist items
        """
        try:
            # Try to parse the result as JSON
            if result.strip().startswith("[") and result.strip().endswith("]"):
                return json.loads(result)
            
            # Look for JSON array in the text
            match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            
            return []
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM response as JSON")
            return []
    
    def _extract_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract JSON from text.
        
        Args:
            text: The text to extract JSON from
            
        Returns:
            List of checklist items
        """
        # Look for JSON-like structures
        items = []
        pattern = r'{\s*"id"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]+)"\s*,\s*"category"\s*:\s*"([^"]+)"\s*(?:,\s*"notes"\s*:\s*"([^"]+)")?\s*}'
        
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            item = {
                "id": match.group(1),
                "description": match.group(2),
                "category": match.group(3),
            }
            if match.group(4):
                item["notes"] = match.group(4)
            items.append(item)
        
        return items
    
    def _extract_li_paper_items_structured(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract Li-Paper SOP items using a structured approach.
        
        Args:
            text: Text to extract items from
            
        Returns:
            List of Li-Paper SOP items
        """
        items = []
        
        # Try to extract items using a more structured prompt
        prompt = """
        You are an expert in biomedical research methodology and reporting checklists.
        
        Extract ALL 35 items from the Li-Paper SOP checklist. For each item, provide:
        1. Item number (1-35)
        2. Field name (e.g., "EndNote_Index", "Pubmed_ID", etc.)
        3. Recommendation or requirement
        4. Where to find the information in the paper
        5. Category or section it belongs to
        
        Format each item as:
        ITEM: [item number]
        FIELD: [field name]
        RECOMMENDATION: [recommendation or requirement]
        WHERE_TO_FIND: [where to find in the paper]
        CATEGORY: [category]
        
        IMPORTANT: Make sure to extract ALL 35 items from the checklist.
        
        TEXT:
        {text}
        """.format(text=text[:50000])
        
        result = self._call_llm(prompt)
        
        # Parse the structured result
        current_item = {}
        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("ITEM:"):
                # Save previous item if it exists
                if current_item and "id" in current_item:
                    items.append(current_item)
                
                # Start new item
                item_id = line[5:].strip()
                try:
                    # Try to convert to integer and then back to string
                    item_id = str(int(item_id))
                except ValueError:
                    # If not a valid integer, keep as is
                    pass
                
                current_item = {"id": item_id, "notes": "Li-Paper SOP"}
            elif line.startswith("FIELD:") and current_item:
                current_item["field"] = line[6:].strip()
            elif line.startswith("RECOMMENDATION:") and current_item:
                current_item["recommendation"] = line[15:].strip()
            elif line.startswith("WHERE_TO_FIND:") and current_item:
                current_item["where_to_find"] = line[14:].strip()
            elif line.startswith("CATEGORY:") and current_item:
                current_item["category"] = line[9:].strip() or "General"
        
        # Add the last item
        if current_item and "id" in current_item:
            items.append(current_item)
        
        # Combine field, recommendation, and where_to_find into description
        for item in items:
            field = item.get("field", "")
            recommendation = item.get("recommendation", "")
            where_to_find = item.get("where_to_find", "")
            
            description_parts = []
            if field:
                description_parts.append(f"{field}:")
            if recommendation:
                description_parts.append(f"{recommendation}")
            if where_to_find:
                description_parts.append(f"(Find in: {where_to_find})")
            
            item["description"] = " ".join(description_parts)
        
        return items
    
    def _ensure_complete_li_paper_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure we have exactly 35 Li-Paper SOP items with the correct categories.
        
        Args:
            items: List of Li-Paper SOP items
            
        Returns:
            Complete list of 35 Li-Paper SOP items
        """
        # Create a dictionary of items by ID
        items_dict = {item["id"]: item for item in items}
        
        # Define the expected categories for each item
        categories = {}
        for i in range(1, 36):
            if i <= 8:
                categories[str(i)] = "Title and Abstract"
            elif i <= 10:
                categories[str(i)] = "Introduction"
            elif i <= 15:
                categories[str(i)] = "Methods - Data Sources"
            elif i <= 18:
                categories[str(i)] = "Methods - Variables"
            elif i <= 25:
                categories[str(i)] = "Methods - Analytic Methods"
            elif i <= 28:
                categories[str(i)] = "Results"
            elif i <= 32:
                categories[str(i)] = "Discussion"
            else:
                categories[str(i)] = "Compliance Check"
        
        # Ensure we have all 35 items
        complete_items = []
        for i in range(1, 36):
            item_id = str(i)
            if item_id in items_dict:
                # Use the extracted item, but ensure it has the correct category
                item = items_dict[item_id]
                item["category"] = categories[item_id]
                item["notes"] = "Li-Paper SOP"
                complete_items.append(item)
            else:
                # Create a placeholder item
                self.logger.warning(f"Missing Li-Paper SOP item {item_id}. Creating placeholder.")
                complete_items.append({
                    "id": item_id,
                    "description": f"Li-Paper SOP item {i}",
                    "category": categories[item_id],
                    "notes": "Li-Paper SOP"
                })
        
        return complete_items
    
    def _extract_items_structured(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract items using a structured approach.
        
        Args:
            texts: List of texts to extract items from
            
        Returns:
            List of checklist items
        """
        items = []
        
        # Try to extract items using a more structured prompt
        prompt = """
        Extract all checklist items from the following text. 
        For each item, provide:
        1. Item number/ID
        2. Description
        3. Category (if available)
        
        Format each item as:
        ITEM: [item number/ID]
        DESCRIPTION: [description]
        CATEGORY: [category]
        
        TEXT:
        {text}
        """.format(text="\n\n".join(texts)[:50000])
        
        result = self._call_llm(prompt)
        
        # Parse the structured result
        current_item = {}
        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("ITEM:"):
                # Save previous item if it exists
                if current_item and "id" in current_item and "description" in current_item:
                    items.append(current_item)
                
                # Start new item
                current_item = {"id": line[5:].strip()}
            elif line.startswith("DESCRIPTION:") and current_item:
                current_item["description"] = line[12:].strip()
            elif line.startswith("CATEGORY:") and current_item:
                current_item["category"] = line[9:].strip() or "General"
        
        # Add the last item
        if current_item and "id" in current_item and "description" in current_item:
            items.append(current_item)
        
        return items
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response
        """
        # Use the parent class's _call_llm method
        return super()._call_llm(prompt)
    
    def _generate_prompt_for_item(self, item: Dict[str, Any]) -> str:
        """
        Generate a prompt for a single checklist item.
        
        Args:
            item: Checklist item
            
        Returns:
            Prompt for the item
        """
        item_id = item["id"]
        description = item["description"]
        category = item.get("category", "")
        notes = item.get("notes", "")
        
        # Determine the item type based on notes only (no hard-coded ID detection)
        if "STROBE" in notes:
            item_type = "STROBE"
        elif "RECORD" in notes:
            item_type = "RECORD item"
        elif "Li-Paper" in notes:
            item_type = "Li-Paper SOP item"
        else:
            item_type = "Checklist item"
        
        prompt = f"""REPORTING GUIDELINE ITEM: {item_id}
            
            DESCRIPTION: {description}
            
            CATEGORY: {category}
            
            ADDITIONAL NOTES: {item_type}
            
            TASK: Please examine the research paper and extract specific information related to this 
            reporting guideline item. Identify whether the paper complies with this item, providing 
            direct quotes and page/section references where the information can be found.
            
            The response should include:
            1. Whether the paper complies with this guideline item (yes, partial, or no)
            2. Evidence from the paper (direct quotes with locations)
            3. Reasoning for your assessment
            4. The correct answer addressing the specific checklist item question
            
            IMPORTANT: Treat the item as a question. What's the correct_answer for the item? For example, 
            for item 1.0.a asking whether the study design is indicated in the title or abstract, the 
            correct_answer should directly address this question, not just state what the study design is.
            
            If the information is missing, clearly state that it is not reported in the paper."""
        
        return prompt
    
    def extract_from_paper_with_rag(self, paper_text: str, prompt: str, item_id: str, paper_id: str) -> Dict[str, Any]:
        """
        Extract information from a paper using RAG.
        
        Args:
            paper_text: Text of the paper
            prompt: Prompt for the extraction
            item_id: ID of the guideline item
            paper_id: ID of the paper
            
        Returns:
            Extraction result
        """
        self.logger.info(f"Extracting information for paper {paper_id}, checklist item: {item_id}")
        
        # Create chunks from the paper text
        chunks = self._create_chunks(paper_text)
        
        # Generate embeddings for the chunks using Voyage AI
        if self.voyage_client:
            chunk_embeddings = self._generate_voyage_embeddings(chunks)
        else:
            # Fall back to OpenAI embeddings if Voyage AI is not available
            chunk_embeddings = self._generate_openai_embeddings(chunks)
        
        # Generate embedding for the prompt
        if self.voyage_client:
            prompt_embedding = self._generate_voyage_embedding(prompt)
        else:
            # Fall back to OpenAI embeddings if Voyage AI is not available
            prompt_embedding = self._generate_openai_embedding(prompt)
        
        # Calculate similarity between prompt and chunks
        similarities = []
        for embedding in chunk_embeddings:
            similarity = self._cosine_similarity(prompt_embedding, embedding)
            similarities.append(similarity)
        
        # Get top-k chunks
        if not similarities:
            relevant_text = paper_text[:8000]  # Fallback to first 8000 chars
        else:
            top_indices = np.argsort(similarities)[-5:]  # Get top 5 chunks
            relevant_chunks = [chunks[i] for i in top_indices]
            relevant_text = "\n\n".join(relevant_chunks)
        
        # Truncate if too long
        if len(relevant_text) > 12000:
            relevant_text = relevant_text[:12000]
        
        # Create a prompt for the LLM
        extraction_prompt = f"""
        {prompt}
        
        PAPER TEXT (RELEVANT SECTIONS):
        {relevant_text}
        
        Please provide your extraction in the following JSON format:
        {{
            "paper_identifier": "{paper_id}",
            "Li-Paper_item_id": "{item_id}",
            "extracted_content": {{
                "compliance": "yes", "partial", "no", or "unknown",
                "evidence": [
                    {{
                        "quote": "direct quote from paper",
                        "location": "section/page information if available"
                    }}
                ],
                "reasoning": "explanation of your assessment",
                "correct_answer": "your answer to the specific checklist item question"
            }}
        }}
        """
        
        # Call LLM to extract information
        result = self._call_llm(extraction_prompt)
        
        # Try to parse the result as JSON
        try:
            extraction = json.loads(result)
            return extraction
        except json.JSONDecodeError:
            # Try to extract JSON from text
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
            "paper_identifier": paper_id,
            "Li-Paper_item_id": item_id,
            "extracted_content": {
                "compliance": "unknown",
                "evidence": [],
                "reasoning": f"Failed to parse result: {result[:500]}...",
                "correct_answer": "unknown"
            }
        }
    
    def validate_extraction_with_rag(self, extraction: Dict[str, Any], guideline_item: Dict[str, Any], paper_text: str, item_id: str, paper_id: str) -> Dict[str, Any]:
        """
        Validate an extraction using RAG.
        
        Args:
            extraction: Extraction result
            guideline_item: Guideline item
            paper_text: Text of the paper
            item_id: ID of the guideline item
            paper_id: ID of the paper
            
        Returns:
            Validation result
        """
        self.logger.info(f"Validating extraction for paper {paper_id}, checklist item: {item_id}")
        
        # Create chunks from the paper text
        chunks = self._create_chunks(paper_text)
        
        # Generate embeddings for the chunks using Voyage AI
        if self.voyage_client:
            chunk_embeddings = self._generate_voyage_embeddings(chunks)
        else:
            # Fall back to OpenAI embeddings if Voyage AI is not available
            chunk_embeddings = self._generate_openai_embeddings(chunks)
        
        # Generate embedding for the guideline item
        guideline_text = f"CHECKLIST ITEM: {item_id}\nDESCRIPTION: {guideline_item.get('description', '')}"
        
        if self.voyage_client:
            guideline_embedding = self._generate_voyage_embedding(guideline_text)
        else:
            # Fall back to OpenAI embeddings if Voyage AI is not available
            guideline_embedding = self._generate_openai_embedding(guideline_text)
        
        # Calculate similarity between guideline and chunks
        similarities = []
        for embedding in chunk_embeddings:
            similarity = self._cosine_similarity(guideline_embedding, embedding)
            similarities.append(similarity)
        
        # Get top-k chunks
        if not similarities:
            relevant_text = "No relevant text found in the paper."
        else:
            top_indices = np.argsort(similarities)[-5:]  # Get top 5 chunks
            relevant_chunks = [chunks[i] for i in top_indices]
            relevant_text = "\n\n".join(relevant_chunks)
        
        # Truncate if too long
        if len(relevant_text) > 12000:
            relevant_text = relevant_text[:12000]
        
        # Extract key information
        extracted_content = extraction.get("extracted_content", {})
        compliance = extracted_content.get("compliance", "unknown")
        evidence = extracted_content.get("evidence", [])
        reasoning = extracted_content.get("reasoning", "")
        
        # Build prompt for validation
        validation_prompt = f"""
        You are an expert validator for biomedical research reporting checklists.
        
        CHECKLIST ITEM: {item_id}
        DESCRIPTION: {guideline_item.get('description', '')}
        
        EXTRACTION RESULTS:
        - Paper Identifier: {paper_id}
        - Compliance: {compliance}
        - Evidence: {json.dumps(evidence, indent=2)}
        - Reasoning: {reasoning}
        
        PAPER TEXT (RELEVANT SECTIONS):
        {relevant_text}
        
        VALIDATION TASK:
        1. Evaluate whether the compliance assessment is correct based on the evidence provided and the paper text.
        2. Assess whether the evidence is sufficient and relevant to the checklist item.
        3. Provide a final assessment of whether you agree with the extractor's assessment.
        4. Provide a correct answer that will be used in the final checklist.
        
        Please provide your validation in the following JSON format:
        {{
            "paper_identifier": "{paper_id}",
            "Li-Paper_item_id": "{item_id}",
            "validate_result": "agree with extractor", "do not agree with extractor", or "unknown",
            "Reason": "your assessment of the extraction",
            "correct_answer": "your answer to the specific checklist item question"
        }}
        """
        
        # Call LLM to validate extraction
        result = self._call_llm(validation_prompt)
        
        # Try to parse the result as JSON
        try:
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
        
        # If parsing fails, return a basic structure
        return {
            "paper_identifier": paper_id,
            "Li-Paper_item_id": item_id,
            "validate_result": compliance,
            "Reason": f"Failed to parse result: {result[:500]}...",
            "correct_answer": "unknown"
        }
    
    def _create_chunks(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Create overlapping chunks from text.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) < 100:  # Skip very small chunks
                continue
            chunks.append(chunk)
            
        return chunks
    
    def _generate_voyage_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Voyage AI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self.voyage_client:
            raise ValueError("Voyage AI client not initialized")
            
        result = self.voyage_client.embed(
            text,
            model="voyage-3",
            input_type="document"
        )
        
        return result.embeddings[0]
    
    def _generate_voyage_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Voyage AI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.voyage_client:
            raise ValueError("Voyage AI client not initialized")
            
        embeddings = []
        for i in range(0, len(texts), 10):  # Process in batches of 10
            batch_texts = texts[i:i+10]
            try:
                result = self.voyage_client.embed(
                    batch_texts, 
                    model="voyage-3", 
                    input_type="document"
                )
                batch_embeddings = result.embeddings
                embeddings.extend(batch_embeddings)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch {i//10}: {e}")
                # Add placeholder embeddings for the failed batch
                for _ in range(len(batch_texts)):
                    embeddings.append([0] * 1024)  # Voyage embeddings are 1024-dimensional
                
        return embeddings
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        from openai import OpenAI
        client = OpenAI(api_key=self.api_keys["openai"])
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536
        )
        
        return response.data[0].embedding
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        from openai import OpenAI
        client = OpenAI(api_key=self.api_keys["openai"])
        
        embeddings = []
        for text in tqdm(texts, desc="Generating OpenAI embeddings"):
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                    dimensions=1536
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Error generating embedding: {e}")
                embeddings.append([0] * 1536)  # Placeholder for failed embedding
                
        return embeddings
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
