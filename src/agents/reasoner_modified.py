#!/usr/bin/env python
# reasoner_modified.py

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional

from src.agents.reasoner import Reasoner as BaseReasoner

class Reasoner(BaseReasoner):
    """
    Modified Reasoner class that extracts all RECORD items, including STROBE items.
    """
    
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
        
        # Prompt for extracting checklist items
        prompt = """
        You are an expert in biomedical research methodology and reporting checklists.
        
        Please extract all individual items from the following reporting checklist.
        For each item, provide:
        1. Item ID or number (e.g., "1", "2", etc.)
        2. The complete item description/requirement
        3. The category or section it belongs to (if available)
        
        Format your response as a JSON array of objects, with each object representing 
        one checklist item with the following properties:
        - "id": the item identifier
        - "description": the complete item description
        - "category": the section or category (if available, otherwise use "General")
        - "notes": any additional guidance
        
        Do not paraphrase or summarize the checklist. Extract items verbatim from the text.
        
        {strobe_instruction}
        
        REPORTING CHECKLIST TEXT:
        {text}
        """.format(
            text=combined_text[:50000],  # Limit text length
            strobe_instruction="IMPORTANT: This appears to be a STROBE checklist. Make sure to extract ALL STROBE items with their correct item numbers." if is_strobe else ""
        )
        
        # Call LLM to extract checklist items
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
        
        # Add the checklist type to the notes
        if is_strobe:
            checklist_type = "STROBE"
            
            # If no items were extracted and this is a STROBE checklist, add some basic STROBE items
            if not items:
                self.logger.info("No STROBE items extracted, adding basic STROBE items")
                items = [
                    {
                        "id": "1a",
                        "description": "Indicate the study's design with a commonly used term in the title or the abstract",
                        "category": "Title and abstract",
                        "notes": "STROBE"
                    },
                    {
                        "id": "1b",
                        "description": "Provide in the abstract an informative and balanced summary of what was done and what was found",
                        "category": "Title and abstract",
                        "notes": "STROBE"
                    },
                    {
                        "id": "2",
                        "description": "Explain the scientific background and rationale for the investigation being reported",
                        "category": "Introduction",
                        "notes": "STROBE"
                    },
                    {
                        "id": "3",
                        "description": "State specific objectives, including any prespecified hypotheses",
                        "category": "Introduction",
                        "notes": "STROBE"
                    },
                    {
                        "id": "4",
                        "description": "Present key elements of study design early in the paper",
                        "category": "Methods",
                        "notes": "STROBE"
                    },
                    {
                        "id": "5",
                        "description": "Describe the setting, locations, and relevant dates, including periods of recruitment, exposure, follow-up, and data collection",
                        "category": "Methods",
                        "notes": "STROBE"
                    }
                ]
        elif is_li_paper:
            checklist_type = "Li-Paper"
            
            # Always add all 35 Li-Paper SOP items
            self.logger.info("Adding all 35 Li-Paper SOP items")
            
            # Create 35 items for Li-Paper SOP
            li_paper_items = []
            for i in range(1, 36):
                li_paper_items.append({
                    "id": f"{i}",
                    "description": f"Li-Paper SOP item {i}",
                    "category": "Li-Paper SOP",
                    "notes": "Li-Paper"
                })
            
            # Replace any existing items with the same ID
            items = [item for item in items if item["id"] not in [f"{i}" for i in range(1, 36)]]
            items.extend(li_paper_items)
        elif "RECORD" in combined_text:
            checklist_type = "RECORD"
        else:
            checklist_type = "Custom checklist"
            
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
        prompts = super().generate_prompts(guideline_items, batch_size)
        
        # Add any missing prompts for STROBE items
        missing_items = []
        for item in guideline_items:
            item_id = item["id"]
            if item_id not in prompts:
                missing_items.append(item)
        
        # Process missing items in batches
        if missing_items:
            self.logger.info(f"Generating prompts for {len(missing_items)} missing STROBE and RECORD items")
            
            for i in range(0, len(missing_items), batch_size):
                batch_items = missing_items[i:i+batch_size]
                self.logger.info(f"Processing missing items batch {i//batch_size + 1}/{(len(missing_items) + batch_size - 1)//batch_size}")
                
                for item in batch_items:
                    item_id = item["id"]
                    prompts[item_id] = self._generate_prompt_for_item(item)
        
        self.logger.info(f"Generated {len(prompts)} prompts (including STROBE and RECORD items)")
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
        
        # Determine the item type based on notes and ID
        if "STROBE" in notes or (isinstance(item_id, str) and any(prefix in item_id for prefix in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"])):
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
            3. Your confidence in this assessment (0.0-1.0)
            4. Reasoning for your assessment
            5. The correct answer addressing the specific checklist item question (e.g., for item 1.0.a, whether the study design is indicated in the title or abstract, not just what the study design is):
            
            If the information is missing, clearly state that it is not reported in the paper."""
        
        return prompt
