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
    
    def _extract_strobe_items_comprehensive(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract STROBE items using a comprehensive approach that ensures all 22 items are captured.
        
        Args:
            text: Text to extract items from
            
        Returns:
            List of STROBE items
        """
        self.logger.info("Using comprehensive STROBE extraction approach")
        
        # First, try to extract items using a detailed STROBE-specific prompt
        prompt = """
        You are an expert in the STROBE (STrengthening the Reporting of OBservational studies in Epidemiology) reporting guidelines.
        
        Extract ALL 22 STROBE checklist items from the following text. STROBE has items numbered 1-22, covering:
        - Title and abstract (item 1)
        - Introduction: Background/rationale (item 2) and Objectives (item 3)
        - Methods: Study design (item 4), Setting (item 5), Participants (item 6), Variables (item 7), Data sources/measurement (item 8), Bias (item 9), Study size (item 10), Quantitative variables (item 11), Statistical methods (item 12)
        - Results: Participants (item 13), Descriptive data (item 14), Outcome data (item 15), Main results (item 16), Other analyses (item 17)
        - Discussion: Key results (item 18), Limitations (item 19), Interpretation (item 20), Generalisability (item 21)
        - Other information: Funding (item 22)
        
        For each item, provide:
        1. Item number (1-22)
        2. The complete item description as it appears in STROBE
        3. The section it belongs to (Title and abstract, Introduction, Methods, Results, Discussion, Other information)
        
        Format your response as a JSON array with exactly 22 objects, each representing one STROBE item:
        [
          {
            "id": "1",
            "description": "[complete STROBE item 1 description]",
            "category": "Title and abstract",
            "notes": "STROBE"
          },
          ...
        ]
        
        IMPORTANT: You must extract all 22 STROBE items. If some items are not clearly visible in the text, 
        use your knowledge of the standard STROBE checklist to include them.
        
        STROBE CHECKLIST TEXT:
        {text}
        """.format(text=text[:50000])
        
        result = self._call_llm(prompt)
        items = self._parse_json_result(result)
        
        # If we didn't get 22 items, try a fallback approach
        if len(items) != 22:
            self.logger.warning(f"First STROBE extraction attempt yielded {len(items)} items instead of 22. Trying fallback approach.")
            items = self._extract_strobe_items_fallback(text)
        
        # Ensure we have exactly 22 items with proper structure
        items = self._ensure_complete_strobe_items(items)
        
        return items
    
    def _extract_strobe_items_fallback(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback approach for STROBE extraction using structured text parsing.
        
        Args:
            text: Text to extract items from
            
        Returns:
            List of STROBE items
        """
        self.logger.info("Using fallback STROBE extraction approach")
        
        # Use a more structured approach
        prompt = """
        Extract all STROBE checklist items from the following text. 
        For each item, format as:
        
        ITEM: [item number]
        DESCRIPTION: [complete description]
        CATEGORY: [section name]
        
        Separate each item with three dashes (---).
        
        STROBE items should cover:
        1. Title and abstract
        2. Background/rationale
        3. Objectives
        4. Study design
        5. Setting
        6. Participants
        7. Variables
        8. Data sources/measurement
        9. Bias
        10. Study size
        11. Quantitative variables
        12. Statistical methods
        13. Participants (results)
        14. Descriptive data
        15. Outcome data
        16. Main results
        17. Other analyses
        18. Key results
        19. Limitations
        20. Interpretation
        21. Generalisability
        22. Funding
        
        TEXT:
        {text}
        """.format(text=text[:50000])
        
        result = self._call_llm(prompt)
        
        # Parse the structured result
        items = []
        current_item = {}
        
        for line in result.split("\n"):
            line = line.strip()
            if not line or line == "---":
                # Save current item if complete
                if current_item and "id" in current_item and "description" in current_item:
                    current_item["notes"] = "STROBE"
                    items.append(current_item)
                current_item = {}
                continue
            
            if line.startswith("ITEM:"):
                current_item["id"] = line[5:].strip()
            elif line.startswith("DESCRIPTION:"):
                current_item["description"] = line[12:].strip()
            elif line.startswith("CATEGORY:"):
                current_item["category"] = line[9:].strip() or "General"
        
        # Add the last item
        if current_item and "id" in current_item and "description" in current_item:
            current_item["notes"] = "STROBE"
            items.append(current_item)
        
        return items
    
    def _ensure_complete_strobe_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure we have exactly 22 STROBE items with proper structure.
        
        Args:
            items: List of STROBE items
            
        Returns:
            Complete list of 22 STROBE items
        """
        # Create a dictionary of items by ID
        items_dict = {item["id"]: item for item in items}
        
        # Define the standard STROBE items
        standard_strobe_items = {
            "1": {"description": "Title and abstract: Indicate the study's design with a commonly used term in the title or the abstract. Provide in the abstract an informative and balanced summary of what was done and what was found.", "category": "Title and abstract"},
            "2": {"description": "Background/rationale: Explain the scientific background and rationale for the investigation being reported.", "category": "Introduction"},
            "3": {"description": "Objectives: State specific objectives, including any prespecified hypotheses.", "category": "Introduction"},
            "4": {"description": "Study design: Present key elements of study design early in the paper.", "category": "Methods"},
            "5": {"description": "Setting: Describe the setting, locations, and relevant dates, including periods of recruitment, exposure, follow-up, and data collection.", "category": "Methods"},
            "6": {"description": "Participants: Give the eligibility criteria, and the sources and methods of selection of participants. Describe methods of follow-up.", "category": "Methods"},
            "7": {"description": "Variables: Clearly define all outcomes, exposures, predictors, potential confounders, and effect modifiers. Give diagnostic criteria, if applicable.", "category": "Methods"},
            "8": {"description": "Data sources/measurement: For each variable of interest, give sources of data and details of methods of assessment (measurement). Describe comparability of assessment methods if there is more than one group.", "category": "Methods"},
            "9": {"description": "Bias: Describe any efforts to address potential sources of bias.", "category": "Methods"},
            "10": {"description": "Study size: Explain how the study size was arrived at.", "category": "Methods"},
            "11": {"description": "Quantitative variables: Explain how quantitative variables were handled in the analyses. If applicable, describe which groupings were chosen and why.", "category": "Methods"},
            "12": {"description": "Statistical methods: Describe all statistical methods, including those used to control for confounding. Describe any methods used to examine subgroups and interactions. Explain how missing data were addressed.", "category": "Methods"},
            "13": {"description": "Participants: Report numbers of individuals at each stage of study—eg numbers potentially eligible, examined for eligibility, confirmed eligible, included in the study, completing follow-up, and analysed.", "category": "Results"},
            "14": {"description": "Descriptive data: Give characteristics of study participants (eg demographic, clinical, social) and information on exposures and potential confounders. Indicate number of participants with missing data for each variable of interest.", "category": "Results"},
            "15": {"description": "Outcome data: Report numbers of outcome events or summary measures over time.", "category": "Results"},
            "16": {"description": "Main results: Give unadjusted estimates and, if applicable, confounder-adjusted estimates and their precision (eg, 95% confidence interval). Make clear which confounders were adjusted for and why they were included.", "category": "Results"},
            "17": {"description": "Other analyses: Report other analyses done—eg analyses of subgroups and interactions, and sensitivity analyses.", "category": "Results"},
            "18": {"description": "Key results: Summarise key results with reference to study objectives.", "category": "Discussion"},
            "19": {"description": "Limitations: Discuss limitations of the study, taking into account sources of potential bias or imprecision. Discuss both direction and magnitude of any potential bias.", "category": "Discussion"},
            "20": {"description": "Interpretation: Give a cautious overall interpretation of results considering objectives, limitations, multiplicity of analyses, results from similar studies, and other relevant evidence.", "category": "Discussion"},
            "21": {"description": "Generalisability: Discuss the generalisability (external validity) of the study results.", "category": "Discussion"},
            "22": {"description": "Funding: Give the source of funding and the role of the funders for the present study and, if applicable, for the original study on which the present article is based.", "category": "Other information"}
        }
        
        # Ensure we have all 22 items
        complete_items = []
        for i in range(1, 23):
            item_id = str(i)
            if item_id in items_dict:
                # Use the extracted item, but ensure it has the correct structure
                item = items_dict[item_id]
                item["notes"] = "STROBE"
                # If category is missing, use the standard one
                if not item.get("category"):
                    item["category"] = standard_strobe_items[item_id]["category"]
                complete_items.append(item)
            else:
                # Create a standard STROBE item
                self.logger.warning(f"Missing STROBE item {item_id}. Using standard definition.")
                complete_items.append({
                    "id": item_id,
                    "description": standard_strobe_items[item_id]["description"],
                    "category": standard_strobe_items[item_id]["category"],
                    "notes": "STROBE"
                })
        
        self.logger.info(f"Ensured complete STROBE checklist with {len(complete_items)} items")
        return complete_items
        
        # For other checklists, use the LLM approach
        if is_strobe:
            checklist_type = "STROBE"
            expected_items = 22  # STROBE has 22 items
            # Use specialized STROBE extraction
            items = self._extract_strobe_items_comprehensive(combined_text)
            if items:
                self.logger.info(f"Extracted {len(items)} STROBE items using comprehensive approach")
                return items
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
            4. The correct answer addressing the specific checklist item question (e.g., for item 1.0.a, whether the study design is indicated in the title or abstract, not just what the study design is):
            
            If the information is missing, clearly state that it is not reported in the paper."""
        
        return prompt
