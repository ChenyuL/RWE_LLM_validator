"""
RECORD Reasoner - Special implementation for RECORD guideline processing.

This module extends the base Reasoner with specific functionality for:
1. Extracting RECORD guideline items from PDF checklist
2. Properly handling STROBE base items and RECORD extensions
3. Generating specific prompts for the Extractor to check compliance
"""

import os
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from src.agents.reasoner import Reasoner

class RecordReasoner(Reasoner):
    """
    Specialized reasoner for handling RECORD guidelines.
    
    This reasoner understands the structure of RECORD items which extend STROBE items,
    with appropriate numbering convention: X.0 for original STROBE items and X.Y for RECORD extensions.
    """
    
    def __init__(self, api_keys: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RECORD Reasoner.
        
        Args:
            api_keys: Dictionary containing API keys for LLM providers
            config: Optional configuration parameters
        """
        super().__init__(api_keys, config)
        self.logger = logging.getLogger(__name__)
        
        # Store processed RECORD items
        self.record_items = []
    
    def extract_record_items(self, checklist_text: str) -> List[Dict[str, Any]]:
        """
        Extract RECORD and STROBE items from the RECORD checklist.
        
        Args:
            checklist_text: Text of the RECORD checklist
            
        Returns:
            List of dictionaries with item information
        """
        self.logger.info("Extracting RECORD guideline items")
        
        # First try using LLM for extraction
        llm_extracted_items = self._extract_with_llm(checklist_text)
        
        # If LLM extraction is successful and has reasonable number of items, use it
        if llm_extracted_items and len(llm_extracted_items) > 10:
            self.logger.info(f"Successfully extracted {len(llm_extracted_items)} items using LLM")
            self.record_items = llm_extracted_items
            return llm_extracted_items
        
        # Otherwise, fall back to rule-based extraction
        self.logger.info("LLM extraction failed or insufficient, using rule-based extraction")
        rule_based_items = self._extract_with_rules(checklist_text)
        self.record_items = rule_based_items
        return rule_based_items
    
    def _extract_with_llm(self, checklist_text: str) -> List[Dict[str, Any]]:
        """
        Extract RECORD items using LLM.
        
        Args:
            checklist_text: Text of the RECORD checklist
            
        Returns:
            List of dictionaries with item information
        """
        prompt = """
        You are an expert in biomedical research methodology and reporting guidelines.
        
        I'm providing you with the RECORD (REporting of studies Conducted using Observational Routinely-collected health Data) checklist,
        which extends the STROBE checklist for observational studies.
        
        Please extract ALL the items from the checklist using the following rules:
        1. For original STROBE items, use item_id format: "X.0.a", "X.0.b", etc. where X is the item number
        2. For RECORD extensions, use item_id format "X.Y" where X is the base item number and Y is the RECORD extension
        3. Include the full text description for each item
        4. Group related items together (e.g., all items under "Title and abstract" should be grouped)
        
        Your output should be a valid JSON array of objects with this structure:
        [
          {
            "item_id": "1.0.a",
            "content": "Indicate the study's design with a commonly used term in the title or the abstract",
            "category": "Title and abstract"
          },
          {
            "item_id": "1.1",
            "content": "The type of data used should be specified in the title or abstract. When possible, the name of the databases used should be included.",
            "category": "Title and abstract"
          }
        ]
        
        Make sure to extract ALL items, both the original STROBE items and their RECORD extensions. Be precise with item numbers.
        
        Here is the RECORD checklist:
        {checklist}
        """.format(checklist=checklist_text)
        
        # Call LLM for extraction
        result = self._call_llm(prompt)
        
        try:
            # Try to parse as JSON
            items = json.loads(result)
            return items
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'(\[\s*\{.*\}\s*\])', result, re.DOTALL)
            if json_match:
                try:
                    items = json.loads(json_match.group(1))
                    return items
                except json.JSONDecodeError:
                    self.logger.error("Failed to extract JSON from LLM response")
            
            return []
    
    def _extract_with_rules(self, checklist_text: str) -> List[Dict[str, Any]]:
        """
        Extract RECORD items using rule-based pattern matching.
        
        Args:
            checklist_text: Text of the RECORD checklist
            
        Returns:
            List of dictionaries with item information
        """
        items = []
        
        # Pattern to match RECORD extension items
        record_pattern = r'RECORD\s+(\d+\.\d+)\s+(.*?)(?=RECORD\s+\d+\.\d+|$)'
        
        # Pattern to match STROBE base items
        strobe_pattern = r'Item\s+(\d+[a-z]?)\s*\(([a-z])\)(.*?)(?=\(([a-z])\)|RECORD|$)'
        
        # Extract categories
        categories = ["Title and abstract", "Introduction", "Methods", "Results", "Discussion", "Other Information"]
        category_map = {}
        
        for category in categories:
            category_pattern = re.compile(f'({category}.*?)(?={"|".join(categories)}|$)', re.DOTALL)
            match = category_pattern.search(checklist_text)
            if match:
                category_text = match.group(1)
                # Extract item numbers in this category
                item_nums = re.findall(r'Item\s+No\.\s*(\d+[a-z]?)', category_text)
                for num in item_nums:
                    category_map[num] = category
        
        # Extract STROBE items
        strobe_matches = re.finditer(strobe_pattern, checklist_text, re.DOTALL)
        
        for match in strobe_matches:
            item_num = match.group(1)
            sub_item = match.group(2)
            content = match.group(3).strip()
            
            # Clean up content
            content = re.sub(r'\s+', ' ', content)
            
            category = category_map.get(item_num, "Unknown")
            
            items.append({
                "item_id": f"{item_num}.0.{sub_item}",
                "content": content,
                "category": category
            })
        
        # Extract RECORD items
        record_matches = re.finditer(record_pattern, checklist_text, re.DOTALL)
        
        for match in record_matches:
            item_id = match.group(1)
            content = match.group(2).strip()
            
            # Clean up content
            content = re.sub(r'\s+', ' ', content)
            
            # Determine category based on item number
            base_num = item_id.split('.')[0]
            category = category_map.get(base_num, "Unknown")
            
            items.append({
                "item_id": item_id,
                "content": content,
                "category": category
            })
        
        return items
    
    def generate_record_prompts(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate prompts for the Extractor based on RECORD items.
        
        Returns:
            Dictionary mapping item IDs to item information and prompts
        """
        if not self.record_items:
            self.logger.warning("No RECORD items to generate prompts from")
            return {}
        
        self.logger.info(f"Generating prompts for {len(self.record_items)} RECORD items")
        
        prompts = {}
        for item in self.record_items:
            item_id = item.get("item_id", "unknown")
            content = item.get("content", "")
            category = item.get("category", "")
            
            # Generate prompt for this item
            prompt = self._generate_item_prompt(item_id, content, category)
            
            # Store item with prompt
            prompts[item_id] = {
                "item_id": item_id,
                "content": content,
                "category": category,
                "prompt": prompt
            }
        
        return prompts
    
    def _generate_item_prompt(self, item_id: str, content: str, category: str) -> str:
        """
        Generate a specific prompt for extracting information related to a RECORD item.
        
        Args:
            item_id: ID of the RECORD item
            content: Content/description of the item
            category: Category of the item
            
        Returns:
            Prompt for the Extractor
        """
        # Customize prompt based on the type of item
        if "title" in category.lower() or "abstract" in category.lower():
            return f"""
            RECORD GUIDELINE ITEM {item_id}: {content}
            
            CATEGORY: {category}
            
            TASK: Examine the title and abstract of the paper to determine if it complies with RECORD item {item_id}.
            
            Look specifically for:
            1. Whether the required information is present in the title or abstract
            2. Exact quotes that demonstrate compliance or non-compliance
            3. Any missing elements that should have been included
            
            Provide your assessment including:
            - Compliance (yes/partial/no)
            - Evidence quotes from the paper
            - Location of the evidence (title or which part of abstract)
            - Your confidence in this assessment (0.0-1.0)
            - Reasoning for your assessment
            """
        
        elif "method" in category.lower():
            return f"""
            RECORD GUIDELINE ITEM {item_id}: {content}
            
            CATEGORY: {category}
            
            TASK: Examine the methods section of the paper to determine if it complies with RECORD item {item_id}.
            
            Look specifically for:
            1. Whether the required methodological details are reported
            2. Exact quotes that demonstrate compliance or non-compliance
            3. Any missing methodological elements
            4. Details about data sources, codes, algorithms, or linkage techniques if relevant
            
            Provide your assessment including:
            - Compliance (yes/partial/no)
            - Evidence quotes from the paper
            - Location of the evidence (section and subsection)
            - Your confidence in this assessment (0.0-1.0)
            - Reasoning for your assessment
            """
        
        elif "result" in category.lower():
            return f"""
            RECORD GUIDELINE ITEM {item_id}: {content}
            
            CATEGORY: {category}
            
            TASK: Examine the results section of the paper to determine if it complies with RECORD item {item_id}.
            
            Look specifically for:
            1. Whether the required results are reported
            2. Exact quotes that demonstrate compliance or non-compliance
            3. Any missing result elements
            4. Relevant tables, figures, or flow diagrams if mentioned in the guideline
            
            Provide your assessment including:
            - Compliance (yes/partial/no)
            - Evidence quotes from the paper
            - Location of the evidence (section, subsection, table, or figure)
            - Your confidence in this assessment (0.0-1.0)
            - Reasoning for your assessment
            """
        
        elif "discussion" in category.lower():
            return f"""
            RECORD GUIDELINE ITEM {item_id}: {content}
            
            CATEGORY: {category}
            
            TASK: Examine the discussion section of the paper to determine if it complies with RECORD item {item_id}.
            
            Look specifically for:
            1. Whether the required discussion elements are included
            2. Exact quotes that demonstrate compliance or non-compliance
            3. Any missing discussion elements
            4. Specific mentions of limitations related to the use of routinely collected data
            
            Provide your assessment including:
            - Compliance (yes/partial/no)
            - Evidence quotes from the paper
            - Location of the evidence (section and subsection)
            - Your confidence in this assessment (0.0-1.0)
            - Reasoning for your assessment
            """
        
        # Default prompt for other categories
        return f"""
        RECORD GUIDELINE ITEM {item_id}: {content}
        
        CATEGORY: {category}
        
        TASK: Examine the paper to determine if it complies with RECORD item {item_id}.
        
        Look specifically for:
        1. Whether the required information is reported
        2. Exact quotes that demonstrate compliance or non-compliance
        3. Any missing elements
        
        Provide your assessment including:
        - Compliance (yes/partial/no)
        - Evidence quotes from the paper
        - Location of the evidence (section and page/paragraph)
        - Your confidence in this assessment (0.0-1.0)
        - Reasoning for your assessment
        """
    
    def process_record_checklist(self, checklist_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Process a RECORD checklist PDF and generate prompts.
        
        Args:
            checklist_path: Path to the RECORD checklist PDF
            
        Returns:
            Dictionary mapping item IDs to item information and prompts
        """
        self.logger.info(f"Processing RECORD checklist: {checklist_path}")
        
        # Import PDF processor (avoiding circular imports)
        from src.utils.pdf_processor import PDFProcessor
        pdf_processor = PDFProcessor()
        
        # Extract text from PDF
        checklist_text = pdf_processor.extract_text(checklist_path)
        
        # Extract RECORD items
        self.extract_record_items(checklist_text)
        
        # Generate prompts
        prompts = self.generate_record_prompts()
        
        self.logger.info(f"Generated {len(prompts)} prompts for RECORD items")
        
        return prompts