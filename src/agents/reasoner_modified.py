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
        Extract guideline items from the RECORD guideline texts.
        This modified version extracts both RECORD-specific items and STROBE items.
        
        Args:
            guideline_texts: List of guideline texts
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of guideline items
        """
        self.logger.info(f"Extracting guideline items from texts using batch size of {batch_size}")
        
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
        Extract additional STROBE and RECORD items that may not be captured by the base extraction.
        
        Args:
            guideline_texts: List of guideline texts
            
        Returns:
            List of additional guideline items
        """
        # Define the additional items that should be included
        additional_items = [
            # RECORD items
            {
                "id": "1.1",
                "description": "The type of data used should be specified in the title or abstract. When possible, the name of the databases used should be included.",
                "category": "Title and abstract",
                "notes": "RECORD item"
            },
            {
                "id": "1.2",
                "description": "If applicable, the geographic region and timeframe within which the study took place should be reported in the title or abstract.",
                "category": "Title and abstract",
                "notes": "RECORD item"
            },
            {
                "id": "1.3",
                "description": "If linkage between databases was conducted for the study, this should be clearly stated in the title or abstract.",
                "category": "Title and abstract",
                "notes": "RECORD item"
            },
            {
                "id": "6.1",
                "description": "The methods of study population selection (such as codes or algorithms used to identify subjects) should be listed in detail. If this is not possible, an explanation should be provided.",
                "category": "Methods (Participants)",
                "notes": "RECORD item"
            },
            {
                "id": "6.2",
                "description": "Any validation studies of the codes or algorithms used to select the population should be referenced. If validation was conducted for this study and not published elsewhere, detailed methods and results should be provided.",
                "category": "Methods (Participants)",
                "notes": "RECORD item"
            },
            {
                "id": "6.3",
                "description": "If the study involved linkage of databases, consider use of a flow diagram or other graphical display to demonstrate the data linkage process, including the number of individuals with linked data at each stage.",
                "category": "Methods (Participants)",
                "notes": "RECORD item"
            },
            {
                "id": "7.1",
                "description": "A complete list of codes and algorithms used to classify exposures, outcomes, confounders, and effect modifiers should be provided. If these cannot be reported, an explanation should be provided.",
                "category": "Methods (Variables)",
                "notes": "RECORD item"
            },
            {
                "id": "12.1",
                "description": "Authors should describe the extent to which the investigators had access to the database population used to create the study population.",
                "category": "Methods (Statistical methods)",
                "notes": "RECORD item"
            },
            {
                "id": "12.2",
                "description": "Authors should provide information on the data cleaning methods used in the study.",
                "category": "Methods (Statistical methods)",
                "notes": "RECORD item"
            },
            {
                "id": "12.3",
                "description": "State whether the study included person-level, institutional-level, or other data linkage across two or more databases. The methods of linkage and methods of linkage quality evaluation should be provided.",
                "category": "Methods (Statistical methods)",
                "notes": "RECORD item"
            },
            {
                "id": "13.1",
                "description": "Describe in detail the selection of the persons included in the study (i.e., study population selection) including filtering based on data quality, data availability and linkage. The selection of included persons can be described in the text and/or by means of the study flow diagram.",
                "category": "Results (Participants)",
                "notes": "RECORD item"
            },
            {
                "id": "19.1",
                "description": "Discuss the implications of using data that were not created or collected to answer the specific research question(s). Include discussion of misclassification bias, unmeasured confounding, missing data, and changing eligibility over time, as they pertain to the study being reported.",
                "category": "Discussion (Limitations)",
                "notes": "RECORD item"
            },
            {
                "id": "22.1",
                "description": "Authors should provide information on how to access any supplemental information such as the study protocol, raw data, or programming code.",
                "category": "Other information (Funding)",
                "notes": "RECORD item"
            },
            
            # STROBE items
            {
                "id": "1.0.a",
                "description": "Indicate the study's design with a commonly used term in the title or the abstract.",
                "category": "Title and abstract",
                "notes": "STROBE item"
            },
            {
                "id": "1.0.b",
                "description": "Provide in the abstract an informative and balanced summary of what was done and what was found.",
                "category": "Title and abstract",
                "notes": "STROBE item"
            },
            {
                "id": "2.0",
                "description": "Explain the scientific background and rationale for the investigation being reported.",
                "category": "Introduction (Background/rationale)",
                "notes": "STROBE item"
            },
            {
                "id": "3.0",
                "description": "State specific objectives, including any prespecified hypotheses.",
                "category": "Introduction (Objectives)",
                "notes": "STROBE item"
            },
            {
                "id": "4.0",
                "description": "Present key elements of study design early in the paper.",
                "category": "Methods (Study design)",
                "notes": "STROBE item"
            },
            {
                "id": "5.0",
                "description": "Describe the setting, locations, and relevant dates, including periods of recruitment, exposure, follow-up, and data collection.",
                "category": "Methods (Setting)",
                "notes": "STROBE item"
            },
            {
                "id": "6.0.a",
                "description": "Cohort study: Give the eligibility criteria, and the sources and methods of selection of participants. Describe methods of follow-up.",
                "category": "Methods (Participants)",
                "notes": "STROBE item"
            },
            {
                "id": "6.0.b",
                "description": "Cohort study: For matched studies, give matching criteria and number of exposed and unexposed.",
                "category": "Methods (Participants)",
                "notes": "STROBE item"
            },
            {
                "id": "7.0",
                "description": "Clearly define all outcomes, exposures, predictors, potential confounders, and effect modifiers. Give diagnostic criteria, if applicable.",
                "category": "Methods (Variables)",
                "notes": "STROBE item"
            },
            {
                "id": "8.0",
                "description": "For each variable of interest, give sources of data and details of methods of assessment (measurement). Describe comparability of assessment methods if there is more than one group.",
                "category": "Methods (Data sources/measurement)",
                "notes": "STROBE item"
            },
            {
                "id": "9.0",
                "description": "Describe any efforts to address potential sources of bias.",
                "category": "Methods (Bias)",
                "notes": "STROBE item"
            },
            {
                "id": "10.0",
                "description": "Explain how the study size was arrived at.",
                "category": "Methods (Study size)",
                "notes": "STROBE item"
            },
            {
                "id": "11.0",
                "description": "Explain how quantitative variables were handled in the analyses. If applicable, describe which groupings were chosen and why.",
                "category": "Methods (Quantitative variables)",
                "notes": "STROBE item"
            },
            {
                "id": "12.0.a",
                "description": "Describe all statistical methods, including those used to control for confounding.",
                "category": "Methods (Statistical methods)",
                "notes": "STROBE item"
            },
            {
                "id": "12.0.b",
                "description": "Describe any methods used to examine subgroups and interactions.",
                "category": "Methods (Statistical methods)",
                "notes": "STROBE item"
            },
            {
                "id": "12.0.c",
                "description": "Explain how missing data were addressed.",
                "category": "Methods (Statistical methods)",
                "notes": "STROBE item"
            },
            {
                "id": "12.0.d",
                "description": "Cohort study: If applicable, explain how loss to follow-up was addressed.",
                "category": "Methods (Statistical methods)",
                "notes": "STROBE item"
            },
            {
                "id": "12.0.e",
                "description": "Describe any sensitivity analyses.",
                "category": "Methods (Statistical methods)",
                "notes": "STROBE item"
            },
            {
                "id": "13.0.a",
                "description": "Report numbers of individuals at each stage of study—eg numbers potentially eligible, examined for eligibility, confirmed eligible, included in the study, completing follow-up, and analysed.",
                "category": "Results (Participants)",
                "notes": "STROBE item"
            },
            {
                "id": "13.0.b",
                "description": "Give reasons for non-participation at each stage.",
                "category": "Results (Participants)",
                "notes": "STROBE item"
            },
            {
                "id": "13.0.c",
                "description": "Consider use of a flow diagram.",
                "category": "Results (Participants)",
                "notes": "STROBE item"
            },
            {
                "id": "14.0.a",
                "description": "Give characteristics of study participants (eg demographic, clinical, social) and information on exposures and potential confounders.",
                "category": "Results (Descriptive data)",
                "notes": "STROBE item"
            },
            {
                "id": "14.0.b",
                "description": "Indicate number of participants with missing data for each variable of interest.",
                "category": "Results (Descriptive data)",
                "notes": "STROBE item"
            },
            {
                "id": "14.0.c",
                "description": "Cohort study: Summarise follow-up time (eg, average and total amount).",
                "category": "Results (Descriptive data)",
                "notes": "STROBE item"
            },
            {
                "id": "15.0",
                "description": "Cohort study: Report numbers of outcome events or summary measures over time.",
                "category": "Results (Outcome data)",
                "notes": "STROBE item"
            },
            {
                "id": "16.0.a",
                "description": "Give unadjusted estimates and, if applicable, confounder-adjusted estimates and their precision (eg, 95% confidence interval). Make clear which confounders were adjusted for and why they were included.",
                "category": "Results (Main results)",
                "notes": "STROBE item"
            },
            {
                "id": "16.0.b",
                "description": "Report category boundaries when continuous variables were categorized.",
                "category": "Results (Main results)",
                "notes": "STROBE item"
            },
            {
                "id": "16.0.c",
                "description": "If relevant, consider translating estimates of relative risk into absolute risk for a meaningful time period.",
                "category": "Results (Main results)",
                "notes": "STROBE item"
            },
            {
                "id": "17.0",
                "description": "Report other analyses done—eg analyses of subgroups and interactions, and sensitivity analyses.",
                "category": "Results (Other analyses)",
                "notes": "STROBE item"
            },
            {
                "id": "18.0",
                "description": "Summarise key results with reference to study objectives.",
                "category": "Discussion (Key results)",
                "notes": "STROBE item"
            },
            {
                "id": "19.0",
                "description": "Discuss limitations of the study, taking into account sources of potential bias or imprecision. Discuss both direction and magnitude of any potential bias.",
                "category": "Discussion (Limitations)",
                "notes": "STROBE item"
            },
            {
                "id": "20.0",
                "description": "Give a cautious overall interpretation of results considering objectives, limitations, multiplicity of analyses, results from similar studies, and other relevant evidence.",
                "category": "Discussion (Interpretation)",
                "notes": "STROBE item"
            },
            {
                "id": "21.0",
                "description": "Discuss the generalisability (external validity) of the study results.",
                "category": "Discussion (Generalisability)",
                "notes": "STROBE item"
            },
            {
                "id": "22.0",
                "description": "Give the source of funding and the role of the funders for the present study and, if applicable, for the original study on which the present article is based.",
                "category": "Other information (Funding)",
                "notes": "STROBE item"
            }
        ]
        
        return additional_items
    
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
    
    def _generate_prompt_for_item(self, item: Dict[str, Any]) -> str:
        """
        Generate a prompt for a single guideline item.
        
        Args:
            item: Guideline item
            
        Returns:
            Prompt for the item
        """
        item_id = item["id"]
        description = item["description"]
        category = item.get("category", "")
        notes = item.get("notes", "")
        
        # Determine if this is a STROBE or RECORD item
        item_type = "STROBE item" if "STROBE" in notes else "RECORD item"
        
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
