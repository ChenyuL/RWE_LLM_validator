"""
FHIR Evidence Generator

This module provides functionality to generate FHIR Evidence resources
from validation report data for clinical research interoperability.
"""

import json
import datetime
import uuid
import streamlit as st
from typing import Dict, List, Any, Optional


def generate_fhir_evidence(report_data: Dict[str, Any], model: str, api_keys: Dict[str, str], 
                          include_evidence: bool = True, include_reasoning: bool = True) -> Optional[Dict[str, Any]]:
    """Generate FHIR Evidence resource from validation report data using LLM."""
    try:
        # Initialize LLM client
        if 'gpt' in model.lower() or 'o1' in model.lower() or 'o3' in model.lower():
            from openai import OpenAI
            client = OpenAI(api_key=api_keys['openai'])
            client_type = 'openai'
        elif 'claude' in model.lower():
            from anthropic import Anthropic
            client = Anthropic(api_key=api_keys['anthropic'])
            client_type = 'anthropic'
        else:
            return None
        
        # Prepare data for FHIR generation
        paper_name = report_data.get('paper', 'Unknown')
        checklist_type = report_data.get('checklist', 'Unknown')
        validation_summary = report_data.get('validation_summary', {})
        items = report_data.get('items', {})
        pdf_content = report_data.get('pdf_content', '')
        
        # Create comprehensive prompt for FHIR Evidence generation
        if pdf_content:
            # Use PDF content for comprehensive FHIR generation
            prompt = f"""
You are an expert in FHIR R4 Evidence resources and clinical research evidence representation.

Generate a complete FHIR R4 Evidence resource based on the following research paper content:

PAPER: {paper_name}
CHECKLIST: {checklist_type}

FULL PAPER CONTENT:
{pdf_content[:15000]}  # Limit content to fit in context window

Extract and analyze the paper content to populate ALL FHIR Evidence fields with relevant information from the paper.
"""
        else:
            # Fallback to report data if no PDF content
            prompt = f"""
You are an expert in FHIR R4 Evidence resources and clinical research evidence representation.

Generate a complete FHIR R4 Evidence resource based on the following validation report data:

PAPER: {paper_name}
CHECKLIST: {checklist_type}
VALIDATION SUMMARY: {json.dumps(validation_summary, indent=2)}
DETAILED ITEMS: {json.dumps(items, indent=2)}
"""
        
        prompt += f"""

You MUST generate a comprehensive FHIR Evidence resource that includes ALL the following fields according to the FHIR R4 Evidence specification:

REQUIRED FIELDS (MUST include):
- resourceType: "Evidence"
- id: unique identifier
- meta: with lastUpdated and profile
- status: "active" (from PublicationStatus value set)

MANDATORY FIELDS TO INCLUDE:
- url: canonical URI for this evidence
- identifier: array with system and value
- version: business version string
- name: machine-friendly name
- title: human-friendly title
- citeAs: markdown citation format
- experimental: boolean (set to false)
- date: current dateTime
- approvalDate: date when approved
- lastReviewDate: date when last reviewed
- author: array of ContactDetail objects
- publisher: string with organization name
- contact: array of ContactDetail objects
- recorder: array of ContactDetail objects
- editor: array of ContactDetail objects
- reviewer: array of ContactDetail objects
- endorser: array of ContactDetail objects
- useContext: array of UsageContext objects
- purpose: markdown explaining purpose
- copyright: markdown with copyright info
- copyrightLabel: string with copyright holder
- relatesTo: array of BackboneElement with type and target
- description: markdown description
- assertion: markdown declarative description
- note: array of Annotation objects
- variableDefinition: array of BackboneElement for each checklist item
- synthesisType: array of CodeableConcept
- studyDesign: array of CodeableConcept
- statistic: array of BackboneElement with comprehensive statistics
- certainty: array of BackboneElement with certainty assessments

SPECIFIC REQUIREMENTS:
1. variableDefinition: Create one entry for each checklist item with:
   - description, note, variableRole ("outcome"), roleSubtype, comparatorCategory
   - observed and intended references, directnessMatch

2. statistic: Include multiple statistics with:
   - description, note, statisticType, category, quantity
   - numberOfEvents, numberAffected, sampleSize (with all sub-fields)
   - attributeEstimate (with description, note, type, quantity, level, range)
   - modelCharacteristic (with code, value, intended, applied, variable, attribute)

3. certainty: Include comprehensive certainty assessment with:
   - description, note, type, rating, rater, subcomponent

4. Use proper FHIR terminology URLs:
   - http://terminology.hl7.org/CodeSystem/publication-status
   - http://terminology.hl7.org/CodeSystem/evidence-variable-role
   - http://terminology.hl7.org/CodeSystem/statistic-type
   - http://terminology.hl7.org/CodeSystem/certainty-type
   - http://terminology.hl7.org/CodeSystem/certainty-rating

5. Include evidence details if enabled: {include_evidence}
6. Include reasoning/confidence intervals if enabled: {include_reasoning}

Generate a COMPLETE FHIR Evidence resource with ALL these fields populated appropriately. Do not omit any of the mandatory fields listed above. Return ONLY the valid JSON resource.
"""
        
        # Call LLM
        if client_type == 'openai':
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000
            )
            result = response.choices[0].message.content
        else:  # anthropic
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text
        
        # Parse and validate JSON
        try:
            fhir_evidence = json.loads(result)
            
            # Ensure required fields are present
            if 'id' not in fhir_evidence:
                fhir_evidence['id'] = str(uuid.uuid4())
            
            if 'meta' not in fhir_evidence:
                fhir_evidence['meta'] = {
                    "lastUpdated": datetime.datetime.now().isoformat(),
                    "profile": ["http://hl7.org/fhir/StructureDefinition/Evidence"]
                }
            
            # Ensure proper resource type
            fhir_evidence['resourceType'] = 'Evidence'
            
            return fhir_evidence
            
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'(\{.*\})', result, re.DOTALL)
            if json_match:
                try:
                    fhir_evidence = json.loads(json_match.group(1))
                    # Ensure required fields
                    if 'id' not in fhir_evidence:
                        fhir_evidence['id'] = str(uuid.uuid4())
                    if 'resourceType' not in fhir_evidence:
                        fhir_evidence['resourceType'] = 'Evidence'
                    return fhir_evidence
                except json.JSONDecodeError:
                    pass
            
            # Fallback: create basic structure if LLM fails
            return {
                "resourceType": "Evidence",
                "id": str(uuid.uuid4()),
                "meta": {
                    "lastUpdated": datetime.datetime.now().isoformat(),
                    "profile": ["http://hl7.org/fhir/StructureDefinition/Evidence"]
                },
                "status": "active",
                "title": f"Reporting Guideline Compliance Evidence for {paper_name}",
                "description": f"Evidence of {checklist_type} reporting guideline compliance assessment",
                "note": [{
                    "text": "LLM generation failed, using fallback structure"
                }]
            }
            
    except Exception as e:
        st.error(f"Error generating FHIR Evidence: {str(e)}")
        return None


def generate_bulk_fhir_evidence(results: List[str], model: str, api_keys: Dict[str, str],
                               include_evidence: bool = True, include_reasoning: bool = True,
                               get_result_files_func=None, get_result_file_path_func=None, 
                               read_json_file_func=None) -> List[Dict[str, Any]]:
    """Generate FHIR Evidence resources for multiple results."""
    bulk_evidence = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, result in enumerate(results):
        status_text.text(f"Processing {i+1}/{len(results)}: {result}")
        progress_bar.progress(i / len(results))
        
        # Load report data for this result
        result_files = get_result_files_func(result)
        if result_files['report']:
            report_file = result_files['report'][0]
            report_path = get_result_file_path_func(result, report_file)
            report_data = read_json_file_func(report_path)
            
            if report_data:
                fhir_evidence = generate_fhir_evidence(
                    report_data, model, api_keys, include_evidence, include_reasoning
                )
                
                if fhir_evidence:
                    # Add result identifier
                    fhir_evidence['identifier'] = [{
                        "system": "http://example.org/validation-results",
                        "value": result
                    }]
                    bulk_evidence.append(fhir_evidence)
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed processing {len(bulk_evidence)} results")
    
    return bulk_evidence


def generate_enhanced_fhir_evidence(report_data: Dict[str, Any], model: str, api_keys: Dict[str, str], 
                                   include_evidence: bool = True, include_reasoning: bool = True,
                                   fhir_base_url: str = "https://build.fhir.org",
                                   evidence_profile: str = None,
                                   org_identifier: str = "http://example.org/validation-evidence") -> Optional[Dict[str, Any]]:
    """Generate FHIR Evidence with custom configuration."""
    # Use the existing function but enhance with custom URLs
    fhir_evidence = generate_fhir_evidence(report_data, model, api_keys, include_evidence, include_reasoning)
    
    if fhir_evidence:
        # Update with custom configuration
        if evidence_profile:
            if 'meta' not in fhir_evidence:
                fhir_evidence['meta'] = {}
            fhir_evidence['meta']['profile'] = [evidence_profile]
        
        if 'identifier' not in fhir_evidence:
            fhir_evidence['identifier'] = []
        
        # Add organization identifier
        fhir_evidence['identifier'].append({
            "system": org_identifier,
            "value": f"{report_data.get('paper', 'unknown')}_{datetime.datetime.now().strftime('%Y%m%d')}"
        })
    
    return fhir_evidence


def extract_pdf_content(pdf_path: str) -> str:
    """Extract text content from PDF file."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            return text_content
    except ImportError:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text_content = ""
            for page in doc:
                text_content += page.get_text() + "\n"
            doc.close()
            return text_content
        except ImportError:
            return f"Error: PDF extraction libraries not available. Please install PyPDF2 or PyMuPDF."
    except Exception as e:
        return f"Error extracting PDF content: {str(e)}"


def generate_fhir_evidence_from_pdf(pdf_path: str, model: str, api_keys: Dict[str, str], 
                                   include_evidence: bool = True, include_reasoning: bool = True,
                                   fhir_base_url: str = "https://build.fhir.org",
                                   evidence_profile: str = None,
                                   org_identifier: str = "http://example.org/validation-evidence") -> Optional[Dict[str, Any]]:
    """Generate FHIR Evidence resource from PDF content."""
    from pathlib import Path
    
    # Extract PDF content
    pdf_content = extract_pdf_content(pdf_path)
    paper_name = Path(pdf_path).stem
    
    # Create enhanced report data with PDF content
    pdf_report_data = {
        "paper": paper_name,
        "pdf_content": pdf_content,
        "pdf_path": pdf_path
    }
    
    return generate_enhanced_fhir_evidence(
        pdf_report_data, model, api_keys, include_evidence, include_reasoning,
        fhir_base_url, evidence_profile, org_identifier
    )


def generate_enhanced_bulk_fhir_evidence(papers: List[str], model: str, api_keys: Dict[str, str], 
                                        include_evidence: bool = True, include_reasoning: bool = True,
                                        fhir_base_url: str = "https://build.fhir.org",
                                        evidence_profile: str = None,
                                        org_identifier: str = "http://example.org/validation-evidence") -> List[Dict[str, Any]]:
    """Generate FHIR Evidence with enhanced configuration for multiple papers using PDF content."""
    from pathlib import Path
    import os
    
    bulk_evidence = []
    
    # Import streamlit only if available
    try:
        import streamlit as st
        progress_bar = st.progress(0)
        status_text = st.empty()
        has_streamlit = True
    except ImportError:
        has_streamlit = False
    
    # Base papers directory
    papers_dir = "/Users/chenyuli/Desktop/Macbookpro/LLMEvaluation/RWE_LLM_validator/data/Papers"
    
    for i, paper_file in enumerate(papers):
        if has_streamlit:
            status_text.text(f"Processing {i+1}/{len(papers)}: {paper_file}")
            progress_bar.progress(i / len(papers))
        
        # Construct full path to PDF
        pdf_path = os.path.join(papers_dir, paper_file)
        
        if os.path.exists(pdf_path):
            # Extract PDF content
            pdf_content = extract_pdf_content(pdf_path)
            paper_id = Path(paper_file).stem
            
            # Create report data with PDF content
            pdf_report_data = {
                "paper": paper_id,
                "pdf_content": pdf_content,
                "pdf_path": pdf_path,
                "checklist": "RECORD"  # Default checklist type
            }
            
            fhir_evidence = generate_enhanced_fhir_evidence(
                pdf_report_data, model, api_keys, include_evidence, include_reasoning,
                fhir_base_url, evidence_profile, org_identifier
            )
            
            if fhir_evidence:
                bulk_evidence.append(fhir_evidence)
        else:
            if has_streamlit:
                st.warning(f"PDF file not found: {pdf_path}")
    
    if has_streamlit:
        progress_bar.progress(1.0)
        status_text.text(f"✅ Completed processing {len(bulk_evidence)} papers")
    
    return bulk_evidence


def create_fhir_bundle(evidence_resources: List[Dict[str, Any]], org_identifier: str) -> Dict[str, Any]:
    """Create a FHIR Bundle containing all Evidence resources."""
    return {
        "resourceType": "Bundle",
        "id": str(datetime.datetime.now().timestamp()).replace('.', ''),
        "meta": {
            "lastUpdated": datetime.datetime.now().isoformat(),
            "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"]
        },
        "identifier": {
            "system": org_identifier,
            "value": f"evidence_bundle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        },
        "type": "collection",
        "timestamp": datetime.datetime.now().isoformat(),
        "total": len(evidence_resources),
        "entry": [
            {
                "resource": resource,
                "fullUrl": f"{org_identifier}/Evidence/{resource.get('id', i)}"
            }
            for i, resource in enumerate(evidence_resources)
        ]
    }


def create_generation_summary(evidence_resources: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary report of the generation process."""
    return {
        "generation_summary": {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_resources": len(evidence_resources),
            "model_used": params['model'],
            "configuration": {
                "fhir_base_url": params['fhir_base_url'],
                "evidence_profile": params['evidence_profile'],
                "organization_identifier": params['org_identifier'],
                "include_evidence_details": params['include_evidence_details'],
                "include_reasoning": params['include_reasoning'],
                "include_statistics": params['include_statistics']
            },
            "papers_processed": params['papers']
        },
        "resource_statistics": {
            "total_size_bytes": len(json.dumps(evidence_resources)),
            "average_items_per_resource": sum(len(r.get('statistic', [])) for r in evidence_resources) / len(evidence_resources) if evidence_resources else 0,
            "resource_ids": [r.get('id', f'resource_{i}') for i, r in enumerate(evidence_resources)]
        }
    }


def create_fhir_evidence_template(paper_name: str, checklist_type: str, 
                                 validation_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Create a basic FHIR Evidence template."""
    return {
        "resourceType": "Evidence",
        "id": str(uuid.uuid4()),
        "meta": {
            "lastUpdated": datetime.datetime.now().isoformat(),
            "profile": ["http://hl7.org/fhir/StructureDefinition/Evidence"]
        },
        "status": "active",
        "title": f"Reporting Guideline Compliance Evidence for {paper_name}",
        "description": f"Evidence of {checklist_type} reporting guideline compliance assessment",
        "identifier": [{
            "system": "http://example.org/validation-evidence",
            "value": f"{paper_name}_{checklist_type}_{datetime.datetime.now().strftime('%Y%m%d')}"
        }],
        "studyType": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/study-type",
                "code": "observational",
                "display": "Observational Study"
            }]
        },
        "synthesisType": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/synthesis-type",
                "code": "classification",
                "display": "Classification"
            }]
        },
        "statistic": [{
            "description": f"Overall compliance rate for {checklist_type}",
            "statisticType": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/statistic-type",
                    "code": "proportion",
                    "display": "Proportion"
                }]
            },
            "quantity": {
                "value": validation_summary.get('agreement_rate', 0),
                "unit": "percent",
                "system": "http://unitsofmeasure.org",
                "code": "%"
            }
        }],
        "certainty": [{
            "description": "Certainty of evidence assessment",
            "rating": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/certainty-rating",
                    "code": "moderate",
                    "display": "Moderate quality"
                }]
            }
        }]
    }
