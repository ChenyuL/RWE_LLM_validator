#!/usr/bin/env python
# rag_extractor_validator_improved.py
# Improved RAG-based extractor and validator for Li-Paper SOP
# - Processes items one by one
# - Uses Voyage AI for embeddings
# - Accepts prompt file as a parameter

import os
import json
import logging
import sys
import datetime
import time
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import voyageai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_extractor_validator_improved.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_extractor_validator_improved")

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Constants
GUIDELINES_PATH = "data/Guidelines"
PAPERS_PATH = "data/Papers"
OUTPUT_PATH = "output"
EMBEDDINGS_PATH = "output/embeddings"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 5

# Create necessary directories
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Get API keys directly from .env file
def get_api_keys_from_env():
    """
    Get API keys directly from the .env file instead of using environment variables.
    """
    api_keys = {}
    
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
        
        for line in env_content.split('\n'):
            if line.startswith('OPENAI_API_KEY='):
                api_keys["openai"] = line.split('=', 1)[1].strip()
            elif line.startswith('ANTHROPIC_API_KEY='):
                api_keys["anthropic"] = line.split('=', 1)[1].strip()
            elif line.startswith('DEEPSEEK_API_KEY='):
                api_keys["deepseek"] = line.split('=', 1)[1].strip()
            elif line.startswith('VOYAGE_API_KEY='):
                api_keys["voyage"] = line.split('=', 1)[1].strip()
        
        logger.info(f"API keys loaded directly from .env file")
        logger.info(f"OpenAI API Key (first 10 chars): {api_keys.get('openai', '')[:10] if 'openai' in api_keys else 'Not found'}...")
        logger.info(f"Anthropic API Key (first 10 chars): {api_keys.get('anthropic', '')[:10] if 'anthropic' in api_keys else 'Not found'}...")
        logger.info(f"Voyage API Key (first 10 chars): {api_keys.get('voyage', '')[:10] if 'voyage' in api_keys else 'Not found'}...")
        
        return api_keys
    except Exception as e:
        logger.error(f"Error loading API keys from .env file: {e}")
        raise

# Get API keys
API_KEYS = get_api_keys_from_env()

class PDFProcessor:
    """
    Process PDF files to extract text and create chunks.
    """
    def __init__(self):
        logger.info("Initializing PDF processor")
        
    def extract_text(self, pdf_path):
        """
        Extract text from a PDF file.
        """
        try:
            import PyPDF2
            logger.info(f"Extracting text from {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                    
                # Add page numbers to the text
                text += f"\nTotal pages: {len(reader.pages)}\n"
                
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def create_chunks(self, text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """
        Create overlapping chunks from text.
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

class VoyageEmbeddingGenerator:
    """
    Generate embeddings for text chunks using Voyage AI's API.
    """
    def __init__(self, api_key, model="voyage-3"):
        self.api_key = api_key
        self.model = model
        logger.info(f"Initializing Voyage embedding generator with model: {model}")
        
        self.client = voyageai.Client(api_key=self.api_key)
    
    def generate_embeddings(self, chunks):
        """
        Generate embeddings for a list of text chunks.
        """
        if not chunks:
            return []
            
        embeddings = []
        for i in range(0, len(chunks), 10):  # Process in batches of 10
            batch_chunks = chunks[i:i+10]
            try:
                result = self.client.embed(
                    batch_chunks, 
                    model=self.model, 
                    input_type="document"
                )
                batch_embeddings = result.embeddings
                embeddings.extend(batch_embeddings)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//10}: {e}")
                # Add placeholder embeddings for the failed batch
                for _ in range(len(batch_chunks)):
                    embeddings.append([0] * 1024)  # Voyage embeddings are 1024-dimensional
                
        return embeddings

class RAGExtractor:
    """
    RAG-based extractor that uses embeddings to find relevant chunks for each prompt.
    """
    def __init__(self, api_key, model="gpt-4o", provider="openai"):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        logger.info(f"Initializing RAG extractor with model: {model} from provider: {provider}")
        
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
    def extract_information(self, paper_text, prompt, item_id, paper_id, embeddings, chunks):
        """
        Extract information from paper text based on a prompt using RAG.
        """
        logger.info(f"Extracting information for paper {paper_id}, checklist item: {item_id}")
        
        # Generate embedding for the prompt using Voyage AI
        voyage_client = voyageai.Client(api_key=API_KEYS["voyage"])
        
        # Generate embedding for the prompt (not the paper text)
        result = voyage_client.embed(
            prompt,
            model="voyage-3",
            input_type="document"
        )
        prompt_embedding = result.embeddings[0]
        
        # Calculate similarity between prompt and chunks
        similarities = []
        for embedding in embeddings:
            similarity = self.cosine_similarity(prompt_embedding, embedding)
            similarities.append(similarity)
        
        # Get top-k chunks
        if not similarities:
            relevant_text = paper_text[:8000]  # Fallback to first 8000 chars
        else:
            top_indices = np.argsort(similarities)[-TOP_K_CHUNKS:]
            relevant_chunks = [chunks[i] for i in top_indices]
            relevant_text = "\n\n".join(relevant_chunks)
        
        # Truncate if too long
        if len(relevant_text) > 12000:
            relevant_text = relevant_text[:12000]
        
        # Create a prompt for OpenAI
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
        
        # Call LLM API based on provider
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.2,
                    max_tokens=2000
                )
                result = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.2,
                    messages=[{"role": "user", "content": extraction_prompt}]
                )
                result = response.content[0].text
            
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
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            
            return {
                "paper_identifier": paper_id,
                "Li-Paper_item_id": item_id,
                "extracted_content": {
                    "compliance": "unknown",
                    "evidence": [],
                    "reasoning": f"Error: {str(e)}",
                    "correct_answer": "unknown"
                }
            }
    
    def cosine_similarity(self, a, b):
        """
        Calculate cosine similarity between two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class RAGValidator:
    """
    RAG-based validator that uses embeddings to find relevant chunks for each prompt.
    """
    def __init__(self, api_key, model="claude-3-5-sonnet-20241022", provider="anthropic"):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        logger.info(f"Initializing RAG validator with model: {model} from provider: {provider}")
        
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
    def validate(self, extraction, guideline_item, item_id, paper_id, embeddings, chunks):
        """
        Validate an extraction against a guideline item using RAG.
        """
        logger.info(f"Validating extraction for paper {paper_id}, checklist item: {item_id}")
        
        # Generate embedding for the guideline item using Voyage AI
        voyage_client = voyageai.Client(api_key=API_KEYS["voyage"])
        
        guideline_text = f"CHECKLIST ITEM: {item_id}\nDESCRIPTION: {guideline_item.get('description', '')}"
        
        result = voyage_client.embed(
            guideline_text,
            model="voyage-3",
            input_type="document"
        )
        guideline_embedding = result.embeddings[0]
        
        # Calculate similarity between guideline and chunks
        similarities = []
        for embedding in embeddings:
            similarity = self.cosine_similarity(guideline_embedding, embedding)
            similarities.append(similarity)
        
        # Get top-k chunks
        if not similarities:
            relevant_text = "No relevant text found in the paper."
        else:
            top_indices = np.argsort(similarities)[-TOP_K_CHUNKS:]
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
        
        # Call LLM API based on provider
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": validation_prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                result = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": validation_prompt}]
                )
                result = response.content[0].text
            
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
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            
            return {
                "paper_identifier": paper_id,
                "Li-Paper_item_id": item_id,
                "validate_result": "unknown",
                "Reason": f"Error: {str(e)}",
                "correct_answer": "unknown"
            }
    
    def cosine_similarity(self, a, b):
        """
        Calculate cosine similarity between two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_prompts_from_file(prompts_file, guideline_type="Li-Paper"):
    """
    Load prompts from a previously saved file.
    """
    logger.info(f"Loading prompts from {prompts_file}")
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # Create a guideline_info structure with the loaded prompts
    guideline_info = {
        "guideline_type": guideline_type,
        "items": [],
        "prompts": prompts
    }
    
    # Create guideline items based on the prompt IDs
    for item_id, prompt_text in prompts.items():
        # Extract description from the prompt text
        description = ""
        if isinstance(prompt_text, str):
            description_match = re.search(r'DESCRIPTION: (.*?)(\n|\r)', prompt_text)
            if description_match:
                description = description_match.group(1).strip()
            
            category_match = re.search(r'CATEGORY: (.*?)(\n|\r)', prompt_text)
            category = category_match.group(1).strip() if category_match else ""
        else:
            description = prompt_text.get("content", "")
            category = prompt_text.get("category", "")
        
        guideline_info["items"].append({
            "id": item_id,
            "description": description,
            "category": category,
            "notes": ""
        })
    
    logger.info(f"Loaded {len(prompts)} prompts")
    return guideline_info

def process_paper_with_rag_improved(paper_path, guideline_info, 
                                   extractor_model="gpt-4o", extractor_provider="openai",
                                   validator_model="claude-3-5-sonnet-20241022", validator_provider="anthropic"):
    """
    Process a paper using improved RAG-based extractor and validator.
    Process each guideline item individually instead of in batches.
    
    Args:
        paper_path: Path to the paper PDF
        guideline_info: Guideline information
        extractor_model: Model to use for extraction
        extractor_provider: Provider to use for extraction (openai or anthropic)
        validator_model: Model to use for validation
        validator_provider: Provider to use for validation (openai or anthropic)
    """
    paper_id = os.path.splitext(os.path.basename(paper_path))[0]
    logger.info(f"Processing paper: {paper_id}")
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Extract text from paper
    paper_text = pdf_processor.extract_text(paper_path)
    if not paper_text:
        logger.error(f"Failed to extract text from {paper_path}")
        return None
    
    # Create chunks
    chunks = pdf_processor.create_chunks(paper_text)
    logger.info(f"Created {len(chunks)} chunks from paper")
    
    # Check if embeddings already exist
    embeddings_file = os.path.join(EMBEDDINGS_PATH, f"{paper_id}_voyage_embeddings.json")
    if os.path.exists(embeddings_file):
        logger.info(f"Loading existing Voyage embeddings from {embeddings_file}")
        with open(embeddings_file, 'r') as f:
            embeddings = json.load(f)
    else:
        # Generate embeddings using Voyage AI
        if "voyage" not in API_KEYS or not API_KEYS["voyage"]:
            logger.error("Voyage API key not found. Please add VOYAGE_API_KEY to .env file.")
            return None
            
        embedding_generator = VoyageEmbeddingGenerator(API_KEYS["voyage"])
        embeddings = embedding_generator.generate_embeddings(chunks)
        
        # Save embeddings
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings, f)
        logger.info(f"Saved Voyage embeddings to {embeddings_file}")
    
    # Initialize RAG extractor and validator with specified models and providers
    if extractor_provider == "openai":
        rag_extractor = RAGExtractor(API_KEYS["openai"], extractor_model, extractor_provider)
    elif extractor_provider == "anthropic":
        rag_extractor = RAGExtractor(API_KEYS["anthropic"], extractor_model, extractor_provider)
    else:
        logger.error(f"Unsupported extractor provider: {extractor_provider}")
        return None
    
    if validator_provider == "openai":
        rag_validator = RAGValidator(API_KEYS["openai"], validator_model, validator_provider)
    elif validator_provider == "anthropic":
        # Ensure Claude model name is lowercase
        if validator_model.startswith("Claude-"):
            validator_model = validator_model.replace("Claude-", "claude-")
        rag_validator = RAGValidator(API_KEYS["anthropic"], validator_model, validator_provider)
    else:
        logger.error(f"Unsupported validator provider: {validator_provider}")
        return None
    
    # Get all item IDs
    item_ids = list(guideline_info["prompts"].keys())
    
    # Process each item individually
    extracted_info = {}
    validation_results = {}
    
    for item_id in tqdm(item_ids, desc="Processing items"):
        prompt = guideline_info["prompts"][item_id]
        
        # Extract information
        extraction = rag_extractor.extract_information(
            paper_text, prompt, item_id, paper_id, embeddings, chunks
        )
        extracted_info[item_id] = extraction
        
        # Get corresponding guideline item
        guideline_item = next((item for item in guideline_info["items"] if item["id"] == item_id), None)
        if guideline_item is None:
            logger.warning(f"No guideline item found for ID: {item_id}")
            continue
        
        # Validate extraction
        validation = rag_validator.validate(
            extraction, guideline_item, item_id, paper_id, embeddings, chunks
        )
        validation_results[item_id] = validation
        
        # Save individual item results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        item_extraction_filename = f"{timestamp}_rag_item_{item_id}_extraction_{paper_id}_{guideline_info['guideline_type']}.json"
        with open(os.path.join(OUTPUT_PATH, item_extraction_filename), "w") as f:
            json.dump(extraction, f, indent=2)
        
        item_validation_filename = f"{timestamp}_rag_item_{item_id}_validation_{paper_id}_{guideline_info['guideline_type']}.json"
        with open(os.path.join(OUTPUT_PATH, item_validation_filename), "w") as f:
            json.dump(validation, f, indent=2)
        
        logger.info(f"Processed item {item_id} for paper {paper_id}")
    
    # Generate final report
    report = {
        "paper": os.path.basename(paper_path),
        "checklist": guideline_info["guideline_type"],
        "validation_summary": calculate_metrics(validation_results),
        "model_info": {
            "extractor": f"rag-{extractor_provider}-{rag_extractor.model}",
            "validator": f"rag-{validator_provider}-{rag_validator.model}",
            "embeddings": "voyage-3"
        },
        "items": {}
    }
    
    # Compile detailed item-by-item results
    for item_id, validation in validation_results.items():
        guideline_item = next((item for item in guideline_info["items"] if item["id"] == item_id), None)
        extraction = extracted_info.get(item_id, {})
        extracted_content = extraction.get("extracted_content", {})
        
        # Get information from the extraction
        evidence = extracted_content.get("evidence", [])
        extractor_reasoning = extracted_content.get("reasoning", "")
        extractor_correct_answer = extracted_content.get("correct_answer", "")
        
        # Get information from the validation
        validator_correct_answer = validation.get("correct_answer", "")
        validator_reasoning = validation.get("Reason", "")
        
        # Use the validator's correct answer if available, otherwise use the extractor's
        final_correct_answer = validator_correct_answer if validator_correct_answer else extractor_correct_answer
        
        report["items"][item_id] = {
            "description": guideline_item["description"] if guideline_item else "Unknown",
            "compliance": validation.get("validate_result", "unknown"),
            "evidence": evidence,
            "extractor_reasoning": extractor_reasoning,
            "extractor_correct_answer": extractor_correct_answer,
            "validator_reasoning": validator_reasoning,
            "validator_correct_answer": validator_correct_answer,
            "final_correct_answer": final_correct_answer,
            "disagreements": []
        }
    
    # Save final report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{timestamp}_rag_improved_report_{paper_id}_{guideline_info['guideline_type']}.json"
    with open(os.path.join(OUTPUT_PATH, report_filename), "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved final report to {report_filename}")
    
    return {
        "paper_path": paper_path,
        "extracted_info": extracted_info,
        "validation_results": validation_results,
        "report": report
    }

def calculate_metrics(validation_results):
    """
    Calculate overall metrics for the validation results.
    """
    # Count validation result categories
    counts = {
        "agree with extractor": 0, 
        "do not agree with extractor": 0, 
        "unknown": 0
    }
    
    for item_id, result in validation_results.items():
        # Get the validation result
        validate_result = result.get("validate_result", "unknown")
        counts[validate_result] = counts.get(validate_result, 0) + 1
    
    # Calculate percentages
    total_items = len(validation_results)
    percentages = {}
    
    if total_items > 0:
        for category, count in counts.items():
            percentages[f"{category}_percent"] = (count / total_items) * 100
    
    # Calculate agreement rate
    if total_items > 0:
        agreement_rate = (counts["agree with extractor"] / total_items) * 100
    else:
        agreement_rate = 0.0
    
    # Compile metrics
    metrics = {
        "total_items": total_items,
        "agree_with_extractor": counts["agree with extractor"],
        "disagree_with_extractor": counts["do not agree with extractor"],
        "unknown": counts["unknown"],
        "agreement_rate": agreement_rate,
        "items_for_review": counts["do not agree with extractor"],
        "review_percentage": (counts["do not agree with extractor"] / total_items * 100) if total_items > 0 else 0.0
    }
    
    # Add percentages
    metrics.update(percentages)
    
    return metrics

def main():
    """
    Main function to run the improved RAG-based extractor and validator.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run improved RAG-based extractor and validator for Li-Paper SOP')
    parser.add_argument('--prompts', type=str, required=True, help='Path to prompts file')
    parser.add_argument('--paper', type=str, required=True, help='Path to paper file')
    parser.add_argument('--checklist', type=str, default='Li-Paper', help='Checklist type (default: Li-Paper)')
    parser.add_argument('--extractor-model', type=str, default='gpt-4o', help='Model to use for extraction (default: gpt-4o)')
    parser.add_argument('--extractor-provider', type=str, default='openai', help='Provider to use for extraction (default: openai)')
    parser.add_argument('--validator-model', type=str, default='claude-3-5-sonnet-20241022', help='Model to use for validation (default: claude-3-5-sonnet-20241022)')
    parser.add_argument('--validator-provider', type=str, default='anthropic', help='Provider to use for validation (default: anthropic)')
    
    args = parser.parse_args()
    
    # Load prompts
    guideline_info = load_prompts_from_file(args.prompts, args.checklist)
    
    # Process paper with specified models and providers
    result = process_paper_with_rag_improved(
        args.paper, 
        guideline_info,
        args.extractor_model,
        args.extractor_provider,
        args.validator_model,
        args.validator_provider
    )
    
    if result:
        logger.info(f"Successfully processed paper: {os.path.basename(args.paper)}")
        
        # Print summary
        report = result["report"]
        metrics = report["validation_summary"]
        
        print("\n" + "="*80)
        print(f"VALIDATION RESULTS FOR {os.path.basename(args.paper)}")
        print("="*80)
        
        print("\nValidation Metrics:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.3f}")
            else:
                print(f"  {metric_name}: {metric_value}")
        
        print("\nModel Information:")
        print(f"  Extractor: {report['model_info']['extractor']}")
        print(f"  Validator: {report['model_info']['validator']}")
        print(f"  Embeddings: {report['model_info']['embeddings']}")
        
        print("\n" + "="*80)
    else:
        logger.error(f"Failed to process paper: {args.paper}")

if __name__ == "__main__":
    main()
