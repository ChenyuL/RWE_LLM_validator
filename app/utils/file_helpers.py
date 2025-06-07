import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKLISTS_PATH = PROJECT_ROOT / "data" / "Guidelines"
PAPERS_PATH = PROJECT_ROOT / "data" / "Papers"
OUTPUT_PATH = PROJECT_ROOT / "output"
RESULTS_PATH = OUTPUT_PATH / "paper_results"
PROMPTS_PATH = OUTPUT_PATH / "prompts"
CUSTOM_PROMPTS_PATH = Path(__file__).parent.parent / "data" / "custom_prompts"

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        CHECKLISTS_PATH,
        PAPERS_PATH,
        OUTPUT_PATH,
        RESULTS_PATH,
        PROMPTS_PATH,
        CUSTOM_PROMPTS_PATH
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_checklist_folders() -> List[str]:
    """Get list of available checklist folders."""
    ensure_directories()
    
    if not CHECKLISTS_PATH.exists():
        return []
    
    folders = []
    for item in CHECKLISTS_PATH.iterdir():
        if item.is_dir():
            folders.append(item.name)
    
    return sorted(folders)

def get_papers() -> List[str]:
    """Get list of available paper files."""
    ensure_directories()
    
    if not PAPERS_PATH.exists():
        return []
    
    papers = []
    for item in PAPERS_PATH.iterdir():
        if item.is_file() and item.suffix.lower() == '.pdf':
            papers.append(item.name)
    
    return sorted(papers)

def get_results() -> List[str]:
    """Get list of result directories."""
    ensure_directories()
    
    if not RESULTS_PATH.exists():
        return []
    
    results = []
    for item in RESULTS_PATH.iterdir():
        if item.is_dir():
            results.append(item.name)
    
    return sorted(results, reverse=True)  # Most recent first

def get_prompts_files() -> List[str]:
    """Get list of available prompts files."""
    ensure_directories()
    
    prompts_files = []
    
    # Check main output directory
    if OUTPUT_PATH.exists():
        for item in OUTPUT_PATH.iterdir():
            if item.is_file() and item.name.endswith('_prompts.json'):
                prompts_files.append(str(item))
    
    # Check prompts directory
    if PROMPTS_PATH.exists():
        for item in PROMPTS_PATH.iterdir():
            if item.is_file() and item.name.endswith('_prompts.json'):
                prompts_files.append(str(item))
    
    return sorted(prompts_files, reverse=True)  # Most recent first

def get_custom_prompts() -> Dict[str, Any]:
    """Get list of custom prompt sets."""
    ensure_directories()
    
    custom_prompts = {}
    
    if CUSTOM_PROMPTS_PATH.exists():
        for item in CUSTOM_PROMPTS_PATH.iterdir():
            if item.is_file() and item.suffix == '.json':
                try:
                    with open(item, 'r') as f:
                        data = json.load(f)
                    custom_prompts[item.stem] = data
                except Exception:
                    continue
    
    return custom_prompts

def save_custom_prompts(name: str, prompts: Dict[str, Any]) -> bool:
    """Save custom prompts to file."""
    ensure_directories()
    
    try:
        file_path = CUSTOM_PROMPTS_PATH / f"{name}.json"
        with open(file_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        return True
    except Exception:
        return False

def get_data_status() -> Dict[str, int]:
    """Get status of available data."""
    return {
        'checklists': len(get_checklist_folders()),
        'papers': len(get_papers()),
        'results': len(get_results())
    }

def get_checklist_files(checklist_type: str) -> List[str]:
    """Get files in a specific checklist folder."""
    checklist_path = CHECKLISTS_PATH / checklist_type
    
    if not checklist_path.exists():
        return []
    
    files = []
    for item in checklist_path.iterdir():
        if item.is_file() and item.suffix.lower() == '.pdf':
            files.append(item.name)
    
    return sorted(files)

def get_result_files(result_name: str) -> Dict[str, List[str]]:
    """Get files in a specific result directory, categorized by type."""
    result_path = RESULTS_PATH / result_name
    
    if not result_path.exists():
        return {}
    
    files = {
        'reasoner': [],
        'extractor': [],
        'validator': [],
        'report': [],
        'checklist': [],
        'other': []
    }
    
    for item in result_path.iterdir():
        if item.is_file():
            filename = item.name.lower()
            
            if 'reasoner' in filename:
                files['reasoner'].append(item.name)
            elif 'extractor' in filename:
                files['extractor'].append(item.name)
            elif 'validator' in filename:
                files['validator'].append(item.name)
            elif 'report' in filename:
                files['report'].append(item.name)
            elif 'checklist' in filename or 'full_' in filename:
                files['checklist'].append(item.name)
            else:
                files['other'].append(item.name)
    
    # Sort files by timestamp (newest first)
    for file_list in files.values():
        file_list.sort(reverse=True)
    
    return files

def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Read and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def display_pdf_base64(file_path: str) -> str:
    """Convert PDF to base64 for display in Streamlit."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    except Exception:
        return '<p>Error loading PDF file.</p>'

def get_paper_path(paper_name: str) -> str:
    """Get full path to a paper file."""
    return str(PAPERS_PATH / paper_name)

def get_checklist_path(checklist_type: str, file_name: str = None) -> str:
    """Get full path to a checklist file or folder."""
    if file_name:
        return str(CHECKLISTS_PATH / checklist_type / file_name)
    else:
        return str(CHECKLISTS_PATH / checklist_type)

def get_result_file_path(result_name: str, file_name: str) -> str:
    """Get full path to a result file."""
    return str(RESULTS_PATH / result_name / file_name)

def save_uploaded_file(uploaded_file, destination_path: str) -> bool:
    """Save an uploaded file to the specified destination."""
    try:
        with open(destination_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception:
        return False

def create_checklist_folder(folder_name: str) -> bool:
    """Create a new checklist folder."""
    try:
        folder_path = CHECKLISTS_PATH / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

def get_default_prompts() -> Dict[str, str]:
    """Get default prompts extracted from current implementations."""
    # This will be populated with prompts extracted from the existing agents
    # For now, return a basic structure
    return {
        "extractor_default": """You are an expert extractor for biomedical research reporting checklists.

Your task is to examine the research paper and extract specific information related to the given checklist item.

Please provide:
1. Whether the paper complies with this guideline item (yes, partial, or no)
2. Evidence from the paper (direct quotes with locations)
3. Reasoning for your assessment
4. The correct answer addressing the specific checklist item question

If the information is missing, clearly state that it is not reported in the paper.""",
        
        "validator_default": """You are an expert validator for biomedical research reporting checklists.

Your task is to validate the extraction results against the guideline item.

Please evaluate:
1. Whether the compliance assessment is correct based on the evidence provided
2. Whether the evidence is sufficient and relevant to the checklist item
3. Provide a final assessment of whether you agree with the extractor's assessment
4. Provide a correct answer that will be used in the final checklist

Respond with your validation assessment and reasoning."""
    }

def filter_prompts_by_checklist(prompts_files: List[str], checklist_type: str) -> List[str]:
    """Filter prompts files by checklist type."""
    filtered = []
    for file_path in prompts_files:
        file_name = Path(file_path).name
        if checklist_type.lower() in file_name.lower():
            filtered.append(file_path)
    
    return filtered if filtered else prompts_files

def get_latest_result_for_paper(paper_name: str) -> Optional[str]:
    """Get the latest result directory for a specific paper."""
    paper_id = Path(paper_name).stem
    if '.' in paper_id:
        paper_id = paper_id.split('.')[0]
    
    results = get_results()
    for result in results:
        if paper_id in result:
            return result
    
    return None

# Initialize directories when module is imported
ensure_directories()
