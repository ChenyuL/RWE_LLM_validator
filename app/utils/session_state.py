import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

def initialize_session_state():
    """Initialize session state variables."""
    
    # API Keys
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'openai': '',
            'anthropic': '',
            'voyage': '',
            'deepseek': ''
        }
    
    # Agent Configuration
    if 'agent_config' not in st.session_state:
        st.session_state.agent_config = {
            'reasoner': {
                'type': 'base',  # 'base' or 'rag'
                'model': 'o3-mini-2025-01-31'
            },
            'extractor': {
                'type': 'base',  # 'base' or 'rag'
                'model': 'gpt-4o',
                'custom_prompt': ''
            },
            'validator': {
                'type': 'base',  # 'base' or 'rag'
                'model': 'claude-3-5-sonnet-20241022',
                'custom_prompt': ''
            }
        }
    
    # RAG Configuration
    if 'rag_config' not in st.session_state:
        st.session_state.rag_config = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'top_k': 5,
            'similarity_threshold': 0.7,
            'embedding_model': 'voyage-3'
        }
    
    # Pipeline Configuration
    if 'pipeline_config' not in st.session_state:
        st.session_state.pipeline_config = {
            'mode': 'full',  # 'full', 'reasoner', 'extractor'
            'batch_size': 5,
            'checklist_type': 'RECORD'
        }
    
    # File paths
    if 'file_paths' not in st.session_state:
        st.session_state.file_paths = {
            'selected_paper': None,
            'selected_checklist': None,
            'prompts_file': None
        }
    
    # Results
    if 'last_validation_result' not in st.session_state:
        st.session_state.last_validation_result = None

def load_api_keys_from_env():
    """Load API keys from .env file."""
    load_dotenv()
    
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY', ''),
        'anthropic': os.getenv('ANTHROPIC_API_KEY', ''),
        'voyage': os.getenv('VOYAGE_API_KEY', ''),
        'deepseek': os.getenv('DEEPSEEK_API_KEY', '')
    }
    
    # Ensure session state is initialized
    if 'api_keys' not in st.session_state:
        initialize_session_state()
    
    # Update session state
    st.session_state.api_keys.update(api_keys)
    
    return api_keys

def save_api_keys_to_env(api_keys):
    """Save API keys to .env file."""
    env_path = Path('.env')
    
    # Read existing .env content
    existing_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    existing_vars[key] = value
    
    # Update with new API keys
    for key, value in api_keys.items():
        if value:  # Only save non-empty keys
            existing_vars[f'{key.upper()}_API_KEY'] = value
    
    # Write back to .env file
    with open(env_path, 'w') as f:
        for key, value in existing_vars.items():
            f.write(f'{key}={value}\n')
    
    # Reload environment variables
    load_dotenv(override=True)
    
    # Update session state
    st.session_state.api_keys.update(api_keys)

def check_api_keys_status():
    """Check which API keys are available."""
    load_api_keys_from_env()
    
    status = {}
    for provider, key in st.session_state.api_keys.items():
        status[provider] = bool(key and len(key) > 10)
    
    return status

def get_available_models():
    """Get available models for each agent type."""
    return {
        'reasoner': {
            'openai': ['o3-mini-2025-01-31', 'o1-2024-12-17', 'o1-mini-2024-09-12', 'gpt-4o', 'gpt-4o-mini'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
            'deepseek': ['deepseek-reasoner', 'deepseek-chat']
        },
        'extractor': {
            'openai': ['gpt-4o', 'gpt-4o-mini-2024-07-18', 'gpt-4.5-preview-2025-02-27', 'gpt-4-turbo'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
            'deepseek': ['deepseek-chat', 'deepseek-coder']
        },
        'validator': {
            'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
            'deepseek': ['deepseek-chat', 'deepseek-coder']
        }
    }

def get_model_provider(model_name):
    """Determine the provider for a given model name."""
    if any(x in model_name.lower() for x in ['gpt', 'o1', 'o3']):
        return 'openai'
    elif 'claude' in model_name.lower():
        return 'anthropic'
    elif 'deepseek' in model_name.lower():
        return 'deepseek'
    else:
        return 'openai'  # Default fallback

def validate_configuration():
    """Validate the current configuration."""
    errors = []
    warnings = []
    
    # Check API keys
    api_status = check_api_keys_status()
    
    # Check if required API keys are available based on selected models
    reasoner_provider = get_model_provider(st.session_state.agent_config['reasoner']['model'])
    extractor_provider = get_model_provider(st.session_state.agent_config['extractor']['model'])
    validator_provider = get_model_provider(st.session_state.agent_config['validator']['model'])
    
    required_providers = {reasoner_provider, extractor_provider, validator_provider}
    
    for provider in required_providers:
        if not api_status.get(provider, False):
            errors.append(f"Missing {provider.title()} API key for selected models")
    
    # Check RAG configuration
    if (st.session_state.agent_config['reasoner']['type'] == 'rag' or 
        st.session_state.agent_config['extractor']['type'] == 'rag' or 
        st.session_state.agent_config['validator']['type'] == 'rag'):
        
        if not api_status.get('voyage', False):
            warnings.append("Voyage AI API key recommended for RAG functionality")
    
    # Check file selections
    if st.session_state.pipeline_config['mode'] == 'extractor':
        if not st.session_state.file_paths['prompts_file']:
            errors.append("Prompts file required for extractor mode")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

# Initialize session state when module is imported
initialize_session_state()
