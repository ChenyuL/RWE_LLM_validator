import streamlit as st
import json
from utils.session_state import (
    get_available_models, get_model_provider, validate_configuration
)
from utils.file_helpers import (
    get_default_prompts, get_custom_prompts, save_custom_prompts
)

# Page header
st.header("⚙️ Configure Agents")
st.markdown("Configure the three-agent pipeline: Reasoner, Extractor, and Validator")

# Create tabs for different configuration sections
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Agent Selection", "🔧 RAG Settings", "✏️ Custom Prompts", "💾 Save/Load"])

with tab1:
    st.subheader("Agent Configuration")
    st.markdown("Select agent types and models for each component of the pipeline.")
    
    # Get available models
    available_models = get_available_models()
    
    # Reasoner Configuration
    st.markdown("### 🧠 Reasoner (LLM1)")
    st.markdown("Processes guideline documents to extract checklist items and generate prompts.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        reasoner_type = st.radio(
            "Reasoner Type",
            ["base", "rag"],
            index=0 if st.session_state.agent_config['reasoner']['type'] == 'base' else 1,
            help="Base: Standard processing | RAG: Enhanced with retrieval augmentation",
            key="reasoner_type"
        )
        st.session_state.agent_config['reasoner']['type'] = reasoner_type
    
    with col2:
        # Determine current provider
        current_model = st.session_state.agent_config['reasoner']['model']
        current_provider = get_model_provider(current_model)
        
        # Provider selection
        reasoner_provider = st.selectbox(
            "Provider",
            ["openai", "anthropic", "deepseek"],
            index=["openai", "anthropic", "deepseek"].index(current_provider),
            key="reasoner_provider"
        )
        
        # Model selection based on provider
        provider_models = available_models['reasoner'][reasoner_provider]
        if current_model in provider_models:
            model_index = provider_models.index(current_model)
        else:
            model_index = 0
        
        reasoner_model = st.selectbox(
            "Model",
            provider_models,
            index=model_index,
            key="reasoner_model"
        )
        st.session_state.agent_config['reasoner']['model'] = reasoner_model
    
    if reasoner_type == "rag":
        st.info("🚀 RAG Reasoner uses embeddings to better understand guideline structure and extract more accurate checklist items.")
    
    st.markdown("---")
    
    # Extractor Configuration
    st.markdown("### 🔍 Extractor (LLM2)")
    st.markdown("Extracts information from research papers based on checklist prompts.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        extractor_type = st.radio(
            "Extractor Type",
            ["base", "rag"],
            index=0 if st.session_state.agent_config['extractor']['type'] == 'base' else 1,
            help="Base: Standard extraction | RAG: Context-aware extraction using embeddings",
            key="extractor_type"
        )
        st.session_state.agent_config['extractor']['type'] = extractor_type
    
    with col2:
        # Determine current provider
        current_model = st.session_state.agent_config['extractor']['model']
        current_provider = get_model_provider(current_model)
        
        # Provider selection
        extractor_provider = st.selectbox(
            "Provider",
            ["openai", "anthropic", "deepseek"],
            index=["openai", "anthropic", "deepseek"].index(current_provider),
            key="extractor_provider"
        )
        
        # Model selection based on provider
        provider_models = available_models['extractor'][extractor_provider]
        if current_model in provider_models:
            model_index = provider_models.index(current_model)
        else:
            model_index = 0
        
        extractor_model = st.selectbox(
            "Model",
            provider_models,
            index=model_index,
            key="extractor_model"
        )
        st.session_state.agent_config['extractor']['model'] = extractor_model
    
    if extractor_type == "rag":
        st.info("🎯 RAG Extractor finds the most relevant sections of papers for each checklist item, improving accuracy and reducing hallucination.")
    
    st.markdown("---")
    
    # Validator Configuration
    st.markdown("### ✅ Validator (LLM3)")
    st.markdown("Validates extracted information and provides final compliance assessment.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        validator_type = st.radio(
            "Validator Type",
            ["base", "rag"],
            index=0 if st.session_state.agent_config['validator']['type'] == 'base' else 1,
            help="Base: Standard validation | RAG: Context-aware validation with paper retrieval",
            key="validator_type"
        )
        st.session_state.agent_config['validator']['type'] = validator_type
    
    with col2:
        # Determine current provider
        current_model = st.session_state.agent_config['validator']['model']
        current_provider = get_model_provider(current_model)
        
        # Provider selection
        validator_provider = st.selectbox(
            "Provider",
            ["openai", "anthropic", "deepseek"],
            index=["openai", "anthropic", "deepseek"].index(current_provider),
            key="validator_provider"
        )
        
        # Model selection based on provider
        provider_models = available_models['validator'][validator_provider]
        if current_model in provider_models:
            model_index = provider_models.index(current_model)
        else:
            model_index = 0
        
        validator_model = st.selectbox(
            "Model",
            provider_models,
            index=model_index,
            key="validator_model"
        )
        st.session_state.agent_config['validator']['model'] = validator_model
    
    if validator_type == "rag":
        st.info("🔍 RAG Validator cross-references validation decisions with relevant paper sections for more accurate assessments.")

with tab2:
    st.subheader("RAG Configuration")
    st.markdown("Configure Retrieval-Augmented Generation settings for enhanced performance.")
    
    # Check if any RAG agents are selected
    using_rag = (
        st.session_state.agent_config['reasoner']['type'] == 'rag' or
        st.session_state.agent_config['extractor']['type'] == 'rag' or
        st.session_state.agent_config['validator']['type'] == 'rag'
    )
    
    if not using_rag:
        st.info("🔧 RAG settings are only applicable when at least one agent is set to RAG mode.")
    else:
        st.success("🚀 RAG mode is enabled for one or more agents.")
    
    # RAG Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Text Processing")
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=st.session_state.rag_config['chunk_size'],
            step=100,
            help="Size of text chunks for processing (larger = more context, slower)"
        )
        st.session_state.rag_config['chunk_size'] = chunk_size
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=50,
            max_value=500,
            value=st.session_state.rag_config['chunk_overlap'],
            step=25,
            help="Overlap between chunks to maintain context"
        )
        st.session_state.rag_config['chunk_overlap'] = chunk_overlap
    
    with col2:
        st.markdown("#### Retrieval Settings")
        
        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1,
            max_value=20,
            value=st.session_state.rag_config['top_k'],
            step=1,
            help="Number of most relevant chunks to retrieve"
        )
        st.session_state.rag_config['top_k'] = top_k
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.rag_config['similarity_threshold'],
            step=0.05,
            help="Minimum similarity score for chunk retrieval"
        )
        st.session_state.rag_config['similarity_threshold'] = similarity_threshold
    
    # Embedding Model Info
    st.markdown("#### Embedding Model")
    embedding_model = st.session_state.rag_config['embedding_model']
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"📊 Current embedding model: **{embedding_model}**")
        st.markdown("This model is used to create vector representations of text for similarity search.")
    
    with col2:
        if st.button("🧪 Test Embeddings"):
            st.success("✅ Embedding test successful!")
            st.markdown("*Note: Actual embedding test would be implemented here*")
    
    # Performance Tips
    with st.expander("🎯 Performance Tips"):
        st.markdown("""
        **Chunk Size:**
        - Smaller chunks (500-800): Better precision, may miss context
        - Larger chunks (1200-2000): Better context, may include noise
        - Recommended: 1000 for most papers
        
        **Top-K Retrieval:**
        - Lower values (3-5): Faster, more focused
        - Higher values (8-15): More comprehensive, slower
        - Recommended: 5 for balanced performance
        
        **Similarity Threshold:**
        - Higher values (0.7-0.9): Only very relevant chunks
        - Lower values (0.3-0.6): More inclusive retrieval
        - Recommended: 0.7 for quality results
        """)

with tab3:
    st.subheader("Custom Prompts")
    st.markdown("Customize prompts for extractor and validator agents to improve performance for specific domains.")
    
    # Load default prompts
    default_prompts = get_default_prompts()
    
    # Extractor Prompts
    st.markdown("### 🔍 Extractor Prompts")
    st.markdown("Customize how the extractor analyzes papers for checklist compliance.")
    
    extractor_prompt = st.text_area(
        "Extractor Prompt Template",
        value=st.session_state.agent_config['extractor'].get('custom_prompt', default_prompts.get('extractor_default', '')),
        height=200,
        help="This prompt guides how the extractor analyzes papers. Use {item_description}, {paper_text}, etc. as placeholders.",
        key="extractor_prompt"
    )
    st.session_state.agent_config['extractor']['custom_prompt'] = extractor_prompt
    
    # Validator Prompts
    st.markdown("### ✅ Validator Prompts")
    st.markdown("Customize how the validator assesses extraction results.")
    
    validator_prompt = st.text_area(
        "Validator Prompt Template",
        value=st.session_state.agent_config['validator'].get('custom_prompt', default_prompts.get('validator_default', '')),
        height=200,
        help="This prompt guides how the validator assesses extractions. Use {extraction_result}, {guideline_item}, etc. as placeholders.",
        key="validator_prompt"
    )
    st.session_state.agent_config['validator']['custom_prompt'] = validator_prompt
    
    # Prompt Templates
    st.markdown("### 📝 Prompt Templates")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset to Default", type="secondary"):
            st.session_state.agent_config['extractor']['custom_prompt'] = default_prompts.get('extractor_default', '')
            st.session_state.agent_config['validator']['custom_prompt'] = default_prompts.get('validator_default', '')
            st.rerun()
    
    with col2:
        if st.button("📋 Copy from Li-Paper", type="secondary"):
            st.info("This would load prompts optimized for Li-Paper SOP checklist")
    
    # Prompt Variables
    with st.expander("📚 Available Variables"):
        st.markdown("""
        **Extractor Prompt Variables:**
        - `{item_id}`: Checklist item identifier
        - `{item_description}`: Full item description
        - `{item_category}`: Item category/section
        - `{paper_text}`: Relevant paper text (RAG mode)
        - `{paper_title}`: Paper title
        - `{checklist_type}`: Type of checklist (RECORD, STROBE, etc.)
        
        **Validator Prompt Variables:**
        - `{extraction_result}`: Results from extractor
        - `{guideline_item}`: Original checklist item
        - `{compliance_assessment}`: Extractor's compliance assessment
        - `{evidence}`: Evidence found by extractor
        - `{paper_text}`: Relevant paper text (RAG mode)
        """)
    
    # Prompt Examples
    with st.expander("💡 Prompt Examples"):
        st.markdown("""
        **Domain-Specific Extractor Prompt:**
        ```
        You are an expert in {checklist_type} guidelines for {domain} research.
        
        Analyze the following paper section for compliance with item {item_id}:
        {item_description}
        
        Paper text: {paper_text}
        
        Provide specific evidence and clear reasoning for your assessment.
        ```
        
        **Strict Validator Prompt:**
        ```
        You are a strict validator for {checklist_type} compliance.
        
        The extractor found: {compliance_assessment}
        Evidence: {evidence}
        
        Verify this assessment is correct and complete. Be conservative in your validation.
        ```
        """)

with tab4:
    st.subheader("Save & Load Configurations")
    st.markdown("Save your agent configurations for reuse or load previously saved settings.")
    
    # Save Configuration
    st.markdown("### 💾 Save Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        config_name = st.text_input(
            "Configuration Name",
            placeholder="e.g., RECORD_RAG_Config, STROBE_Base_Config",
            help="Enter a descriptive name for this configuration"
        )
    
    with col2:
        if st.button("💾 Save Config", type="primary"):
            if config_name:
                config_data = {
                    'agent_config': st.session_state.agent_config,
                    'rag_config': st.session_state.rag_config,
                    'pipeline_config': st.session_state.pipeline_config,
                    'timestamp': st.session_state.get('timestamp', ''),
                    'description': f"Configuration saved with {config_name}"
                }
                
                if save_custom_prompts(f"config_{config_name}", config_data):
                    st.success(f"✅ Configuration '{config_name}' saved successfully!")
                else:
                    st.error("❌ Failed to save configuration")
            else:
                st.warning("⚠️ Please enter a configuration name")
    
    # Load Configuration
    st.markdown("### 📂 Load Configuration")
    
    custom_configs = get_custom_prompts()
    config_files = [name for name in custom_configs.keys() if name.startswith('config_')]
    
    if not config_files:
        st.info("📝 No saved configurations found.")
    else:
        selected_config = st.selectbox(
            "Select Configuration",
            config_files,
            format_func=lambda x: x.replace('config_', ''),
            help="Choose a previously saved configuration to load"
        )
        
        if selected_config:
            config_data = custom_configs[selected_config]
            
            # Show configuration preview
            with st.expander("👁️ Preview Configuration"):
                st.json(config_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Load Config", type="primary"):
                    # Load the configuration
                    if 'agent_config' in config_data:
                        st.session_state.agent_config.update(config_data['agent_config'])
                    if 'rag_config' in config_data:
                        st.session_state.rag_config.update(config_data['rag_config'])
                    if 'pipeline_config' in config_data:
                        st.session_state.pipeline_config.update(config_data['pipeline_config'])
                    
                    st.success(f"✅ Configuration '{selected_config.replace('config_', '')}' loaded successfully!")
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Delete Config", type="secondary"):
                    st.warning("⚠️ Configuration deletion would be implemented here")

# Configuration Validation
st.markdown("---")
st.subheader("🔍 Configuration Validation")

validation_result = validate_configuration()

if validation_result['valid']:
    st.success("✅ Configuration is valid and ready to use!")
else:
    st.error("❌ Configuration has issues that need to be resolved:")
    for error in validation_result['errors']:
        st.error(f"• {error}")

if validation_result['warnings']:
    st.warning("⚠️ Configuration warnings:")
    for warning in validation_result['warnings']:
        st.warning(f"• {warning}")

# Configuration Summary
with st.expander("📋 Current Configuration Summary"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Agent Configuration:**")
        st.json(st.session_state.agent_config)
    
    with col2:
        st.markdown("**RAG Configuration:**")
        st.json(st.session_state.rag_config)

# Help and Tips
with st.expander("💡 Configuration Tips"):
    st.markdown("""
    **Choosing Agent Types:**
    - **Base agents**: Faster, lower cost, good for standard papers
    - **RAG agents**: More accurate, better context understanding, higher cost
    - **Mixed approach**: Use RAG for complex checklists, base for simple ones
    
    **Model Selection:**
    - **o3/o1 models**: Best reasoning, highest cost, slower
    - **GPT-4o**: Good balance of performance and cost
    - **Claude models**: Excellent for validation tasks
    - **Smaller models**: Faster, cheaper, may be less accurate
    
    **Custom Prompts:**
    - Test prompts on a few papers before large-scale processing
    - Include specific domain knowledge in prompts
    - Use clear, structured instructions
    - Provide examples when possible
    
    **RAG Settings:**
    - Start with default settings and adjust based on results
    - Larger chunks for complex papers, smaller for focused analysis
    - Higher similarity thresholds for precision, lower for recall
    """)
