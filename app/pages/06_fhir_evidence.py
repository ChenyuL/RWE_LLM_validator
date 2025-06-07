import streamlit as st
import pandas as pd
import json
import datetime
from pathlib import Path
from utils.file_helpers import (
    get_results, get_result_files, get_result_file_path, read_json_file
)
from utils.session_state import load_api_keys_from_env
from utils.fhir_generator import (
    generate_fhir_evidence, generate_bulk_fhir_evidence, 
    generate_enhanced_fhir_evidence, generate_enhanced_bulk_fhir_evidence,
    create_fhir_bundle, create_generation_summary
)

# Page header
st.header("🔬 FHIR Evidence Generation")
st.markdown("Generate standardized FHIR Evidence resources for clinical research interoperability and future evidence synthesis.")

# Information section
with st.expander("ℹ️ About FHIR Evidence Resources", expanded=False):
    st.markdown("""
    **FHIR Evidence Resources** are standardized data structures that represent research evidence in a machine-readable format.
    
    **Key Benefits:**
    - **Interoperability**: Compatible with clinical research systems worldwide
    - **Standardization**: Follows HL7 FHIR R4 specifications
    - **Evidence Synthesis**: Supports systematic reviews and meta-analyses
    - **Quality Assessment**: Includes certainty and quality ratings
    - **Future Use**: Ready for evidence databases and guideline development
    
    **Learn More:**
    - [FHIR Evidence Resource Specification](https://build.fhir.org/evidence.html)
    - [HL7 FHIR Documentation](https://www.hl7.org/fhir/)
    """)

# Check for available papers
import os
from pathlib import Path

papers_dir = "/Users/chenyuli/Desktop/Macbookpro/LLMEvaluation/RWE_LLM_validator/data/Papers"
if os.path.exists(papers_dir):
    paper_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
else:
    paper_files = []

if not paper_files:
    st.warning("⚠️ No PDF papers found in the data directory.")
    st.info("👉 Please upload papers to the data/Papers directory first.")
    st.stop()

# Configuration Section
st.subheader("⚙️ Configuration")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### LLM Model Selection")
    
    # Load API keys and determine available models
    api_keys = load_api_keys_from_env()
    available_models = []
    
    if api_keys.get('openai'):
        available_models.extend([
            'gpt-4o',
            'gpt-4',
            'gpt-3.5-turbo'
        ])
    
    if api_keys.get('anthropic'):
        available_models.extend([
            'claude-3-5-sonnet-20241022',
            'claude-3-haiku-20240307'
        ])
    
    if not available_models:
        st.error("❌ No API keys configured.")
        st.info("👉 Please configure API keys in the **API Keys** page.")
        st.stop()
    
    selected_model = st.selectbox(
        "Select LLM Model",
        available_models,
        help="Choose the model to generate FHIR Evidence representations"
    )
    
    # Model information
    if 'gpt' in selected_model.lower():
        st.info(f"🤖 Using OpenAI model: {selected_model}")
    elif 'claude' in selected_model.lower():
        st.info(f"🤖 Using Anthropic model: {selected_model}")

with col2:
    st.markdown("#### FHIR Configuration")
    
    # FHIR domain configuration
    default_fhir_base = "https://build.fhir.org"
    fhir_base_url = st.text_input(
        "FHIR Base URL",
        value=default_fhir_base,
        help="Base URL for FHIR resource references"
    )
    
    # Evidence profile URL
    evidence_profile = st.text_input(
        "Evidence Profile URL",
        value=f"{fhir_base_url}/evidence.html",
        help="URL for the FHIR Evidence profile specification"
    )
    
    # Organization identifier
    org_identifier = st.text_input(
        "Organization System",
        value="http://example.org/validation-evidence",
        help="System identifier for your organization"
    )

# Generation Options
st.markdown("#### Generation Options")

col1, col2, col3 = st.columns(3)

with col1:
    include_evidence_details = st.checkbox(
        "Include Evidence Details",
        value=True,
        help="Include specific quotes and locations from papers"
    )

with col2:
    include_reasoning = st.checkbox(
        "Include Reasoning",
        value=True,
        help="Include validation reasoning and confidence scores"
    )

with col3:
    include_statistics = st.checkbox(
        "Include Statistics",
        value=True,
        help="Include compliance rates and statistical measures"
    )

# Paper Selection Section
st.markdown("---")
st.subheader("📄 Paper Selection")

# Display available papers
st.info(f"📊 Found {len(paper_files)} PDF papers in the data directory")

# Paper selection mode
selection_mode = st.radio(
    "Selection Mode",
    ["Select Individual Papers", "Select All Papers"],
    help="Choose how to select papers for FHIR generation"
)

if selection_mode == "Select Individual Papers":
    # Create a more informative display for paper selection
    paper_options = []
    for paper_file in paper_files:
        # Extract paper ID from filename
        paper_id = Path(paper_file).stem
        display_name = f"{paper_id} ({paper_file})"
        paper_options.append((paper_file, display_name))
    
    if paper_options:
        selected_papers = st.multiselect(
            "Select Papers",
            options=[option[0] for option in paper_options],
            format_func=lambda x: next(display for paper, display in paper_options if paper == x),
            help="Choose specific papers to generate FHIR Evidence for"
        )
    else:
        selected_papers = []
        st.warning("⚠️ No papers available.")
else:
    selected_papers = paper_files
    st.info(f"📊 Selected all {len(selected_papers)} papers for FHIR generation.")

# Display selection summary
if selected_papers:
    st.markdown("#### Selection Summary")
    
    # Create summary DataFrame
    summary_data = []
    for paper_file in selected_papers:
        paper_id = Path(paper_file).stem
        summary_data.append({
            'Paper ID': paper_id,
            'Filename': paper_file,
            'Size': 'Unknown'  # Could add file size if needed
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Selected Papers", len(selected_papers))
        
        with col2:
            st.metric("Total Files", len(paper_files))
        
        with col3:
            selection_rate = (len(selected_papers) / len(paper_files)) * 100 if paper_files else 0
            st.metric("Selection Rate", f"{selection_rate:.1f}%")

# Generation Section
st.markdown("---")
st.subheader("🚀 Generate FHIR Evidence")

if selected_papers:
    # Estimated processing time and cost
    estimated_time = len(selected_papers) * 30  # 30 seconds per paper estimate
    estimated_cost = len(selected_papers) * 0.50  # $0.50 per paper estimate
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"⏱️ Estimated time: {estimated_time // 60}m {estimated_time % 60}s")
    
    with col2:
        st.info(f"💰 Estimated cost: ${estimated_cost:.2f}")
    
    # Generation button
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔬 Generate FHIR Evidence Resources", type="primary", use_container_width=True):
            # Store generation parameters in session state
            st.session_state.fhir_generation_params = {
                'papers': selected_papers,
                'model': selected_model,
                'fhir_base_url': fhir_base_url,
                'evidence_profile': evidence_profile,
                'org_identifier': org_identifier,
                'include_evidence_details': include_evidence_details,
                'include_reasoning': include_reasoning,
                'include_statistics': include_statistics
            }
            
            # Start generation
            with st.spinner("Generating FHIR Evidence resources..."):
                # Enhanced bulk generation with custom parameters
                bulk_fhir = generate_enhanced_bulk_fhir_evidence(
                    selected_papers,
                    selected_model,
                    api_keys,
                    include_evidence_details,
                    include_reasoning,
                    fhir_base_url,
                    evidence_profile,
                    org_identifier
                )
                
                if bulk_fhir:
                    st.session_state.generated_fhir_evidence = bulk_fhir
                    st.success(f"✅ Successfully generated FHIR Evidence for {len(bulk_fhir)} papers!")
                else:
                    st.error("❌ Failed to generate FHIR Evidence")
    
    with col2:
        # Validate FHIR
        if st.button("✅ Validate FHIR", use_container_width=True):
            st.info("🔧 FHIR validation feature coming soon!")

# Results Section
if hasattr(st.session_state, 'generated_fhir_evidence') and st.session_state.generated_fhir_evidence:
    st.markdown("---")
    st.subheader("📥 Download Results")
    
    fhir_evidence = st.session_state.generated_fhir_evidence
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download as single bundle
        bundle_fhir = create_fhir_bundle(fhir_evidence, org_identifier)
        bundle_json = json.dumps(bundle_fhir, indent=2)
        
        st.download_button(
            label="📦 Download FHIR Bundle",
            data=bundle_json,
            file_name=f"fhir_evidence_bundle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Download individual resources
        individual_json = json.dumps(fhir_evidence, indent=2)
        
        st.download_button(
            label="📄 Download Individual Resources",
            data=individual_json,
            file_name=f"fhir_evidence_resources_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Download summary report
        summary_report = create_generation_summary(fhir_evidence, st.session_state.fhir_generation_params)
        summary_json = json.dumps(summary_report, indent=2)
        
        st.download_button(
            label="📊 Download Summary Report",
            data=summary_json,
            file_name=f"fhir_generation_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Display results summary
    st.markdown("#### Generation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Generated Resources", len(fhir_evidence))
    
    with col2:
        total_size = len(json.dumps(fhir_evidence))
        st.metric("Total Size", f"{total_size // 1024} KB")
    
    with col3:
        avg_items = sum(len(resource.get('statistic', [])) for resource in fhir_evidence) / len(fhir_evidence) if fhir_evidence else 0
        st.metric("Avg Items/Resource", f"{avg_items:.1f}")
    
    with col4:
        generation_time = datetime.datetime.now()
        st.metric("Generated", generation_time.strftime("%H:%M"))

# Preview Section
if hasattr(st.session_state, 'preview_fhir') and st.session_state.preview_fhir:
    st.markdown("---")
    st.subheader("👁️ FHIR Preview")
    
    with st.expander("🔍 View FHIR Evidence Resource", expanded=True):
        st.json(st.session_state.preview_fhir)

# Display generated FHIR evidence
if hasattr(st.session_state, 'generated_fhir_evidence') and st.session_state.generated_fhir_evidence:
    with st.expander("🔍 View All Generated FHIR Evidence", expanded=False):
        for i, resource in enumerate(st.session_state.generated_fhir_evidence):
            st.markdown(f"**Resource {i+1}: {resource.get('title', 'Untitled')}**")
            st.json(resource)
            if i < len(st.session_state.generated_fhir_evidence) - 1:
                st.markdown("---")


# Help section
with st.expander("💡 FHIR Evidence Generation Help"):
    st.markdown("""
    **Getting Started:**
    1. **Configure**: Select your preferred LLM model and FHIR settings
    2. **Select Papers**: Choose individual papers or select all filtered results
    3. **Generate**: Click "Generate FHIR Evidence" to create resources
    4. **Download**: Use download buttons to save FHIR resources
    
    **Configuration Options:**
    - **LLM Model**: Choose the model for generating FHIR content
    - **FHIR Base URL**: Customize the base URL for FHIR references
    - **Evidence Profile**: Specify the FHIR Evidence profile URL
    - **Organization System**: Set your organization's identifier system
    
    **Generation Options:**
    - **Evidence Details**: Include specific quotes and locations from papers
    - **Reasoning**: Include validation reasoning and confidence scores
    - **Statistics**: Include compliance rates and statistical measures
    
    **Output Formats:**
    - **FHIR Bundle**: All resources packaged in a single FHIR Bundle
    - **Individual Resources**: Separate FHIR Evidence resources
    - **Summary Report**: Generation metadata and statistics
    
    **Best Practices:**
    - Start with a small number of papers to test configuration
    - Use preview function to verify FHIR structure before bulk generation
    - Customize FHIR domains to match your organization's requirements
    - Include evidence details for comprehensive documentation
    
    **FHIR Compliance:**
    - All generated resources follow FHIR R4 specifications
    - Resources include required metadata and identifiers
    - Compatible with FHIR-compliant clinical research systems
    """)
