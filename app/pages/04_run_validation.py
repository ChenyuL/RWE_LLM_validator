import streamlit as st
import json
import os
import sys
import datetime
from pathlib import Path
from utils.file_helpers import (
    get_checklist_folders, get_papers, get_prompts_files,
    filter_prompts_by_checklist, get_paper_path
)
from utils.session_state import validate_configuration, check_api_keys_status, load_api_keys_from_env

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.core.pipeline import EnhancedValidationPipeline

# Page header
st.header("🚀 Run Validation")
st.markdown("Execute the validation pipeline to analyze papers against reporting checklists.")

# Check configuration first
validation_result = validate_configuration()
api_status = check_api_keys_status()

if not validation_result['valid']:
    st.error("❌ Configuration issues detected. Please fix these before running validation:")
    for error in validation_result['errors']:
        st.error(f"• {error}")
    st.info("👉 Go to the **Configure Agents** page to fix these issues.")
    st.stop()

# Get available data
checklist_folders = get_checklist_folders()
papers = get_papers()
prompts_files = get_prompts_files()

if not checklist_folders:
    st.warning("⚠️ No checklist folders found. Please upload checklists first.")
    st.info("👉 Go to the **Upload Data** page to add checklists.")
    st.stop()

if not papers:
    st.warning("⚠️ No papers found. Please upload papers first.")
    st.info("👉 Go to the **Upload Data** page to add papers.")
    st.stop()

# Main configuration
st.subheader("📋 Validation Configuration")

col1, col2 = st.columns(2)

with col1:
    # Pipeline mode selection
    st.markdown("#### Pipeline Mode")
    mode = st.radio(
        "Select mode",
        ["full", "reasoner", "extractor"],
        index=["full", "reasoner", "extractor"].index(st.session_state.pipeline_config['mode']),
        help="Full: Complete pipeline | Reasoner: Generate prompts only | Extractor: Use existing prompts"
    )
    st.session_state.pipeline_config['mode'] = mode
    
    # Checklist selection
    st.markdown("#### Checklist Type")
    selected_checklist = st.selectbox(
        "Select checklist",
        checklist_folders,
        index=checklist_folders.index(st.session_state.pipeline_config['checklist_type']) 
        if st.session_state.pipeline_config['checklist_type'] in checklist_folders else 0
    )
    st.session_state.pipeline_config['checklist_type'] = selected_checklist

with col2:
    # Paper selection
    st.markdown("#### Paper Selection")
    paper_selection_mode = st.radio(
        "Select papers",
        ["single", "multiple", "all"],
        help="Single: One paper | Multiple: Select several | All: Process all papers"
    )
    
    if paper_selection_mode == "single":
        selected_papers = [st.selectbox("Select paper", papers)]
    elif paper_selection_mode == "multiple":
        selected_papers = st.multiselect("Select papers", papers)
    else:
        selected_papers = papers
    
    # Batch size for processing (only show for multiple papers)
    if paper_selection_mode in ["multiple", "all"] and len(selected_papers) > 1:
        st.markdown("#### Processing Settings")
        batch_size = st.slider(
            "Batch size",
            min_value=1,
            max_value=10,
            value=st.session_state.pipeline_config['batch_size'],
            help="Number of checklist items to process at once"
        )
        st.session_state.pipeline_config['batch_size'] = batch_size
    else:
        batch_size = st.session_state.pipeline_config['batch_size']

# Mode-specific configuration
if mode == "extractor":
    st.markdown("#### Prompts File (Required for Extractor Mode)")
    
    # Filter prompts by checklist type
    filtered_prompts = filter_prompts_by_checklist(prompts_files, selected_checklist)
    
    if not filtered_prompts:
        st.error(f"❌ No prompts files found for {selected_checklist}. Please run in 'reasoner' mode first.")
        st.stop()
    
    selected_prompts_file = st.selectbox(
        "Select prompts file",
        filtered_prompts,
        format_func=lambda x: Path(x).name
    )
    st.session_state.file_paths['prompts_file'] = selected_prompts_file

# Agent configuration summary
st.markdown("---")
st.subheader("🤖 Current Agent Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧠 Reasoner**")
    reasoner_config = st.session_state.agent_config['reasoner']
    st.write(f"Type: {reasoner_config['type'].upper()}")
    st.write(f"Model: {reasoner_config['model']}")

with col2:
    st.markdown("**🔍 Extractor**")
    extractor_config = st.session_state.agent_config['extractor']
    st.write(f"Type: {extractor_config['type'].upper()}")
    st.write(f"Model: {extractor_config['model']}")

with col3:
    st.markdown("**✅ Validator**")
    validator_config = st.session_state.agent_config['validator']
    st.write(f"Type: {validator_config['type'].upper()}")
    st.write(f"Model: {validator_config['model']}")

# RAG configuration display
using_rag = any(
    st.session_state.agent_config[agent]['type'] == 'rag' 
    for agent in ['reasoner', 'extractor', 'validator']
)

if using_rag:
    with st.expander("🔧 RAG Configuration"):
        rag_config = st.session_state.rag_config
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Chunk Size: {rag_config['chunk_size']}")
            st.write(f"Chunk Overlap: {rag_config['chunk_overlap']}")
        
        with col2:
            st.write(f"Top-K: {rag_config['top_k']}")
            st.write(f"Similarity Threshold: {rag_config['similarity_threshold']}")

# Validation summary
st.markdown("---")
st.subheader("📊 Validation Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Papers to Process", len(selected_papers) if selected_papers else 0)

with col2:
    st.metric("Checklist Type", selected_checklist)

with col3:
    st.metric("Pipeline Mode", mode.upper())

with col4:
    estimated_time = len(selected_papers) * 5 if selected_papers else 0  # Rough estimate
    st.metric("Est. Time (min)", estimated_time)

# Cost estimation
if using_rag:
    cost_multiplier = 1.5
else:
    cost_multiplier = 1.0

estimated_cost = len(selected_papers) * 1.0 * cost_multiplier if selected_papers else 0
st.info(f"💰 Estimated cost: ${estimated_cost:.2f} (approximate)")

# Run validation
st.markdown("---")
st.subheader("▶️ Execute Validation")

if not selected_papers:
    st.warning("⚠️ Please select at least one paper to process.")
else:
    # Show what will be processed
    with st.expander("📋 Processing Details"):
        st.markdown("**Papers to process:**")
        for paper in selected_papers[:10]:  # Show first 10
            st.write(f"• {paper}")
        if len(selected_papers) > 10:
            st.write(f"... and {len(selected_papers) - 10} more papers")
        
        st.markdown("**Configuration:**")
        st.write(f"• Mode: {mode}")
        st.write(f"• Checklist: {selected_checklist}")
        st.write(f"• Batch size: {batch_size}")
        if mode == "extractor":
            st.write(f"• Prompts file: {Path(selected_prompts_file).name}")
    
    # Confirmation and execution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        confirm_run = st.checkbox(
            "I confirm the configuration and want to proceed",
            help="Check this box to enable the run button"
        )
    
    with col2:
        run_button = st.button(
            "🚀 Start Validation",
            type="primary",
            disabled=not confirm_run,
            help="Start the validation process"
        )
    
    if run_button and confirm_run:
        # Load API keys
        api_keys = load_api_keys_from_env()
        
        # Create configuration for the pipeline
        config_data = {
            "reasoner": st.session_state.agent_config['reasoner'],
            "extractor": st.session_state.agent_config['extractor'],
            "validator": st.session_state.agent_config['validator'],
            "rag_config": st.session_state.rag_config if using_rag else None
        }
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()
        
        try:
            # Initialize pipeline
            status_text.text("Initializing validation pipeline...")
            pipeline = EnhancedValidationPipeline(api_keys, config_data)
            
            total_papers = len(selected_papers)
            successful_papers = 0
            
            for i, paper in enumerate(selected_papers):
                status_text.text(f"Processing paper {i+1}/{total_papers}: {paper}")
                progress_bar.progress(i / total_papers)
                
                paper_path = get_paper_path(paper)
                
                with log_container:
                    with st.expander(f"📄 {paper} - Processing Log", expanded=False):
                        log_placeholder = st.empty()
                        
                        try:
                            if mode == "full":
                                log_placeholder.text("Running full validation pipeline...")
                                result = pipeline.run_full_pipeline(paper_path, selected_checklist)
                                
                            elif mode == "reasoner":
                                log_placeholder.text("Running reasoner to generate prompts...")
                                guideline_info = pipeline.process_guideline(selected_checklist)
                                
                                # Save prompts
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                prompts_filename = f"{timestamp}_reasoner_{selected_checklist}_prompts.json"
                                prompts_path = os.path.join("output", prompts_filename)
                                
                                os.makedirs("output", exist_ok=True)
                                with open(prompts_path, "w") as f:
                                    json.dump(guideline_info["prompts"], f, indent=2)
                                
                                result = {"prompts_saved": prompts_path, "prompts_count": len(guideline_info["prompts"])}
                                
                            elif mode == "extractor":
                                log_placeholder.text("Running extractor with existing prompts...")
                                
                                # Load prompts
                                with open(selected_prompts_file, 'r') as f:
                                    prompts = json.load(f)
                                
                                # Create mock guideline info
                                guideline_info = {
                                    "guideline_type": selected_checklist,
                                    "items": [],
                                    "prompts": prompts
                                }
                                
                                # Create mock items based on prompts
                                for item_id in prompts.keys():
                                    guideline_info["items"].append({
                                        "id": item_id,
                                        "description": f"Checklist item {item_id}",
                                        "category": "General"
                                    })
                                
                                # Process paper
                                paper_info = pipeline.process_paper(paper_path, guideline_info)
                                
                                # Validate extraction
                                validation_results = pipeline.validate_extraction(paper_info, guideline_info)
                                
                                # Generate final report
                                result = pipeline.generate_report(paper_info, guideline_info, validation_results)
                                
                                # Save results
                                pipeline._save_results(paper_path, guideline_info, paper_info, validation_results, result)
                            
                            log_placeholder.text("✅ Processing completed successfully!")
                            st.success(f"✅ {paper} processed successfully")
                            successful_papers += 1
                            
                        except Exception as paper_error:
                            error_msg = f"❌ Error processing {paper}: {str(paper_error)}"
                            log_placeholder.text(error_msg)
                            st.error(error_msg)
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text(f"✅ Processing completed! {successful_papers}/{total_papers} papers successful")
            
            # Success message
            if successful_papers > 0:
                st.success(f"🎉 Validation completed for {successful_papers} out of {total_papers} paper(s)!")
                st.info("👉 Go to the **View Results** page to see the analysis results.")
            else:
                st.error("❌ No papers were processed successfully. Please check the logs above.")
            
            # Store last validation info
            st.session_state.last_validation_result = {
                'papers': selected_papers,
                'checklist': selected_checklist,
                'mode': mode,
                'successful': successful_papers,
                'total': total_papers,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"❌ Error during validation: {str(e)}")
            st.error("Please check your configuration and try again.")

# Recent validations
if hasattr(st.session_state, 'last_validation_result') and st.session_state.last_validation_result:
    st.markdown("---")
    st.subheader("📈 Recent Validation")
    
    last_result = st.session_state.last_validation_result
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_papers = last_result.get('total', len(last_result.get('papers', [])))
        successful_papers = last_result.get('successful', 0)
        st.write(f"**Papers:** {successful_papers}/{total_papers}")
        st.write(f"**Checklist:** {last_result['checklist']}")
    
    with col2:
        st.write(f"**Mode:** {last_result['mode'].upper()}")
        timestamp = datetime.datetime.fromisoformat(last_result['timestamp'])
        st.write(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
    
    with col3:
        if st.button("📊 View Results"):
            st.info("👉 Navigate to the **View Results** page to see detailed analysis.")

# Help and tips
with st.expander("💡 Validation Tips"):
    st.markdown("""
    **Pipeline Modes:**
    - **Full Mode**: Complete analysis from guidelines to final validation
    - **Reasoner Mode**: Only generate prompts from guidelines (useful for testing)
    - **Extractor Mode**: Use existing prompts to analyze papers (faster for batch processing)
    
    **Performance Optimization:**
    - Start with a single paper to test your configuration
    - Use smaller batch sizes for complex papers
    - RAG mode is more accurate but slower and more expensive
    - Process papers in groups rather than all at once for large datasets
    
    **Troubleshooting:**
    - Check API key status if validation fails
    - Ensure papers are readable PDFs
    - Verify checklist guidelines are properly uploaded
    - Monitor costs, especially with o3/o1 models
    
    **Best Practices:**
    - Test configuration on a few papers first
    - Save successful configurations for reuse
    - Monitor progress logs for any issues
    - Keep track of processing costs
    """)

with st.expander("⚠️ Important Notes"):
    st.markdown("""
    **Before Running:**
    - Ensure you have sufficient API credits
    - Check that all required files are uploaded
    - Verify your configuration is correct
    - Consider the estimated processing time
    
    **During Processing:**
    - Don't close the browser tab
    - Monitor the progress logs
    - Check for any error messages
    - Be patient - processing can take time
    
    **After Processing:**
    - Review results in the View Results page
    - Check for any failed papers
    - Save successful configurations
    - Clean up temporary files if needed
    """)
