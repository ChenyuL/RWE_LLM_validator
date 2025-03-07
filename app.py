import streamlit as st
import os
import json
import subprocess
import pandas as pd
import time
import shutil
from pathlib import Path
import base64
import tempfile

# Define paths
GUIDELINES_PATH = os.path.join("data", "Guidelines")
PAPERS_PATH = os.path.join("data", "Papers")
OUTPUT_PATH = "output"
RESULTS_PATH = os.path.join(OUTPUT_PATH, "paper_results")

# Define available models
REASONER_MODELS = ["o3-mini-2025-01-31", "gpt-4o", "gpt-4-turbo"]
EXTRACTOR_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
VALIDATOR_MODELS = ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]

# Ensure directories exist
def ensure_directories():
    os.makedirs(GUIDELINES_PATH, exist_ok=True)
    os.makedirs(PAPERS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

# Get list of guideline folders
def get_guideline_folders():
    if not os.path.exists(GUIDELINES_PATH):
        return []
    return [d for d in os.listdir(GUIDELINES_PATH) if os.path.isdir(os.path.join(GUIDELINES_PATH, d))]

# Get list of papers
def get_papers():
    if not os.path.exists(PAPERS_PATH):
        return []
    return [f for f in os.listdir(PAPERS_PATH) if f.lower().endswith('.pdf')]

# Get list of results
def get_results():
    if not os.path.exists(RESULTS_PATH):
        return []
    return [d for d in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, d))]

# Run validation process
def run_validation(paper_path, guideline_type, mode="full", reasoner_model="o3-mini-2025-01-31", 
                  extractor_model="gpt-4o", validator_model="claude-3-sonnet-20240229", prompts_file=None):
    # Create a temporary config file
    config = {
        "reasoner": {
            "openai_model": reasoner_model
        },
        "extractor": {
            "model": extractor_model
        },
        "validator": {
            "model": validator_model
        }
    }
    
    with open("temp_config.json", "w") as f:
        json.dump(config, f)
    
    # Build the command
    cmd = ["python", "run_all_papers.py"]
    
    if mode != "full":
        cmd.extend(["--mode", mode])
    
    if prompts_file:
        cmd.extend(["--prompts", prompts_file])
    
    cmd.extend(["--paper", paper_path, "--config", "temp_config.json", "--guideline", guideline_type])
    
    # Run the command
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Create a placeholder for the output
        output_placeholder = st.empty()
        
        # Stream the output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_placeholder.text(output_placeholder.text + output)
        
        # Get the return code
        return_code = process.poll()
        
        # Clean up the temporary config file
        if os.path.exists("temp_config.json"):
            os.remove("temp_config.json")
        
        if return_code == 0:
            st.success("Validation completed successfully!")
            return True
        else:
            st.error(f"Validation failed with return code {return_code}")
            return False
    except Exception as e:
        st.error(f"Error running validation: {e}")
        return False

# Display PDF
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Display JSON
def display_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    st.json(data)

# Main app
def main():
    st.set_page_config(page_title="RWE LLM Validator", page_icon="📊", layout="wide")
    
    st.title("RWE LLM Validator")
    st.markdown("A tool for validating research papers against reporting guidelines using Large Language Models (LLMs).")
    
    # Ensure directories exist
    ensure_directories()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Guidelines", "Upload Papers", "Run Validation", "View Results"])
    
    if page == "Upload Guidelines":
        st.header("Upload Guidelines")
        
        # Create new guideline folder
        st.subheader("Create New Guideline Folder")
        new_folder = st.text_input("Enter folder name (e.g., RECORD, STROBE, etc.)")
        
        if st.button("Create Folder") and new_folder:
            folder_path = os.path.join(GUIDELINES_PATH, new_folder)
            if os.path.exists(folder_path):
                st.warning(f"Folder '{new_folder}' already exists!")
            else:
                os.makedirs(folder_path)
                st.success(f"Folder '{new_folder}' created successfully!")
        
        # Upload guideline files
        st.subheader("Upload Guideline Files")
        guideline_folders = get_guideline_folders()
        
        if not guideline_folders:
            st.warning("No guideline folders found. Please create a folder first.")
        else:
            selected_folder = st.selectbox("Select guideline folder", guideline_folders)
            uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
            
            if uploaded_files and st.button("Upload Files"):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(GUIDELINES_PATH, selected_folder, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
        
        # Display existing guideline files
        st.subheader("Existing Guideline Files")
        
        if guideline_folders:
            folder_to_view = st.selectbox("Select folder to view", guideline_folders, key="view_folder")
            folder_path = os.path.join(GUIDELINES_PATH, folder_to_view)
            
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
                
                if not files:
                    st.info(f"No PDF files found in '{folder_to_view}'.")
                else:
                    st.write(f"Files in '{folder_to_view}':")
                    for file in files:
                        col1, col2 = st.columns([3, 1])
                        col1.write(file)
                        if col2.button("View", key=f"view_{file}"):
                            st.session_state.viewing_file = os.path.join(folder_path, file)
                            st.session_state.viewing_file_name = file
            
            if 'viewing_file' in st.session_state and os.path.exists(st.session_state.viewing_file):
                st.subheader(f"Viewing: {st.session_state.viewing_file_name}")
                display_pdf(st.session_state.viewing_file)
    
    elif page == "Upload Papers":
        st.header("Upload Papers")
        
        st.markdown("""
        Upload research papers to validate. Papers should be named with their PubMed ID (e.g., `34923518.pdf`).
        """)
        
        uploaded_files = st.file_uploader("Upload PDF papers", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and st.button("Upload Papers"):
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                
                # Check if filename is a valid PubMed ID format
                if not file_name.lower().endswith('.pdf'):
                    st.warning(f"File '{file_name}' is not a PDF. Skipping.")
                    continue
                
                file_path = os.path.join(PAPERS_PATH, file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"{len(uploaded_files)} paper(s) uploaded successfully!")
        
        # Display existing papers
        st.subheader("Existing Papers")
        papers = get_papers()
        
        if not papers:
            st.info("No papers found.")
        else:
            st.write(f"Found {len(papers)} papers:")
            
            for paper in papers:
                col1, col2 = st.columns([3, 1])
                col1.write(paper)
                if col2.button("View", key=f"view_{paper}"):
                    st.session_state.viewing_paper = os.path.join(PAPERS_PATH, paper)
                    st.session_state.viewing_paper_name = paper
            
            if 'viewing_paper' in st.session_state and os.path.exists(st.session_state.viewing_paper):
                st.subheader(f"Viewing: {st.session_state.viewing_paper_name}")
                display_pdf(st.session_state.viewing_paper)
    
    elif page == "Run Validation":
        st.header("Run Validation")
        
        # Get available guidelines and papers
        guideline_folders = get_guideline_folders()
        papers = get_papers()
        
        if not guideline_folders:
            st.warning("No guideline folders found. Please upload guidelines first.")
        elif not papers:
            st.warning("No papers found. Please upload papers first.")
        else:
            # Select guideline
            selected_guideline = st.selectbox("Select guideline", guideline_folders)
            
            # Select paper
            selected_paper = st.selectbox("Select paper", papers)
            
            # Select mode
            selected_mode = st.selectbox("Select mode", ["full", "reasoner", "extractor"])
            
            # Select models
            st.subheader("Select Models")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("Reasoner Model")
                reasoner_model = st.selectbox("", REASONER_MODELS)
            
            with col2:
                st.write("Extractor Model")
                extractor_model = st.selectbox("", EXTRACTOR_MODELS)
            
            with col3:
                st.write("Validator Model")
                validator_model = st.selectbox("", VALIDATOR_MODELS)
            
            # Prompts file (for extractor mode)
            prompts_file = None
            if selected_mode == "extractor":
                st.subheader("Select Prompts File")
                prompts_files = [f for f in os.listdir(OUTPUT_PATH) if f.endswith('_RECORD_prompts.json')]
                
                if not prompts_files:
                    st.warning("No prompts files found. Please run in 'reasoner' mode first.")
                else:
                    selected_prompts_file = st.selectbox("Select prompts file", prompts_files)
                    prompts_file = os.path.join(OUTPUT_PATH, selected_prompts_file)
            
            # Run validation
            if st.button("Run Validation"):
                paper_path = os.path.join(PAPERS_PATH, selected_paper)
                
                with st.spinner("Running validation..."):
                    success = run_validation(
                        paper_path=paper_path,
                        guideline_type=selected_guideline,
                        mode=selected_mode,
                        reasoner_model=reasoner_model,
                        extractor_model=extractor_model,
                        validator_model=validator_model,
                        prompts_file=prompts_file
                    )
                
                if success:
                    # Get the PubMed ID from the paper filename
                    pubmed_id = os.path.splitext(selected_paper)[0]
                    if '.' in pubmed_id:
                        pubmed_id = pubmed_id.split('.')[0]
                    
                    st.session_state.last_validated_paper = pubmed_id
                    st.success(f"Validation completed for {selected_paper}!")
                    st.markdown(f"[View Results](#View Results)")
    
    elif page == "View Results":
        st.header("View Results")
        
        # Get available results
        results = get_results()
        
        if not results:
            st.warning("No results found. Please run validation first.")
        else:
            # Select result to view
            if 'last_validated_paper' in st.session_state and st.session_state.last_validated_paper in results:
                default_idx = results.index(st.session_state.last_validated_paper)
            else:
                default_idx = 0
            
            selected_result = st.selectbox("Select result", results, index=default_idx)
            
            # Display result
            result_path = os.path.join(RESULTS_PATH, selected_result)
            
            if os.path.exists(result_path):
                # Find the most recent files
                files = os.listdir(result_path)
                
                # Group files by type
                reasoner_files = [f for f in files if "openai_reasoner" in f and not f.endswith("_process_log.txt")]
                extractor_files = [f for f in files if "openai-gpt4o_extractor" in f]
                validator_files = [f for f in files if "claude-sonnet_validator" in f]
                report_files = [f for f in files if "openai_claude_report" in f]
                checklist_files = [f for f in files if "full_record_checklist" in f]
                
                # Sort files by timestamp (newest first)
                for file_list in [reasoner_files, extractor_files, validator_files, report_files, checklist_files]:
                    file_list.sort(reverse=True)
                
                # Create tabs for different result types
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Final Report", "Checklist", "Reasoner", "Extractor", "Validator"])
                
                with tab1:
                    if report_files:
                        st.subheader("Final Report")
                        report_file = os.path.join(result_path, report_files[0])
                        display_json(report_file)
                    else:
                        st.info("No final report found.")
                
                with tab2:
                    if checklist_files:
                        st.subheader("RECORD Checklist")
                        checklist_file = os.path.join(result_path, checklist_files[0])
                        
                        # Display as a table
                        with open(checklist_file, "r") as f:
                            checklist_data = json.load(f)
                        
                        if "checklist" in checklist_data:
                            items = []
                            for item_id, item_data in checklist_data["checklist"].items():
                                items.append({
                                    "Item ID": item_id,
                                    "Description": item_data.get("description", ""),
                                    "Answer": item_data.get("answer", "")
                                })
                            
                            if items:
                                df = pd.DataFrame(items)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("No checklist items found.")
                        else:
                            display_json(checklist_file)
                    else:
                        st.info("No checklist found.")
                
                with tab3:
                    if reasoner_files:
                        st.subheader("Reasoner Output")
                        reasoner_file = os.path.join(result_path, reasoner_files[0])
                        display_json(reasoner_file)
                    else:
                        st.info("No reasoner output found.")
                
                with tab4:
                    if extractor_files:
                        st.subheader("Extractor Output")
                        extractor_file = os.path.join(result_path, extractor_files[0])
                        display_json(extractor_file)
                    else:
                        st.info("No extractor output found.")
                
                with tab5:
                    if validator_files:
                        st.subheader("Validator Output")
                        validator_file = os.path.join(result_path, validator_files[0])
                        display_json(validator_file)
                    else:
                        st.info("No validator output found.")

if __name__ == "__main__":
    main()
