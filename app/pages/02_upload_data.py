import streamlit as st
from utils.file_helpers import (
    get_checklist_folders, get_papers, get_checklist_files,
    create_checklist_folder, save_uploaded_file, get_checklist_path,
    get_paper_path, display_pdf_base64
)

# Page header
st.header("📁 Upload Data")
st.markdown("Upload checklist guidelines and research papers for validation.")

# Create tabs for different upload types
tab1, tab2 = st.tabs(["📋 Checklists", "📄 Papers"])

with tab1:
    st.subheader("Checklist Guidelines")
    st.markdown("Upload reporting guideline documents (RECORD, STROBE, Li-Paper, etc.)")
    
    # Create new checklist folder
    st.markdown("### Create New Checklist Type")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_folder = st.text_input(
            "Checklist Type Name",
            placeholder="e.g., RECORD, STROBE, Li-Paper, CHEERS",
            help="Enter a name for the new checklist type"
        )
    
    with col2:
        if st.button("📁 Create Folder", type="secondary"):
            if new_folder:
                if create_checklist_folder(new_folder):
                    st.success(f"✅ Created folder '{new_folder}'")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to create folder '{new_folder}'")
            else:
                st.warning("⚠️ Please enter a folder name")
    
    # Upload files to existing folders
    st.markdown("### Upload Checklist Files")
    checklist_folders = get_checklist_folders()
    
    if not checklist_folders:
        st.info("📝 No checklist folders found. Create a folder first.")
    else:
        selected_folder = st.selectbox(
            "Select Checklist Type",
            checklist_folders,
            help="Choose which checklist type to upload files to"
        )
        
        uploaded_files = st.file_uploader(
            "Upload PDF Files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files containing the checklist guidelines"
        )
        
        if uploaded_files:
            if st.button("📤 Upload Files", type="primary"):
                success_count = 0
                for uploaded_file in uploaded_files:
                    file_path = get_checklist_path(selected_folder, uploaded_file.name)
                    if save_uploaded_file(uploaded_file, file_path):
                        success_count += 1
                    else:
                        st.error(f"❌ Failed to upload {uploaded_file.name}")
                
                if success_count > 0:
                    st.success(f"✅ Successfully uploaded {success_count} file(s)")
                    st.rerun()
    
    # Display existing checklist files
    st.markdown("### Existing Checklist Files")
    
    if checklist_folders:
        folder_to_view = st.selectbox(
            "Select folder to view",
            checklist_folders,
            key="view_folder"
        )
        
        files = get_checklist_files(folder_to_view)
        
        if not files:
            st.info(f"📂 No PDF files found in '{folder_to_view}'")
        else:
            st.markdown(f"**Files in '{folder_to_view}':**")
            
            # Create a grid layout for files
            cols = st.columns(3)
            for i, file in enumerate(files):
                with cols[i % 3]:
                    if st.button(f"👁️ {file}", key=f"view_checklist_{file}"):
                        st.session_state.viewing_checklist_file = get_checklist_path(folder_to_view, file)
                        st.session_state.viewing_checklist_name = file
            
            # Display selected file
            if hasattr(st.session_state, 'viewing_checklist_file'):
                st.markdown("---")
                st.subheader(f"📖 Viewing: {st.session_state.viewing_checklist_name}")
                
                try:
                    pdf_display = display_pdf_base64(st.session_state.viewing_checklist_file)
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error displaying PDF: {str(e)}")

with tab2:
    st.subheader("Research Papers")
    st.markdown("Upload research papers to validate against checklists.")
    
    # Important notice
    st.info("""
    📋 **Important Notes:**
    - Papers should be named with their identifier (e.g., `34923518.pdf`)
    - Ensure you have permission to upload and analyze the papers
    - For copyrighted papers, consider uploading only necessary supplements
    - Include any essential supplementary materials with the main paper
    """)
    
    # Upload papers
    uploaded_papers = st.file_uploader(
        "Upload PDF Papers",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more research papers in PDF format"
    )
    
    if uploaded_papers:
        # Show preview of files to be uploaded
        st.markdown("**Files to upload:**")
        for paper in uploaded_papers:
            st.write(f"📄 {paper.name} ({paper.size / 1024 / 1024:.1f} MB)")
        
        if st.button("📤 Upload Papers", type="primary"):
            success_count = 0
            for uploaded_paper in uploaded_papers:
                if not uploaded_paper.name.lower().endswith('.pdf'):
                    st.warning(f"⚠️ Skipping '{uploaded_paper.name}' (not a PDF)")
                    continue
                
                file_path = get_paper_path(uploaded_paper.name)
                if save_uploaded_file(uploaded_paper, file_path):
                    success_count += 1
                else:
                    st.error(f"❌ Failed to upload {uploaded_paper.name}")
            
            if success_count > 0:
                st.success(f"✅ Successfully uploaded {success_count} paper(s)")
                st.rerun()
    
    # Display existing papers
    st.markdown("### Existing Papers")
    papers = get_papers()
    
    if not papers:
        st.info("📂 No papers found. Upload some papers to get started.")
    else:
        st.markdown(f"**Found {len(papers)} papers:**")
        
        # Search/filter papers
        search_term = st.text_input("🔍 Search papers", placeholder="Enter paper ID or filename")
        
        if search_term:
            filtered_papers = [p for p in papers if search_term.lower() in p.lower()]
        else:
            filtered_papers = papers
        
        # Display papers in a grid
        if filtered_papers:
            # Pagination
            papers_per_page = 12
            total_pages = (len(filtered_papers) + papers_per_page - 1) // papers_per_page
            
            if total_pages > 1:
                page = st.selectbox("Page", range(1, total_pages + 1)) - 1
            else:
                page = 0
            
            start_idx = page * papers_per_page
            end_idx = min(start_idx + papers_per_page, len(filtered_papers))
            page_papers = filtered_papers[start_idx:end_idx]
            
            # Display papers in grid
            cols = st.columns(4)
            for i, paper in enumerate(page_papers):
                with cols[i % 4]:
                    # Extract paper ID for display
                    paper_id = paper.replace('.pdf', '')
                    if '.' in paper_id:
                        paper_id = paper_id.split('.')[0]
                    
                    st.markdown(f"**{paper_id}**")
                    if st.button(f"👁️ View", key=f"view_paper_{paper}"):
                        st.session_state.viewing_paper_file = get_paper_path(paper)
                        st.session_state.viewing_paper_name = paper
            
            # Show pagination info
            if total_pages > 1:
                st.markdown(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_papers)} papers")
        else:
            st.info("🔍 No papers match your search criteria")
        
        # Display selected paper
        if hasattr(st.session_state, 'viewing_paper_file'):
            st.markdown("---")
            st.subheader(f"📖 Viewing: {st.session_state.viewing_paper_name}")
            
            try:
                pdf_display = display_pdf_base64(st.session_state.viewing_paper_file)
                st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error displaying PDF: {str(e)}")

# Data summary
st.markdown("---")
st.subheader("📊 Data Summary")

col1, col2, col3 = st.columns(3)

with col1:
    checklist_count = len(get_checklist_folders())
    st.metric("Checklist Types", checklist_count)

with col2:
    paper_count = len(get_papers())
    st.metric("Papers", paper_count)

with col3:
    # Calculate total checklist files
    total_checklist_files = 0
    for folder in get_checklist_folders():
        total_checklist_files += len(get_checklist_files(folder))
    st.metric("Checklist Files", total_checklist_files)

# Tips and guidelines
with st.expander("💡 Tips and Guidelines"):
    st.markdown("""
    **Checklist Guidelines:**
    - Upload official guideline documents (PDF format)
    - Organize by checklist type (RECORD, STROBE, etc.)
    - Include all relevant sections and appendices
    - Use clear, descriptive folder names
    
    **Research Papers:**
    - Use consistent naming conventions (e.g., PubMed ID)
    - Ensure papers are complete and readable
    - Include supplementary materials if relevant
    - Check file sizes (large files may take longer to process)
    
    **File Organization:**
    - Keep related files together
    - Use descriptive names
    - Regularly clean up old or duplicate files
    - Back up important data
    """)

with st.expander("🔒 Privacy and Copyright"):
    st.markdown("""
    **Important Considerations:**
    - Only upload papers you have permission to analyze
    - Respect copyright and licensing restrictions
    - Consider using open access papers when possible
    - Remove or redact sensitive information if necessary
    
    **Data Handling:**
    - Files are stored locally on your system
    - Papers are only sent to selected LLM providers for analysis
    - No data is permanently stored by external services
    - You can delete files at any time
    
    **Best Practices:**
    - Review your institution's data handling policies
    - Consider anonymizing papers when possible
    - Keep track of which papers you've uploaded
    - Regularly review and clean up uploaded files
    """)
