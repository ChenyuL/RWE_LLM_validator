import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Configure Streamlit page
st.set_page_config(
    page_title="RWE LLM Validator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stSelectbox > div > div > select {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state first
    from utils.session_state import initialize_session_state
    initialize_session_state()
    
    # Main header
    st.markdown('<div class="main-header">🔬 RWE LLM Validator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Agent Framework for Clinical Checklist Compliance Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    # Navigation options
    pages = {
        "🔑 API Keys": "pages/01_api_keys.py",
        "📁 Upload Data": "pages/02_upload_data.py", 
        "⚙️ Configure Agents": "pages/03_configure_agents.py",
        "🚀 Run Validation": "pages/04_run_validation.py",
        "📊 View Results": "pages/05_view_results.py",
        "🔬 FHIR Evidence": "pages/06_fhir_evidence.py"
    }
    
    # Page selection
    selected_page = st.sidebar.radio(
        "Select a page:",
        list(pages.keys()),
        index=0
    )
    
    # Add some spacing
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.subheader("📈 System Status")
    
    # Check API keys
    from utils.session_state import check_api_keys_status
    api_status = check_api_keys_status()
    
    if api_status["openai"]:
        st.sidebar.success("✅ OpenAI API")
    else:
        st.sidebar.error("❌ OpenAI API")
        
    if api_status["anthropic"]:
        st.sidebar.success("✅ Anthropic API")
    else:
        st.sidebar.error("❌ Anthropic API")
        
    if api_status["voyage"]:
        st.sidebar.success("✅ Voyage AI API")
    else:
        st.sidebar.warning("⚠️ Voyage AI API (Optional)")
    
    # Data status
    from utils.file_helpers import get_data_status
    data_status = get_data_status()
    
    st.sidebar.markdown("**📂 Data Status:**")
    st.sidebar.write(f"Checklists: {data_status['checklists']} types")
    st.sidebar.write(f"Papers: {data_status['papers']} files")
    st.sidebar.write(f"Results: {data_status['results']} papers")
    
    # Load the selected page
    page_file = pages[selected_page]
    
    try:
        # Import and run the selected page
        page_path = Path(__file__).parent / page_file
        if page_path.exists():
            with open(page_path, 'r') as f:
                page_code = f.read()
            exec(page_code, globals())
        else:
            st.error(f"Page not found: {page_file}")
            st.info("This page is under development.")
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.info("Please check the page implementation.")

if __name__ == "__main__":
    main()
