import streamlit as st
from utils.session_state import load_api_keys_from_env, save_api_keys_to_env, check_api_keys_status, initialize_session_state

# Ensure session state is initialized
initialize_session_state()

# Page header
st.header("🔑 API Keys Configuration")
st.markdown("Configure your API keys for different LLM providers. These keys are required for the validation pipeline to work.")

# Load current API keys
load_api_keys_from_env()

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("API Key Settings")
    
    # OpenAI API Key
    st.markdown("**OpenAI API Key** (Required)")
    st.markdown("Used for GPT models (reasoner, extractor)")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_keys.get('openai', ''),
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys",
        label_visibility="collapsed"
    )
    
    # Anthropic API Key
    st.markdown("**Anthropic API Key** (Required)")
    st.markdown("Used for Claude models (validator)")
    anthropic_key = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_keys.get('anthropic', ''),
        type="password",
        help="Get your API key from https://console.anthropic.com/",
        label_visibility="collapsed"
    )
    
    # Voyage AI API Key
    st.markdown("**Voyage AI API Key** (Optional)")
    st.markdown("Used for RAG embeddings (recommended for better performance)")
    voyage_key = st.text_input(
        "Voyage AI API Key",
        value=st.session_state.api_keys.get('voyage', ''),
        type="password",
        help="Get your API key from https://www.voyageai.com/",
        label_visibility="collapsed"
    )
    
    # DeepSeek API Key
    st.markdown("**DeepSeek API Key** (Optional)")
    st.markdown("Used for DeepSeek models (alternative option)")
    deepseek_key = st.text_input(
        "DeepSeek API Key",
        value=st.session_state.api_keys.get('deepseek', ''),
        type="password",
        help="Get your API key from https://platform.deepseek.com/",
        label_visibility="collapsed"
    )
    
    # Save button
    if st.button("💾 Save API Keys", type="primary"):
        api_keys = {
            'openai': openai_key,
            'anthropic': anthropic_key,
            'voyage': voyage_key,
            'deepseek': deepseek_key
        }
        
        try:
            save_api_keys_to_env(api_keys)
            st.success("✅ API keys saved successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error saving API keys: {str(e)}")

with col2:
    st.subheader("Status")
    
    # Check API key status
    api_status = check_api_keys_status()
    
    # Display status for each provider
    providers = [
        ('openai', 'OpenAI', '🤖'),
        ('anthropic', 'Anthropic', '🧠'),
        ('voyage', 'Voyage AI', '🚀'),
        ('deepseek', 'DeepSeek', '🔍')
    ]
    
    for provider_key, provider_name, icon in providers:
        if api_status.get(provider_key, False):
            st.success(f"{icon} {provider_name} ✅")
        else:
            if provider_key in ['openai', 'anthropic']:
                st.error(f"{icon} {provider_name} ❌")
            else:
                st.warning(f"{icon} {provider_name} ⚠️")
    
    # Overall status
    st.markdown("---")
    required_keys = ['openai', 'anthropic']
    all_required_present = all(api_status.get(key, False) for key in required_keys)
    
    if all_required_present:
        st.success("🎉 Ready to proceed!")
        st.markdown("All required API keys are configured.")
    else:
        st.error("⚠️ Missing required keys")
        missing = [key for key in required_keys if not api_status.get(key, False)]
        st.markdown(f"Missing: {', '.join(missing)}")

# Information section
st.markdown("---")
st.subheader("ℹ️ Information")

with st.expander("How to get API keys"):
    st.markdown("""
    **OpenAI API Key:**
    1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
    2. Sign in or create an account
    3. Click "Create new secret key"
    4. Copy the key and paste it above
    
    **Anthropic API Key:**
    1. Go to [Anthropic Console](https://console.anthropic.com/)
    2. Sign in or create an account
    3. Navigate to API Keys section
    4. Create a new key and copy it
    
    **Voyage AI API Key:**
    1. Go to [Voyage AI](https://www.voyageai.com/)
    2. Sign up for an account
    3. Get your API key from the dashboard
    4. This is used for better embeddings in RAG mode
    
    **DeepSeek API Key:**
    1. Go to [DeepSeek Platform](https://platform.deepseek.com/)
    2. Create an account and get your API key
    3. This provides alternative model options
    """)

with st.expander("Security and Privacy"):
    st.markdown("""
    **Security Notes:**
    - API keys are stored locally in a `.env` file
    - Keys are not transmitted to any external services except the respective API providers
    - Use environment variables in production environments
    - Regularly rotate your API keys for security
    
    **Privacy:**
    - Your research papers and data are only sent to the selected LLM providers
    - No data is stored on external servers beyond the API calls
    - Review each provider's privacy policy for their data handling practices
    """)

with st.expander("Cost Estimation"):
    st.markdown("""
    **Typical costs per paper validation:**
    - **OpenAI GPT-4o**: ~$0.50-2.00 per paper
    - **Anthropic Claude**: ~$0.30-1.50 per paper
    - **Voyage AI Embeddings**: ~$0.05-0.20 per paper (RAG mode)
    
    **Factors affecting cost:**
    - Paper length and complexity
    - Number of checklist items
    - Selected models (o3 models cost more)
    - RAG vs. base mode
    
    **Cost optimization tips:**
    - Use smaller models for testing
    - Process papers in batches
    - Use RAG mode for better accuracy with potentially lower costs
    """)

# Test connection section
st.markdown("---")
st.subheader("🔧 Test Connections")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Test OpenAI"):
        if api_status.get('openai', False):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=st.session_state.api_keys['openai'])
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                st.success("✅ OpenAI connection successful!")
            except Exception as e:
                st.error(f"❌ OpenAI connection failed: {str(e)}")
        else:
            st.warning("⚠️ OpenAI API key not configured")

with col2:
    if st.button("Test Anthropic"):
        if api_status.get('anthropic', False):
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=st.session_state.api_keys['anthropic'])
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                st.success("✅ Anthropic connection successful!")
            except Exception as e:
                st.error(f"❌ Anthropic connection failed: {str(e)}")
        else:
            st.warning("⚠️ Anthropic API key not configured")

with col3:
    if st.button("Test Voyage AI"):
        if api_status.get('voyage', False):
            try:
                import voyageai
                client = voyageai.Client(api_key=st.session_state.api_keys['voyage'])
                result = client.embed(
                    "Hello world",
                    model="voyage-3",
                    input_type="document"
                )
                st.success("✅ Voyage AI connection successful!")
            except Exception as e:
                st.error(f"❌ Voyage AI connection failed: {str(e)}")
        else:
            st.warning("⚠️ Voyage AI API key not configured")
