# Enhanced RWE LLM Validator

A comprehensive multi-agent framework for automated validation of reporting checklist compliance in observational studies, featuring both base and RAG-enhanced agents with an intuitive Streamlit interface.

## 🚀 Features

### Multi-Agent Architecture
- **Reasoner (LLM1)**: Processes guideline documents to extract checklist items and generate prompts
- **Extractor (LLM2)**: Extracts information from research papers based on checklist prompts
- **Validator (LLM3)**: Validates extracted information and provides final compliance assessment

### Agent Types
- **Base Agents**: Standard LLM processing for fast, cost-effective analysis
- **RAG Agents**: Retrieval-Augmented Generation for enhanced accuracy and context understanding

### Supported Checklists
- **RECORD**: REporting of studies Conducted using Observational Routinely-collected Data
- **STROBE**: STrengthening the Reporting of OBservational studies in Epidemiology
- **Li-Paper**: Custom checklist for specific research domains
- **Custom**: Upload your own reporting guidelines

### User Interface
- **Streamlit Web App**: Intuitive interface for configuration and execution
- **Command Line**: Flexible CLI for batch processing and automation

## 📁 Project Structure

```
RWE_LLM_validator/
├── app/                          # Streamlit web application
│   ├── main.py                   # Main app entry point
│   ├── pages/                    # Individual app pages
│   │   ├── 01_api_keys.py       # API key configuration
│   │   ├── 02_upload_data.py    # Data upload interface
│   │   ├── 03_configure_agents.py # Agent configuration
│   │   ├── 04_run_validation.py # Validation execution
│   │   └── 05_view_results.py   # Results visualization
│   ├── utils/                    # Utility modules
│   │   ├── session_state.py     # Session state management
│   │   └── file_helpers.py      # File operations
│   └── data/                     # App-specific data
│       └── custom_prompts/       # User-saved prompts
├── src/                          # Core framework
│   ├── core/                     # Enhanced pipeline
│   │   └── pipeline.py          # Main pipeline implementation
│   ├── agents/                   # Agent implementations
│   │   ├── reasoner_modified.py # Base reasoner
│   │   ├── reasoner_rag.py      # RAG reasoner
│   │   ├── extractor.py         # Base extractor
│   │   └── validator.py         # Base validator
│   ├── utils/                    # Utility modules
│   │   └── pdf_processor.py     # PDF text extraction
│   └── config.py                # Configuration settings
├── data/                         # Input data
│   ├── Guidelines/              # Checklist guidelines
│   │   ├── RECORD/             # RECORD guideline PDFs
│   │   ├── STROBE/             # STROBE guideline PDFs
│   │   └── Li-Paper/           # Li-Paper guideline PDFs
│   └── Papers/                  # Research papers to validate
├── output/                       # Results and outputs
│   ├── paper_results/           # Individual paper results
│   └── prompts/                 # Generated prompts
├── archive/                      # Legacy implementations
├── run_validation.py            # Enhanced CLI runner
└── requirements.txt             # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- API keys for:
  - OpenAI (required)
  - Anthropic (required)
  - Voyage AI (optional, for better RAG performance)
  - DeepSeek (optional)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RWE_LLM_validator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   VOYAGE_API_KEY=your_voyage_api_key  # Optional
   DEEPSEEK_API_KEY=your_deepseek_api_key  # Optional
   ```

4. **Prepare data directories:**
   ```bash
   mkdir -p data/Guidelines/RECORD
   mkdir -p data/Guidelines/STROBE
   mkdir -p data/Papers
   mkdir -p output
   ```

## 🖥️ Usage

### Streamlit Web Interface (Recommended)

1. **Start the application:**
   ```bash
   streamlit run app/main.py
   ```

2. **Navigate through the interface:**
   - **API Keys**: Configure your API keys
   - **Upload Data**: Upload checklists and papers
   - **Configure Agents**: Select agent types, models, and customize prompts
   - **Run Validation**: Execute the validation pipeline
   - **View Results**: Analyze and visualize results

### Command Line Interface

1. **Full pipeline (recommended for first-time users):**
   ```bash
   python run_validation.py --mode full --paper data/Papers/example.pdf --checklist RECORD
   ```

2. **Reasoner only (generate prompts):**
   ```bash
   python run_validation.py --mode reasoner --checklist RECORD --output output/
   ```

3. **Extractor mode (use existing prompts):**
   ```bash
   python run_validation.py --mode extractor --paper data/Papers/example.pdf --prompts output/prompts.json --checklist RECORD
   ```

4. **With custom configuration:**
   ```bash
   python run_validation.py --mode full --paper data/Papers/example.pdf --checklist RECORD --config config.json
   ```

## ⚙️ Configuration

### Agent Configuration

Create a configuration file (JSON format) to specify agent types and models:

```json
{
  "reasoner": {
    "type": "rag",
    "model": "o3-mini-2025-01-31"
  },
  "extractor": {
    "type": "rag",
    "model": "gpt-4o",
    "custom_prompt": "Your custom extraction prompt..."
  },
  "validator": {
    "type": "base",
    "model": "claude-3-5-sonnet-20241022",
    "custom_prompt": "Your custom validation prompt..."
  },
  "rag_config": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "similarity_threshold": 0.7,
    "embedding_model": "voyage-3"
  }
}
```

### Agent Types

- **Base Agents**: Standard LLM processing
  - Faster execution
  - Lower cost
  - Good for standard papers

- **RAG Agents**: Retrieval-Augmented Generation
  - Better accuracy
  - Context-aware processing
  - Higher cost but better results

### Model Options

**Reasoner Models:**
- OpenAI: o3-mini, o1, o1-mini, gpt-4o
- Anthropic: claude-3-5-sonnet, claude-3-opus
- DeepSeek: deepseek-reasoner

**Extractor Models:**
- OpenAI: gpt-4o, gpt-4o-mini, gpt-4.5-preview
- Anthropic: claude-3-5-sonnet, claude-3-opus
- DeepSeek: deepseek-chat

**Validator Models:**
- OpenAI: gpt-4o, gpt-4o-mini
- Anthropic: claude-3-5-sonnet, claude-3-opus
- DeepSeek: deepseek-chat

## 📊 Results

### Output Structure

For each processed paper, the system generates:

```
output/paper_results/PAPER_ID_CHECKLIST/
├── TIMESTAMP_reasoner_PAPER_ID_CHECKLIST.json      # Generated prompts
├── TIMESTAMP_extractor_PAPER_ID_CHECKLIST.json     # Extracted information
├── TIMESTAMP_validator_PAPER_ID_CHECKLIST.json     # Validation results
├── TIMESTAMP_report_PAPER_ID_CHECKLIST.json        # Final report
└── TIMESTAMP_full_CHECKLIST_checklist_PAPER_ID.json # Complete checklist
```

### Result Components

- **Reasoner Output**: Generated prompts for each checklist item
- **Extractor Output**: Information extracted from the paper with evidence
- **Validator Output**: Validation of extracted information with reasoning
- **Final Report**: Comprehensive analysis with compliance assessment
- **Full Checklist**: Complete checklist with final answers

## 🔧 Advanced Features

### Custom Prompts

Customize extraction and validation prompts for specific domains:

```python
# Example custom extractor prompt
extractor_prompt = """
You are an expert in {checklist_type} guidelines for epidemiological research.

Analyze the following paper section for compliance with item {item_id}:
{item_description}

Paper text: {paper_text}

Provide specific evidence and clear reasoning for your assessment.
Focus on methodological rigor and reporting completeness.
"""
```

### RAG Configuration

Fine-tune RAG parameters for optimal performance:

- **Chunk Size**: 500-2000 characters (default: 1000)
- **Chunk Overlap**: 50-500 characters (default: 200)
- **Top-K**: 1-20 chunks (default: 5)
- **Similarity Threshold**: 0.1-0.9 (default: 0.7)

### Batch Processing

Process multiple papers efficiently:

```bash
# Process all papers in directory
for paper in data/Papers/*.pdf; do
    python run_validation.py --mode full --paper "$paper" --checklist RECORD
done
```

## 📈 Performance Optimization

### Cost Optimization
- Use smaller models for testing (gpt-4o-mini, claude-3-haiku)
- Process papers in reasoner → extractor mode for batch processing
- Use RAG selectively for complex papers

### Accuracy Optimization
- Use RAG agents for better context understanding
- Customize prompts for specific domains
- Use o3/o1 models for complex reasoning tasks

### Speed Optimization
- Use base agents for faster processing
- Process papers in parallel (manual implementation)
- Cache embeddings for repeated analysis

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify API keys in `.env` file
   - Check API key permissions and credits

2. **PDF Processing Errors**
   - Ensure PDFs are text-readable (not scanned images)
   - Check file permissions and paths

3. **Memory Issues**
   - Reduce batch size for large papers
   - Use smaller chunk sizes for RAG processing

4. **Model Errors**
   - Verify model names and availability
   - Check API provider status

### Debug Mode

Enable detailed logging:

```bash
python run_validation.py --mode full --paper example.pdf --checklist RECORD --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{rwe_llm_validator,
  title={Enhanced RWE LLM Validator: Multi-Agent Framework for Automated Checklist Compliance},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/RWE_LLM_validator}
}
```

## 🆘 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

## 🔄 Version History

### v2.0.0 (Enhanced Version)
- Added RAG-enhanced agents
- Streamlit web interface
- Improved configuration system
- Better result visualization
- Enhanced error handling

### v1.0.0 (Original Version)
- Basic three-agent pipeline
- RECORD checklist support
- Command-line interface
- Basic result generation
