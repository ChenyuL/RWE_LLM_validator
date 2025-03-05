import pytest
from pathlib import Path
from llm_validation_framework import GuidelineProcessor

def test_process_guidelines_valid_pdf():
    """Test processing of a valid PDF file"""
    processor = GuidelineProcessor({})
    pdf_path = "data/Guidelines/RECORD/RECORD_Checklist.pdf"
    
    chunks = processor.process_guidelines(pdf_path)
    
    assert chunks is not None
    assert len(chunks) > 0
    assert all('content' in chunk for chunk in chunks)
    assert all('metadata' in chunk for chunk in chunks)

def test_process_guidelines_invalid_path():
    """Test handling of invalid PDF path"""
    processor = GuidelineProcessor({})
    pdf_path = "nonexistent.pdf"
    
    with pytest.raises(Exception):
        processor.process_guidelines(pdf_path)

def test_process_guidelines_chunk_size():
    """Test different chunk sizes"""
    processor = GuidelineProcessor({})
    pdf_path = "data/Guidelines/RECORD/RECORD_Checklist.pdf"
    
    # Test with different chunk sizes
    chunks_1000 = processor.process_guidelines(pdf_path, chunk_size=1000)
    chunks_500 = processor.process_guidelines(pdf_path, chunk_size=500)
    
    assert len(chunks_1000) < len(chunks_500)  # Smaller chunk size should result in more chunks
