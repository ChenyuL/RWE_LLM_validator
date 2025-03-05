import pytest
from pathlib import Path
from llm_validation_framework import LLMValidationFramework

class MockEmbeddings:
    def embed_documents(self, texts):
        # Return mock embeddings for testing
        return [[0.1, 0.2, 0.3] for _ in texts]

@pytest.fixture
def mock_api_keys():
    return {
        "openai": "test-key",
        "anthropic": "test-key",
        "deepseek": "test-key"
    }

@pytest.fixture
def framework(mock_api_keys):
    return LLMValidationFramework(mock_api_keys, embeddings=MockEmbeddings())

def test_end_to_end_workflow(framework):
    """Test complete workflow with a sample paper"""
    pdf_path = "data/Guidelines/RECORD/RECORD_Checklist.pdf"
    
    results = framework.process_document(pdf_path)
    
    # Check high-level structure
    assert results['resourceType'] == "Evidence"
    assert results['status'] == "active"
    assert 'statisticalAnalysis' in results
    assert 'certainty' in results
    
    # Check metrics
    metrics = results['statisticalAnalysis']['modelCharacteristics']['value']
    assert 'average_kappa' in metrics
    assert 'average_accuracy' in metrics
    assert 'average_precision' in metrics
    assert 'average_recall' in metrics
    assert 'average_f1' in metrics
    
    # Validate metric ranges
    assert 0 <= metrics['average_kappa'] <= 1
    assert 0 <= metrics['average_accuracy'] <= 1
    assert 0 <= metrics['average_precision'] <= 1
    assert 0 <= metrics['average_recall'] <= 1
    assert 0 <= metrics['average_f1'] <= 1

def test_multiple_papers_processing(framework):
    """Test processing multiple papers"""
    paper_paths = [
        "data/Guidelines/RECORD/RECORD_Checklist.pdf",
        "data/Guidelines/RECORD/RECORD_publication-BMJ.pdf"
    ]
    
    results = []
    for path in paper_paths:
        try:
            result = framework.process_document(path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    # Check structure of successful results
    for result in results:
        assert result['resourceType'] == "Evidence"
        assert 'statisticalAnalysis' in result

def test_error_handling_integration(framework):
    """Test error handling in the complete workflow"""
    
    # Test with nonexistent file
    with pytest.raises(Exception):
        framework.process_document("nonexistent.pdf")
    
    # Test with empty file path
    with pytest.raises(Exception):
        framework.process_document("")
    
    # Test with directory instead of file
    with pytest.raises(Exception):
        framework.process_document("data/Papers")

def test_result_consistency(framework):
    """Test consistency of results across multiple runs"""
    pdf_path = "data/Guidelines/RECORD/RECORD_Checklist.pdf"
    
    # Run the same document twice
    result1 = framework.process_document(pdf_path)
    result2 = framework.process_document(pdf_path)
    
    # Compare key metrics
    metrics1 = result1['statisticalAnalysis']['modelCharacteristics']['value']
    metrics2 = result2['statisticalAnalysis']['modelCharacteristics']['value']
    
    # Results should be identical or very close (allowing for minor floating point differences)
    for key in metrics1.keys():
        assert abs(metrics1[key] - metrics2[key]) < 0.0001

def test_guideline_validation(framework):
    """Test validation against RECORD guidelines"""
    pdf_path = "data/Guidelines/RECORD/RECORD_Checklist.pdf"
    
    # Process the RECORD guidelines
    results = framework.process_document(pdf_path)
    
    # Check if the results contain guideline-specific elements
    assert results['resourceType'] == "Evidence"
    assert 'variableDefinition' in results['statisticalAnalysis']['modelCharacteristics']
    
    # The guidelines should be processed without errors
    assert results['status'] == "active"
