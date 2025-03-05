import pytest
import numpy as np
from llm_validation_framework import Reasoner

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
def reasoner(mock_api_keys):
    return Reasoner(
        models=["model1", "model2"],
        api_keys=mock_api_keys,
        embeddings=MockEmbeddings()
    )

def test_calculate_similarity(reasoner):
    """Test cosine similarity calculation"""
    # Test with simple vectors
    vec1 = [1, 0, 0]
    vec2 = [1, 0, 0]
    similarity = reasoner._calculate_similarity(vec1, vec2)
    assert similarity == pytest.approx(1.0)
    
    # Test orthogonal vectors
    vec1 = [1, 0, 0]
    vec2 = [0, 1, 0]
    similarity = reasoner._calculate_similarity(vec1, vec2)
    assert similarity == pytest.approx(0.0)
    
    # Test with empty vectors
    assert reasoner._calculate_similarity([], []) == 0.0

def test_calculate_content_overlap(reasoner):
    """Test content overlap calculation"""
    # Test identical texts
    text1 = "The quick brown fox"
    text2 = "The quick brown fox"
    overlap = reasoner._calculate_content_overlap(text1, text2)
    assert overlap == 1.0
    
    # Test completely different texts
    text1 = "The quick brown fox"
    text2 = "jumps over lazy dog"
    overlap = reasoner._calculate_content_overlap(text1, text2)
    assert overlap == 0.0
    
    # Test partial overlap
    text1 = "The quick brown fox"
    text2 = "The quick black fox"
    overlap = reasoner._calculate_content_overlap(text1, text2)
    assert 0 < overlap < 1
    
    # Test empty texts
    assert reasoner._calculate_content_overlap("", "") == 0.0

def test_generate_prompts(reasoner):
    """Test prompt generation with context"""
    context = "This is a test context for prompt generation."
    
    results = reasoner.generate_prompts(context)
    
    assert 'primary' in results
    assert 'secondary' in results
    assert 'embeddings' in results
    assert isinstance(results['embeddings'], list)
    assert len(results['embeddings']) == 3  # Mock embeddings length

def test_analyze_differences(reasoner):
    """Test difference analysis between LLM outputs"""
    output1 = {
        'embeddings': [1, 0, 0],
        'text': 'The quick brown fox'
    }
    output2 = {
        'embeddings': [0, 1, 0],
        'text': 'The quick black fox'
    }
    
    metrics = reasoner.analyze_differences(output1, output2)
    
    assert 'similarity_score' in metrics
    assert 'content_overlap' in metrics
    assert 0 <= metrics['similarity_score'] <= 1
    assert 0 <= metrics['content_overlap'] <= 1

def test_error_handling(reasoner):
    """Test error handling in the Reasoner class"""
    
    # Test with None inputs
    with pytest.raises(Exception):
        reasoner.generate_prompts(None)
    
    with pytest.raises(Exception):
        reasoner.analyze_differences(None, None)
    
    # Test with invalid embeddings
    with pytest.raises(Exception):
        reasoner._calculate_similarity([1, 2], [1, 2, 3])  # Different dimensions
