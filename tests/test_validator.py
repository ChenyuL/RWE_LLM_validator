import pytest
from llm_validation_framework import Validator

@pytest.fixture
def mock_api_keys():
    return {
        
        "openai": "test-key",
        "anthropic": "test-key",
        "deepseek": "test-key"
    }

@pytest.fixture
def validator(mock_api_keys):
    return Validator(
        models=["claude-3.5", "gpt-4"],
        api_keys=mock_api_keys
    )

def test_calculate_metrics(validator):
    """Test validation metrics calculation"""
    # Test perfect agreement
    validation1 = {'decisions': [1, 1, 1, 0, 0]}
    validation2 = {'decisions': [1, 1, 1, 0, 0]}
    metrics = validator.calculate_metrics(validation1, validation2)
    
    assert metrics['kappa_score'] == 1.0
    assert metrics['accuracy'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1'] == 1.0
    
    # Test no agreement
    validation1 = {'decisions': [1, 1, 1, 1, 1]}
    validation2 = {'decisions': [0, 0, 0, 0, 0]}
    metrics = validator.calculate_metrics(validation1, validation2)
    
    assert metrics['kappa_score'] == -1.0
    assert metrics['accuracy'] == 0.0
    
    # Test partial agreement
    validation1 = {'decisions': [1, 1, 1, 0, 0]}
    validation2 = {'decisions': [1, 1, 0, 0, 0]}
    metrics = validator.calculate_metrics(validation1, validation2)
    
    assert -1.0 <= metrics['kappa_score'] <= 1.0
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['recall'] <= 1.0
    assert 0.0 <= metrics['f1'] <= 1.0

def test_calculate_agreement_level(validator):
    """Test agreement level calculation based on kappa score"""
    assert validator._calculate_agreement_level(0.9) == "Almost Perfect Agreement"
    assert validator._calculate_agreement_level(0.7) == "Substantial Agreement"
    assert validator._calculate_agreement_level(0.5) == "Moderate Agreement"
    assert validator._calculate_agreement_level(0.3) == "Fair Agreement"
    assert validator._calculate_agreement_level(0.1) == "Slight Agreement"

def test_generate_fhir_output(validator):
    """Test FHIR output generation"""
    original_data = {"test": "data"}
    validation1 = {"result": "pass"}
    validation2 = {"result": "pass"}
    metrics = {
        'kappa_score': 0.8,
        'accuracy': 0.9,
        'precision': 0.85,
        'recall': 0.87,
        'f1': 0.86
    }
    
    fhir_output = validator.generate_fhir_output(
        original_data,
        validation1,
        validation2,
        metrics
    )
    
    # Check FHIR structure
    assert fhir_output['resourceType'] == "Evidence"
    assert fhir_output['status'] == "active"
    assert 'statisticalAnalysis' in fhir_output
    assert 'modelCharacteristics' in fhir_output['statisticalAnalysis']
    assert 'variableDefinition' in fhir_output
    assert 'certainty' in fhir_output
    
    # Check metrics are included
    stats = fhir_output['statisticalAnalysis']['modelCharacteristics']['value']
    assert all(metric in metrics for metric in stats.keys())
    
    # Check agreement level is appropriate
    certainty = fhir_output['certainty'][0]['rating']['value']
    assert certainty == "Almost Perfect Agreement"

def test_validate_extracted_info(validator):
    """Test validation of extracted information"""
    extracted_info = {
        'content': 'Test content',
        'reasoning': {
            'primary': {'text': 'Primary analysis'},
            'secondary': {'text': 'Secondary analysis'}
        }
    }
    
    result = validator.validate(extracted_info)
    
    # Check FHIR structure
    assert result['resourceType'] == "Evidence"
    assert result['status'] == "active"
    assert 'statisticalAnalysis' in result
    assert 'variableDefinition' in result
    assert 'certainty' in result

def test_error_handling(validator):
    """Test error handling in validation process"""
    
    # Test with invalid validation data
    with pytest.raises(ValueError):
        validator.calculate_metrics(
            {'decisions': []},  # Empty decisions
            {'decisions': [1, 0]}  # Non-empty decisions
        )
    
    # Test with missing validation data
    with pytest.raises(Exception):
        validator.validate({})  # Empty extracted info
    
    # Test with invalid metrics
    with pytest.raises(Exception):
        validator.generate_fhir_output(
            {},  # original data
            {},  # validation1
            {},  # validation2
            {}   # empty metrics
        )
