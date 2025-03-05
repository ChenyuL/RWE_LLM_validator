import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory"""
    return Path("data")

@pytest.fixture(scope="session")
def guidelines_dir(test_data_dir):
    """Return the path to the guidelines directory"""
    return test_data_dir / "Guidelines"

@pytest.fixture(scope="session")
def papers_dir(test_data_dir):
    """Return the path to the papers directory"""
    return test_data_dir / "Papers"

@pytest.fixture(scope="session")
def record_guidelines_dir(guidelines_dir):
    """Return the path to the RECORD guidelines directory"""
    return guidelines_dir / "RECORD"

@pytest.fixture(scope="session")
def mock_api_keys():
    """Return mock API keys for testing"""
    return {
        "openai": "test-key",
        "anthropic": "test-key",
        "deepseek": "test-key"
    }

@pytest.fixture(scope="session")
def sample_pdf_content():
    """Return sample PDF content for testing"""
    return """
    Title: Test Paper
    Abstract: This is a test paper for validation framework testing.
    Methods: We conducted a study using routinely collected health data.
    Results: The study showed significant results.
    Conclusion: The framework successfully validated the methodology.
    """

@pytest.fixture(scope="session")
def sample_embeddings():
    """Return sample embeddings for testing"""
    return [0.1, 0.2, 0.3, 0.4, 0.5]

@pytest.fixture(scope="session")
def sample_validation_decisions():
    """Return sample validation decisions for testing"""
    return {
        'decisions': [1, 1, 0, 1, 0]
    }
