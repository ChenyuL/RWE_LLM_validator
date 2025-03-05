# LLM Validation Framework Test Suite

This directory contains the test suite for the LLM Validation Framework. The tests are organized to validate different components of the framework and ensure proper integration.

## Test Structure

- `test_guideline_processor.py`: Tests for PDF processing and guideline extraction
- `test_reasoner.py`: Tests for the reasoning component that handles LLM interactions
- `test_validator.py`: Tests for validation logic and FHIR output generation
- `test_integration.py`: End-to-end tests using real papers and guidelines
- `conftest.py`: Shared pytest fixtures and utilities

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Running All Tests

```bash
pytest
```

### Running Specific Test Categories

Run unit tests only:
```bash
pytest -m unit
```

Run integration tests only:
```bash
pytest -m integration
```

Run tests that don't require API access:
```bash
pytest -m "not api"
```

### Test Coverage

Generate coverage report:
```bash
pytest --cov=llm_validation_framework --cov-report=html
```

The HTML coverage report will be available in the `htmlcov` directory.

## Test Data

The tests use sample data from:
- `data/Guidelines/RECORD/`: RECORD guidelines and checklists
- `data/Papers/`: Sample research papers for validation

## Adding New Tests

When adding new tests:

1. Follow the existing file naming convention: `test_*.py`
2. Use appropriate markers for test categorization
3. Add fixtures to `conftest.py` if they can be reused
4. Update this README if adding new test categories

## Mocking

The test suite uses pytest-mock for mocking API calls and external dependencies. When adding tests that require API access:

1. Use the `mock_api_keys` fixture for API credentials
2. Mark tests requiring API access with `@pytest.mark.api`
3. Implement graceful fallbacks when API access fails

## Continuous Integration

The test suite is configured to run in CI environments. The following aspects are automatically checked:

- Code formatting (black)
- Import sorting (isort)
- Type checking (mypy)
- Code style (flake8)
- Test coverage (pytest-cov)

## Troubleshooting

Common issues and solutions:

1. API rate limiting:
   - Use mock responses for development
   - Implement proper retry logic in tests

2. PDF processing issues:
   - Ensure test PDFs are accessible
   - Check PDF file permissions

3. Coverage reporting:
   - Clear `.coverage` file if results seem incorrect
   - Ensure all source files are included in coverage

## Contributing

When contributing new tests:

1. Follow the existing code style
2. Add appropriate documentation
3. Include both positive and negative test cases
4. Consider edge cases and error conditions
5. Update requirements if adding new dependencies
