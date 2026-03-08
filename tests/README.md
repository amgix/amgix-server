# Testing Guide

This directory contains the API integration test suite for the project.

## Structure

```
tests/
├── test_api_integration.py    # Comprehensive API integration tests
├── test_validation.py         # Model validation tests
├── test_model_validation.py   # Additional model validation tests
├── test_simple_api.py         # Simple API endpoint tests
├── requirements.txt           # Test dependencies
└── README.md                 # This file
```

## Running Tests

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run all tests
```bash
pytest
```

### Run specific test files
```bash
# Run comprehensive API integration tests
pytest test_api_integration.py -v

# Run validation tests
pytest test_validation.py -v

# Run model validation tests
pytest test_model_validation.py -v

# Run simple API tests
pytest test_simple_api.py -v
```

### Run tests with coverage
```bash
pytest --cov=src
```

## Test Categories

### API Integration Tests (`test_api_integration.py`)
- **Purpose**: End-to-end testing of the complete API
- **Speed**: Slower (seconds to minutes)
- **Dependencies**: External services (Qdrant, MariaDB, RabbitMQ)
- **Coverage**:
  - Collection management (create, list, delete, empty)
  - Document operations (add, update, delete, status)
  - Vector search (all vector types and combinations)
  - Document type filtering
  - Queue processing and status tracking

### Validation Tests
- **`test_validation.py`**: API request/response validation
- **`test_model_validation.py`**: Pydantic model validation
- **`test_simple_api.py`**: Basic API endpoint functionality

## Prerequisites

Before running tests, ensure these services are running:
- **Qdrant**: Vector database on port 6334
- **MariaDB**: SQL database (if testing SQL backend)
- **RabbitMQ**: Message broker for queue processing
- **API Server**: FastAPI server on port 8234

## Test Configuration

Tests use the following configuration:
- **API Base URL**: `http://localhost:8234`
- **Database Backends**: Both Qdrant and MariaDB are tested
- **Vector Models**: Various HuggingFace models for different vector types
- **Test Collections**: Automatically created and cleaned up

## Writing Tests

### Test Structure
```python
def test_specific_functionality():
    """Test description explaining what is being tested."""
    # Arrange: Set up test data and conditions
    collection_name = "test_collection"
    
    # Act: Perform the operation being tested
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
    
    # Assert: Verify the expected outcome
    assert response.status_code == 200
    assert response.json() == expected_result
```

### Using Test Fixtures
```python
@pytest.fixture(scope="function")
def setup_collection(request):
    """Create a test collection with vector configurations."""
    # Collection setup logic
    yield collection_name
    
    # Cleanup logic
    try:
        requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
    except Exception:
        pass
```

## Best Practices

1. **Test names**: Use descriptive names that explain what is being tested
2. **Arrange-Act-Assert**: Structure tests in three clear sections
3. **One assertion per test**: Each test should verify one specific behavior
4. **Clean up**: Always clean up after tests using fixtures
5. **Error handling**: Include proper error handling and cleanup in tests
6. **Documentation**: Add clear docstrings explaining test purpose and setup 