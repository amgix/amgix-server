# Backend-Agnostic Test Framework

This framework allows you to write tests that automatically adapt to different backend capabilities without duplicating code. It's designed to handle the case where some backends support dense vectors while others only support sparse vectors.

## Key Features

- **Simple Capability Detection**: Just specify whether dense vectors are supported
- **Test Data Factories**: Generate appropriate test configurations based on backend support
- **Pytest Markers**: Mark tests to run only on specific backend types
- **No Code Duplication**: Single test logic that adapts to different backends
- **Flexible Execution**: Run tests for specific backend capabilities

## How It Works

### 1. Simple Capability Detection

The framework simply checks if dense vectors are enabled:

```python
@pytest.fixture(scope="session")
def dense_vectors_enabled(request):
    """Determine if dense vectors are enabled for testing."""
    if request.config.getoption("--no-dense-vectors"):
        return False
    elif request.config.getoption("--dense-vectors"):
        return True
    else:
        # Default: assume no dense vectors (for older MariaDB/MySQL)
        return False
```

### 2. Test Data Factory

The `TestDataFactory` class generates appropriate test configurations:

```python
class TestDataFactory:
    def create_vector_configs(self, test_type: str = "basic") -> List[VectorConfig]:
        """Create vector configurations based on backend capabilities."""
        if test_type == "basic":
            # Always include sparse vectors
            if self.supports_sparse:
                configs.append(VectorConfig(
                    name="trigrams",
                    type=VectorType.TRIGRAMS,
                    index_fields=["name", "content"]
                ))
        
        elif test_type == "advanced":
            # Include dense vectors if supported
            if self.supports_dense:
                configs.append(VectorConfig(
                    name="embeddings",
                    type=VectorType.DENSE_MODEL,
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    index_fields=["name", "content"]
                ))
        
        return configs
```

### 3. Pytest Markers

Use markers to control which tests run on which backends:

```python
@pytest.mark.all_backends
def test_basic_functionality():
    """This test runs on ALL backends."""
    pass

@pytest.mark.dense_vectors_only
def test_dense_vector_features():
    """This test only runs on backends with dense vector support."""
    pass

@pytest.mark.sparse_only
def test_sparse_only_features():
    """This test only runs on backends without dense vector support."""
    pass
```

## Usage Examples

### Running Tests

#### 1. Backend with Dense Vector Support
```bash
# Run all tests including dense vector tests
pytest tests/ --dense-vectors -v

# Or just run pytest (default behavior)
pytest tests/ -v
```

#### 2. Backend WITHOUT Dense Vector Support (e.g., MariaDB < 11.6)
```bash
# Run all tests EXCEPT dense vector specific tests
pytest tests/ --nodense -v
```

#### 3. Selective Test Execution
```bash
# Run only tests that work on all backends
pytest tests/ -m "all_backends" -v

# Run only dense vector tests (will skip on sparse-only backends)
pytest tests/ -m "dense_vectors_only" -v

# Run only sparse-only tests (will skip on dense vector backends)
pytest tests/ -m "sparse_only" -v
```

### Writing Tests

#### 1. Tests for All Backends
```python
@pytest.mark.all_backends
def test_basic_search_functionality(setup_collection, test_data_factory):
    """Test basic search functionality that works on all backends."""
    collection_name = setup_collection
    
    # Add test documents
    doc = create_test_document("doc1", "Python", "Python is a programming language")
    
    # Upsert document
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
    assert response.status_code == 200
    
    # Create search query based on backend capabilities
    search_query = test_data_factory.create_search_query("basic", "Python")
    
    # Execute search
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
```

#### 2. Tests for Dense Vector Backends Only
```python
@pytest.mark.dense_vectors_only
def test_dense_vector_search(setup_collection, test_data_factory, backend_capabilities):
    """Test dense vector search functionality."""
    from tests.conftest import skip_if_not_supported
    
    # Skip if backend doesn't support dense vectors
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    collection_name = setup_collection
    
    # Use advanced config that includes dense vectors
    # The setup_collection fixture automatically adapts to backend capabilities
    
    # Test dense vector specific functionality
    search_query = test_data_factory.create_search_query("advanced", "query")
    # ... test logic
```

#### 3. Tests for Sparse-Only Backends
```python
@pytest.mark.sparse_only
def test_sparse_only_features(setup_collection, test_data_factory, backend_capabilities):
    """Test features that are specific to sparse-only backends."""
    # Skip if backend supports dense vectors
    if backend_capabilities.get("dense_vectors", False):
        pytest.skip("Backend supports dense vectors, testing sparse-only features")
    
    collection_name = setup_collection
    
    # Test sparse-only functionality
    # This ensures sparse-only backends work correctly
```

### 4. Parameterized Tests
```python
@pytest.mark.parametrize("setup_collection", ["basic", "advanced", "sparse_only"], indirect=True)
@pytest.mark.all_backends
def test_different_collection_configs(setup_collection, test_data_factory, backend_capabilities):
    """Test different collection configurations automatically adapted to backend capabilities."""
    collection_name = setup_collection
    
    # The setup_collection fixture automatically creates the right configuration
    # based on what the backend supports
    
    # Get expected features for this configuration
    test_type = getattr(setup_collection, 'param', 'basic')
    expected_features = test_data_factory.get_expected_features(test_type)
    
    # Test based on expected capabilities
    if expected_features["sparse_search"]:
        # Test sparse search
        pass
    
    if expected_features["dense_search"]:
        # Test dense search
        pass
```

## Configuration Options

### Command Line Options

- `--dense`: Enable dense vector support for testing
- `--nodense`: Disable dense vector support for testing (sparse-only)

### Default Behavior

If neither flag is specified, the framework defaults to **enabling dense vectors**, which means all tests will run by default. This gives you complete test coverage when you just run `pytest` without any parameters.

To run only sparse-only tests (for backends without dense vector support), explicitly use `--no-dense-vectors`.

## Test Data Factory Methods

### Vector Configurations

```python
# Basic configuration (sparse vectors only)
config = test_data_factory.create_vector_configs("basic")

# Advanced configuration (sparse + dense if supported)
config = test_data_factory.create_vector_configs("advanced")

# Sparse-only configuration
config = test_data_factory.create_vector_configs("sparse_only")

# Dense-only configuration
config = test_data_factory.create_vector_configs("dense_only")
```

### Collection Configurations

```python
# Create collection config based on backend capabilities
collection_config = test_data_factory.create_collection_config("basic")
```

### Search Queries

```python
# Create search query based on backend capabilities
search_query = test_data_factory.create_search_query("basic", "search text")
search_query = test_data_factory.create_search_query("advanced", "search text")
```

### Expected Features

```python
# Get expected features for a test type
expected_features = test_data_factory.get_expected_features("basic")
expected_features = test_data_factory.get_expected_features("advanced")

# Check specific features
if expected_features["hybrid_search"]:
    # Test hybrid search
    pass
```

## Helper Functions

### Skip Tests Based on Capabilities

```python
from tests.conftest import skip_if_not_supported

def test_feature(backend_capabilities):
    skip_if_not_supported(backend_capabilities, "dense_vectors", "Backend doesn't support dense vectors")
    # Test dense vector functionality
```

## Best Practices

1. **Use the `@pytest.mark.all_backends` marker** for tests that should work on all backends
2. **Use the `@pytest.mark.dense_vectors_only` marker** for dense vector specific tests
3. **Use the `@pytest.mark.sparse_only` marker** for sparse-only specific tests
4. **Always use the `test_data_factory`** to create test configurations
5. **Use the `setup_collection` fixture** for collection setup
6. **Check backend capabilities** before testing specific features
7. **Use parameterized tests** for testing different configurations

## Migration from Existing Tests

To migrate existing tests to use this framework:

1. **Replace hardcoded vector configurations** with `test_data_factory.create_vector_configs()`
2. **Add appropriate pytest markers** to control test execution
3. **Use the `setup_collection` fixture** instead of manual collection setup
4. **Replace hardcoded search queries** with `test_data_factory.create_search_query()`
5. **Add capability checks** for feature-specific tests

## Example Migration

### Before (Hardcoded)
```python
def test_search():
    config = {
        "vectors": [
            {"name": "trigrams", "type": "trigrams", "index_fields": ["content"]},
            {"name": "embeddings", "type": "dense_model", "model": "model-name"}
        ]
    }
    # ... test logic
```

### After (Backend-Agnostic)
```python
@pytest.mark.all_backends
def test_search(setup_collection, test_data_factory):
    collection_name = setup_collection  # Automatically configured
    
    # Create search query based on backend capabilities
    search_query = test_data_factory.create_search_query("basic", "search text")
    
    # ... test logic that works on all backends
```

This framework gives you complete test coverage for all backend types without code duplication, while maintaining the flexibility to test backend-specific features when needed.
