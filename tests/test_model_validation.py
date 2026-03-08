import pytest
import pytest
import requests
from typing import Dict, Any
from tests.conftest import skip_if_not_supported

API_BASE_URL = "http://localhost:8234/v1"
COLLECTION_NAME = "test_model_validation"


@pytest.mark.dense_vectors_only
def test_create_collection_with_dense_model_no_dimensions(backend_capabilities):
    """Test creating a collection with dense model without specifying dimensions."""
    
    # Skip if backend doesn't support dense vectors
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Create collection config with dense model (no dimensions specified)
    config = {
        "vectors": [
            {
                "name": "dense_test",
                "type": "dense_model",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should automatically discover dimensions via encoder
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_no_dimensions", json=config)
    assert response.status_code == 200
    
    # Clean up
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_no_dimensions")


@pytest.mark.all_backends
def test_create_collection_with_sparse_model():
    """Test creating a collection with sparse model (no dimensions needed)."""
    
    # Create collection config with sparse model
    config = {
        "vectors": [
            {
                "name": "sparse_test",
                "type": "sparse_model",
                "model": "prithivida/Splade_PP_en_v1",
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should validate model loads
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_sparse", json=config)
    assert response.status_code == 200
    
    # Clean up
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_sparse")


@pytest.mark.dense_vectors_only
def test_create_collection_with_invalid_model(backend_capabilities):
    """Test creating a collection with an invalid model name."""
    
    # Skip if backend doesn't support dense vectors
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Create collection config with invalid model
    config = {
        "vectors": [
            {
                "name": "invalid_test",
                "type": "dense_model",
                "model": "this-model-does-not-exist",
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should fail validation
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_invalid", json=config)
    assert response.status_code == 400
    # Handle both string and list error response formats
    error_detail = response.json()["detail"]
    if isinstance(error_detail, list):
        # Handle list of dictionaries format
        if error_detail and isinstance(error_detail[0], dict):
            # Extract error messages from dict structure
            error_detail = " ".join([str(item.get("msg", item)) for item in error_detail])
        else:
            # Handle list of strings
            error_detail = " ".join(error_detail)
    assert "Model validation failed" in error_detail


@pytest.mark.dense_vectors_only
def test_create_collection_with_mixed_vectors(backend_capabilities):
    """Test creating a collection with multiple vector types."""
    
    # Skip if backend doesn't support dense vectors
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Create collection config with mixed vector types
    config = {
        "vectors": [
            {
                "name": "dense_test",
                "type": "dense_model",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "index_fields": ["content"]
            },
            {
                "name": "sparse_test",
                "type": "sparse_model",
                "model": "prithivida/Splade_PP_en_v1",
                "index_fields": ["content"]
            },
            {
                "name": "trigrams_test",
                "type": "trigrams",
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should validate all models
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_mixed", json=config)
    assert response.status_code == 200
    
    # Clean up
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_mixed")


@pytest.mark.dense_vectors_only
def test_create_collection_with_dimensions_specified(backend_capabilities):
    """Test that API accepts dimensions field for DENSE_MODEL (will be validated during vectorization)."""
    
    # Skip if backend doesn't support dense vectors
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Create collection config with dimensions specified (this should be accepted)
    config = {
        "vectors": [
            {
                "name": "dense_test",
                "type": "dense_model",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,  # This should be accepted by API
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should succeed (dimensions are allowed for DENSE_MODEL)
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_with_dimensions", json=config)
    assert response.status_code == 200  # Success - dimensions are allowed
    
    # Clean up
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_with_dimensions")


@pytest.mark.dense_vectors_only
def test_create_collection_with_mismatched_dimensions_fails(backend_capabilities):
    """Test that API rejects dimensions that don't match the actual model dimensions."""
    
    # Skip if backend doesn't support dense vectors
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Create collection config with wrong dimensions (this should fail during vectorization)
    config = {
        "vectors": [
            {
                "name": "dense_test",
                "type": "dense_model",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 999,  # Wrong dimensions - should fail during vectorization
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should fail during model validation (dimensions mismatch)
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_wrong_dimensions", json=config)
    assert response.status_code == 400  # Model validation failed
    assert "Model validation failed" in response.json()["detail"]
    # Handle both string and list error response formats
    error_detail = response.json()["detail"]
    if isinstance(error_detail, list):
        error_detail = " ".join(error_detail)
    assert "dimensions" in error_detail.lower()  # Should mention dimensions in error
    
    # Clean up (just in case)
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_wrong_dimensions")


@pytest.mark.all_backends
@pytest.mark.dense_vectors_only
def test_create_collection_with_custom_vectors_validation_mode(backend_capabilities):
    """Test that custom vector validation works correctly in validation mode."""
    
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Create collection config with valid DENSE_CUSTOM and SPARSE_CUSTOM vectors
    # Note: These are just configs - the actual custom vectors would be provided when uploading documents
    config = {
        "vectors": [
            {
                "name": "custom_dense",
                "type": "dense_custom",
                "dimensions": 384,
                "index_fields": ["content"]
            },
            {
                "name": "custom_sparse",
                "type": "sparse_custom",
                "top_k": 100,
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should succeed (all custom vector configs are valid)
    # The system validates the config structure, not the actual vectors
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_valid", json=config)
    assert response.status_code == 200  # Success - all custom vector configs are valid
    
    # Clean up
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_valid")


@pytest.mark.all_backends
@pytest.mark.dense_vectors_only
def test_create_collection_with_custom_vectors_rejects_model(backend_capabilities):
    """Test that API rejects model field for custom vector types (this validation actually exists)."""
    
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Create collection config with DENSE_CUSTOM but with model field (this should fail validation)
    config = {
        "vectors": [
            {
                "name": "custom_dense",
                "type": "dense_custom",
                "dimensions": 384,
                "model": "sentence-transformers/all-MiniLM-L6-v2",  # Should not be allowed
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should fail validation (model not allowed for DENSE_CUSTOM)
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_with_model", json=config)
    assert response.status_code == 422  # Validation error
    # Handle both string and list error response formats
    error_detail = response.json()["detail"]
    if isinstance(error_detail, list):
        # Handle list of dictionaries format
        if error_detail and isinstance(error_detail[0], dict):
            # Extract error messages from dict structure
            error_detail = " ".join([str(item.get("msg", item)) for item in error_detail])
        else:
            # Handle list of strings
            error_detail = " ".join(error_detail)
    assert "model" in error_detail.lower()  # Should mention model in error
    
    # Clean up (just in case)
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_with_model")


@pytest.mark.all_backends
@pytest.mark.dense_vectors_only
def test_create_collection_with_dense_custom_requires_dimensions(backend_capabilities):
    """Test that API requires dimensions for DENSE_CUSTOM vector type (this validation actually exists)."""
    
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Create collection config with DENSE_CUSTOM but no dimensions (this should fail validation)
    config = {
        "vectors": [
            {
                "name": "custom_dense",
                "type": "dense_custom",
                "index_fields": ["content"]
                # Missing dimensions - should fail validation
            }
        ]
    }
    
    # Create collection - should fail validation (dimensions required for DENSE_CUSTOM)
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_no_dimensions", json=config)
    assert response.status_code == 422  # Validation error
    # Handle both string and list error response formats
    error_detail = response.json()["detail"]
    if isinstance(error_detail, list):
        # Handle list of dictionaries format
        if error_detail and isinstance(error_detail[0], dict):
            # Extract error messages from dict structure
            error_detail = " ".join([str(item.get("msg", item)) for item in error_detail])
        else:
            # Handle list of strings
            error_detail = " ".join(error_detail)
    assert "dimensions" in error_detail.lower()  # Should mention dimensions in error
    
    # Clean up (just in case)
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_no_dimensions")


@pytest.mark.all_backends
def test_create_collection_with_sparse_custom_top_k_validation():
    """Test that SPARSE_CUSTOM top_k validation works for invalid values and defaults are applied."""
    
    # Test 1: SPARSE_CUSTOM with valid top_k should succeed
    config_valid = {
        "vectors": [
            {
                "name": "custom_sparse",
                "type": "sparse_custom",
                "top_k": 100,
                "index_fields": ["content"]
            }
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_valid_top_k", json=config_valid)
    assert response.status_code == 200  # Success - valid top_k
    
    # Clean up
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_valid_top_k")
    
    # Test 2: SPARSE_CUSTOM with no top_k should use default and succeed
    config_default = {
        "vectors": [
            {
                "name": "custom_sparse",
                "type": "sparse_custom",
                "index_fields": ["content"]
                # No top_k specified - should use default value
            }
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_default_top_k", json=config_default)
    assert response.status_code == 200  # Success - default top_k used
    
    # Clean up
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_default_top_k")
    
    # Test 3: SPARSE_CUSTOM with invalid top_k should fail validation
    config_invalid = {
        "vectors": [
            {
                "name": "custom_sparse",
                "type": "sparse_custom",
                "top_k": 0,  # Invalid - must be positive
                "index_fields": ["content"]
            }
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_invalid_top_k", json=config_invalid)
    assert response.status_code == 422  # Validation error - top_k must be positive
    
    # Clean up (just in case)
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_custom_invalid_top_k")


@pytest.mark.all_backends
def test_create_collection_with_negative_dimensions_fails():
    """Test that API rejects negative dimensions (this validation actually exists)."""
    
    # Create collection config with negative dimensions (this should fail validation)
    config = {
        "vectors": [
            {
                "name": "dense_test",
                "type": "dense_model",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": -384,  # Negative dimensions - should fail validation
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should fail validation (negative dimensions not allowed)
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_negative_dimensions", json=config)
    assert response.status_code == 422  # Validation error
    # Handle both string and list error response formats
    error_detail = response.json()["detail"]
    if isinstance(error_detail, list):
        # Handle list of dictionaries format
        if error_detail and isinstance(error_detail[0], dict):
            # Extract error messages from dict structure
            error_detail = " ".join([str(item.get("msg", item)) for item in error_detail])
        else:
            # Handle list of strings
            error_detail = " ".join(error_detail)
    assert "dimensions" in error_detail.lower()  # Should mention dimensions in error
    
    # Clean up (just in case)
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_negative_dimensions")


@pytest.mark.all_backends
def test_create_collection_with_invalid_top_k_fails():
    """Test that API rejects invalid top_k values (this validation actually exists)."""
    
    # Create collection config with invalid top_k (this should fail validation)
    config = {
        "vectors": [
            {
                "name": "sparse_test",
                "type": "sparse_model",
                "model": "prithivida/Splade_PP_en_v1",
                "top_k": 0,  # Invalid top_k - should fail validation
                "index_fields": ["content"]
            }
        ]
    }
    
    # Create collection - should fail validation (top_k must be positive)
    response = requests.post(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_invalid_top_k", json=config)
    assert response.status_code == 422  # Validation error
    # Handle both string and list error response formats
    error_detail = response.json()["detail"]
    if isinstance(error_detail, list):
        # Handle list of dictionaries format
        if error_detail and isinstance(error_detail[0], dict):
            # Extract error messages from dict structure
            error_detail = " ".join([str(item.get("msg", item)) for item in error_detail])
        else:
            # Handle list of strings
            error_detail = " ".join(error_detail)
    assert "top_k" in error_detail.lower()  # Should mention top_k in error
    
    # Clean up (just in case)
    requests.delete(f"{API_BASE_URL}/collections/{COLLECTION_NAME}_invalid_top_k")

