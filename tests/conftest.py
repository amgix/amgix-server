import pytest
import os
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import Mock

# Import your models
from src.core.common import VectorType, DatabaseFeatures
from src.core.models.vector import VectorConfig, CollectionConfig


def pytest_addoption(parser):
    """Add custom command line options for test configuration."""
    parser.addoption(
        "--dense",
        action="store_true",
        help="Enable dense vector support for testing"
    )
    parser.addoption(
        "--nodense",
        action="store_true",
        help="Disable dense vector support for testing (sparse-only)"
    )


@pytest.fixture(scope="session")
def dense_vectors_enabled(request):
    """Determine if dense vectors are enabled for testing."""
    if request.config.getoption("--nodense"):
        return False
    elif request.config.getoption("--dense"):
        return True
    else:
        # Default: enable dense vectors (run all tests)
        return True


@pytest.fixture(scope="session")
def backend_capabilities(dense_vectors_enabled):
    """
    Determine backend capabilities based on dense vector support.
    
    Returns:
        Dict containing backend capabilities
    """
    return {
        "dense_vectors": dense_vectors_enabled,
        "sparse_vectors": True,  # All backends support sparse vectors
    }


class TestDataFactory:
    """
    Factory for creating test data based on backend capabilities.
    
    This class generates appropriate test configurations and data based on
    what the backend supports, eliminating the need to duplicate tests.
    """
    
    def __init__(self, backend_capabilities: Dict[str, Any]):
        self.supports_dense = backend_capabilities.get("dense_vectors", False)
        self.supports_sparse = backend_capabilities.get("sparse_vectors", True)
    
    def create_vector_configs(self, test_type: str = "basic") -> List[VectorConfig]:
        """
        Create vector configurations based on backend capabilities.
        
        Args:
            test_type: Type of test ("basic", "advanced", "sparse_only", "dense_only")
            
        Returns:
            List of VectorConfig objects appropriate for the backend
        """
        configs = []
        
        if test_type == "basic":
            # Basic test - always include sparse vectors
            if self.supports_sparse:
                configs.append(VectorConfig(
                    name="trigrams",
                    type=VectorType.TRIGRAMS,
                    index_fields=["name", "content"]
                ))
                
                configs.append(VectorConfig(
                    name="full_text",
                    type=VectorType.FULL_TEXT,
                    index_fields=["name", "content"],
                    language_detect=True,
                    language_default_code="en",
                    language_confidence=0.8
                ))
                
                configs.append(VectorConfig(
                    name="whitespace",
                    type=VectorType.WHITESPACE,
                    index_fields=["name", "content"],
                    language_detect=True,
                    language_default_code="en",
                    language_confidence=0.8
                ))
                
                configs.append(VectorConfig(
                    name="wmtr",
                    type=VectorType.WMTR,
                    index_fields=["name", "content"],
                    language_detect=True,
                    language_default_code="en",
                    language_confidence=0.8
                ))
        
        elif test_type == "advanced":
            # Advanced test - include both sparse and dense if supported
            if self.supports_sparse:
                configs.append(VectorConfig(
                    name="trigrams",
                    type=VectorType.TRIGRAMS,
                    index_fields=["name", "content"]
                ))
                
                configs.append(VectorConfig(
                    name="full_text",
                    type=VectorType.FULL_TEXT,
                    index_fields=["name", "content"],
                    language_detect=True,
                    language_default_code="en",
                    language_confidence=0.8
                ))
                
                configs.append(VectorConfig(
                    name="whitespace",
                    type=VectorType.WHITESPACE,
                    index_fields=["name", "content"],
                    language_detect=True,
                    language_default_code="en",
                    language_confidence=0.8
                ))
                
                configs.append(VectorConfig(
                    name="wmtr",
                    type=VectorType.WMTR,
                    index_fields=["name", "content"],
                    language_detect=True,
                    language_default_code="en",
                    language_confidence=0.8
                ))
                
                configs.append(VectorConfig(
                    name="dense_custom",
                    type=VectorType.DENSE_CUSTOM,
                    dimensions=384,
                    index_fields=["name", "content"]
                ))
                
                configs.append(VectorConfig(
                    name="sparse_custom",
                    type=VectorType.SPARSE_CUSTOM,
                    top_k=100,
                    index_fields=["name", "content"]
                ))
            
            if self.supports_dense:
                configs.append(VectorConfig(
                    name="embeddings",
                    type=VectorType.DENSE_MODEL,
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    index_fields=["name", "content"],
                    normalization=True
                ))
        
        elif test_type == "sparse_only":
            # Sparse-only test - only sparse vectors
            if self.supports_sparse:
                configs.append(VectorConfig(
                    name="trigrams",
                    type=VectorType.TRIGRAMS,
                    index_fields=["name", "content"]
                ))
                
                configs.append(VectorConfig(
                    name="full_text",
                    type=VectorType.FULL_TEXT,
                    index_fields=["name", "content"],
                    language_detect=True,
                    language_default_code="en",
                    language_confidence=0.8
                ))
                
                configs.append(VectorConfig(
                    name="whitespace",
                    type=VectorType.WHITESPACE,
                    index_fields=["name", "content"],
                    language_detect=True,
                    language_default_code="en",
                    language_confidence=0.8
                ))
                
                configs.append(VectorConfig(
                    name="wmtr",
                    type=VectorType.WMTR,
                    index_fields=["name", "content"],
                    language_detect=True,
                    language_default_code="en",
                    language_confidence=0.8
                ))
                
                configs.append(VectorConfig(
                    name="dense_custom",
                    type=VectorType.DENSE_CUSTOM,
                    dimensions=384,
                    index_fields=["name", "content"]
                ))
                
                configs.append(VectorConfig(
                    name="sparse_custom",
                    type=VectorType.SPARSE_CUSTOM,
                    top_k=100,
                    index_fields=["name", "content"]
                ))
                
                configs.append(VectorConfig(
                    name="sparse_model",
                    type=VectorType.SPARSE_MODEL,
                    model="prithivida/Splade_PP_en_v1",
                    top_k=100,
                    index_fields=["name", "content"]
                ))
        
        elif test_type == "dense_only":
            # Dense-only test - only dense vectors
            if self.supports_dense:
                configs.append(VectorConfig(
                    name="embeddings",
                    type=VectorType.DENSE_MODEL,
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    index_fields=["name", "content"],
                    normalization=True
                ))
        
        return configs
    
    def create_collection_config(self, test_type: str = "basic") -> CollectionConfig:
        """
        Create a collection configuration based on backend capabilities.
        
        Args:
            test_type: Type of test configuration to create
            
        Returns:
            CollectionConfig appropriate for the backend
        """
        vectors = self.create_vector_configs(test_type)
        return CollectionConfig(vectors=vectors)
    
    def create_search_query(self, test_type: str = "basic", query_text: str = "test query") -> Dict[str, Any]:
        """
        Create a search query based on backend capabilities.
        
        Args:
            test_type: Type of test query to create
            query_text: The search query text
            
        Returns:
            Search query dictionary appropriate for the backend
        """
        base_query = {
            "query": query_text,
            "limit": 10
        }
        
        if test_type == "basic" and self.supports_sparse:
            base_query["vector_weights"] = [
                {"vector_name": "trigrams", "field": "name", "weight": 0.5},
                {"vector_name": "trigrams", "field": "content", "weight": 0.5}
            ]
        
        elif test_type == "advanced":
            if self.supports_sparse:
                base_query["vector_weights"] = [
                    {"vector_name": "trigrams", "field": "name", "weight": 0.3},
                    {"vector_name": "trigrams", "field": "content", "weight": 0.3}
                ]
            
            if self.supports_dense:
                if "vector_weights" not in base_query:
                    base_query["vector_weights"] = []
                base_query["vector_weights"].append(
                    {"vector_name": "embeddings", "field": "content", "weight": 0.4}
                )
        
        return base_query
    
    def get_expected_features(self, test_type: str = "basic") -> Dict[str, Any]:
        """
        Get expected features that should work with the current backend.
        
        Args:
            test_type: Type of test being run
            
        Returns:
            Dict of expected features and their support status
        """
        features = {
            "sparse_search": self.supports_sparse,
            "dense_search": self.supports_dense,
            "hybrid_search": self.supports_sparse and self.supports_dense,
            "language_detection": self.supports_sparse,
            "trigram_search": self.supports_sparse,
            "full_text_search": self.supports_sparse
        }
        
        if test_type == "basic":
            features.update({
                "basic_sparse": self.supports_sparse,
                "basic_full_text": self.supports_sparse
            })
        elif test_type == "advanced":
            features.update({
                "advanced_sparse": self.supports_sparse,
                "advanced_dense": self.supports_dense,
                "hybrid_combinations": self.supports_sparse and self.supports_dense
            })
        
        return features


@pytest.fixture
def test_data_factory(backend_capabilities):
    """Provide a test data factory instance."""
    return TestDataFactory(backend_capabilities)


# Pytest markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "dense_vectors_only: mark test to run only on backends with dense vector support"
    )
    config.addinivalue_line(
        "markers", "sparse_only: mark test to run only on backends without dense vector support"
    )
    config.addinivalue_line(
        "markers", "all_backends: mark test to run on all backends"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring backend"
    )


# Helper function to skip tests based on backend capabilities
def skip_if_not_supported(backend_capabilities: Dict[str, Any], feature: str, reason: str = None):
    """
    Skip test if backend doesn't support required feature.
    
    Args:
        backend_capabilities: Backend capabilities dict
        feature: Feature to check (e.g., "dense_vectors")
        reason: Optional reason for skipping
    """
    if not backend_capabilities.get(feature, False):
        pytest.skip(reason or f"Backend doesn't support {feature}")


# Example of how to use these fixtures in tests:
# 
# @pytest.mark.all_backends
# def test_basic_search(backend_capabilities, test_data_factory):
#     """Test that runs on all backends."""
#     config = test_data_factory.create_collection_config("basic")
#     # ... test logic ...
# 
# @pytest.mark.dense_vectors_only
# def test_dense_vector_search(backend_capabilities, test_data_factory):
#     """Test that only runs on backends with dense vector support."""
#     skip_if_not_supported(backend_capabilities, "dense_vectors")
#     config = test_data_factory.create_collection_config("advanced")
#     # ... test logic ...
# 
# @pytest.mark.sparse_only
# def test_sparse_only_features(backend_capabilities, test_data_factory):
#     """Test that only runs on backends without dense vector support."""
#     if backend_capabilities.get("dense_vectors", False):
#         pytest.skip("Backend supports dense vectors, testing sparse-only features")
#     config = test_data_factory.create_collection_config("sparse_only")
#     # ... test logic ...
