import pytest
import requests

API_BASE_URL = "http://localhost:8234/v1"


@pytest.mark.all_backends
class TestCollectionNameValidation:
    """Test collection name validation rules."""
    
    def test_collection_name_empty(self):
        """Collection name cannot be empty."""
        response = requests.post(f"{API_BASE_URL}/collections/", json={
            "vectors": [{"name": "test", "type": "trigrams", "index_fields": ["name"]}]
        })
        assert response.status_code == 405  # FastAPI returns 405 for empty path parameter
    
    def test_collection_name_whitespace_only(self):
        """Collection name cannot be whitespace only."""
        response = requests.post(f"{API_BASE_URL}/collections/   ", json={
            "vectors": [{"name": "test", "type": "trigrams", "index_fields": ["name"]}]
        })
        assert response.status_code == 422  # FastAPI validates and returns 422
    
    def test_collection_name_with_spaces(self):
        """Collection name cannot contain spaces."""
        response = requests.post(f"{API_BASE_URL}/collections/test collection", json={
            "vectors": [{"name": "test", "type": "trigrams", "index_fields": ["name"]}]
        })
        assert response.status_code == 422  # FastAPI validation error
        error_detail = response.json()["detail"]
        # Check that it's a validation error
        assert isinstance(error_detail, list) and len(error_detail) > 0
    
    def test_collection_name_with_url_special_chars(self):
        """Test collection names with URL special characters that get truncated."""
        # These characters have special meaning in URLs and won't reach the API as intended
        url_special_chars = [
            "test#collection",  # URL fragment - only "test" reaches the API
            "test?collection",  # URL query - only "test" reaches the API  
            "test/collection",  # Path separator - only "test" reaches the API
            "test&collection",  # Query param separator - only "test" reaches the API
            "test=collection",  # Query param assignment - only "test" reaches the API
        ]
        
        for name in url_special_chars:
            # The API should only see the part before the special character
            # This could cause confusion - user thinks they're creating "test#collection"
            # but actually creating "test"
            response = requests.post(f"{API_BASE_URL}/collections/{name}", json={
                "vectors": [{"name": "test", "type": "trigrams", "index_fields": ["name"]}]
            })
            
            # These should succeed because the API only sees the valid part
            # But this demonstrates why our validation is important
            if response.status_code == 200:
                # Clean up if it was created - the same URL truncation applies
                requests.delete(f"{API_BASE_URL}/collections/{name}")
    
    def test_collection_name_with_other_special_chars(self):
        """Test collection names with other special characters that should be rejected."""
        # These characters don't have URL meaning but should be rejected by our validation
        invalid_names = [
            "test@collection",
            "test$collection", 
            "test%collection",
            "test*collection",
            "test+collection",
            "test[collection",
            "test]collection",
            "test{collection",
            "test}collection",
            "test|collection",
            "test\\collection",
            "test:collection",
            "test;collection",
            "test<collection",
            "test>collection",
            "test,collection",
            "test.collection",
            "test!collection"
        ]
        
        for invalid_name in invalid_names:
            response = requests.post(f"{API_BASE_URL}/collections/{invalid_name}", json={
                "vectors": [{"name": "test", "type": "trigrams", "index_fields": ["name"]}]
            })
            # These should be rejected by our validation
            assert response.status_code == 422, f"Should reject '{invalid_name}'"
            error_detail = response.json()["detail"]
            # Check that it's a validation error
            assert isinstance(error_detail, list) and len(error_detail) > 0
    
    def test_collection_name_too_long(self):
        """Collection name cannot exceed 100 characters."""
        long_name = "a" * 101
        response = requests.post(f"{API_BASE_URL}/collections/{long_name}", json={
            "vectors": [{"name": "test", "type": "trigrams", "index_fields": ["name"]}]
        })
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        # Check that it's a validation error
        assert isinstance(error_detail, list) and len(error_detail) > 0
    
    def test_collection_name_valid_characters(self):
        """Collection name with valid characters should work."""
        valid_names = [
            "test_collection",
            "test-collection",
            "test123collection",
            "TestCollection",
            "test_collection_123",
            "test-collection-123"
        ]
        
        for valid_name in valid_names:
            # Clean up first
            requests.delete(f"{API_BASE_URL}/collections/{valid_name}")
            
            response = requests.post(f"{API_BASE_URL}/collections/{valid_name}", json={
                "vectors": [{"name": "test", "type": "trigrams", "index_fields": ["name"]}]
            })
            assert response.status_code == 200, f"Should accept '{valid_name}'"
            
            # Cleanup
            requests.delete(f"{API_BASE_URL}/collections/{valid_name}")


class TestVectorNameValidation:
    """Test vector name validation rules."""
    
    def test_vector_name_empty(self):
        """Vector name cannot be empty."""
        response = requests.post(f"{API_BASE_URL}/collections/test_validation", json={
            "vectors": [{"name": "", "type": "trigrams", "index_fields": ["name"]}]
        })
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("empty" in str(detail) for detail in error_detail)
    
    def test_vector_name_whitespace_only(self):
        """Vector name cannot be whitespace only."""
        response = requests.post(f"{API_BASE_URL}/collections/test_validation", json={
            "vectors": [{"name": "   ", "type": "trigrams", "index_fields": ["name"]}]
        })
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("empty" in str(detail) for detail in error_detail)
    
    def test_vector_name_with_spaces(self):
        """Vector name cannot contain spaces."""
        response = requests.post(f"{API_BASE_URL}/collections/test_validation", json={
            "vectors": [{"name": "test vector", "type": "trigrams", "index_fields": ["name"]}]
        })
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("letters, numbers, underscores, and hyphens" in str(detail) for detail in error_detail)
    
    def test_vector_name_with_special_chars(self):
        """Vector name cannot contain special characters."""
        invalid_names = [
            "test@vector",
            "test#vector",
            "test$vector",
            "test%vector",
            "test&vector",
            "test*vector",
            "test+vector",
            "test=vector",
            "test[vector",
            "test]vector",
            "test{vector",
            "test}vector",
            "test|vector",
            "test\\vector",
            "test/vector",
            "test:vector",
            "test;vector",
            "test<vector",
            "test>vector",
            "test,vector",
            "test.vector",
            "test?vector",
            "test!vector"
        ]
        
        for invalid_name in invalid_names:
            response = requests.post(f"{API_BASE_URL}/collections/test_validation", json={
                "vectors": [{"name": invalid_name, "type": "trigrams", "index_fields": ["name"]}]
            })
            assert response.status_code == 422, f"Should reject vector name '{invalid_name}'"
            error_detail = response.json()["detail"]
            assert any("letters, numbers, underscores, and hyphens" in str(detail) for detail in error_detail)
    
    def test_vector_name_too_long(self):
        """Vector name cannot exceed 100 characters."""
        long_name = "a" * 101
        response = requests.post(f"{API_BASE_URL}/collections/test_validation", json={
            "vectors": [{"name": long_name, "type": "trigrams", "index_fields": ["name"]}]
        })
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("100 characters" in str(detail) for detail in error_detail)
    
    def test_vector_name_valid_characters(self):
        """Vector name with valid characters should work."""
        # Clean up first
        requests.delete(f"{API_BASE_URL}/collections/test_validation")
        
        valid_names = [
            "test_vector",
            "test-vector",
            "test123vector",
            "TestVector",
            "test_vector_123",
            "test-vector-123"
        ]
        
        for valid_name in valid_names:
            response = requests.post(f"{API_BASE_URL}/collections/test_validation", json={
                "vectors": [{"name": valid_name, "type": "trigrams", "index_fields": ["name"]}]
            })
            assert response.status_code == 200, f"Should accept vector name '{valid_name}'"
            
            # Clean up after each test
            requests.delete(f"{API_BASE_URL}/collections/test_validation")


class TestSearchQueryValidation:
    """Test search query vector name validation."""
    
    def setup_method(self):
        """Set up test collection for search tests."""
        self.collection_name = "test_search_validation"
        config = {
            "vectors": [{"name": "test_vector", "type": "trigrams", "index_fields": ["name"]}]
        }
        requests.delete(f"{API_BASE_URL}/collections/{self.collection_name}")
        response = requests.post(f"{API_BASE_URL}/collections/{self.collection_name}", json=config)
        assert response.status_code == 200
    
    def teardown_method(self):
        """Clean up test collection."""
        requests.delete(f"{API_BASE_URL}/collections/{self.collection_name}")
    
    def test_search_invalid_vector_name(self):
        """Search with invalid vector name should fail."""
        response = requests.post(f"{API_BASE_URL}/collections/{self.collection_name}/search", json={
            "query": "test query",
            "vector_weights": [{"vector_name": "invalid vector name", "weight": 1.0, "field": "name"}]
        })
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("letters, numbers, underscores, and hyphens" in str(detail) for detail in error_detail)
    
    def test_search_valid_vector_name(self):
        """Search with valid vector name should work."""
        response = requests.post(f"{API_BASE_URL}/collections/{self.collection_name}/search", json={
            "query": "test query",
            "vector_weights": [{"vector_name": "test_vector", "weight": 1.0, "field": "name"}]
        })
        # Should not fail validation (might fail for other reasons, but not validation)
        assert response.status_code != 422


if __name__ == "__main__":
    pytest.main([__file__])
