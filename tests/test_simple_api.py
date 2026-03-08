import pytest
import requests
from datetime import datetime, timezone

API_BASE_URL = "http://localhost:8234/v1"


@pytest.mark.all_backends
def test_health_check():
    """Test that the API is running and healthy."""
    response = requests.get(f"{API_BASE_URL}/health/check")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.all_backends
def test_collections_list():
    """Test listing collections."""
    response = requests.get(f"{API_BASE_URL}/collections")
    assert response.status_code == 200
    collections = response.json()
    assert isinstance(collections, list)


@pytest.mark.all_backends
def test_create_simple_collection():
    """Test creating a simple collection."""
    collection_name = "test_simple"
    
    # Clean up any existing collection first
    requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
    
    config = {
        "vectors": [
            {
                "name": "trigrams",
                "type": "trigrams",
                "top_k": 1000,
                "index_fields": ["name"]
            }
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
    assert response.status_code == 200
    
    # Cleanup
    requests.delete(f"{API_BASE_URL}/collections/{collection_name}")


@pytest.mark.all_backends
def test_simple_document_upsert():
    """Test simple document upsert."""
    collection_name = "test_upsert"
    
    # Clean up any existing collection first
    requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
    
    # Create collection
    config = {
        "vectors": [
            {
                "name": "trigrams",
                "type": "trigrams",
                "top_k": 1000,
                "index_fields": ["name"]
            }
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
    assert response.status_code == 200, f"Failed to create collection: {response.text}"
    
    # Upsert document
    doc = {
        "id": "test_doc",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": "Test Document",
        "tags": ["test"]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
    
    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is True
    
    # Try to retrieve the document with simple polling
    def _got():
        r = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/test_doc")
        return r.status_code == 200
    import time as _t
    deadline = _t.time() + 8
    while _t.time() < deadline and not _got():
        _t.sleep(0.25)
    
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/test_doc")
    assert response.status_code == 200
    
    # Cleanup
    requests.delete(f"{API_BASE_URL}/collections/{collection_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

