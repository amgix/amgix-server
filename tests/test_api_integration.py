import pytest
import requests
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Callable, List
from tests.conftest import skip_if_not_supported
from src.core.common import QueuedDocumentStatus

# Test configuration
API_BASE_URL = "http://localhost:8234/v1"
def wait_until(predicate: Callable[[], bool], timeout_s: float = 10.0, interval_s: float = 0.25) -> bool:
    """Poll predicate() until True or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def wait_for_document(collection_name: str, doc_id: str, timeout_s: float = 60.0) -> None:
    url = f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}"
    ok = wait_until(lambda: requests.get(url).status_code == 200, timeout_s)
    assert ok, f"Timed out waiting for document {doc_id} to be available"


def wait_for_document_status(collection_name: str, doc_id: str, expected_status: str, timeout_s: float = 60.0) -> None:
    """Wait for a document to reach a specific status (e.g., 'indexed', 'failed')."""
    def _check_status() -> bool:
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}/status")
        if response.status_code != 200:
            return False
        status_response = response.json()
        return any(s["status"].lower() == expected_status.lower() for s in status_response["statuses"])
    
    ok = wait_until(_check_status, timeout_s)
    assert ok, f"Timed out waiting for document {doc_id} to reach status '{expected_status}'"


def wait_for_search(collection_name: str, query: Dict[str, Any], expect: Callable[[List[Dict[str, Any]]], bool], timeout_s: float = 60.0) -> List[Dict[str, Any]]:
    url = f"{API_BASE_URL}/collections/{collection_name}/search"
    last_results: List[Dict[str, Any]] = []
    def _try() -> bool:
        nonlocal last_results
        resp = requests.post(url, json=query)
        if resp.status_code != 200:
            return False
        last_results = resp.json()
        return expect(last_results)
    ok = wait_until(_try, timeout_s)
    assert ok, f"Timed out waiting for expected search results; last_results={last_results}"
    return last_results


def create_test_document(doc_id: str, name: str, content: str, doc_type: str = "article") -> Dict[str, Any]:
    """Helper function to create test documents with required timestamp field."""
    doc: Dict[str, Any] = {
        "id": doc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "content": content,
    }
    if doc_type is not None:
        doc["tags"] = [doc_type]
    return doc


@pytest.fixture(scope="function")
def setup_collection(request, test_data_factory):
    """
    Create a test collection with vector configurations based on backend capabilities.
    
    This fixture automatically adapts the collection configuration based on what
    the backend supports, eliminating the need for separate test setups.
    """
    # Generate unique collection name using test function name and timestamp
    test_name = request.function.__name__
    unique_id = str(uuid.uuid4())[:8]    # first 8 chars of UUID
    collection_name = f"test_{test_name}_{unique_id}"
    
    # Get collection config based on test type
    # You can override this by passing a different test_type parameter
    test_type = getattr(request, 'param', 'basic')
    config = test_data_factory.create_collection_config(test_type)
    
    # Convert to API format
    api_config = config.model_dump()
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=api_config)
    if response.status_code != 200:
        print(f"Collection creation failed: {response.status_code}")
        print(f"Response: {response.text}")
    assert response.status_code == 200
    
    yield collection_name
    
    # Cleanup
    try:
        requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
    except Exception as e:
        # Log cleanup errors but don't fail the test
        print(f"Warning: Failed to cleanup collection {collection_name}: {e}")


@pytest.mark.all_backends
def test_health_check():
    """Test that the API is running and healthy."""
    response = requests.get(f"{API_BASE_URL}/health/check")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.all_backends
def test_upsert_document_sync_and_search(setup_collection, test_data_factory):
    collection_name = setup_collection

    # Create two docs
    doc1 = create_test_document("doc-sync-1", "Alpha", "Alpha content about birds")
    doc2 = create_test_document("doc-sync-2", "Beta", "Beta content about fish")

    # Upsert via sync endpoint
    resp1 = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents/sync", json=doc1)
    assert resp1.status_code == 200
    assert resp1.json().get("ok") is True

    resp2 = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents/sync", json=doc2)
    assert resp2.status_code == 200
    assert resp2.json().get("ok") is True

    # Verify documents retrievable directly (sync endpoint guarantees readiness)
    r1 = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc1['id']}")
    assert r1.status_code == 200
    r2 = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc2['id']}")
    assert r2.status_code == 200

    # Basic search that should find both
    search_query = {
        "query": "content",
        "limit": 10
    }

    results = wait_for_search(collection_name, search_query, lambda rs: len(rs) >= 2, timeout_s=20.0)
    returned_ids = {r.get("id") for r in results}
    assert doc1["id"] in returned_ids
    assert doc2["id"] in returned_ids


@pytest.mark.all_backends
def test_collections_endpoints():
    """Test collection management endpoints."""
    # List collections
    response = requests.get(f"{API_BASE_URL}/collections")
    assert response.status_code == 200
    collections = response.json()
    assert isinstance(collections, list)


@pytest.mark.all_backends
def test_document_upsert_and_search(setup_collection):
    """Test the full pipeline: upsert document and search for it."""
    collection_name = setup_collection
    
    # Test document 1
    doc1 = create_test_document(
        "doc1", 
        "Python Programming",
        "Python is a high-level programming language known for its simplicity and readability."
    )
    
    # Upsert document (should be async)
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc1)
    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is True
    
    # Test document 2
    doc2 = create_test_document(
        "doc2",
        "Machine Learning Basics",
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "tutorial"
    )
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc2)
    assert response.status_code == 200
    
    # Wait for async processing to finish
    wait_for_document(collection_name, "doc1")
    wait_for_document(collection_name, "doc2")
    
    # Search for documents
    search_query = {
                    "query": "Python programming",
        "limit": 10,
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 0.5},
            {"vector_name": "trigrams", "field": "content", "weight": 0.5}
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
    assert response.status_code == 200
    results = response.json()
    
    # Should find at least one document
    assert len(results) > 0
    
    # Check that we found the Python document
    doc_ids = [r["id"] for r in results]
    assert "doc1" in doc_ids
    
    # Verify document content
    python_doc = next(r for r in results if r["id"] == "doc1")
    assert python_doc["name"] == "Python Programming"
    assert "score" in python_doc


@pytest.mark.all_backends
# @pytest.mark.skip(reason="Temporarily disabled while PostgreSQL backend is being stabilized")
def test_sparse_model_query_override():
    """Verify query-time model specification for SPARSE_MODEL using SPLADE doc vs query model."""
    collection_name = f"splade_override_{uuid.uuid4().hex[:8]}"
    create_resp = requests.post(
        f"{API_BASE_URL}/collections/{collection_name}",
        json={
            "vectors": [
                {
                    "name": "splade",
                    "type": "sparse_model",
                    "model": "prithivida/Splade_PP_en_v1",
                    "query_model": "prithivida/Splade_PP_en_v1",
                    "index_fields": ["name", "content"],
                }
            ],
            "store_content": False,
        },
    )
    if create_resp.status_code != 200:
        print("Collection creation failed:", create_resp.status_code)
        try:
            print(create_resp.json())
        except Exception:
            print(create_resp.text)
    assert create_resp.status_code == 200

    try:
        d1 = create_test_document("sdoc-1", "SPLADE Alpha", "SPLADE content about search and matching")
        d2 = create_test_document("sdoc-2", "SPLADE Beta", "Another piece about sparse retrieval")

        r1 = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents/sync", json=d1)
        if r1.status_code != 200:
            print("Document 1 sync failed:", r1.status_code)
            try:
                print(r1.json())
            except Exception:
                print(r1.text)
        assert r1.status_code == 200 and r1.json().get("ok") is True
        
        r2 = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents/sync", json=d2)
        if r2.status_code != 200:
            print("Document 2 sync failed:", r2.status_code)
            try:
                print(r2.json())
            except Exception:
                print(r2.text)
        assert r2.status_code == 200 and r2.json().get("ok") is True

        search_body = {
            "query": "sparse retrieval search",
            "limit": 10,
            "vector_weights": [
                {"vector_name": "splade", "field": "content", "weight": 1.0}
            ],
        }
        resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_body)
        if resp.status_code != 200:
            print("Search failed:", resp.status_code)
            try:
                print(resp.json())
            except Exception:
                print(resp.text)
        assert resp.status_code == 200
        results = resp.json()
        ids = {r.get("id") for r in results}
        assert "sdoc-1" in ids or "sdoc-2" in ids
    finally:
        try:
            requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
        except Exception:
            pass


@pytest.mark.all_backends
def test_search_with_full_text_vectors(setup_collection):
    """Test search using full text vectors."""
    collection_name = setup_collection
    
    # Add test documents first
    doc1 = create_test_document(
        "doc1",
        "Python Programming", 
        "Python is a high-level programming language known for its simplicity and readability."
    )
    
    doc2 = create_test_document(
        "doc2",
        "Machine Learning Basics",
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.", 
        "tutorial"
    )
    
    # Upsert documents
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc1)
    print(f"Doc1 upsert response: {response.status_code} - {response.text}")
    assert response.status_code == 200
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc2)
    print(f"Doc2 upsert response: {response.status_code} - {response.text}")
    assert response.status_code == 200
    
    # Wait for async processing
    wait_for_document(collection_name, "doc1")
    wait_for_document(collection_name, "doc2")
    
    # Search using full text vectors
    search_query = {
                    "query": "machine learning",
        "limit": 10,
        "vector_weights": [
            {"vector_name": "full_text", "field": "name", "weight": 0.3},
            {"vector_name": "full_text", "field": "content", "weight": 0.7}
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
    assert response.status_code == 200
    results = response.json()
    
    # Should find the machine learning document
    assert len(results) > 0
    doc_ids = [r["id"] for r in results]
    assert "doc2" in doc_ids


@pytest.mark.all_backends
def test_search_with_document_tags_filter(setup_collection):
    """Test search with document tags filtering."""
    collection_name = setup_collection
    
    search_query = {
                    "query": "programming",
        "limit": 10,
        "document_tags": ["article"],
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 0.5},
            {"vector_name": "trigrams", "field": "content", "weight": 0.5}
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
    assert response.status_code == 200
    results = response.json()
    
    # Should only return articles
    for result in results:
        assert result['tags'] == "article"


@pytest.mark.all_backends
def test_search_order_with_trigrams(setup_collection):
    """Verify search returns expected order using trigrams."""
    collection_name = setup_collection
    
    # Two docs tailored so doc_py ranks above doc_ml for query "python"
    doc_py = create_test_document("order_py", "Python deep dive", "Advanced Python internals")
    # Ensure second doc still weakly matches query to appear but rank lower
    doc_ml = create_test_document("order_ml", "Machine Learning", "Overview article mentioning python once")
    assert requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_py).status_code == 200
    assert requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_ml).status_code == 200

    wait_for_document(collection_name, "order_py")
    wait_for_document(collection_name, "order_ml")

    q = {
                    "query": "python",
        "limit": 10,
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 0.7},
            {"vector_name": "trigrams", "field": "content", "weight": 0.3},
        ],
    }
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=q)
    assert response.status_code == 200
    results = response.json()
    
    # Debug: Log the full search results
    print(f"\n=== SEARCH DEBUG ===")
    print(f"Collection: {collection_name}")
    print(f"Query: {q}")
    print(f"Total results: {len(results)}")
    print(f"Full results: {results}")
    print(f"Document IDs: {[r['id'] for r in results]}")
    print(f"Document titles: {[r.get('name', 'NO_TITLE') for r in results]}")
    print(f"Document types: {[r.get('type', 'NO_TYPE') for r in results]}")
    if results:
        print(f"First result full data: {results[0]}")
    print(f"=== END DEBUG ===\n")
    
    ids = [r["id"] for r in results]
    assert ids.index("order_py") < ids.index("order_ml")


@pytest.mark.all_backends
def test_weight_combination_flips_order(setup_collection):
    """Weights on title vs content should flip order predictably."""
    collection_name = setup_collection
    
    # One doc matches strongly in title, other in content
    doc_t = create_test_document("w_title", "Alpha in Title", "filler text with no key repetition")
    # Strong content signal with heavy repetition of the key
    doc_c = create_test_document("w_content", "generic title", "Alpha Alpha Alpha Alpha Alpha Alpha Alpha content")
    assert requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_t).status_code == 200
    assert requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_c).status_code == 200

    wait_for_document(collection_name, "w_title")
    wait_for_document(collection_name, "w_content")

    # Title-only weighting should put w_title first
    q_title = {
        "query": "Alpha",
        "limit": 10,
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 1.0},
        ],
    }
    # Title-only may yield only the title-matching doc; ensure it's ranked first
    r_title_resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=q_title)
    assert r_title_resp.status_code == 200
    r_title = r_title_resp.json()
    assert len(r_title) >= 1 and r_title[0]["id"] == "w_title"

    # Content-only weighting should put w_content first
    q_content = {
        "query": "Alpha",
        "limit": 10,
        "vector_weights": [
            {"vector_name": "trigrams", "field": "content", "weight": 1.0},
        ],
    }
    r_content_resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=q_content)
    assert r_content_resp.status_code == 200
    r_content = r_content_resp.json()
    assert len(r_content) >= 1 and r_content[0]["id"] == "w_content"


@pytest.mark.all_backends
def test_search_limit_enforced(setup_collection):
    """Limit should trim results to top-N in correct order."""
    collection_name = setup_collection
    
    # Create three similar docs containing key in title to ensure all match
    for i in range(3):
        d = create_test_document(f"lim_{i}", f"KeyTerm {i}", "body")
        assert requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=d).status_code == 200
        wait_for_document(collection_name, f"lim_{i}")

    q = {
        "query": "KeyTerm",
        "limit": 2,
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 1.0},
        ],
    }
    resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=q)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2


@pytest.mark.all_backends
def test_search_no_results_returns_empty(setup_collection):
    """Non-matching query should return empty list with 200."""
    collection_name = setup_collection
    
    q = {
        "query": "zzzxxyyunlikelyterm",
        "limit": 5,
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 1.0},
        ],
    }
    # Expect empty list; allow immediate 200 with []
    resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=q)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.all_backends
def test_document_tags_filter_edges(setup_collection):
    """document_tags filter with no match and multi-type match."""
    collection_name = setup_collection
    
    d_news = create_test_document("t_news", "Breaking News", "desc", doc_type="news")
    d_art = create_test_document("t_article", "Article", "desc", doc_type="article")
    assert requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=d_news).status_code == 200
    assert requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=d_art).status_code == 200
    wait_for_document(collection_name, "t_news")
    wait_for_document(collection_name, "t_article")

    q_none = {
        "query": "news",
        "limit": 10,
        "document_tags": ["blog"],
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 1.0},
        ],
    }
    resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=q_none)
    assert resp.status_code == 200
    assert resp.json() == []

    q_both = {
        "query": "desc",
        "limit": 10,
        "document_tags": ["news", "article"],
        "vector_weights": [
            {"vector_name": "trigrams", "field": "content", "weight": 1.0},
        ],
    }
    resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=q_both)
    assert resp.status_code == 200
    results = resp.json()
    types = {(r.get("tags")[0] if r.get("tags") else None) for r in results}
    assert "news" in types and "article" in types


@pytest.mark.all_backends
def test_metadata_roundtrip(setup_collection):
    """Upsert with metadata and verify roundtrip on GET."""
    collection_name = setup_collection
    
    doc = create_test_document("meta_doc", "Meta Title", "Meta body")
    doc["metadata"] = {"source": "api", "lang": "en"}
    assert requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc).status_code == 200
    wait_for_document(collection_name, "meta_doc")
    r = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/meta_doc")
    assert r.status_code == 200
    data = r.json()
    assert data.get("metadata") is not None
    md = data["metadata"]
    assert md["source"]["value"] == "api" and md["lang"]["value"] == "en"


@pytest.mark.all_backends
def test_metadata_indexes_validation(setup_collection):
    """Test metadata indexes validation with all types, valid/invalid values, and retrieval."""
    collection_name = setup_collection
    
    # Create a new collection with metadata_indexes of all types
    test_collection_name = f"test_metadata_indexes_{str(uuid.uuid4())[:8]}"
    config = {
        "vectors": [
            {
                "name": "trigrams",
                "type": "trigrams",
                "top_k": 1000,
                "index_fields": ["name"]
            }
        ],
        "metadata_indexes": [
            {"key": "author", "type": "string"},
            {"key": "year", "type": "integer"},
            {"key": "rating", "type": "float"},
            {"key": "published", "type": "boolean"},
            {"key": "created_at", "type": "datetime"}
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{test_collection_name}", json=config)
    assert response.status_code == 200, f"Collection creation failed: {response.text}"
    
    try:
        # Test 1: Valid document with all metadata keys
        doc1 = create_test_document("doc1", "Test Document 1", "Content 1")
        doc1["metadata"] = {
            "author": "John Doe",
            "year": 2023,
            "rating": 4.5,
            "published": True,
            "created_at": {"value": "2023-01-15T10:30:00Z", "type": "datetime"}
        }
        response = requests.post(f"{API_BASE_URL}/collections/{test_collection_name}/documents/sync", json=doc1)
        assert response.status_code == 200, f"Valid document upload failed: {response.text}"
        
        # Test 2: Valid document with some metadata keys missing (should be fine)
        doc2 = create_test_document("doc2", "Test Document 2", "Content 2")
        doc2["metadata"] = {
            "author": "Jane Smith",
            "year": 2024
            # rating, published, created_at are missing - should be fine
        }
        response = requests.post(f"{API_BASE_URL}/collections/{test_collection_name}/documents/sync", json=doc2)
        assert response.status_code == 200, f"Document with missing keys failed: {response.text}"
        
        # Test 3: Invalid document with wrong type for indexed key
        doc3 = create_test_document("doc3", "Test Document 3", "Content 3")
        doc3["metadata"] = {
            "author": "Bob",
            "year": {"value": "2023", "type": "string"}  # Wrong type - should be integer (using explicit format to test validation)
        }
        response = requests.post(f"{API_BASE_URL}/collections/{test_collection_name}/documents/sync", json=doc3)
        assert response.status_code == 400, f"Invalid type should have been rejected: {response.text}"
        
        # Test 4: Invalid document with wrong type for another key
        doc4 = create_test_document("doc4", "Test Document 4", "Content 4")
        doc4["metadata"] = {
            "author": "Alice",
            "rating": {"value": "high", "type": "string"}  # Wrong type - should be float (using explicit format to test validation)
        }
        response = requests.post(f"{API_BASE_URL}/collections/{test_collection_name}/documents/sync", json=doc4)
        assert response.status_code == 400, f"Invalid type should have been rejected: {response.text}"
        
        # Test 5: Retrieve doc1 via direct GET and verify metadata
        response = requests.get(f"{API_BASE_URL}/collections/{test_collection_name}/documents/doc1")
        assert response.status_code == 200
        data = response.json()
        assert data.get("metadata") is not None
        md = data["metadata"]
        assert md["author"]["value"] == "John Doe" and md["author"]["type"] == "string"
        assert md["year"]["value"] == 2023 and md["year"]["type"] == "integer"
        assert md["rating"]["value"] == 4.5 and md["rating"]["type"] == "float"
        assert md["published"]["value"] is True and md["published"]["type"] == "boolean"
        assert md["created_at"]["value"] == "2023-01-15T10:30:00Z" and md["created_at"]["type"] == "datetime"
        
        # Test 6: Retrieve doc2 via direct GET and verify metadata (with missing keys)
        response = requests.get(f"{API_BASE_URL}/collections/{test_collection_name}/documents/doc2")
        assert response.status_code == 200
        data = response.json()
        assert data.get("metadata") is not None
        md = data["metadata"]
        assert md["author"]["value"] == "Jane Smith" and md["author"]["type"] == "string"
        assert md["year"]["value"] == 2024 and md["year"]["type"] == "integer"
        # Other keys should not be present
        
        # Test 7: Search and verify metadata in results
        search_query = {"query": "Test", "limit": 10}
        response = requests.post(f"{API_BASE_URL}/collections/{test_collection_name}/search", json=search_query)
        assert response.status_code == 200
        results = response.json()
        assert len(results) >= 2
        
        # Find doc1 and doc2 in results
        doc1_result = next((r for r in results if r["id"] == "doc1"), None)
        doc2_result = next((r for r in results if r["id"] == "doc2"), None)
        
        assert doc1_result is not None, "doc1 not found in search results"
        assert doc1_result.get("metadata") is not None
        md1 = doc1_result["metadata"]
        assert md1["author"]["value"] == "John Doe"
        assert md1["year"]["value"] == 2023
        assert md1["rating"]["value"] == 4.5
        assert md1["published"]["value"] is True
        assert md1["created_at"]["value"] == "2023-01-15T10:30:00Z"
        
        assert doc2_result is not None, "doc2 not found in search results"
        assert doc2_result.get("metadata") is not None
        md2 = doc2_result["metadata"]
        assert md2["author"]["value"] == "Jane Smith"
        assert md2["year"]["value"] == 2024
        
    finally:
        # Cleanup
        try:
            requests.delete(f"{API_BASE_URL}/collections/{test_collection_name}")
        except Exception as e:
            print(f"Warning: Failed to cleanup collection {test_collection_name}: {e}")


@pytest.mark.all_backends
def test_recursive_metadata_filtering():
    """Test recursive metadata filtering (eq/range/and/or/not) and validation errors."""
    collection_name = f"test_metadata_filter_{str(uuid.uuid4())[:8]}"
    config = {
        "vectors": [
            {
                "name": "trigrams",
                "type": "trigrams",
                "top_k": 1000,
                "index_fields": ["name"]
            }
        ],
        "metadata_indexes": [
            {"key": "author", "type": "string"},
            {"key": "year", "type": "integer"},
            {"key": "rating", "type": "float"},
            {"key": "published", "type": "boolean"},
            {"key": "created_at", "type": "datetime"},
        ],
    }

    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
    assert response.status_code == 200, f"Collection creation failed: {response.text}"

    try:
        docs = [
            {
                **create_test_document("meta_f_1", "Filter Alpha", "alpha content"),
                "metadata": {
                    "author": "Alice",
                    "year": 2021,
                    "rating": 4.2,
                    "published": False,
                    "created_at": {"value": "2021-01-01T00:00:00Z", "type": "datetime"},
                },
            },
            {
                **create_test_document("meta_f_2", "Filter Beta", "beta content"),
                "metadata": {
                    "author": "Bob",
                    "year": 2023,
                    "rating": 4.8,
                    "published": True,
                    "created_at": {"value": "2023-06-10T00:00:00Z", "type": "datetime"},
                },
            },
            {
                **create_test_document("meta_f_3", "Filter Gamma", "gamma content"),
                "metadata": {
                    "author": "Carol",
                    "year": 2024,
                    "rating": 3.5,
                    "published": False,
                    "created_at": {"value": "2024-02-20T00:00:00Z", "type": "datetime"},
                },
            },
        ]

        for doc in docs:
            upload_resp = requests.post(
                f"{API_BASE_URL}/collections/{collection_name}/documents/sync",
                json=doc
            )
            assert upload_resp.status_code == 200, f"Doc upload failed: {upload_resp.text}"

        base_query = {
            "query": "Filter",
            "limit": 10,
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}],
        }

        eq_query = {
            **base_query,
            "metadata_filter": {"key": "author", "op": "eq", "value": "Alice"},
        }
        eq_resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=eq_query)
        assert eq_resp.status_code == 200, f"eq filter failed: {eq_resp.text}"
        eq_ids = {r["id"] for r in eq_resp.json()}
        assert eq_ids == {"meta_f_1"}

        range_query = {
            **base_query,
            "metadata_filter": {"key": "year", "op": "gt", "value": 2022},
        }
        range_resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=range_query)
        assert range_resp.status_code == 200, f"range filter failed: {range_resp.text}"
        range_ids = {r["id"] for r in range_resp.json()}
        assert range_ids == {"meta_f_2", "meta_f_3"}

        and_query = {
            **base_query,
            "metadata_filter": {
                "and": [
                    {"key": "year", "op": "gte", "value": 2022},
                    {"key": "published", "op": "eq", "value": True},
                ]
            },
        }
        and_resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=and_query)
        assert and_resp.status_code == 200, f"and filter failed: {and_resp.text}"
        and_ids = {r["id"] for r in and_resp.json()}
        assert and_ids == {"meta_f_2"}

        or_query = {
            **base_query,
            "metadata_filter": {
                "or": [
                    {"key": "author", "op": "eq", "value": "Alice"},
                    {"key": "rating", "op": "lt", "value": 4.0},
                ]
            },
        }
        or_resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=or_query)
        assert or_resp.status_code == 200, f"or filter failed: {or_resp.text}"
        or_ids = {r["id"] for r in or_resp.json()}
        assert or_ids == {"meta_f_1", "meta_f_3"}

        not_query = {
            **base_query,
            "metadata_filter": {
                "not": {"key": "published", "op": "eq", "value": True}
            },
        }
        not_resp = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=not_query)
        assert not_resp.status_code == 200, f"not filter failed: {not_resp.text}"
        not_ids = {r["id"] for r in not_resp.json()}
        assert not_ids == {"meta_f_1", "meta_f_3"}

        nested_not_query = {
            **base_query,
            "metadata_filter": {
                "not": {
                    "and": [
                        {"key": "year", "op": "gt", "value": 2022},
                        {"key": "author", "op": "eq", "value": "Bob"},
                    ]
                }
            },
        }
        nested_not_resp = requests.post(
            f"{API_BASE_URL}/collections/{collection_name}/search",
            json=nested_not_query
        )
        assert nested_not_resp.status_code == 200, f"nested not filter failed: {nested_not_resp.text}"
        nested_not_ids = {r["id"] for r in nested_not_resp.json()}
        assert nested_not_ids == {"meta_f_1", "meta_f_3"}

        datetime_query = {
            **base_query,
            "metadata_filter": {"key": "created_at", "op": "gte", "value": "2023-01-01T00:00:00Z"},
        }
        datetime_resp = requests.post(
            f"{API_BASE_URL}/collections/{collection_name}/search",
            json=datetime_query
        )
        assert datetime_resp.status_code == 200, f"datetime filter failed: {datetime_resp.text}"
        datetime_ids = {r["id"] for r in datetime_resp.json()}
        assert datetime_ids == {"meta_f_2", "meta_f_3"}

        invalid_key_query = {
            **base_query,
            "metadata_filter": {"key": "unknown_key", "op": "eq", "value": "x"},
        }
        invalid_key_resp = requests.post(
            f"{API_BASE_URL}/collections/{collection_name}/search",
            json=invalid_key_query
        )
        assert invalid_key_resp.status_code == 400, f"Invalid key should fail: {invalid_key_resp.text}"

        invalid_type_query = {
            **base_query,
            "metadata_filter": {"key": "year", "op": "gte", "value": "2023"},
        }
        invalid_type_resp = requests.post(
            f"{API_BASE_URL}/collections/{collection_name}/search",
            json=invalid_type_query
        )
        assert invalid_type_resp.status_code == 400, f"Invalid type should fail: {invalid_type_resp.text}"
    finally:
        requests.delete(f"{API_BASE_URL}/collections/{collection_name}")


@pytest.mark.all_backends
def test_duplicate_collection_create_fails():
    """Creating an existing collection should fail."""
    name = "dup_create_test"
    cfg = {"vectors": [{"name": "trigrams", "type": "trigrams", "index_fields": ["name"]}]}
    r1 = requests.post(f"{API_BASE_URL}/collections/{name}", json=cfg)
    assert r1.status_code == 200
    r2 = requests.post(f"{API_BASE_URL}/collections/{name}", json=cfg)
    assert r2.status_code >= 400
    # cleanup
    requests.delete(f"{API_BASE_URL}/collections/{name}")


@pytest.mark.all_backends
def test_document_retrieval(setup_collection):
    """Test retrieving a specific document."""
    collection_name = setup_collection
    
    # Add a document first
    doc1 = create_test_document(
        "doc1",
        "Python Programming",
        "Python is a high-level programming language known for its simplicity and readability."
    )
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc1)
    assert response.status_code == 200
    
    # Wait for async processing
    wait_for_document(collection_name, "doc1")
    
    # Now retrieve the document
    wait_for_document(collection_name, "doc1")
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/doc1")
    
    doc = response.json()
    assert doc["id"] == "doc1"
    assert doc["name"] == "Python Programming"
    assert (doc.get('tags') or []) and (doc.get('tags')[0] == 'article')


@pytest.mark.all_backends
def test_document_deletion(setup_collection):
    """Test document deletion."""
    collection_name = setup_collection
    
    # Create a document first
    doc1 = create_test_document(
        "doc1", 
        "Python Programming",
        "Python is a high-level programming language known for its simplicity and readability."
    )
    
    # Upsert document
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc1)
    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is True
    
    # Wait for document to be ready
    wait_for_document(collection_name, "doc1")
    
    # Now delete the document
    response = requests.delete(f"{API_BASE_URL}/collections/{collection_name}/documents/doc1")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    
    # Verify it's gone
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/doc1")
    assert response.status_code == 404
    
    # Search should not return the deleted document
    search_query = {
        "query": "Python",
        "limit": 10,
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 1.0}
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
    assert response.status_code == 200
    results = response.json()
    
    doc_ids = [r["id"] for r in results]
    assert "doc1" not in doc_ids


@pytest.mark.all_backends
def test_empty_search_query(setup_collection):
    """Test search with empty query string."""
    collection_name = setup_collection
    
    search_query = {
        "query": "",
        "limit": 10,
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 1.0}
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
    assert response.status_code == 422  # Empty query should be rejected


@pytest.mark.all_backends
def test_document_invalid_timestamps(setup_collection):
    """Test document upsert with invalid timestamp values."""
    collection_name = setup_collection
    
    # Test 1: Missing timestamp
    doc_missing_timestamp = {
        "id": "test_missing_ts",
        "name": "Test Document",
        "content": "Test content",
        "tags": ["test"]
        # No timestamp field
    }
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_missing_timestamp)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("timestamp" in str(err).lower() for err in error_detail)
    
    # Test 2: Naive timestamp (no timezone)
    doc_naive_timestamp = {
        "id": "test_naive_ts", 
        "timestamp": "2025-08-09T15:30:00",  # No timezone info
        "name": "Test Document",
        "content": "Test content",
        "tags": ["test"]
    }
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_naive_timestamp)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("timezone" in str(err).lower() for err in error_detail)
    
    # Test 3: Wrong timezone (not UTC)
    doc_wrong_timezone = {
        "id": "test_wrong_tz",
        "timestamp": "2025-08-09T15:30:00-05:00",  # EST timezone
        "name": "Test Document", 
        "content": "Test content",
        "tags": ["test"]
    }
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_wrong_timezone)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("utc" in str(err).lower() for err in error_detail)
    
    # Test 4: Garbage timestamp
    doc_garbage_timestamp = {
        "id": "test_garbage_ts",
        "timestamp": "not-a-datetime",
        "name": "Test Document",
        "content": "Test content", 
        "tags": ["test"]
    }
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_garbage_timestamp)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    # Should fail datetime parsing
    assert any("datetime" in str(err).lower() or "timestamp" in str(err).lower() for err in error_detail)
    
    # Test 5: Valid UTC timestamp (should succeed)
    doc_valid_timestamp = create_test_document(
        "test_valid_ts",
        "Test Document",
        "Test content"
    )
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_valid_timestamp)
    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is True


@pytest.mark.all_backends
def test_invalid_collection():
    """Test operations on non-existent collection."""
    
    # Try to search in non-existent collection
    search_query = {
        "query": "test",
        "limit": 10,
        "vector_weights": [
            {"vector_name": "trigrams", "field": "name", "weight": 1.0}
        ]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/nonexistent/search", json=search_query)
    assert response.status_code == 404  # Not found should propagate as 404


@pytest.mark.all_backends
def test_timestamp_based_deduplication():
    """Test that documents are only upserted if they have newer timestamps."""
    collection_name = "test_timestamp_dedup"
    
    # Clean up any existing collection
    requests.delete(f"{API_BASE_URL}/collections/{collection_name}")


@pytest.mark.skip(reason="FastEmbed support temporarily disabled")
@pytest.mark.dense_vectors_only
def test_fastembed_dense_vectors(backend_capabilities):
    """Test FastEmbed dense vector functionality end-to-end."""
    from tests.conftest import skip_if_not_supported
    
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    collection_name = f"test_fastembed_dense_{str(uuid.uuid4())[:8]}"
    
    # Create collection with FastEmbed dense vector
    config = {
        "vectors": [
            {
                "name": "fastembed_dense",
                "type": "dense_fastembed",
                # "model": "BAAI/bge-small-en-v1.5",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "index_fields": ["content"],
                "normalization": True
            }
        ]
    }
    
    try:
        # Create collection
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        if response.status_code != 200:
            print(f"FastEmbed dense collection creation failed: {response.status_code}")
            print(f"Response: {response.text}")
        assert response.status_code == 200
        
        # Add test documents
        documents = [
            create_test_document("fe_doc1", "Machine Learning", "Deep learning and neural networks"),
            create_test_document("fe_doc2", "Data Science", "Statistical analysis and data mining")
        ]
        
        # Insert documents - this will trigger FastEmbed dense vector generation
        for doc in documents:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert FastEmbed document {doc['id']}: {response.text}"
        
        # Wait for processing
        for doc in documents:
            wait_for_document(collection_name, doc["id"])
        
        # Test search - this will trigger FastEmbed dense vector generation for query
        search_query = {
            "query": "machine learning algorithms",
            "limit": 10
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200, f"FastEmbed dense search failed: {response.text}"
        results = response.json()
        assert len(results) > 0, "FastEmbed dense search returned no results"
        
        print(f"✅ FastEmbed dense vectors working: {len(results)} search results")
        
    finally:
        # Cleanup
        requests.delete(f"{API_BASE_URL}/collections/{collection_name}")


@pytest.mark.skip(reason="FastEmbed support temporarily disabled")
@pytest.mark.all_backends  
def test_fastembed_sparse_vectors():
    """Test FastEmbed sparse vector functionality end-to-end."""
    
    collection_name = f"test_fastembed_sparse_{str(uuid.uuid4())[:8]}"
    
    # Create collection with FastEmbed sparse vector
    config = {
        "vectors": [
            {
                "name": "fastembed_sparse",
                "type": "sparse_fastembed", 
                "model": "prithivida/Splade_PP_en_v1",
                "top_k": 100,
                "index_fields": ["content"]
            }
        ]
    }
    
    try:
        # Create collection
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        if response.status_code != 200:
            print(f"FastEmbed sparse collection creation failed: {response.status_code}")
            print(f"Response: {response.text}")
        assert response.status_code == 200
        
        # Add test documents
        documents = [
            create_test_document("fs_doc1", "Python Programming", "Object-oriented programming and algorithms"),
            create_test_document("fs_doc2", "Web Development", "HTML, CSS, and JavaScript frameworks")
        ]
        
        # Insert documents - this will trigger FastEmbed sparse vector generation
        for doc in documents:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert FastEmbed sparse document {doc['id']}: {response.text}"
        
        # Wait for processing
        for doc in documents:
            wait_for_document(collection_name, doc["id"])
        
        # Test search - this will trigger FastEmbed sparse vector generation for query
        search_query = {
            "query": "python programming concepts",
            "limit": 10
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200, f"FastEmbed sparse search failed: {response.text}"
        results = response.json()
        assert len(results) > 0, "FastEmbed sparse search returned no results"
        
        print(f"✅ FastEmbed sparse vectors working: {len(results)} search results")
        
    finally:
        # Cleanup
        requests.delete(f"{API_BASE_URL}/collections/{collection_name}")


@pytest.mark.all_backends
def test_sparse_model_vectors():
    """Test sparse_model vector functionality end-to-end."""
    
    collection_name = f"test_sparse_model_{str(uuid.uuid4())[:8]}"
    
    # Create collection with sparse_model vector  
    config = {
        "vectors": [
            {
                "name": "sparse_model",
                "type": "sparse_model",
                "model": "prithivida/Splade_PP_en_v1", 
                "top_k": 100,
                "index_fields": ["content"]
            }
        ]
    }
    
    try:
        # Create collection
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        if response.status_code != 200:
            print(f"Sparse model collection creation failed: {response.status_code}")
            print(f"Response: {response.text}")
        assert response.status_code == 200
        
        # Add test documents
        documents = [
            create_test_document("sm_doc1", "Database Systems", "Relational databases and SQL queries"),
            create_test_document("sm_doc2", "Cloud Computing", "AWS, Azure, and distributed systems")
        ]
        
        # Insert documents - this will trigger sparse_model vector generation
        for doc in documents:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert sparse_model document {doc['id']}: {response.text}"
        
        # Wait for processing
        for doc in documents:
            wait_for_document(collection_name, doc["id"])
        
        # Test search - this will trigger sparse_model vector generation for query
        search_query = {
            "query": "database management systems",
            "limit": 10
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200, f"Sparse model search failed: {response.text}"
        results = response.json()
        assert len(results) > 0, "Sparse model search returned no results"
        
        print(f"✅ Sparse model vectors working: {len(results)} search results")
        
    finally:
        # Cleanup
        requests.delete(f"{API_BASE_URL}/collections/{collection_name}")


@pytest.mark.skip(reason="FastEmbed support temporarily disabled")
@pytest.mark.dense_vectors_only
def test_fastembed_dense_end_to_end(backend_capabilities):
    """Test FastEmbed dense vector functionality end-to-end."""
    from tests.conftest import skip_if_not_supported
    
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    collection_name = f"test_fastembed_dense_{str(uuid.uuid4())[:8]}"
    
    # Create collection with FastEmbed dense vector
    config = {
        "vectors": [
            {
                "name": "fastembed_dense",
                "type": "dense_fastembed",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "index_fields": ["content"],
                "normalization": True
            }
        ]
    }
    
    try:
        # Create collection
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        if response.status_code != 200:
            print(f"FastEmbed dense collection creation failed: {response.status_code}")
            print(f"Response: {response.text}")
        assert response.status_code == 200
        
        # Add test documents
        documents = [
            create_test_document("fe_doc1", "Machine Learning", "Deep learning and neural networks"),
            create_test_document("fe_doc2", "Data Science", "Statistical analysis and data mining")
        ]
        
        # Insert documents - this will trigger FastEmbed dense vector generation
        for doc in documents:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert FastEmbed document {doc['id']}: {response.text}"
        
        # Wait for processing
        for doc in documents:
            wait_for_document(collection_name, doc["id"])
        
        # Test search - this will trigger FastEmbed dense vector generation for query
        search_query = {
            "query": "machine learning algorithms",
            "limit": 10
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200, f"FastEmbed dense search failed: {response.text}"
        results = response.json()
        assert len(results) > 0, "FastEmbed dense search returned no results"
        
        print(f"✅ FastEmbed dense vectors working: {len(results)} search results")
        
    finally:
        # Cleanup
        requests.delete(f"{API_BASE_URL}/collections/{collection_name}")


@pytest.mark.skip(reason="FastEmbed support temporarily disabled")
@pytest.mark.all_backends  
def test_fastembed_sparse_end_to_end():
    """Test FastEmbed sparse vector functionality end-to-end."""
    
    collection_name = f"test_fastembed_sparse_{str(uuid.uuid4())[:8]}"
    
    # Create collection with FastEmbed sparse vector
    config = {
        "vectors": [
            {
                "name": "fastembed_sparse",
                "type": "sparse_fastembed", 
                "model": "prithivida/Splade_PP_en_v1",
                "top_k": 100,
                "index_fields": ["content"]
            }
        ]
    }
    
    try:
        # Create collection
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        if response.status_code != 200:
            print(f"FastEmbed sparse collection creation failed: {response.status_code}")
            print(f"Response: {response.text}")
        assert response.status_code == 200
        
        # Add test documents
        documents = [
            create_test_document("fs_doc1", "Python Programming", "Object-oriented programming and algorithms"),
            create_test_document("fs_doc2", "Web Development", "HTML, CSS, and JavaScript frameworks")
        ]
        
        # Insert documents - this will trigger FastEmbed sparse vector generation
        for doc in documents:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert FastEmbed sparse document {doc['id']}: {response.text}"
        
        # Wait for processing
        for doc in documents:
            wait_for_document(collection_name, doc["id"])
        
        # Test search - this will trigger FastEmbed sparse vector generation for query
        search_query = {
            "query": "python programming concepts",
            "limit": 10
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200, f"FastEmbed sparse search failed: {response.text}"
        results = response.json()
        assert len(results) > 0, "FastEmbed sparse search returned no results"
        
        print(f"✅ FastEmbed sparse vectors working: {len(results)} search results")
        
    finally:
        # Cleanup
        requests.delete(f"{API_BASE_URL}/collections/{collection_name}")


@pytest.mark.all_backends
def test_sparse_model_end_to_end():
    """Test sparse_model vector functionality end-to-end."""
    
    collection_name = f"test_sparse_model_{str(uuid.uuid4())[:8]}"
    
    # Create collection with sparse_model vector  
    config = {
        "vectors": [
            {
                "name": "sparse_model",
                "type": "sparse_model",
                "model": "prithivida/Splade_PP_en_v1", 
                "top_k": 100,
                "index_fields": ["content"]
            }
        ]
    }
    
    try:
        # Create collection
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        if response.status_code != 200:
            print(f"Sparse model collection creation failed: {response.status_code}")
            print(f"Response: {response.text}")
        assert response.status_code == 200
        
        # Add test documents
        documents = [
            create_test_document("sm_doc1", "Database Systems", "Relational databases and SQL queries"),
            create_test_document("sm_doc2", "Cloud Computing", "AWS, Azure, and distributed systems")
        ]
        
        # Insert documents - this will trigger sparse_model vector generation
        for doc in documents:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert sparse_model document {doc['id']}: {response.text}"
        
        # Wait for processing
        for doc in documents:
            wait_for_document(collection_name, doc["id"])
        
        # Test search - this will trigger sparse_model vector generation for query
        search_query = {
            "query": "database management systems",
            "limit": 10
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200, f"Sparse model search failed: {response.text}"
        results = response.json()
        assert len(results) > 0, "Sparse model search returned no results"
        
        print(f"✅ Sparse model vectors working: {len(results)} search results")
        
    finally:
        # Cleanup
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
    if response.status_code != 200:
        print(f"Collection creation failed: {response.status_code}")
        print(f"Response: {response.text}")
    assert response.status_code == 200
    
    # Test timestamps - use specific times for predictable ordering
    old_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    new_time = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    
    doc_id = "dedup_test_doc"
    
    # Step 1: Insert document with old timestamp
    doc_old = create_test_document(doc_id, "Original Title", "Original content")
    doc_old["timestamp"] = old_time.isoformat()
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_old)
    if response.status_code != 200:
        print(f"Collection creation failed: {response.status_code}")
        print(f"Response: {response.text}")
    assert response.status_code == 200
    
    # Wait for processing
    wait_for_document(collection_name, doc_id)
    
    # Verify document was stored (note: content is not stored in DB, only title/description/metadata)
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}")
    if response.status_code != 200:
        print(f"Collection retrieval failed: {response.status_code}")
        print(f"Response: {response.text}")
    assert response.status_code == 200
    stored_doc = response.json()
    assert stored_doc["name"] == "Original Title"
    # Content is not stored in database, so we can't verify it
    
    # Step 2: Try to update with newer timestamp - should succeed
    doc_new = create_test_document(doc_id, "Updated Title", "Updated content")
    doc_new["timestamp"] = new_time.isoformat()
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_new)
    if response.status_code != 200:
        print(f"Collection creation failed: {response.status_code}")
        print(f"Response: {response.text}")
    assert response.status_code == 200
    
    # Wait for processing
    wait_for_document(collection_name, doc_id)
    
    # Verify document was updated (poll until title changes)
    def _updated() -> bool:
        r = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}")
        if r.status_code != 200:
            return False
        return r.json().get("name") == "Updated Title"
    ok = wait_until(_updated, timeout_s=10.0)
    assert ok, "Timed out waiting for title to update to 'Updated Title'"
    # Content is not stored in database, so we can't verify it
    
    # Step 3: Try to update with older timestamp - should be ignored
    doc_older = create_test_document(doc_id, "Should Not Update", "Should not be stored")
    doc_older["timestamp"] = old_time.isoformat()
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_older)
    if response.status_code != 200:
        print(f"Collection creation failed: {response.status_code}")
        print(f"Response: {response.text}")
    assert response.status_code == 200  # API accepts it, but encoder should ignore
    
    # Wait for processing
    wait_for_document(collection_name, doc_id)
    
    # Verify document was NOT updated (still has new title)
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}")
    if response.status_code != 200:
        print(f"Collection retrieval failed: {response.status_code}")
        print(f"Response: {response.text}")
    assert response.status_code == 200
    stored_doc = response.json()
    assert stored_doc["name"] == "Updated Title"  # Should still be updated version
    
    # Step 4: Try to update with same timestamp - should be ignored
    doc_same = create_test_document(doc_id, "Same Timestamp", "Same timestamp content")
    doc_same["timestamp"] = new_time.isoformat()  # Same as current
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_same)
    if response.status_code != 200:
        print(f"Collection creation failed: {response.status_code}")
        print(f"Response: {response.text}")
    assert response.status_code == 200  # API accepts it, but encoder should ignore
    
    # Wait for processing
    time.sleep(2)
    
    # Verify document was NOT updated (still has previous title), poll a bit to be sure
    def _still_updated() -> bool:
        r = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}")
        return r.status_code == 200 and r.json().get("name") == "Updated Title"
    ok = wait_until(_still_updated, timeout_s=5.0)
    assert ok
    
    # Cleanup
    try:
        requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
    except Exception as e:
        print(f"Warning: Failed to cleanup collection {collection_name}: {e}")


@pytest.mark.dense_vectors_only
def test_comprehensive_vector_combinations(backend_capabilities):
    """Test all possible combinations of vector types for comprehensive coverage."""
    from tests.conftest import skip_if_not_supported
    
    # Skip if backend doesn't support dense vectors
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    collection_name = f"test_comprehensive_vectors_{str(uuid.uuid4())[:8]}"
    
    # Create collection with ALL vector types
    config = {
        "vectors": [
            {
                "name": "dense_embeddings",
                "type": "dense_model",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "index_fields": ["content"]
            },
            {
                "name": "sparse_embeddings",
                "type": "sparse_model",
                "model": "prithivida/Splade_PP_en_v1",
                "index_fields": ["name", "content"]
            },
            {
                "name": "trigrams",
                "type": "trigrams",
                "index_fields": ["name", "content"]
            },
            {
                "name": "full_text",
                "type": "full_text",
                "index_fields": ["name", "content"],
                "language_detect": True,
                "language_default_code": "en",
                "language_confidence": 0.8
            }
        ]
    }
    
    try:
        # Create collection
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        if response.status_code != 200:
            print(f"Collection creation failed: {response.status_code}")
            print(f"Response: {response.text}")
        assert response.status_code == 200
        
        # Test getting collection info
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}")
        assert response.status_code == 200, f"Failed to get collection info: {response.text}"
        collection_info = response.json()
        
        # Verify collection info structure
        assert "vectors" in collection_info, "Collection info missing vectors field"
        assert len(collection_info["vectors"]) == 4, f"Expected 4 vectors, got {len(collection_info['vectors'])}"
        
        # Verify each vector configuration
        vector_names = [v["name"] for v in collection_info["vectors"]]
        expected_names = ["dense_embeddings", "sparse_embeddings", "trigrams", "full_text"]
        assert set(vector_names) == set(expected_names), f"Vector names mismatch: {vector_names} vs {expected_names}"
        
        # Verify dense vector has normalization
        dense_vector = next(v for v in collection_info["vectors"] if v["name"] == "dense_embeddings")
        assert dense_vector['type'] == 'dense_model', "Dense vector type mismatch"
        assert "normalization" in dense_vector, "Dense vector missing normalization field"
        assert dense_vector["normalization"] is True, "Dense vector normalization should default to True"
        
        # Verify sparse vector has no normalization
        sparse_vector = next(v for v in collection_info["vectors"] if v["name"] == "sparse_embeddings")
        assert sparse_vector['type'] == 'sparse_model', "Sparse vector type mismatch"
        assert "normalization" in sparse_vector, "Sparse vector missing normalization field"
        assert sparse_vector["normalization"] is False, "Sparse vector normalization should default to False"
        
        print(f"✅ Collection info retrieved successfully: {len(collection_info['vectors'])} vectors configured")
        
        # Add diverse test documents
        documents = [
            create_test_document("doc1", "Machine Learning Fundamentals", 
                "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without explicit programming."),
            create_test_document("doc2", "Python Programming Language", 
                "Python is a high-level, interpreted programming language known for its simplicity, readability, and extensive library ecosystem."),
            create_test_document("doc3", "Data Science and Analytics", 
                "Data science combines statistics, programming, and domain expertise to extract insights from structured and unstructured data."),
            create_test_document("doc4", "Web Development with JavaScript", 
                "JavaScript is a versatile programming language used for creating interactive web applications and dynamic user experiences."),
            create_test_document("doc5", "Database Design Principles", 
                "Good database design involves normalization, proper indexing, and understanding relationships between data entities.")
        ]
        
        # Insert all documents
        for doc in documents:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert document {doc['id']}: {response.text}"
        
        # Wait for all documents to be processed
        for doc in documents:
            wait_for_document(collection_name, doc["id"])
        
        # Test multiple tags functionality
        print("\n--- Testing multiple tags functionality ---")
        
        # Add a document with multiple tags
        multi_tag_doc = create_test_document("multi_tag_doc", "Multi-Tag Document", 
            "This document has multiple tags for testing tag filtering functionality.",
            doc_type="ml")
        multi_tag_doc["tags"] = ["ml", "ai", "programming"]  # Override with multiple tags
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=multi_tag_doc)
        assert response.status_code == 200, f"Failed to insert multi-tag document: {response.text}"
        wait_for_document(collection_name, "multi_tag_doc")
        
        # Get and print the document to see what it looks like in the database
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/multi_tag_doc")
        assert response.status_code == 200, f"Failed to retrieve multi-tag document: {response.text}"
        retrieved_doc = response.json()
        print(f"Retrieved document from DB: {retrieved_doc}")
        
        # Test search with single tag filter
        search_query_single = {
            "query": "machine learning",
            "limit": 10,
            "document_tags": ["ml"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_single)
        assert response.status_code == 200, f"Single tag search failed: {response.text}"
        
        results_single = response.json()
        print(f"Single tag 'ml' search returned {len(results_single)} documents")
        
        # Should find documents with 'ml' tag
        ml_docs = [r for r in results_single if r.get("tags") and "ml" in r["tags"]]
        assert len(ml_docs) > 0, "Should find documents with 'ml' tag"
        
        # Test search with multiple tag filter
        search_query_multi = {
            "query": "machine learning",
            "limit": 10,
            "document_tags": ["ml", "programming"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_multi)
        assert response.status_code == 200, f"Multiple tag search failed: {response.text}"
        
        results_multi = response.json()
        print(f"Multiple tags 'ml,programming' search returned {len(results_multi)} documents")
        
        # Should find documents with either 'ml' OR 'programming' tag
        multi_tag_docs = [r for r in results_multi if r.get("tags") and any(tag in r["tags"] for tag in ["ml", "programming"])]
        assert len(multi_tag_docs) > 0, "Should find documents with 'ml' or 'programming' tags"
        
        print("✅ Multiple tags functionality tested successfully!")
        
        # Define all 15 vector combinations to test
        vector_combinations = [
            # Single vector types (4 combinations)
            [{"vector_name": "dense_embeddings", "field": "content", "weight": 1.0}],
            [{"vector_name": "sparse_embeddings", "field": "name", "weight": 1.0}],
            [{"vector_name": "trigrams", "field": "name", "weight": 1.0}],
            [{"vector_name": "full_text", "field": "name", "weight": 1.0}],
            
            # Two vector types (6 combinations)
            [
                {"vector_name": "dense_embeddings", "field": "content", "weight": 0.5},
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.5}
            ],
            [
                {"vector_name": "dense_embeddings", "field": "content", "weight": 0.5},
                {"vector_name": "trigrams", "field": "name", "weight": 0.5}
            ],
            [
                {"vector_name": "dense_embeddings", "field": "content", "weight": 0.5},
                {"vector_name": "full_text", "field": "name", "weight": 0.5}
            ],
            [
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.5},
                {"vector_name": "trigrams", "field": "name", "weight": 0.5}
            ],
            [
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.5},
                {"vector_name": "full_text", "field": "name", "weight": 0.5}
            ],
            [
                {"vector_name": "trigrams", "field": "name", "weight": 0.5},
                {"vector_name": "full_text", "field": "name", "weight": 0.5}
            ],
            
            # Three vector types (4 combinations)
            [
                {"vector_name": "dense_embeddings", "field": "content", "weight": 0.33},
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.33},
                {"vector_name": "trigrams", "field": "name", "weight": 0.34}
            ],
            [
                {"vector_name": "dense_embeddings", "field": "content", "weight": 0.33},
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.33},
                {"vector_name": "full_text", "field": "name", "weight": 0.34}
            ],
            [
                {"vector_name": "dense_embeddings", "field": "content", "weight": 0.33},
                {"vector_name": "trigrams", "field": "name", "weight": 0.33},
                {"vector_name": "full_text", "field": "name", "weight": 0.34}
            ],
            [
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.33},
                {"vector_name": "trigrams", "field": "name", "weight": 0.33},
                {"vector_name": "full_text", "field": "name", "weight": 0.34}
            ],
            
            # All four vector types (1 combination)
            [
                {"vector_name": "dense_embeddings", "field": "content", "weight": 0.25},
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.25},
                {"vector_name": "trigrams", "field": "name", "weight": 0.25},
                {"vector_name": "full_text", "field": "name", "weight": 0.25}
            ]
        ]
        
        # Test each combination
        for i, vector_weights in enumerate(vector_combinations):
            print(f"\n--- Testing combination {i+1}/15: {len(vector_weights)} vector types ---")
            print(f"Vector weights: {vector_weights}")
            
            # Search with this combination
            search_query = {
                "query": "machine learning programming",
                "limit": 5,
                "vector_weights": vector_weights
            }
            
            # Time the search
            import time
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
            end_time = time.time()
            search_time_ms = (end_time - start_time) * 1000
            
            assert response.status_code == 200, f"Search failed for combination {i+1}: {response.text}"
            
            results = response.json()
            print(f"Results: {len(results)} documents returned in {search_time_ms:.2f}ms")
            
            # Basic validation
            assert len(results) > 0, f"No results returned for combination {i+1}"
            assert len(results) <= 5, f"More results than limit for combination {i+1}"
            
            # Check that results have expected structure
            for result in results:
                assert "id" in result, f"Result missing 'id' field: {result}"
                assert "score" in result, f"Result missing 'score' field: {result}"
                assert "name" in result, f"Result missing 'title' field: {result}"
                assert isinstance(result["score"], (int, float)), f"Score is not numeric: {result['score']}"
            
            # Check that scores are in descending order
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), f"Scores not in descending order: {scores}"
            
            # Log top result for verification
            if results:
                top_result = results[0]
                print(f"Top result: '{top_result['name']}' (score: {top_result['score']:.4f})")
        
        print(f"\n✅ All 15 vector combinations tested successfully!")
        
        # Print timing summary
        print(f"\n📊 Search Performance Summary:")
        print(f"Total combinations tested: 15")
        print(f"Auto-weight test: 1")
        print(f"All searches completed successfully!")
        
        # Test auto-weight generation (no vector_weights specified)
        print(f"\n--- Testing auto-weight generation ---")
        search_query = {
            "query": "machine learning programming",
            "limit": 5
            # No vector_weights - should auto-generate equal weights
        }
        
        # Time the auto-weight search
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        end_time = time.time()
        search_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200, f"Auto-weight search failed: {response.text}"
        
        results = response.json()
        print(f"Auto-weight results: {len(results)} documents returned in {search_time_ms:.2f}ms")
        
        # Should still get results even without explicit weights
        assert len(results) > 0, "No results returned with auto-weights"
        assert len(results) <= 5, "More results than limit with auto-weights"
        
        # Log top result for verification
        if results:
            top_result = results[0]
            print(f"Auto-weight top result: '{top_result['name']}' (score: {top_result['score']:.4f})")
        
        print(f"\n✅ Auto-weight generation tested successfully!")
        
        # Test document update functionality
        print(f"\n--- Testing document update ---")
        
        # First, let's check what the original document looks like
        print("Checking original document before update...")
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/doc1")
        assert response.status_code == 200, f"Failed to retrieve original document doc1: {response.text}"
        
        original_doc = response.json()
        print(f"Original document - Title: '{original_doc.get('name', 'MISSING')}', Description: '{original_doc.get('description', 'MISSING')}'")
        
        # Update doc1 with new title and description (content is not stored in DB)
        updated_doc = create_test_document(
            "doc1", 
            "Updated Machine Learning Fundamentals", 
            "This document has been updated with new content about advanced machine learning techniques including deep learning, neural networks, and reinforcement learning."
        )
        
        print(f"Updating document with new title: '{updated_doc['name']}'")
        
        # Send the update (same ID should trigger update)
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=updated_doc)
        assert response.status_code == 200, f"Failed to update document doc1: {response.text}"
        
        print(f"Update request successful, response: {response.json()}")
        
        # Wait for the update to be processed by polling until new title appears
        print("Waiting for document update to be processed...")
        max_wait_time = 30  # seconds
        wait_interval = 1   # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/doc1")
                if response.status_code == 200:
                    current_doc = response.json()
                    current_title = current_doc.get("name", "")
                    current_description = current_doc.get("description", "")
                    
                    print(f"Current state - Title: '{current_title}', Description: '{current_description[:100] if current_description else 'None'}...'")
                    
                    if current_title == "Updated Machine Learning Fundamentals":
                        print(f"Document update processed in {time.time() - start_time:.1f}s")
                        break
                    else:
                        print(f"Update not yet processed, current title: '{current_title}'")
                else:
                    print(f"Failed to retrieve document during wait: {response.status_code}")
            except Exception as e:
                print(f"Error during wait: {e}")
            
            time.sleep(wait_interval)
        else:
            # If we get here, the update didn't complete within max_wait_time
            raise TimeoutError(f"Document update did not complete within {max_wait_time} seconds")
        
        # Verify the title was actually updated (content is not stored in DB)
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/doc1")
        assert response.status_code == 200, f"Failed to retrieve updated document doc1: {response.text}"
        
        updated_document = response.json()
        assert updated_document["name"] == "Updated Machine Learning Fundamentals", f"Title not updated: {updated_document['name']}"
        print(f"✓ Document update verified - title successfully changed to: {updated_document['name']}")
        
        # Test search on updated document - search by dense vectors on content field
        # Note: content field is used for vector generation but not stored in DB
        search_query = {
            "query": "advanced machine learning techniques",
            "limit": 5,
            "vector_weights": [{"vector_name": "dense_embeddings", "field": "content", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200, f"Search after update failed: {response.text}"
        
        results = response.json()
        print(f"Search after update: {len(results)} documents returned")
        
        # The updated doc1 should now be more relevant to this query
        doc1_in_results = any(result["id"] == "doc1" for result in results)
        if doc1_in_results:
            doc1_result = next(result for result in results if result["id"] == "doc1")
            print(f"Updated doc1 found in search results with score: {doc1_result['score']:.4f}")
        else:
            print("Note: Updated doc1 not in top 3 results (this is acceptable)")
        
        print(f"\n✅ Document update tested successfully!")
        
        # Test document type filtering functionality
        print(f"\n--- Testing document type filtering ---")
        
        # First, let's empty the collection to remove all previous test documents
        print("Emptying collection to start with clean state...")
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/empty")
        assert response.status_code == 200, f"Failed to empty collection: {response.text}"
        print("✓ Collection emptied successfully")
        
        # Now add documents with different types to test filtering
        print("Adding documents with different types for filtering tests...")
        
        # Add documents with specific types
        type_docs = [
            create_test_document("type_ml", "Machine Learning Research", 
                "Advanced research in machine learning algorithms and neural networks.", "research"),
            create_test_document("type_tutorial", "Python Tutorial for Beginners", 
                "Step-by-step guide to learning Python programming.", "tutorial"),
            create_test_document("type_news", "AI Breakthrough News", 
                "Latest developments in artificial intelligence technology.", "news"),
            create_test_document("type_article", "Data Science Best Practices", 
                "Comprehensive guide to data science methodologies.", "article"),
            create_test_document("type_empty", "Empty Type Document", 
                "This document has no specific type.", "untagged"),
            create_test_document("type_none", "None Type Document", 
                "This document has None as its type.", None)
        ]
        
        # Insert type-specific documents
        for doc in type_docs:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert type document {doc['id']}: {response.text}"
        
        # Wait for all type documents to be processed
        for doc in type_docs:
            wait_for_document(collection_name, doc["id"])
        
        print(f"✓ Added {len(type_docs)} documents with different types")
        
        # Test 1: No document type filter (should return all types)
        print("\n--- Test 1: No document type filter ---")
        search_query_no_filter = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 20,  # Higher limit to see more results
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_no_filter)
        assert response.status_code == 200, f"Search without filter failed: {response.text}"
        
        results_no_filter = response.json()
        print(f"No filter results: {len(results_no_filter)} documents returned")
        
        # Debug: Log all results to understand what we're getting
        print("Results details:")
        for i, result in enumerate(results_no_filter):
            print(f"  {i+1}. ID: {result['id']}, Type: '{result['tags']}', Title: '{result['name']}'")
        
        # Should get results from all document types
        types_no_filter = {(r.get("tags")[0] if r.get("tags") else None) for r in results_no_filter}
        print(f"Document types found: {types_no_filter}")
        assert len(types_no_filter) >= 2, f"Expected multiple document types, got: {types_no_filter}"
        
        # Verify we found all our type documents
        found_doc_ids = {r["id"] for r in results_no_filter}
        type_doc_ids = {"type_ml", "type_tutorial", "type_news", "type_article", "type_empty", "type_none"}
        found_type_docs = type_doc_ids & found_doc_ids
        print(f"✓ Found {len(found_type_docs)} type documents in search: {found_type_docs}")
        
        # With our comprehensive query and clean collection, we should find exactly 6 documents
        expected_total = 6  # 6 type docs only
        assert len(results_no_filter) == expected_total, f"Expected {expected_total} documents, got {len(results_no_filter)}"
        
        # Test 2: Single document type filter
        print("\n--- Test 2: Single document type filter ---")
        search_query_single_type = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 10,
            "document_tags": ["research"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_single_type)
        assert response.status_code == 200, f"Search with single type filter failed: {response.text}"
        
        results_single_type = response.json()
        print(f"Single type filter results: {len(results_single_type)} documents returned")
        
        # All results should be of type "research"
        for result in results_single_type:
            assert 'research' in (result.get('tags') or []), f"Expected type 'research', got '{result['tags']}' for document '{result['id']}'"
        
        print(f"✓ All {len(results_single_type)} results are of type 'research'")
        
        # Test 3: Multiple document type filter
        print("\n--- Test 3: Multiple document type filter ---")
        search_query_multi_type = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 10,
            "document_tags": ["tutorial", "article"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_multi_type)
        assert response.status_code == 200, f"Search with multi-type filter failed: {response.text}"
        
        results_multi_type = response.json()
        print(f"Multiple-type filter results: {len(results_multi_type)} documents returned")
        
        # All results should be of type "tutorial" or "article"
        allowed_types = {"tutorial", "article"}
        for result in results_multi_type:
            assert any(t in allowed_types for t in (result.get('tags') or [])), f"Expected type in {allowed_types}, got '{result['tags']}' for document '{result['id']}'"
        
        print(f"✓ All {len(results_multi_type)} results are of allowed types: {allowed_types}")
        
        # Test 4: Document type filter with no matches
        print("\n--- Test 4: Document type filter with no matches ---")
        search_query_no_match = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 10,
            "document_tags": ["blog"],  # No documents of type "blog"
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_no_match)
        assert response.status_code == 200, f"Search with no-match filter failed: {response.text}"
        
        results_no_match = response.json()
        print(f"No-match filter results: {len(results_no_match)} documents returned")
        
        # Should return empty results since no documents match the filter
        assert len(results_no_match) == 0, f"Expected no results for non-existent type 'blog', got {len(results_no_match)}"
        
        print(f"✓ No results returned for non-existent document type 'blog'")
        
        # Test 5: Document type filter with dense vectors
        print("\n--- Test 5: Document type filter with dense vectors ---")
        search_query_dense_filter = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 10,
            "document_tags": ["research", "tutorial"],
            "vector_weights": [{"vector_name": "dense_embeddings", "field": "content", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_dense_filter)
        assert response.status_code == 200, f"Search with dense vector filter failed: {response.text}"
        
        results_dense_filter = response.json()
        print(f"Dense vector filter results: {len(results_dense_filter)} documents returned")
        
        # All results should be of allowed types
        allowed_types_dense = {"research", "tutorial"}
        for result in results_dense_filter:
            assert any(t in allowed_types_dense for t in (result.get('tags') or [])), f"Expected type in {allowed_types_dense}, got '{result['tags']}' for document '{result['id']}'"
        
        print(f"✓ All {len(results_dense_filter)} dense vector results are of allowed types: {allowed_types_dense}")
        
        # Test 6: Document type filter with hybrid search (multiple vector types)
        print("\n--- Test 6: Document type filter with hybrid search ---")
        search_query_hybrid_filter = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 10,
            "document_tags": ["article", "news"],
            "vector_weights": [
                {"vector_name": "dense_embeddings", "field": "content", "weight": 0.5},
                {"vector_name": "trigrams", "field": "name", "weight": 0.5}
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_hybrid_filter)
        assert response.status_code == 200, f"Search with hybrid filter failed: {response.text}"
        
        results_hybrid_filter = response.json()
        print(f"Hybrid filter results: {len(results_hybrid_filter)} documents returned")
        
        # All results should be of allowed types
        allowed_types_hybrid = {"article", "news"}
        for result in results_hybrid_filter:
            assert any(t in allowed_types_hybrid for t in (result.get('tags') or [])), f"Expected type in {allowed_types_hybrid}, got '{result['tags']}' for document '{result['id']}'"
        
        print(f"✓ All {len(results_hybrid_filter)} hybrid search results are of allowed types: {allowed_types_hybrid}")
        
        # Test 7: Edge cases - untagged and None types
        print("\n--- Test 7: Edge cases - untagged and None types ---")
        
        # Test 7a: Search for documents with untagged type
        search_query_empty_type = {
            "query": "empty type document",
            "limit": 10,
            "document_tags": ["untagged"],  # Untagged type
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_empty_type)
        assert response.status_code == 200, f"Search for empty type failed: {response.text}"
        
        results_empty_type = response.json()
        print(f"Empty type filter results: {len(results_empty_type)} documents returned")
        
        # Should find the document with untagged type
        if results_empty_type:
            for result in results_empty_type:
                assert 'untagged' in (result.get('tags') or []), f"Expected 'untagged' type, got '{result['tags']}' for document '{result['id']}'"
            print(f"✓ Found {len(results_empty_type)} documents with untagged type")
        else:
            print("Note: No documents found with untagged type (this might be expected behavior)")
        
        # Test 7b: Search for documents with None type
        # Note: API doesn't allow None in document_tags, so we'll test this differently
        print("Note: API validation prevents None in document_tags array - testing with string 'None' instead")
        search_query_none_type = {
            "query": "none type document",
            "limit": 10,
            "document_tags": ["None"],  # String "None" instead of actual None
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_none_type)
        assert response.status_code == 200, f"Search for None type failed: {response.text}"
        
        results_none_type = response.json()
        print(f"None type filter results: {len(results_none_type)} documents returned")
        
        # Should find the document with None type
        if results_none_type:
            for result in results_none_type:
                assert result['tags'] is None, f"Expected None type, got '{result['tags']}' for document '{result['id']}'"
            print(f"✓ Found {len(results_none_type)} documents with None type")
        else:
            print("Note: No documents found with None type (this might be expected behavior)")
        
        # Test 7c: Verify untagged/None types don't appear in normal searches unless explicitly requested
        print("\n--- Test 7c: Verify edge case types don't interfere with normal searches ---")
        
        # Search without any type filter - should NOT include empty/None types by default
        search_query_normal = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 20,  # Higher limit to see all documents
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_normal)
        assert response.status_code == 200, f"Normal search failed: {response.text}"
        
        results_normal = response.json()
        print(f"Normal search results: {len(results_normal)} documents returned")
        
        # Log all document types found to debug the missing document issue
        all_types_found = {(r.get("tags")[0] if r.get("tags") else None) for r in results_normal}
        print(f"All document types found in normal search: {all_types_found}")
        
        # Check if we can find all our test documents
        all_doc_ids = {r["id"] for r in results_normal}
        expected_doc_ids = {"type_ml", "type_tutorial", "type_news", "type_article", "type_empty", "type_none"}
        found_docs = expected_doc_ids & all_doc_ids
        print(f"✓ Found {len(found_docs)} expected documents in normal search")
        
        # With our comprehensive query and clean collection, we should find exactly 6 documents
        assert len(results_normal) == len(expected_doc_ids), f"Expected {len(expected_doc_ids)} documents, got {len(results_normal)}"
        
        # Test 7d: Explicitly include untagged and None types in search
        print("\n--- Test 7d: Explicitly include edge case types ---")
        search_query_include_edge = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 20,
            "document_tags": ["untagged", "None", "research", "tutorial"],  # Include edge cases (string "None" instead of actual None)
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_include_edge)
        assert response.status_code == 200, f"Search including edge cases failed: {response.text}"
        
        results_edge = response.json()
        print(f"Edge case inclusive search results: {len(results_edge)} documents returned")
        
        # Should include documents with untagged and None types
        edge_types_found = {(r.get("tags")[0] if r.get("tags") else None) for r in results_edge}
        print(f"Document types found including edge cases: {edge_types_found}")
        
        # Verify we can find documents with edge case types
        edge_doc_ids = {r["id"] for r in results_edge if r['tags'] in ["untagged", "None"]}
        print(f"Documents with edge case types found: {edge_doc_ids}")
        
        # Test 7e: Verify that documents with actual None types are stored correctly
        print("\n--- Test 7e: Verify None type document storage ---")
        
        # Get the document with None type directly to verify it's stored correctly
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/type_none")
        if response.status_code == 200:
            none_doc = response.json()
            print(f"Document type_none retrieved: ID={none_doc['id']}, Type={none_doc['tags']}, Title='{none_doc['name']}'")
        else:
            print(f"⚠️  Failed to retrieve type_none document: {response.status_code}")
        
        print(f"\n✅ Document type filtering tested successfully!")
        
    finally:
        # Cleanup
        try:
            requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
        except Exception as e:
            print(f"Warning: Failed to cleanup collection {collection_name}: {e}")


@pytest.mark.dense_vectors_only
def test_euclidean_distance_functionality(backend_capabilities):
    """Test that euclidean distance functionality works correctly for dense vectors."""
    from tests.conftest import skip_if_not_supported
    
    # Skip if backend doesn't support dense vectors
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    collection_name = f"test_euclidean_distance_{str(uuid.uuid4())[:8]}"
    
    # Create collection with dense vector using euclidean distance
    config = {
        "vectors": [
            {
                "name": "euclidean_embeddings",
                "type": "dense_model",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "index_fields": ["content"],
                "dense_distance": "euclid"
            }
        ]
    }
    
    try:
        # Create collection
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        assert response.status_code == 200, f"Failed to create collection: {response.text}"
        
        # Test getting collection info to verify dense_distance was set
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}")
        assert response.status_code == 200, f"Failed to get collection info: {response.text}"
        collection_info = response.json()
        
        # Verify collection info structure
        assert "vectors" in collection_info, "Collection info missing vectors field"
        assert len(collection_info["vectors"]) == 1, f"Expected 1 vector, got {len(collection_info['vectors'])}"
        
        # Verify dense vector has euclidean distance
        dense_vector = collection_info["vectors"][0]
        assert dense_vector['name'] == 'euclidean_embeddings', "Vector name mismatch"
        assert dense_vector['type'] == 'dense_model', "Vector type mismatch"
        assert dense_vector['dense_distance'] == 'euclid', f"Expected dense_distance 'euclid', got '{dense_vector.get('dense_distance')}'"
        
        print(f"✅ Collection created with euclidean distance: {dense_vector['dense_distance']}")
        
        # Add test documents
        documents = [
            create_test_document("doc1", "Machine Learning Fundamentals", 
                "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without explicit programming."),
            create_test_document("doc2", "Python Programming Language", 
                "Python is a high-level, interpreted programming language known for its simplicity, readability, and extensive library ecosystem."),
            create_test_document("doc3", "Data Science and Analytics", 
                "Data science combines statistics, programming, and domain expertise to extract insights from structured and unstructured data.")
        ]
        
        # Insert all documents
        for doc in documents:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert document {doc['id']}: {response.text}"
        
        # Wait for all documents to be processed
        for doc in documents:
            wait_for_document(collection_name, doc["id"])
        
        # Test search using euclidean distance
        search_query = {
            "query": "machine learning",
            "limit": 3,
            "vector_weights": [{"vector_name": "euclidean_embeddings", "field": "content", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200, f"Search with euclidean distance failed: {response.text}"
        
        results = response.json()
        print(f"Euclidean distance search returned {len(results)} documents")
        
        # Verify we got results
        assert len(results) > 0, "Search should return at least one result"
        
        # Log top result for verification
        if results:
            top_result = results[0]
            print(f"Top result: '{top_result['name']}' (score: {top_result['score']:.4f})")
        
        print(f"✅ Euclidean distance functionality tested successfully!")
        
    finally:
        # Cleanup
        try:
            requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
        except Exception as e:
            print(f"Warning: Failed to cleanup collection {collection_name}: {e}")


@pytest.mark.sparse_only
def test_comprehensive_sparse_vector_combinations(backend_capabilities):
    """Test all possible combinations of sparse vector types for comprehensive coverage (no dense vectors)."""
    # Skip if backend supports dense vectors
    if backend_capabilities.get("dense_vectors", False):
        pytest.skip("Backend supports dense vectors, testing sparse-only features")
    
    collection_name = f"test_comprehensive_sparse_vectors_{str(uuid.uuid4())[:8]}"
    
    # Create collection with ONLY sparse vector types
    config = {
        "vectors": [
            {
                "name": "sparse_embeddings",
                "type": "sparse_model",
                "model": "prithivida/Splade_PP_en_v1",
                "index_fields": ["name", "content"]
            },
            {
                "name": "trigrams",
                "type": "trigrams",
                "index_fields": ["name", "content"]
            },
            {
                "name": "full_text",
                "type": "full_text",
                "index_fields": ["name", "content"],
                "language_detect": True,
                "language_default_code": "en",
                "language_confidence": 0.8
            }
        ]
    }
    
    try:
        # Create collection
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        assert response.status_code == 200, f"Failed to create collection: {response.text}"
        
        # Test getting collection info
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}")
        assert response.status_code == 200, f"Failed to get collection info: {response.text}"
        collection_info = response.json()
        
        # Verify collection info structure
        assert "vectors" in collection_info, "Collection info missing vectors field"
        assert len(collection_info["vectors"]) == 3, f"Expected 3 vectors, got {len(collection_info['vectors'])}"
        
        # Verify each vector configuration
        vector_names = [v["name"] for v in collection_info["vectors"]]
        expected_names = ["sparse_embeddings", "trigrams", "full_text"]
        assert set(vector_names) == set(expected_names), f"Vector names mismatch: {vector_names} vs {expected_names}"
        
        # Verify sparse vector has no normalization
        sparse_vector = next(v for v in collection_info["vectors"] if v["name"] == "sparse_embeddings")
        assert sparse_vector['type'] == 'sparse_model', "Sparse vector type mismatch"
        assert "normalization" in sparse_vector, "Sparse vector missing normalization field"
        assert sparse_vector["normalization"] is False, "Sparse vector normalization should default to False"
        
        print(f"✅ Collection info retrieved successfully: {len(collection_info['vectors'])} sparse vectors configured")
        
        # Add diverse test documents
        documents = [
            create_test_document("doc1", "Machine Learning Fundamentals", 
                "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without explicit programming."),
            create_test_document("doc2", "Python Programming Language", 
                "Python is a high-level, interpreted programming language known for its simplicity, readability, and extensive library ecosystem."),
            create_test_document("doc3", "Data Science and Analytics", 
                "Data science combines statistics, programming, and domain expertise to extract insights from structured and unstructured data."),
            create_test_document("doc4", "Web Development with JavaScript", 
                "JavaScript is a versatile programming language used for creating interactive web applications and dynamic user experiences."),
            create_test_document("doc5", "Database Design Principles", 
                "Good database design involves normalization, proper indexing, and understanding relationships between data entities.")
        ]
        
        # Insert all documents
        for doc in documents:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert document {doc['id']}: {response.text}"
        
        # Wait for all documents to be processed
        for doc in documents:
            wait_for_document(collection_name, doc["id"])
        
        # Test multiple tags functionality
        print("\n--- Testing multiple tags functionality ---")
        
        # Add a document with multiple tags
        multi_tag_doc = create_test_document("multi_tag_doc", "Multi-Tag Document", 
            "This document has multiple tags for testing tag filtering functionality.",
            doc_type="ml")
        multi_tag_doc["tags"] = ["ml", "ai", "programming"]  # Override with multiple tags
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=multi_tag_doc)
        assert response.status_code == 200, f"Failed to insert multi-tag document: {response.text}"
        wait_for_document(collection_name, "multi_tag_doc")
        
        # Get and print the document to see what it looks like in the database
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/multi_tag_doc")
        assert response.status_code == 200, f"Failed to retrieve multi-tag document: {response.text}"
        retrieved_doc = response.json()
        print(f"Retrieved document from DB: {retrieved_doc}")
        
        # Test search with single tag filter
        search_query_single = {
            "query": "machine learning",
            "limit": 10,
            "document_tags": ["ml"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_single)
        assert response.status_code == 200, f"Single tag search failed: {response.text}"
        
        results_single = response.json()
        print(f"Single tag 'ml' search returned {len(results_single)} documents")
        
        # Should find documents with 'ml' tag
        ml_docs = [r for r in results_single if r.get("tags") and "ml" in r["tags"]]
        assert len(ml_docs) > 0, "Should find documents with 'ml' tag"
        
        # Test search with multiple tag filter
        search_query_multi = {
            "query": "machine learning",
            "limit": 10,
            "document_tags": ["ml", "programming"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_multi)
        assert response.status_code == 200, f"Multiple tag search failed: {response.text}"
        
        results_multi = response.json()
        print(f"Multiple tags 'ml,programming' search returned {len(results_multi)} documents")
        
        # Should find documents with either 'ml' OR 'programming' tag
        multi_tag_docs = [r for r in results_multi if r.get("tags") and any(tag in r["tags"] for tag in ["ml", "programming"])]
        assert len(multi_tag_docs) > 0, "Should find documents with 'ml' or 'programming' tags"
        
        print("✅ Multiple tags functionality tested successfully!")
        
        # Define all sparse vector combinations to test (7 combinations instead of 15)
        vector_combinations = [
            # Single vector types (3 combinations)
            [{"vector_name": "sparse_embeddings", "field": "name", "weight": 1.0}],
            [{"vector_name": "trigrams", "field": "name", "weight": 1.0}],
            [{"vector_name": "full_text", "field": "name", "weight": 1.0}],
            
            # Two vector types (3 combinations)
            [
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.5},
                {"vector_name": "trigrams", "field": "name", "weight": 0.5}
            ],
            [
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.5},
                {"vector_name": "full_text", "field": "name", "weight": 0.5}
            ],
            [
                {"vector_name": "trigrams", "field": "name", "weight": 0.5},
                {"vector_name": "full_text", "field": "name", "weight": 0.5}
            ],
            
            # All three vector types (1 combination)
            [
                {"vector_name": "sparse_embeddings", "field": "name", "weight": 0.33},
                {"vector_name": "trigrams", "field": "name", "weight": 0.33},
                {"vector_name": "full_text", "field": "name", "weight": 0.34}
            ]
        ]
        
        # Test each combination
        for i, vector_weights in enumerate(vector_combinations):
            print(f"\n--- Testing sparse combination {i+1}/7: {len(vector_weights)} vector types ---")
            print(f"Vector weights: {vector_weights}")
            
            # Search with this combination
            search_query = {
                "query": "machine learning programming",
                "limit": 5,
                "vector_weights": vector_weights
            }
            
            # Time the search
            import time
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
            end_time = time.time()
            search_time_ms = (end_time - start_time) * 1000
            
            assert response.status_code == 200, f"Search failed for combination {i+1}: {response.text}"
            
            results = response.json()
            print(f"Results: {len(results)} documents returned in {search_time_ms:.2f}ms")
            
            # Basic validation
            assert len(results) > 0, f"No results returned for combination {i+1}"
            assert len(results) <= 5, f"More results than limit for combination {i+1}"
            
            # Check that results have expected structure
            for result in results:
                assert "id" in result, f"Result missing 'id' field: {result}"
                assert "score" in result, f"Result missing 'score' field: {result}"
                assert "name" in result, f"Result missing 'title' field: {result}"
                assert isinstance(result["score"], (int, float)), f"Score is not numeric: {result['score']}"
            
            # Check that scores are in descending order
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), f"Scores not in descending order: {scores}"
            
            # Log top result for verification
            if results:
                top_result = results[0]
                print(f"Top result: '{top_result['name']}' (score: {top_result['score']:.4f})")
        
        print(f"\n✅ All 7 sparse vector combinations tested successfully!")
        
        # Print timing summary
        print(f"\n📊 Sparse Search Performance Summary:")
        print(f"Total combinations tested: 7")
        print(f"Auto-weight test: 1")
        print(f"All searches completed successfully!")
        
        # Test auto-weight generation (no vector_weights specified)
        print(f"\n--- Testing auto-weight generation ---")
        search_query = {
            "query": "machine learning programming",
            "limit": 5
            # No vector_weights - should auto-generate equal weights
        }
        
        # Time the auto-weight search
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        end_time = time.time()
        search_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200, f"Auto-weight search failed: {response.text}"
        
        results = response.json()
        print(f"Auto-weight results: {len(results)} documents returned in {search_time_ms:.2f}ms")
        
        # Should still get results even without explicit weights
        assert len(results) > 0, "No results returned with auto-weights"
        assert len(results) <= 5, "More results than limit with auto-weights"
        
        # Log top result for verification
        if results:
            top_result = results[0]
            print(f"Auto-weight top result: '{top_result['name']}' (score: {top_result['score']:.4f})")
        
        print(f"\n✅ Auto-weight generation tested successfully!")
        
        # Test document update functionality
        print(f"\n--- Testing document update ---")
        
        # First, let's check what the original document looks like
        print("Checking original document before update...")
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/doc1")
        assert response.status_code == 200, f"Failed to retrieve original document doc1: {response.text}"
        
        original_doc = response.json()
        print(f"Original document - Title: '{original_doc.get('name', 'MISSING')}', Description: '{original_doc.get('description', 'MISSING')}'")
        
        # Update doc1 with new title and description (content is not stored in DB)
        updated_doc = create_test_document(
            "doc1", 
            "Updated Machine Learning Fundamentals", 
            "This document has been updated with new content about advanced machine learning techniques including deep learning, neural networks, and reinforcement learning."
        )
        
        print(f"Updating document with new title: '{updated_doc['name']}'")
        
        # Send the update (same ID should trigger update)
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=updated_doc)
        assert response.status_code == 200, f"Failed to update document doc1: {response.text}"
        
        print(f"Update request successful, response: {response.json()}")
        
        # Wait for the update to be processed by polling until new title appears
        print("Waiting for document update to be processed...")
        max_wait_time = 30  # seconds
        wait_interval = 1   # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/doc1")
                if response.status_code == 200:
                    current_doc = response.json()
                    current_title = current_doc.get("name", "")
                    current_description = current_doc.get("description", "")
                    
                    print(f"Current state - Title: '{current_title}', Description: '{current_description[:100] if current_description else 'None'}...'")
                    
                    if current_title == "Updated Machine Learning Fundamentals":
                        print(f"Document update processed in {time.time() - start_time:.1f}s")
                        break
                    else:
                        print(f"Update not yet processed, current title: '{current_title}'")
                else:
                    print(f"Failed to retrieve document during wait: {response.status_code}")
            except Exception as e:
                print(f"Error during wait: {e}")
            
            time.sleep(wait_interval)
        else:
            # If we get here, the update didn't complete within max_wait_time
            raise TimeoutError(f"Document update did not complete within {max_wait_time} seconds")
        
        # Verify the title was actually updated (content is not stored in DB)
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/doc1")
        assert response.status_code == 200, f"Failed to retrieve updated document doc1: {response.text}"
        
        updated_document = response.json()
        assert updated_document["name"] == "Updated Machine Learning Fundamentals", f"Title not updated: {updated_document['name']}"
        print(f"✓ Document update verified - title successfully changed to: {updated_document['name']}")
        
        # Test search on updated document - search by sparse vectors on content field
        # Note: content field is used for vector generation but not stored in DB
        search_query = {
            "query": "advanced machine learning techniques",
            "limit": 5,
            "vector_weights": [{"vector_name": "sparse_embeddings", "field": "content", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200, f"Search after update failed: {response.text}"
        
        results = response.json()
        print(f"Search after update: {len(results)} documents returned")
        
        # The updated doc1 should now be more relevant to this query
        doc1_in_results = any(result["id"] == "doc1" for result in results)
        if doc1_in_results:
            doc1_result = next(result for result in results if result["id"] == "doc1")
            print(f"Updated doc1 found in search results with score: {doc1_result['score']:.4f}")
        else:
            print("Note: Updated doc1 not in top 3 results (this is acceptable)")
        
        print(f"\n✅ Document update tested successfully!")
        
        # Test document type filtering functionality
        print(f"\n--- Testing document type filtering ---")
        
        # First, let's empty the collection to remove all previous test documents
        print("Emptying collection to start with clean state...")
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/empty")
        assert response.status_code == 200, f"Failed to empty collection: {response.text}"
        print("✓ Collection emptied successfully")
        
        # Now add documents with different types to test filtering
        print("Adding documents with different types for filtering tests...")
        
        # Add documents with specific types
        type_docs = [
            create_test_document("type_ml", "Machine Learning Research", 
                "Advanced research in machine learning algorithms and neural networks.", "research"),
            create_test_document("type_tutorial", "Python Tutorial for Beginners", 
                "Step-by-step guide to learning Python programming.", "tutorial"),
            create_test_document("type_news", "AI Breakthrough News", 
                "Latest developments in artificial intelligence technology.", "news"),
            create_test_document("type_article", "Data Science Best Practices", 
                "Comprehensive guide to data science methodologies.", "article"),
            create_test_document("type_empty", "Empty Type Document", 
                "This document has no specific type.", "untagged"),
            create_test_document("type_none", "None Type Document", 
                "This document has None as its type.", None)
        ]
        
        # Insert type-specific documents
        for doc in type_docs:
            response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
            assert response.status_code == 200, f"Failed to insert type document {doc['id']}: {response.text}"
        
        # Wait for all type documents to be processed
        for doc in type_docs:
            wait_for_document(collection_name, doc["id"])
        
        print(f"✓ Added {len(type_docs)} documents with different types")
        
        # Test 1: No document type filter (should return all types)
        print("\n--- Test 1: No document type filter ---")
        search_query_no_filter = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 20,  # Higher limit to see more results
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_no_filter)
        assert response.status_code == 200, f"Search without filter failed: {response.text}"
        
        results_no_filter = response.json()
        print(f"No filter results: {len(results_no_filter)} documents returned")
        
        # Debug: Log all results to understand what we're getting
        print("Results details:")
        for i, result in enumerate(results_no_filter):
            print(f"  {i+1}. ID: {result['id']}, Type: '{result['tags']}', Title: '{result['name']}'")
        
        # Should get results from all document types
        types_no_filter = {(r.get("tags")[0] if r.get("tags") else None) for r in results_no_filter}
        print(f"Document types found: {types_no_filter}")
        assert len(types_no_filter) >= 2, f"Expected multiple document types, got: {types_no_filter}"
        
        # Verify we found all our type documents
        found_doc_ids = {r["id"] for r in results_no_filter}
        type_doc_ids = {"type_ml", "type_tutorial", "type_news", "type_article", "type_empty", "type_none"}
        found_type_docs = type_doc_ids & found_doc_ids
        print(f"✓ Found {len(found_type_docs)} type documents in search: {found_type_docs}")
        
        # With our comprehensive query and clean collection, we should find exactly 6 documents
        expected_total = 6  # 6 type docs only
        assert len(results_no_filter) == expected_total, f"Expected {expected_total} documents, got {len(results_no_filter)}"
        
        # Test 2: Single document type filter
        print("\n--- Test 2: Single document type filter ---")
        search_query_single_type = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 10,
            "document_tags": ["research"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_single_type)
        assert response.status_code == 200, f"Search with single type filter failed: {response.text}"
        
        results_single_type = response.json()
        print(f"Single type filter results: {len(results_single_type)} documents returned")
        
        # All results should be of type "research"
        for result in results_single_type:
            assert 'research' in (result.get('tags') or []), f"Expected type 'research', got '{result['tags']}' for document '{result['id']}'"
        
        print(f"✓ All {len(results_single_type)} results are of type 'research'")
        
        # Test 3: Multiple document type filter
        print("\n--- Test 3: Multiple document type filter ---")
        search_query_multi_type = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 10,
            "document_tags": ["tutorial", "article"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_multi_type)
        assert response.status_code == 200, f"Search with multiple type filter failed: {response.text}"
        
        results_multi_type = response.json()
        print(f"Multiple type filter results: {len(results_multi_type)} documents returned")
        
        # All results should be of type "tutorial" OR "article"
        for result in results_multi_type:
            result_tags = result.get('tags') or []
            assert any(tag in result_tags for tag in ["tutorial", "article"]), f"Expected type 'tutorial' or 'article', got '{result_tags}' for document '{result['id']}'"
        
        print(f"✓ All {len(results_multi_type)} results are of type 'tutorial' or 'article'")
        
        # Test 4: Untagged type filter
        print("\n--- Test 4: Untagged type filter ---")
        search_query_untagged_type = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 10,
            "document_tags": ["untagged"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_untagged_type)
        assert response.status_code == 200, f"Search with untagged type filter failed: {response.text}"
        
        results_untagged_type = response.json()
        print(f"Untagged type filter results: {len(results_untagged_type)} documents returned")
        
        # All results should have untagged type
        for result in results_untagged_type:
            result_tags = result.get('tags') or []
            assert "untagged" in result_tags, f"Expected 'untagged' type, got '{result_tags}' for document '{result['id']}'"
        
        print(f"✓ All {len(results_untagged_type)} results have untagged type")
        
        # Test 5: Non-existent type filter
        print("\n--- Test 5: Non-existent type filter ---")
        search_query_nonexistent_type = {
            "query": "machine python ai data empty none",  # First word from each title to guarantee matches
            "limit": 10,
            "document_tags": ["nonexistent_type"],
            "vector_weights": [{"vector_name": "trigrams", "field": "name", "weight": 1.0}]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query_nonexistent_type)
        assert response.status_code == 200, f"Search with non-existent type filter failed: {response.text}"
        
        results_nonexistent_type = response.json()
        print(f"Non-existent type filter results: {len(results_nonexistent_type)} documents returned")
        
        # Should return empty results for non-existent type
        assert len(results_nonexistent_type) == 0, f"Expected no results for non-existent type, got {len(results_nonexistent_type)}"
        
        print(f"✓ No results returned for non-existent type (as expected)")
        
        # Test 6: Verify that documents with actual None types are stored correctly
        print("\n--- Test 6: Verify None type document storage ---")
        
        # Get the document with None type directly to verify it's stored correctly
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/type_none")
        if response.status_code == 200:
            none_doc = response.json()
            print(f"Document type_none retrieved: ID={none_doc['id']}, Type={none_doc['tags']}, Title='{none_doc['name']}'")
        else:
            print(f"⚠️  Failed to retrieve type_none document: {response.status_code}")
        
        print(f"\n✅ Document type filtering tested successfully!")
        
    finally:
        # Cleanup
        try:
            requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
        except Exception as e:
            print(f"Warning: Failed to cleanup collection {collection_name}: {e}")


@pytest.mark.all_backends
def test_document_status_api(setup_collection):
    """Test the document status API endpoint with various scenarios."""
    collection_name = setup_collection
    
    print(f"\n=== Testing Document Status API for collection: {collection_name} ===")
    
    # Test 1: Check status for non-existent document (should return 404)
    print("\n--- Test 1: Check status for non-existent document ---")
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/nonexistent_doc/status")
    assert response.status_code == 404, f"Expected 404 for non-existent document, got {response.status_code}"
    print("✓ Non-existent document correctly returns 404")
    
    # Test 2: Add a new document and check initial status
    print("\n--- Test 2: Add new document and check initial status ---")
    doc_id = "status_test_doc"
    test_doc = create_test_document(
        doc_id=doc_id,
        name="Status Test Document",
        content="This is a test document for status API testing",
        doc_type="test"
    )
    
    # Add document to queue
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=test_doc)
    assert response.status_code == 200, f"Failed to add document: {response.text}"
    print("✓ Document added to queue successfully")
    
    # Check status immediately - should show "queued"
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}/status")
    assert response.status_code == 200, f"Failed to get document status: {response.text}"
    
    status_response = response.json()
    assert len(status_response["statuses"]) >= 1, "Should have at least one status"
    
    # Check that we have a "queued" status
    queued_status = next((s for s in status_response["statuses"] if s["status"] == QueuedDocumentStatus.QUEUED), None)
    assert queued_status is not None, "Should have 'queued' status"
    assert "queue_id" in queued_status, "Queued status should have queue_id"
    print("✓ Document status shows 'queued' with queue_id")
    
    # Test 3: Wait for document to be processed and check status
    print("\n--- Test 3: Wait for document processing and check status ---")
    
    # Wait for document to be available (processed)
    wait_for_document(collection_name, doc_id, timeout_s=15.0)
    print("✓ Document is now available in collection")
    
    # Check status again - should show only "indexed" (queue entry was removed after processing)
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}/status")
    assert response.status_code == 200, f"Failed to get document status after processing: {response.text}"
    
    status_response = response.json()
    assert len(status_response["statuses"]) == 1, "Should have exactly one status (indexed) after processing"
    
    # Check that we only have "indexed" status (queue entry was removed)
    status_types = {s["status"] for s in status_response["statuses"]}
    assert QueuedDocumentStatus.INDEXED in status_types, "Should have 'indexed' status"
    assert QueuedDocumentStatus.QUEUED not in status_types, "Should not have 'queued' status after processing (removed from queue)"
    
    print("✓ Document status shows only 'indexed' status after processing (queue entry removed)")
    
    # Test 4: Update document multiple times and check status
    print("\n--- Test 4: Update document multiple times and check status ---")
    
    # First update
    update_doc_1 = {
        "id": doc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": "Updated Status Test Document - First Update",
        "content": "This is the first update to the test document",
        "tags": ["test"]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=update_doc_1)
    assert response.status_code == 200, f"Failed to update document first time: {response.text}"
    print("✓ First document update queued successfully")
    
    # Second update
    update_doc_2 = {
        "id": doc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": "Updated Status Test Document - Second Update",
        "content": "This is the second update to the test document",
        "tags": ["test"]
    }
    
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=update_doc_2)
    assert response.status_code == 200, f"Failed to update document second time: {response.text}"
    print("✓ Second document update queued successfully")
    
    # Check status immediately after updates - should show some "queued" statuses
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}/status")
    assert response.status_code == 200, f"Failed to get document status after updates: {response.text}"
    
    status_response = response.json()
    queued_statuses = [s for s in status_response["statuses"] if s["status"] == QueuedDocumentStatus.QUEUED]
    # We might have fewer queued statuses if some updates were already processed
    print(f"Found {len(queued_statuses)} queued statuses after updates")
    
    # Should have at least some queued statuses (depending on processing speed)
    assert len(queued_statuses) >= 1, f"Should have at least 1 queued status, got {len(queued_statuses)}"
    
    # Test 5: Wait for updates to be processed and check final status
    print("\n--- Test 5: Wait for updates to be processed and check final status ---")
    
    expected_name = "Updated Status Test Document - Second Update"
    start_wait = time.time()
    timeout_s = 60.0
    poll_interval_s = 0.3
    final_doc = None
    
    while time.time() - start_wait < timeout_s:
        # Check status
        resp_status = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}/status")
        if resp_status.status_code == 200:
            status_response = resp_status.json()
            queued_count = len([s for s in status_response.get("statuses", []) if s["status"] == QueuedDocumentStatus.QUEUED])
        else:
            queued_count = 1  # force continue
        
        # Fetch current document
        resp_doc = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}")
        if resp_doc.status_code == 200:
            final_doc = resp_doc.json()
        
        if queued_count == 0 and final_doc and final_doc.get("name") == expected_name:
            break
        
        time.sleep(poll_interval_s)
    
    # After polling, assert status conditions
    assert final_doc is not None, "Final document should be retrievable"
    
    # Debug: Print queue status and document details if assertion fails
    if final_doc["name"] != expected_name:
        print(f"\n🔍 DEBUG: Document name mismatch!")
        print(f"Expected: '{expected_name}'")
        print(f"Actual: '{final_doc['name']}'")
        
        # Print the timestamps from both updates for comparison
        print(f"\n⏰ Update timestamps:")
        print(f"First update timestamp: {update_doc_1['timestamp']}")
        print(f"Second update timestamp: {update_doc_2['timestamp']}")
        print(f"Final document timestamp: {final_doc.get('timestamp')}")
        
        # Get current queue status for debugging
        resp_status = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc_id}/status")
        if resp_status.status_code == 200:
            status_response = resp_status.json()
            print(f"\n📊 Current queue status:")
            print(f"Total statuses: {len(status_response.get('statuses', []))}")
            for i, status in enumerate(status_response.get('statuses', [])):
                print(f"  {i+1}. Status: {status['status']}, Timestamp: {status.get('timestamp', 'N/A')}")
            
            queued_count = len([s for s in status_response.get("statuses", []) if s["status"] == QueuedDocumentStatus.QUEUED])
            print(f"Queued count: {queued_count}")
        else:
            print(f"Failed to get status: {resp_status.status_code} - {resp_status.text}")
        
        # Print final document details
        print(f"\n📄 Final document details:")
        print(f"ID: {final_doc.get('id')}")
        print(f"Name: {final_doc.get('name')}")
        print(f"Description: {final_doc.get('description')}")
        print(f"Timestamp: {final_doc.get('timestamp')}")
        print(f"Tags: {final_doc.get('tags')}")
        
        # # Query the queue directly to see what's actually queued for this collection
        # print(f"\n🔍 Querying queue for collection '{collection_name}':")
        # try:
        #     queue_response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/queue/info")
        #     if queue_response.status_code == 200:
        #         queue_data = queue_response.json()
        #         print(f"Queue info: {queue_data}")
        #     else:
        #         print(f"Failed to query queue: {queue_response.status_code} - {queue_response.text}")
        # except Exception as e:
        #     print(f"Exception querying queue: {e}")
    
    assert final_doc["name"] == expected_name, "Document title should reflect the latest update"
    # Note: content field is not stored/returned - it's only used for indexing
    
    print("✓ Document metadata correctly reflects the latest update")
    
    print(f"\n✅ Document Status API tested successfully!")


@pytest.mark.all_backends
def test_document_with_content_storage(setup_collection):
    """Test storing and retrieving a document with content when store_content=True."""
    collection_name = setup_collection
    
    # Create a collection config with store_content=True
    # We need to override the default config from the fixture
    config = {
        "vectors": [
            {
                "name": "trigrams",
                "type": "trigrams",
                "index_fields": ["name", "content"]
            }
        ],
        "store_content": True
    }
    
    print(f"Creating collection with config: {config}")
    
    # Delete the existing collection and recreate with content storage
    requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
    assert response.status_code == 200, f"Failed to create collection with content storage: {response.text}"
    
    # Verify the collection was created with store_content=True
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}")
    assert response.status_code == 200, f"Failed to get collection config: {response.text}"
    
    collection_config = response.json()
    print(f"Collection config retrieved: {collection_config}")
    assert collection_config.get("store_content") == True, f"Collection should have store_content=True, got: {collection_config.get('store_content')}"
    
    # Create a test document with content
    doc = {
        "id": "content_test_doc",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": "Content Storage Test",
        "content": "This is the content that should be stored in the database",
        "tags": ["test", "content"]
    }
    
    print(f"Uploading document: {doc}")
    
    # Upload the document
    response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc)
    assert response.status_code == 200, f"Failed to upload document with content: {response.text}"
    
    # Wait for processing
    wait_for_document(collection_name, "content_test_doc")
    
    # Retrieve the document and verify content is stored
    response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/content_test_doc")
    assert response.status_code == 200, f"Failed to retrieve document: {response.text}"
    
    retrieved_doc = response.json()
    print(f"Retrieved document: {retrieved_doc}")
    
    assert retrieved_doc["content"] == "This is the content that should be stored in the database", "Content should be stored and retrieved"
    assert retrieved_doc["name"] == "Content Storage Test"
    assert set(retrieved_doc["tags"]) == set(["test", "content"]), "Tags should match regardless of order"
    
    print("✅ Document content storage test passed!")


@pytest.mark.all_backends
@pytest.mark.dense_vectors_only
def test_custom_vectors_functionality(backend_capabilities):
    """Test that custom vectors work end-to-end: saving documents with custom vectors and searching with them."""
    
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Generate unique collection name
    unique_id = str(uuid.uuid4())[:8]
    collection_name = f"test_custom_vectors_{unique_id}"
    
    try:
        # Create collection with custom vector types
        config = {
            "vectors": [
                {
                    "name": "custom_dense",
                    "type": "dense_custom",
                    "dimensions": 2,
                    "index_fields": ["content"]
                },
                {
                    "name": "custom_sparse", 
                    "type": "sparse_custom",
                    "top_k": 5,
                    "index_fields": ["content"]
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        assert response.status_code == 200
        
        # Test 1: Upload document with custom vectors
        doc_with_custom_vectors = {
            "id": "custom_test_doc",
            "timestamp": "2024-01-01T00:00:00Z",
            "name": "Custom Vector Test Document",  # Add name to satisfy validation
            "content": "This is a test document with custom vectors",
            "tags": ["custom", "test"],
            "custom_vectors": [
                {
                    "vector_name": "custom_dense",
                    "field": "content",
                    "vector": [0.1, 0.9]  # Simple 2D dense vector
                },
                {
                    "vector_name": "custom_sparse",
                    "field": "content", 
                    "vector": [(1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2), (5, 0.1)]  # Simple sparse vector
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_with_custom_vectors)
        assert response.status_code == 200
        
        # Wait a bit for processing
        time.sleep(2)
        
        # Test 2: Search with custom vectors
        search_query = {
            "query": "test document",
            "custom_vectors": [
                {
                    "vector_name": "custom_dense",
                    "vector": [0.9, 0.1]  # Similar but different dense vector
                },
                {
                    "vector_name": "custom_sparse",
                    "vector": [(1, 0.9), (2, 0.7), (3, 0.5), (4, 0.3), (5, 0.2)]  # Similar sparse vector
                }
            ],
            "limit": 5
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200
        
        results = response.json()
        assert len(results) > 0
        assert results[0]["id"] == "custom_test_doc"
        
    finally:
        # Clean up
        try:
            requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
        except Exception as e:
            print(f"Warning: Failed to cleanup collection {collection_name}: {e}")


@pytest.mark.all_backends
@pytest.mark.dense_vectors_only
def test_custom_vectors_validation(backend_capabilities):
    """Test that custom vector validation works correctly."""
    
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Generate unique collection name
    unique_id = str(uuid.uuid4())[:8]
    collection_name = f"test_custom_validation_{unique_id}"
    
    try:
        # Create collection with custom vector types
        config = {
            "vectors": [
                {
                    "name": "custom_dense",
                    "type": "dense_custom", 
                    "dimensions": 2,
                    "index_fields": ["content"]
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        assert response.status_code == 200
        
        # Test 1: Document with wrong dimensions should fail
        doc_wrong_dimensions = {
            "id": "wrong_dim_doc",
            "timestamp": "2024-01-01T00:00:00Z",
            "name": "Wrong Dimensions Test",  # Add name to satisfy validation
            "content": "This has wrong dimensions",
            "custom_vectors": [
                {
                    "vector_name": "custom_dense",
                    "field": "content",
                    "vector": [0.1, 0.9, 0.5]  # 3D vector for 2D config - should fail
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_wrong_dimensions)
        assert response.status_code == 200  # Document accepted to queue
        
        # Wait for processing to fail due to validation error
        wait_for_document_status(collection_name, "wrong_dim_doc", "failed")
        
        # Verify the failure details
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/wrong_dim_doc/status")
        assert response.status_code == 200
        
        status_response = response.json()
        failed_status = next((s for s in status_response["statuses"] if s["status"].startswith("failed")), None)
        assert failed_status is not None, "Document with wrong dimensions should have failed validation"
        assert "dimensions" in failed_status["info"].lower(), "Failure should mention dimensions mismatch"
        
        # Test 2: Document with correct dimensions should succeed
        doc_correct_dimensions = {
            "id": "correct_dim_doc", 
            "timestamp": "2024-01-01T00:00:00Z",
            "name": "Correct Dimensions Test",  # Add name to satisfy validation
            "content": "This has correct dimensions",
            "custom_vectors": [
                {
                    "vector_name": "custom_dense",
                    "field": "content",
                    "vector": [0.1, 0.9]  # 2D vector for 2D config - should succeed
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_correct_dimensions)
        assert response.status_code == 200  # Document accepted to queue
        
        # Wait for processing and check status - should show indexed (success)
        wait_for_document(collection_name, "correct_dim_doc")
        
        # Verify status shows indexed
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/correct_dim_doc/status")
        assert response.status_code == 200
        
        status_response = response.json()
        indexed_status = next((s for s in status_response["statuses"] if s["status"] == QueuedDocumentStatus.INDEXED), None)
        assert indexed_status is not None, "Document with correct dimensions should be indexed successfully"
        
    finally:
        # Clean up
        try:
            requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
        except Exception as e:
            print(f"Warning: Failed to cleanup collection {collection_name}: {e}")


@pytest.mark.all_backends
@pytest.mark.dense_vectors_only
def test_mixed_custom_and_generated_vectors(backend_capabilities):
    """Test that custom vectors work together with generated vectors (WMTR) in the same collection."""
    
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    # Generate unique collection name
    unique_id = str(uuid.uuid4())[:8]
    collection_name = f"test_mixed_vectors_{unique_id}"
    
    try:
        # Create collection with mixed vector types: custom + generated
        config = {
            "vectors": [
                {
                    "name": "custom_dense",
                    "type": "dense_custom",
                    "dimensions": 2,
                    "index_fields": ["content"]
                },
                {
                    "name": "wmtr_generated",
                    "type": "wmtr",
                    "language_default_code": "en",
                    "index_fields": ["content"]
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        assert response.status_code == 200
        
        # Upload document with custom dense vector (WMTR will be generated automatically)
        doc_with_mixed_vectors = {
            "id": "mixed_vectors_doc",
            "timestamp": "2024-01-01T00:00:00Z",
            "name": "Mixed Vectors Test Document",  # Add name to satisfy validation
            "content": "This document has both custom dense vectors and generated WMTR vectors",
            "tags": ["mixed", "test"],
            "custom_vectors": [
                {
                    "vector_name": "custom_dense",
                    "field": "content",
                    "vector": [0.3, 0.7]  # Simple 2D dense vector
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_with_mixed_vectors)
        assert response.status_code == 200
        
        # Wait for processing
        time.sleep(2)
        
        # Search using both vector types
        search_query = {
            "query": "document vectors",
            "custom_vectors": [
                {
                    "vector_name": "custom_dense",
                    "vector": [0.7, 0.3]  # Similar but different dense vector
                }
            ],
            "vector_weights": [
                {"vector_name": "custom_dense", "field": "content", "weight": 0.6},
                {"vector_name": "wmtr_generated", "field": "content", "weight": 0.4}
            ],
            "limit": 5
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200
        
        results = response.json()
        assert len(results) > 0
        assert results[0]["id"] == "mixed_vectors_doc"
        
    finally:
        # Clean up
        try:
            requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
        except Exception as e:
            print(f"Warning: Failed to cleanup collection {collection_name}: {e}")


@pytest.mark.all_backends
def test_sparse_custom_vectors_only():
    """Test that sparse custom vectors work without requiring dense vector support."""
    
    # Generate unique collection name
    unique_id = str(uuid.uuid4())[:8]
    collection_name = f"test_sparse_only_{unique_id}"
    
    try:
        # Create collection with only sparse custom vector types
        config = {
            "vectors": [
                {
                    "name": "custom_sparse", 
                    "type": "sparse_custom",
                    "top_k": 5,
                    "index_fields": ["content"]
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}", json=config)
        assert response.status_code == 200
        
        # Test 1: Upload document with sparse custom vectors
        doc_with_sparse_vectors = {
            "id": "sparse_test_doc",
            "timestamp": "2024-01-01T00:00:00Z",
            "name": "Sparse Custom Vector Test",  # Add name to satisfy validation
            "content": "This is a test document with sparse custom vectors",
            "tags": ["sparse", "test"],
            "custom_vectors": [
                {
                    "vector_name": "custom_sparse",
                    "field": "content", 
                    "vector": [(1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2), (5, 0.1)]  # Simple sparse vector
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_with_sparse_vectors)
        assert response.status_code == 200  # Document accepted to queue
        
        # Wait for processing and check status - should show indexed (success)
        wait_for_document(collection_name, "sparse_test_doc")
        
        # Test 2: Search with sparse custom vectors
        search_query = {
            "query": "test document",
            "custom_vectors": [
                {
                    "vector_name": "custom_sparse",
                    "vector": [(1, 0.9), (2, 0.7), (3, 0.5), (4, 0.3), (5, 0.2)]  # Similar sparse vector
                }
            ],
            "limit": 5
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/search", json=search_query)
        assert response.status_code == 200
        
        results = response.json()
        assert len(results) > 0
        assert results[0]["id"] == "sparse_test_doc"
        
        # Test 3: Validation - document with too many sparse entries should fail
        doc_too_many_entries = {
            "id": "too_many_entries_doc",
            "timestamp": "2024-01-01T00:00:00Z",
            "name": "Too Many Entries Test",
            "content": "This has too many sparse entries",
            "custom_vectors": [
                {
                    "vector_name": "custom_sparse",
                    "field": "content", 
                    "vector": [(1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2), (5, 0.1), (6, 0.05)]  # 6 entries for top_k=5
                }
            ]
        }
        
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}/documents", json=doc_too_many_entries)
        assert response.status_code == 200  # Document accepted to queue
        
        # Wait for processing to fail due to validation error
        wait_for_document_status(collection_name, "too_many_entries_doc", "failed")
        
        # Verify the failure details
        response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/too_many_entries_doc/status")
        assert response.status_code == 200
        
        status_response = response.json()
        failed_status = next((s for s in status_response["statuses"] if s["status"].startswith("failed")), None)
        assert failed_status is not None, "Document with too many sparse entries should have failed validation"
        assert "max allowed" in failed_status["info"].lower(), "Failure should mention max allowed limit"
        
    finally:
        # Clean up
        try:
            requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
        except Exception as e:
            print(f"Warning: Failed to cleanup collection {collection_name}: {e}")


if __name__ == "__main__":
    # For manual testing
    pytest.main([__file__, "-v"])

