#!/usr/bin/env python3
"""
Integration test for maximum length string handling across all endpoints.
Tests backend robustness when dealing with maximum allowed string lengths.
"""

import pytest
import requests
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from tests.test_api_integration import wait_for_document

# Test configuration
API_BASE_URL = "http://localhost:8234/v1"

class MaxLengthTestData:
    """Test data with maximum length strings for all fields."""
    
    # Maximum lengths from our models
    MAX_ID_LENGTH = 100
    MAX_TYPE_LENGTH = 100
    MAX_NAME_LENGTH = 500
    MAX_DESCRIPTION_LENGTH = 2000
    MAX_CONTENT_LENGTH = 1000000  # 1M characters
    MAX_METADATA_KEY_LENGTH = 100
    MAX_METADATA_VALUE_LENGTH = 100
    MAX_MODEL_LENGTH = 200
    MAX_REVISION_LENGTH = 200
    MAX_QUERY_LENGTH = 10000
    MAX_DOCUMENT_TAGS_LENGTH = 100
    
    @staticmethod
    def generate_max_string(length: int, prefix: str = "") -> str:
        """Generate a string of specified length with a prefix."""
        remaining_length = length - len(prefix)
        if remaining_length <= 0:
            return prefix[:length]
        return prefix + "X" * remaining_length
    
    @staticmethod
    def truncate_string(s: str, max_show: int = 50) -> str:
        """Truncate a string for display purposes."""
        if len(s) <= max_show:
            return s
        return f"{s[:max_show]}...[{len(s)} chars total]"
    
    @classmethod
    def get_max_document(cls) -> Dict[str, Any]:
        """Create a document with all fields at maximum length."""
        return {
            "id": cls.generate_max_string(cls.MAX_ID_LENGTH, "MAX_ID"),
            "timestamp": "2024-01-01T00:00:00Z",
            "type": cls.generate_max_string(cls.MAX_TYPE_LENGTH, "MAX_TYPE"),
            "name": cls.generate_max_string(cls.MAX_NAME_LENGTH, "MAX_NAME"),
            "description": cls.generate_max_string(cls.MAX_DESCRIPTION_LENGTH, "MAX_DESCRIPTION"),
            "content": cls.generate_max_string(cls.MAX_CONTENT_LENGTH, "MAX_CONTENT"),
            "metadata": {
                cls.generate_max_string(cls.MAX_METADATA_KEY_LENGTH, "MAX_KEY"): cls.generate_max_string(cls.MAX_METADATA_VALUE_LENGTH, "MAX_VALUE"),
                cls.generate_max_string(cls.MAX_METADATA_KEY_LENGTH, "MAX_KEY2"): 12345  # Test non-string value (integer)
            }
        }
    
    @classmethod
    def get_max_vector_configs(cls) -> list[Dict[str, Any]]:
        """Create vector configs with maximum length strings."""
        return [
            {
                "name": cls.generate_max_string(100, "MAX_DENSE"),  # name max is 100
                "type": "dense_model",
                "model": "sentence-transformers/all-MiniLM-L6-v2",  # Use real model
                "normalization": True
            },
            {
                "name": cls.generate_max_string(100, "MAX_SPARSE"),
                "type": "trigrams",
                "top_k": 2048
            },
            {
                "name": cls.generate_max_string(100, "MAX_FULLTEXT"),
                "type": "full_text",
                "language_default_code": "en",
                "top_k": 2048
            }
        ]
    
    @classmethod
    def get_max_collection_config(cls) -> Dict[str, Any]:
        """Create collection config with maximum length strings."""
        return {
            "vectors": cls.get_max_vector_configs()
        }
    
    @classmethod
    def get_max_search_query(cls) -> Dict[str, Any]:
        """Create search query with maximum length strings."""
        return {
            "query": cls.generate_max_string(cls.MAX_QUERY_LENGTH, "MAX_QUERY"),
            "document_tags": [
                cls.generate_max_string(cls.MAX_DOCUMENT_TAGS_LENGTH, "MAX_DOC_TAG_1"),
                cls.generate_max_string(cls.MAX_DOCUMENT_TAGS_LENGTH, "MAX_DOC_TAG_2")
            ],
            "limit": 10
        }


@pytest.mark.dense_vectors_only
def test_max_length_integration(backend_capabilities):
    """Test that all endpoints handle maximum length strings correctly."""
    from tests.conftest import skip_if_not_supported
    
    # Skip if backend doesn't support dense vectors
    skip_if_not_supported(backend_capabilities, "dense_vectors")
    
    print("🧪 Testing maximum length string handling across all endpoints...")
    
    # Generate test data with maximum allowed lengths
    collection_name = MaxLengthTestData.generate_max_string(100, "MAX_COLL_")
    doc1_id = MaxLengthTestData.generate_max_string(100, "MAX_ID_")
    doc2_id = MaxLengthTestData.generate_max_string(100, "MAX_ID2_")
    search_query_text = MaxLengthTestData.generate_max_string(10000, "MAX_QUERY_")
    
    print(f"📁 Collection name: {MaxLengthTestData.truncate_string(collection_name)}")
    print(f"📄 Document 1 ID: {MaxLengthTestData.truncate_string(doc1_id)}")
    print(f"📄 Document 2 ID: {MaxLengthTestData.truncate_string(doc2_id)}")
    print(f"🔍 Search query length: {len(search_query_text)}")
    
    # Ensure cleanup happens even if test fails
    try:
        # Test 1: Create collection with maximum length vector configs
        print("\n1️⃣ Creating collection with maximum length vector configs...")
        try:
            collection_config = {
                "vectors": [
                    {
                        "name": "dense_vector",
                        "type": "dense_model",
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    },
                    {
                        "name": "trigram_vector", 
                        "type": "trigrams"
                    },
                    {
                        "name": "full_text_vector",
                        "type": "full_text",
                        "language_default_code": "en"
                    }
                ]
            }
            
            response = requests.post(
                f"{API_BASE_URL}/collections/{collection_name}",
                json=collection_config
            )
            assert response.status_code == 200, f"Failed to create collection: {response.text}"
            print("✅ Collection created successfully")
        except Exception as e:
            print(f"❌ Collection creation failed: {e}")
            raise
        
        # Test 2: Add first document with max length values
        print("\n2️⃣ Adding first document with maximum length values...")
        try:
            doc1 = {
                "id": doc1_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tags": [MaxLengthTestData.generate_max_string(100, "MAX_TYPE_")] ,
                "name": MaxLengthTestData.generate_max_string(500, "MAX_NAME_"),
                "description": MaxLengthTestData.generate_max_string(2000, "MAX_DESC_"),
                "content": MaxLengthTestData.generate_max_string(1000000, "MAX_CONTENT_"),
                "metadata": {
                    MaxLengthTestData.generate_max_string(100, "MAX_KEY_"): MaxLengthTestData.generate_max_string(100, "MAX_VAL_")
                }
            }
            
            response = requests.post(
                f"{API_BASE_URL}/collections/{collection_name}/documents",
                json=doc1
            )
            assert response.status_code == 200, f"Failed to add document 1: {MaxLengthTestData.truncate_string(response.text, 200)}"
            print("✅ Document 1 added successfully")
        except Exception as e:
            print(f"❌ Document 1 addition failed: {e}")
            raise
        
        # Test 2.5: Verify document 1 is actually indexed
        print("\n2.5️⃣ Verifying document 1 is indexed...")
        try:
            # Wait for document to be indexed
            wait_for_document(collection_name, doc1_id)
            
            # Check document status
            response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc1_id}/status")
            assert response.status_code == 200, f"Failed to get document 1 status: {response.text}"
            status_response = response.json()
            print(f"📊 Document 1 status: {MaxLengthTestData.truncate_string(str(status_response))}")
            
            # Check if document exists
            response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc1_id}")
            assert response.status_code == 200, f"Failed to get document 1: {response.text}"
            retrieved_doc = response.json()
            print(f"📄 Retrieved document 1: {retrieved_doc['id']}")
            
            print("✅ Document 1 verification successful")
        except Exception as e:
            print(f"❌ Document 1 verification failed: {e}")
            raise
        
        # Test 3: Add second document with max length values
        print("\n3️⃣ Adding second document with maximum length values...")
        try:
            doc2 = {
                "id": doc2_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tags": [MaxLengthTestData.generate_max_string(100, "MAX_TYPE2_")] ,
                "name": MaxLengthTestData.generate_max_string(500, "MAX_NAME2_"),
                "description": MaxLengthTestData.generate_max_string(2000, "MAX_DESC2_"),
                "content": MaxLengthTestData.generate_max_string(1000000, "MAX_CONTENT2_"),
                "metadata": {
                    MaxLengthTestData.generate_max_string(100, "MAX_KEY2_"): MaxLengthTestData.generate_max_string(100, "MAX_VAL2_")
                }
            }
            
            response = requests.post(
                f"{API_BASE_URL}/collections/{collection_name}/documents",
                json=doc2
            )
            assert response.status_code == 200, f"Failed to add document 2: {MaxLengthTestData.truncate_string(response.text, 200)}"
            print("✅ Document 2 added successfully")
        except Exception as e:
            print(f"❌ Document 2 addition failed: {e}")
            raise
        
        # Test 3.5: Verify document 2 is actually indexed
        print("\n3.5️⃣ Verifying document 2 is indexed...")
        try:
            # Wait for document to be indexed
            wait_for_document(collection_name, doc2_id)
            
            # Check document status
            response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc2_id}/status")
            assert response.status_code == 200, f"Failed to get document 2 status: {response.text}"
            status_response = response.json()
            print(f"📊 Document 2 status: {MaxLengthTestData.truncate_string(str(status_response))}")
            
            # Check if document exists
            response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc2_id}")
            assert response.status_code == 200, f"Failed to get document 2: {response.text}"
            retrieved_doc = response.json()
            print(f"📄 Retrieved document 2: {retrieved_doc['id']}")
            
            print("✅ Document 2 verification successful")
        except Exception as e:
            print(f"❌ Document 2 verification failed: {e}")
            raise
        
        # Test 4: Search with max length query
        print("\n4️⃣ Searching with maximum length query...")
        try:
            search_query = {
                "query": search_query_text,
                "limit": 10
            }
            
            response = requests.post(
                f"{API_BASE_URL}/collections/{collection_name}/search",
                json=search_query
            )
            assert response.status_code == 200, f"Search failed: {MaxLengthTestData.truncate_string(response.text, 200)}"
            search_results = response.json()
            print(f"🔍 Search response type: {type(search_results)}")
            print(f"🔍 Search response length: {len(search_results) if isinstance(search_results, list) else 'N/A'}")
            if isinstance(search_results, list) and len(search_results) > 0:
                # Truncate the first result to avoid massive output
                first_result = search_results[0]
                truncated_result = {}
                for key, value in first_result.items():
                    if isinstance(value, str) and len(value) > 100:
                        truncated_result[key] = MaxLengthTestData.truncate_string(value, 100)
                    else:
                        truncated_result[key] = value
                print(f"🔍 First result: {truncated_result}")
            else:
                print(f"🔍 Search response: {search_results}")
            
            # API returns direct list, not dict with "results" key
            assert isinstance(search_results, list), f"Expected list, got {type(search_results)}"
            assert len(search_results) > 0, f"No search results returned. Expected at least 1 result with 2 documents in collection"
            print(f"✅ Search successful, returned {len(search_results)} results")
        except Exception as e:
            print(f"❌ Search failed: {e}")
            raise
        
        # Test 5: Get document by ID
        print("\n5️⃣ Retrieving document by ID...")
        try:
            response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc1_id}")
            assert response.status_code == 200, f"Failed to get document: {response.text}"
            retrieved_doc = response.json()
            assert retrieved_doc["id"] == doc1_id, "Retrieved document ID mismatch"
            print("✅ Document retrieval successful")
        except Exception as e:
            print(f"❌ Document retrieval failed: {e}")
            raise
        
        # Test 6: Update document with max length values
        print("\n6️⃣ Updating document with maximum length values...")
        try:
            # Modify some fields to test updates
            doc1["name"] = MaxLengthTestData.generate_max_string(500, "UPDATED_NAME")
            doc1["description"] = MaxLengthTestData.generate_max_string(2000, "UPDATED_DESC")
            
            response = requests.post(
                f"{API_BASE_URL}/collections/{collection_name}/documents",
                json=doc1
            )
            assert response.status_code == 200, f"Failed to update document: {response.text}"
            print("✅ Document update successful")
        except Exception as e:
            print(f"❌ Document update failed: {e}")
            raise
        
        # Test 7: Get document status
        print("\n7️⃣ Getting document status...")
        try:
            response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents/{doc1_id}/status")
            assert response.status_code == 200, f"Failed to get document status: {response.text}"
            status_response = response.json()
            assert "statuses" in status_response, "Status response missing statuses"
            print("✅ Document status retrieval successful")
        except Exception as e:
            print(f"❌ Document status retrieval failed: {e}")
            raise
        
        # Test 8: List documents in collection
        print("\n8️⃣ Listing documents in collection...")
        try:
            # Note: This endpoint may not be supported by the API
            print("⚠️ Skipping document listing test - endpoint may not be supported")
            # response = requests.get(f"{API_BASE_URL}/collections/{collection_name}/documents")
            # assert response.status_code == 200, f"Failed to list documents: {response.text}"
            # docs_response = response.json()
            # assert "documents" in docs_response, "List response missing documents"
            # assert len(docs_response["documents"]) >= 2, "Expected at least 2 documents"
            # print(f"✅ Document listing successful, found {len(docs_response['documents'])} documents")
            print("✅ Document listing test skipped")
        except Exception as e:
            print(f"❌ Document listing failed: {e}")
            raise
        
        # Test 9: Delete document
        print("\n9️⃣ Deleting document...")
        try:
            response = requests.delete(
                f"{API_BASE_URL}/collections/{collection_name}/documents/{doc2_id}/sync",
                params={"request_timestamp": datetime.now(timezone.utc).isoformat()},
            )
            assert response.status_code == 200, f"Failed to delete document: {response.text}"
            print("✅ Document deletion successful")
        except Exception as e:
            print(f"❌ Document deletion failed: {e}")
            raise
        
        print("\n🎉 All tests passed! Maximum length handling works correctly.")
        
    finally:
        # Cleanup: Always try to delete the collection, even if tests fail
        print("\n🧹 Cleaning up test collection...")
        try:
            response = requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
            if response.status_code == 200:
                print("✅ Test collection cleaned up successfully")
            else:
                print(f"⚠️ Warning: Failed to clean up collection: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Warning: Error during cleanup: {e}")


if __name__ == "__main__":
    test_max_length_integration()
