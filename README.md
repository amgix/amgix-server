# **Amgix** - Hybrid Search Engine for Applications

> **Amgix** (pronounced `a-MAG-ix`) - short for Amalgam Index  
> *amalgam: a mixture or blend of different elements*

---

Amgix is an open-source hybrid search engine for applications.

It combines keyword and semantic search in a single system, with built-in embedding pipelines, multi-vector ranking, and production-ready ingestion.

---

## How It Works

**1. Define your collection:**
```json
POST /v1/collections/products
{
  "vectors": [
    {"name": "keyword", "type": "keyword", "index_fields": ["name", "content"]},
    {"name": "semantic", "type": "dense_model", "model": "sentence-transformers/all-MiniLM-L6-v2", "index_fields": ["content"]}
  ]
}
```

**2. Upload your data:**
```json
POST /v1/collections/products/documents
{
  "id": "part-001",
  "timestamp": "2026-01-01T00:00:00Z",
  "name": "Pin 12LP'-x03/5-XL",
  "content": "Precision pinch roller assembly for industrial use"
}
```

**3. Search:**
```json
POST /v1/collections/products/search
{
  "query": "pinch roller"
}
```

That's it. Amgix handles vectorization, fusion, and ranking.

---

## Why Amgix

Because integrating modern search into applications is hard. You usually have to:

1. Stitch together multiple systems for retrieval, embeddings, and indexing.
2. Build indexing and re-indexing pipelines that handle deduplication and failures.
3. Build and maintain embedding pipelines for semantic search.
4. Combine keyword and semantic results into one ranking.
5. Tune relevance for real queries, filters, typos, and identifier-heavy data.
6. Keep the whole thing in sync, reliable, and scalable.

Amgix was built to make this easier. It combines ingestion pipelines, on-the-fly embeddings, hybrid retrieval, ranking, and search in one system that can scale with your needs.

---

More documentation is coming soon...