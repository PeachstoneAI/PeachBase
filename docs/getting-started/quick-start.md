# Quick Start

Get up and running with PeachBase in 5 minutes.

---

## Your First Search

```python
import peachbase

# 1. Connect to database
db = peachbase.connect("./my_db")

# 2. Create collection
collection = db.create_collection("docs", dimension=384)

# 3. Add documents with embeddings
collection.add([
    {
        "id": "doc1",
        "text": "Python is a programming language",
        "vector": [0.1, 0.2, ...],  # 384-dim embedding
        "metadata": {"category": "programming"}
    },
    {
        "id": "doc2",
        "text": "Machine learning is a subset of AI",
        "vector": [0.3, 0.4, ...],
        "metadata": {"category": "ai"}
    }
])

# 4. Search
results = collection.search(
    query_vector=[0.15, 0.25, ...],
    mode="semantic",
    limit=5
)

# 5. Get results
for result in results:
    print(f"{result['text']} (score: {result['score']:.3f})")
```

---

## Three Search Modes

### Semantic Search (Vector Similarity)

Best for concept matching:

```python
results = collection.search(
    query_vector=[0.1, 0.2, ...],
    mode="semantic",
    metric="cosine",  # or "l2", "dot"
    limit=10
)
```

### Lexical Search (BM25)

Best for keyword matching:

```python
results = collection.search(
    query_text="machine learning algorithms",
    mode="lexical",
    limit=10
)
```

### Hybrid Search (Best of Both)

**Recommended for production** - combines semantic and lexical:

```python
results = collection.search(
    query_text="machine learning algorithms",
    query_vector=[0.1, 0.2, ...],
    mode="hybrid",
    alpha=0.5,  # 0=semantic only, 1=lexical only
    limit=10
)
```

See [Search Modes Guide](../guides/search-modes.md) for details.

---

## Complete Example with Real Embeddings

```python
import peachbase
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

# Connect to database
db = peachbase.connect("./rag_db")
collection = db.create_collection("articles", dimension=384)

# Add documents
documents = [
    {
        "id": "1",
        "text": "Python is a high-level programming language",
        "metadata": {"topic": "programming"}
    },
    {
        "id": "2",
        "text": "Machine learning enables computers to learn from data",
        "metadata": {"topic": "ai"}
    },
    {
        "id": "3",
        "text": "Deep learning uses neural networks",
        "metadata": {"topic": "ai"}
    }
]

# Generate embeddings
for doc in documents:
    doc["vector"] = model.encode(doc["text"]).tolist()

# Add to collection
collection.add(documents)

# Search with query
query = "artificial intelligence and neural networks"
query_vector = model.encode(query).tolist()

# Hybrid search (best results)
results = collection.search(
    query_text=query,
    query_vector=query_vector,
    mode="hybrid",
    alpha=0.5,
    limit=3
)

# Display results
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['text']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Topic: {result['metadata']['topic']}")

# Save collection
collection.save()
```

---

## Metadata Filtering

Filter results by metadata:

```python
results = collection.search(
    query_vector=query_vector,
    filter={"topic": "ai"},
    limit=10
)

# Complex filters (MongoDB-like)
results = collection.search(
    query_vector=query_vector,
    filter={
        "topic": "ai",
        "year": {"$gte": 2020},
        "tags": {"$in": ["deep-learning", "nlp"]}
    },
    limit=10
)
```

---

## Save and Load

```python
# Save to disk
collection.save()

# Load from disk
db = peachbase.connect("./my_db")
collection = db.get_collection("docs")

# Now you can search
results = collection.search(query_vector=vec, limit=10)
```

---

## Performance Tips

### For Speed
- Use `mode="semantic"` with `metric="dot"` (fastest)
- Reduce `limit` parameter
- Use metadata filters to narrow search space

### For Quality
- Use `mode="hybrid"` (best balance)
- Tune `alpha` parameter (0.4-0.6 usually optimal)
- Use meaningful metadata for filtering

### For Large Collections (10K+)
- Build with OpenMP for multi-core acceleration
- Use pre-flattened vectors (automatic)
- Consider sharding for 100K+ documents

See [Performance Tuning](../guides/performance.md) for details.

---

## Common Patterns

### RAG (Retrieval-Augmented Generation)

```python
# 1. Search for relevant context
results = collection.search(
    query_text=question,
    query_vector=model.encode(question).tolist(),
    mode="hybrid",
    limit=5
)

# 2. Build context
context = "\n\n".join([r["text"] for r in results])

# 3. Send to LLM
response = llm.generate(f"Question: {question}\n\nContext: {context}")
```

### Document Q&A

```python
# Add document chunks
for chunk in document_chunks:
    collection.add({
        "id": chunk.id,
        "text": chunk.text,
        "vector": model.encode(chunk.text).tolist(),
        "metadata": {
            "source": chunk.source,
            "page": chunk.page
        }
    })

# Search with metadata filter
results = collection.search(
    query_text="what is the main conclusion?",
    query_vector=query_vec,
    mode="hybrid",
    filter={"source": "report_2024.pdf"},
    limit=3
)
```

### Semantic Deduplication

```python
# Find duplicates using L2 distance
for doc in documents:
    similar = collection.search(
        query_vector=doc["vector"],
        mode="semantic",
        metric="l2",  # Lower = more similar
        limit=5
    )

    # If very close match exists (L2 < 0.1), it's likely a duplicate
    if similar[0]["score"] < 0.1:
        print(f"Potential duplicate: {doc['id']}")
```

---

## Run Examples

PeachBase includes several working examples:

```bash
cd examples

# Quick test (30 seconds)
python quick_test.py

# Basic usage
python basic_usage.py

# Hybrid search comparison
python hybrid_search.py

# Full Wikipedia RAG (2-3 minutes)
python wikipedia_rag.py

# Large scale (10K+ docs, ~2 minutes)
python wikipedia_rag_large_hf.py
```

See [Examples Documentation](../examples/) for details.

---

## Next Steps

1. **[Basic Concepts](basic-concepts.md)** - Understand core concepts
2. **[Search Modes Guide](../guides/search-modes.md)** - Deep dive into search modes
3. **[Scoring Guide](../guides/scoring.md)** - Understand how scores work
4. **[Examples](../examples/)** - More complex examples
5. **[API Reference](../reference/api.md)** - Complete API documentation

---

## Quick Reference Card

```python
# Connect
db = peachbase.connect("./db_path")

# Create collection
col = db.create_collection("name", dimension=384)

# Add documents
col.add([{"id": "1", "text": "...", "vector": [...], "metadata": {...}}])

# Semantic search
results = col.search(query_vector=[...], mode="semantic", limit=10)

# Lexical search
results = col.search(query_text="...", mode="lexical", limit=10)

# Hybrid search (recommended)
results = col.search(
    query_text="...",
    query_vector=[...],
    mode="hybrid",
    alpha=0.5,
    limit=10
)

# With filter
results = col.search(query_vector=[...], filter={"key": "value"}, limit=10)

# Save
col.save()

# Load
col = db.get_collection("name")

# Collection info
print(col.size)  # Number of documents
print(col.dimension)  # Vector dimension
```

---

[← Back to Installation](installation.md) | [Next: Basic Concepts →](basic-concepts.md)
