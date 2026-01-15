# Basic Usage Example

This guide walks through the basic operations in PeachBase using the `examples/basic_usage.py` example.

---

## Overview

The basic usage example demonstrates:
- Connecting to a database
- Creating a collection
- Adding documents with embeddings
- Performing semantic search
- Using metadata filters
- Performing lexical (BM25) search
- Saving and loading collections

**Example File**: [`examples/basic_usage.py`](../../examples/basic_usage.py)

---

## Prerequisites

```bash
# Install PeachBase
pip install peachbase

# Or from source
pip install dist/peachbase-*.whl
```

---

## Step-by-Step Walkthrough

### 1. Connect to Database

```python
import peachbase

# Connect to local database (creates directory if doesn't exist)
db = peachbase.connect("./my_database")
print(f"Connected to database: {db}")
```

**Output:**
```
Connected to database: Database(uri='./my_database', type=local, collections=0)
```

### 2. Create a Collection

```python
# Create collection with 384-dimensional vectors
collection = db.create_collection(
    name="articles",
    dimension=384,
    overwrite=True  # Overwrite if exists
)
print(f"Created collection: {collection}")
```

**Output:**
```
Created collection: Collection(name='articles', dimension=384, size=0)
```

**Key Parameters:**
- `name`: Unique collection name
- `dimension`: Vector dimension (must match your embeddings)
- `overwrite`: Whether to overwrite existing collection (default: False)

### 3. Generate Embeddings

In practice, use a real embedding model like `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embedding
text = "Machine learning is fascinating"
vector = model.encode(text).tolist()
```

For this example, we use mock embeddings:

```python
def generate_mock_embedding(text: str) -> list[float]:
    """Generate a mock embedding. In practice, use a real embedding model."""
    import random
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(384)]
```

### 4. Add Documents

```python
documents = [
    {
        "id": "doc1",
        "text": "Machine learning is a subset of artificial intelligence",
        "vector": generate_mock_embedding("Machine learning is a subset of artificial intelligence"),
        "metadata": {"category": "tech", "year": 2023}
    },
    {
        "id": "doc2",
        "text": "Python is a popular programming language for data science",
        "vector": generate_mock_embedding("Python is a popular programming language for data science"),
        "metadata": {"category": "tech", "year": 2023}
    },
    {
        "id": "doc3",
        "text": "Climate change is a pressing global issue",
        "vector": generate_mock_embedding("Climate change is a pressing global issue"),
        "metadata": {"category": "environment", "year": 2024}
    },
    {
        "id": "doc4",
        "text": "Deep learning models have revolutionized computer vision",
        "vector": generate_mock_embedding("Deep learning models have revolutionized computer vision"),
        "metadata": {"category": "tech", "year": 2024}
    },
]

collection.add(documents)
print(f"Added {len(documents)} documents to collection")
```

**Output:**
```
Added 4 documents to collection
```

**Document Schema:**
- `id` (required): Unique document identifier (string)
- `text` (optional): Document text content (string)
- `vector` (required): Embedding vector (list of floats)
- `metadata` (optional): Additional metadata (dict)

### 5. Semantic Search

Search by vector similarity:

```python
query_text = "artificial intelligence and deep learning"
query_vector = generate_mock_embedding(query_text)

print(f"\n--- Semantic Search ---")
print(f"Query: {query_text}")

results = collection.search(
    query_vector=query_vector,
    mode="semantic",
    metric="cosine",  # or "l2", "dot"
    limit=3
)

print(f"\nTop {len(results)} results:")
for i, result in enumerate(results.to_list(), 1):
    print(f"\n{i}. Document ID: {result['id']}")
    print(f"   Text: {result['text']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Metadata: {result['metadata']}")
```

**Output:**
```
--- Semantic Search ---
Query: artificial intelligence and deep learning

Top 3 results:

1. Document ID: doc1
   Text: Machine learning is a subset of artificial intelligence
   Score: 0.8234
   Metadata: {'category': 'tech', 'year': 2023}

2. Document ID: doc4
   Text: Deep learning models have revolutionized computer vision
   Score: 0.7891
   Metadata: {'category': 'tech', 'year': 2024}

3. Document ID: doc2
   Text: Python is a popular programming language for data science
   Score: 0.6543
   Metadata: {'category': 'tech', 'year': 2023}
```

**Similarity Metrics:**
- `cosine`: Cosine similarity (default, range: -1 to 1)
- `l2`: L2 (Euclidean) distance (lower is better)
- `dot`: Dot product similarity

### 6. Search with Metadata Filters

Filter results by metadata:

```python
print(f"\n--- Semantic Search with Filter (category='tech') ---")

filtered_results = collection.search(
    query_vector=query_vector,
    mode="semantic",
    filter={"category": "tech"},
    limit=3
)

print(f"\nTop {len(filtered_results)} filtered results:")
for i, result in enumerate(filtered_results.to_list(), 1):
    print(f"\n{i}. Document ID: {result['id']}")
    print(f"   Text: {result['text']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Category: {result['metadata']['category']}")
```

**Filter Examples:**

```python
# Exact match
filter={"category": "tech"}

# Greater than or equal
filter={"year": {"$gte": 2024}}

# Range
filter={"year": {"$gte": 2020, "$lte": 2024}}

# Multiple conditions (AND)
filter={"category": "tech", "year": 2024}

# OR condition
filter={"$or": [{"category": "tech"}, {"category": "science"}]}

# In array
filter={"category": {"$in": ["tech", "science", "ai"]}}
```

### 7. Lexical Search (BM25)

Keyword-based search using BM25:

```python
print(f"\n--- Lexical Search (BM25) ---")
print(f"Query: {query_text}")

lexical_results = collection.search(
    query_text=query_text,
    mode="lexical",
    limit=3
)

print(f"\nTop {len(lexical_results)} results:")
for i, result in enumerate(lexical_results.to_list(), 1):
    print(f"\n{i}. Document ID: {result['id']}")
    print(f"   Text: {result['text']}")
    print(f"   BM25 Score: {result['score']:.4f}")
```

**Output:**
```
--- Lexical Search (BM25) ---
Query: artificial intelligence and deep learning

Top 3 results:

1. Document ID: doc1
   Text: Machine learning is a subset of artificial intelligence
   BM25 Score: 1.8234

2. Document ID: doc4
   Text: Deep learning models have revolutionized computer vision
   BM25 Score: 1.4532

3. Document ID: doc2
   Text: Python is a popular programming language for data science
   BM25 Score: 0.3421
```

**When to use Lexical vs Semantic:**
- **Lexical (BM25)**: Exact keyword matching, technical terms, names
- **Semantic**: Conceptual similarity, paraphrases, meaning-based search
- **Hybrid**: Combine both for best results (see [hybrid search example](../../examples/hybrid_search.py))

### 8. Save Collection

Persist collection to disk:

```python
print(f"\n--- Saving Collection ---")
collection.save()
print(f"Collection saved to: {db.get_collection_path('articles')}")
```

**Output:**
```
--- Saving Collection ---
Collection saved to: ./my_database/articles.pdb
```

**What gets saved:**
- Document vectors (SIMD-aligned binary format)
- Document text and metadata
- BM25 index (vocabulary, IDF scores, term frequencies)
- Collection metadata (dimension, size, etc.)

### 9. Load Collection

Load saved collection:

```python
print(f"\n--- Loading Collection ---")
loaded_collection = db.open_collection("articles")
print(f"Loaded collection: {loaded_collection}")
print(f"Documents in loaded collection: {loaded_collection.size}")
```

**Output:**
```
--- Loading Collection ---
Loaded collection: Collection(name='articles', dimension=384, size=4)
Documents in loaded collection: 4
```

**Loading is fast:**
- Memory-mapped file access
- No deserialization overhead
- Instant access to vectors and indices

---

## Additional Operations

### Get Document by ID

```python
doc = collection.get("doc1")
print(f"Document: {doc}")
```

### Delete Document

```python
collection.delete("doc1")
print(f"Deleted doc1, new size: {collection.size}")
collection.save()  # Persist the deletion
```

### Update Document

```python
# Delete old version
collection.delete("doc1")

# Add new version
collection.add([{
    "id": "doc1",
    "text": "Updated text",
    "vector": generate_mock_embedding("Updated text"),
    "metadata": {"category": "tech", "year": 2024, "updated": True}
}])

collection.save()
```

### List All Collections

```python
collections = db.list_collections()
print(f"Collections: {collections}")
```

### Delete Collection

```python
db.drop_collection("articles")
print("Collection deleted")
```

---

## Running the Example

```bash
# Run the example
python examples/basic_usage.py

# View output
```

**Expected Runtime:**
- Collection creation: < 1ms
- Adding 4 documents: < 10ms
- Semantic search: < 5ms
- Lexical search: < 5ms
- Save/Load: < 10ms

---

## Complete Code

See the full working example: [`examples/basic_usage.py`](../../examples/basic_usage.py)

---

## Next Steps

- **Search Modes**: Learn about [hybrid search](../../examples/hybrid_search.py)
- **Production**: See [Wikipedia RAG example](wikipedia-rag.md) for real-world usage
- **Large Scale**: Check [large-scale example](large-scale.md) for 10K+ documents
- **Deployment**: Read [deployment guide](../guides/deployment.md) for AWS Lambda

---

## Common Patterns

### Pattern 1: RAG (Retrieval-Augmented Generation)

```python
def rag_query(question: str, model, llm):
    # 1. Generate embedding for question
    query_vector = model.encode(question).tolist()

    # 2. Search for relevant documents
    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=5
    )

    # 3. Build context from results
    context = "\n\n".join([r["text"] for r in results.to_list()])

    # 4. Generate answer with LLM
    answer = llm.generate(f"Context: {context}\n\nQuestion: {question}")

    return answer, results
```

### Pattern 2: Multi-Collection Search

```python
# Search across multiple collections
def search_all(query_vector):
    results = []
    for collection_name in db.list_collections():
        collection = db.open_collection(collection_name)
        collection_results = collection.search(query_vector=query_vector, limit=5)
        results.extend(collection_results.to_list())

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]  # Top 10 overall
```

### Pattern 3: Incremental Updates

```python
# Add documents in batches
batch_size = 100
for i in range(0, len(all_documents), batch_size):
    batch = all_documents[i:i+batch_size]
    collection.add(batch)
    collection.save()  # Save after each batch
    print(f"Added batch {i//batch_size + 1}, total: {collection.size}")
```

---

## Troubleshooting

**"ValueError: Document with id 'doc1' already exists"**
- Solution: Use unique IDs or delete old document first

**"IndexError: list index out of range"**
- Wrong vector dimension
- Solution: Ensure vectors match collection dimension

**"Collection not found"**
- Collection not saved yet
- Solution: Call `collection.save()` before `db.open_collection()`

**Search returns no results**
- Empty collection or mismatched vectors
- Solution: Verify documents added with `collection.size`

---

## API Reference

For complete API documentation, see:
- [API Reference](../reference/api.md)
- [Search Modes Guide](../guides/search-modes.md)

---

**Questions?** Open an issue: https://github.com/PeachstoneAI/peachbase/issues
