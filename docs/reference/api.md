# PeachBase API Reference

Complete API documentation for PeachBase.

---

## Module: `peachbase`

### `connect(path: str) -> Database`

Connect to a PeachBase database.

**Parameters**:
- `path` (str): Path to database directory (local or S3)
  - Local: `"./my_db"` or `"/absolute/path/to/db"`
  - S3: `"s3://bucket-name/path/to/db"`

**Returns**: `Database` object

**Example**:
```python
import peachbase

# Local database
db = peachbase.connect("./my_db")

# S3 database
db = peachbase.connect("s3://my-bucket/vector-db")
```

---

## Class: `Database`

Represents a PeachBase database instance.

### `create_collection(name: str, dimension: int) -> Collection`

Create a new collection in the database.

**Parameters**:
- `name` (str): Name of the collection
- `dimension` (int): Dimension of embedding vectors (e.g., 384, 768, 1536)

**Returns**: `Collection` object

**Raises**:
- `ValueError`: If collection already exists or dimension is invalid

**Example**:
```python
collection = db.create_collection("documents", dimension=384)
```

### `get_collection(name: str) -> Collection`

Get an existing collection.

**Parameters**:
- `name` (str): Name of the collection

**Returns**: `Collection` object

**Raises**:
- `KeyError`: If collection doesn't exist

**Example**:
```python
collection = db.get_collection("documents")
```

### `list_collections() -> List[str]`

List all collection names in the database.

**Returns**: List of collection names

**Example**:
```python
collections = db.list_collections()
print(collections)  # ['documents', 'embeddings', 'vectors']
```

### `delete_collection(name: str) -> None`

Delete a collection from the database.

**Parameters**:
- `name` (str): Name of the collection to delete

**Example**:
```python
db.delete_collection("old_collection")
```

---

## Class: `Collection`

Represents a collection of documents with embeddings.

### Properties

#### `name: str`
Name of the collection.

#### `dimension: int`
Dimension of vectors in this collection.

#### `size: int`
Number of documents in the collection.

### `add(documents: List[Dict]) -> None`

Add documents to the collection.

**Parameters**:
- `documents` (List[Dict]): List of document dictionaries

**Document Schema**:
```python
{
    "id": str,              # Required: Unique document ID
    "text": str,            # Required: Document text for lexical search
    "vector": List[float],  # Required: Embedding vector
    "metadata": Dict,       # Optional: Arbitrary metadata
}
```

**Example**:
```python
collection.add([
    {
        "id": "doc1",
        "text": "Machine learning is a subset of AI",
        "vector": [0.1, 0.2, ..., 0.5],  # 384-dim
        "metadata": {"category": "ai", "year": 2024}
    },
    {
        "id": "doc2",
        "text": "Deep learning uses neural networks",
        "vector": [0.3, 0.1, ..., 0.8],
        "metadata": {"category": "ai", "year": 2024}
    }
])
```

**Notes**:
- Duplicate IDs will overwrite existing documents
- All vectors must have the same dimension as the collection
- Text is tokenized and indexed for lexical search

### `search(query_text: str = None, query_vector: List[float] = None, mode: str = "semantic", **kwargs) -> QueryResult`

Search the collection.

**Parameters**:
- `query_text` (str, optional): Query text for lexical/hybrid search
- `query_vector` (List[float], optional): Query embedding for semantic/hybrid search
- `mode` (str): Search mode - `"semantic"`, `"lexical"`, or `"hybrid"`
- `limit` (int): Maximum results to return (default: 10)
- `filter` (Dict, optional): Metadata filter (MongoDB-like syntax)
- `metric` (str): Distance metric for semantic search - `"cosine"`, `"l2"`, or `"dot"` (default: "cosine")
- `alpha` (float): Weight for hybrid search (0=semantic, 1=lexical, default: 0.5)

**Returns**: `QueryResult` object

**Raises**:
- `ValueError`: If required parameters are missing for the search mode

**Examples**:

**Semantic Search**:
```python
results = collection.search(
    query_vector=[0.2, 0.3, ..., 0.4],
    mode="semantic",
    metric="cosine",
    limit=10
)
```

**Lexical Search**:
```python
results = collection.search(
    query_text="machine learning algorithms",
    mode="lexical",
    limit=10
)
```

**Hybrid Search**:
```python
results = collection.search(
    query_text="machine learning algorithms",
    query_vector=[0.2, 0.3, ..., 0.4],
    mode="hybrid",
    alpha=0.5,  # Balanced
    limit=10
)
```

**With Metadata Filter**:
```python
results = collection.search(
    query_vector=[0.2, 0.3, ..., 0.4],
    mode="semantic",
    filter={
        "category": "ai",
        "year": {"$gte": 2023}
    },
    limit=10
)
```

### `get(id: str) -> Dict`

Get a document by ID.

**Parameters**:
- `id` (str): Document ID

**Returns**: Document dictionary

**Raises**:
- `KeyError`: If document doesn't exist

**Example**:
```python
doc = collection.get("doc1")
print(doc["text"])
```

### `delete(id: str) -> None`

Delete a document by ID.

**Parameters**:
- `id` (str): Document ID

**Example**:
```python
collection.delete("doc1")
```

### `update(id: str, document: Dict) -> None`

Update a document.

**Parameters**:
- `id` (str): Document ID
- `document` (Dict): New document data (same schema as `add()`)

**Example**:
```python
collection.update("doc1", {
    "id": "doc1",
    "text": "Updated text",
    "vector": [0.5, 0.6, ..., 0.9],
    "metadata": {"category": "ml"}
})
```

### `save() -> None`

Persist the collection to disk.

**Example**:
```python
collection.save()
```

**Notes**:
- Collections are automatically saved on database close
- Manual save ensures data is persisted immediately
- Save includes both document data and search indices

### `load() -> None`

Reload the collection from disk.

**Example**:
```python
collection.load()
```

---

## Class: `QueryResult`

Represents search results.

### `to_list() -> List[Dict]`

Convert results to a list of dictionaries.

**Returns**: List of result dictionaries

**Result Schema**:
```python
{
    "id": str,              # Document ID
    "text": str,            # Document text
    "vector": List[float],  # Embedding vector
    "metadata": Dict,       # Document metadata
    "score": float,         # Relevance score
}
```

**Example**:
```python
results = collection.search(query_vector=vec, limit=5)
for result in results.to_list():
    print(f"ID: {result['id']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text'][:100]}...")
    print()
```

### `__iter__()` and `__len__()`

QueryResult objects are iterable and have length.

**Example**:
```python
results = collection.search(query_vector=vec, limit=5)

# Iterate
for result in results:
    print(result['id'], result['score'])

# Length
print(f"Found {len(results)} results")
```

---

## Metadata Filtering

PeachBase supports MongoDB-like query syntax for filtering.

### Exact Match
```python
filter = {"category": "ai"}
```

### Comparison Operators
```python
# Greater than or equal
filter = {"year": {"$gte": 2023}}

# Less than or equal
filter = {"year": {"$lte": 2024}}

# Greater than
filter = {"score": {"$gt": 0.8}}

# Less than
filter = {"score": {"$lt": 1.0}}
```

### In Operator
```python
filter = {"category": {"$in": ["ai", "ml", "nlp"]}}
```

### Logical Operators
```python
# AND (implicit)
filter = {
    "category": "ai",
    "year": {"$gte": 2023}
}

# OR
filter = {
    "$or": [
        {"category": "ai"},
        {"category": "ml"}
    ]
}

# AND with explicit $and
filter = {
    "$and": [
        {"category": "ai"},
        {"year": {"$gte": 2023}}
    ]
}
```

### Complex Queries
```python
filter = {
    "$and": [
        {
            "$or": [
                {"category": "ai"},
                {"category": "ml"}
            ]
        },
        {"year": {"$gte": 2023}},
        {"score": {"$gt": 0.7}}
    ]
}
```

---

## Search Modes

### Semantic Search

Uses vector similarity to find documents.

**Required**: `query_vector`

**Metrics**:
- `cosine` (default): Cosine similarity, range [-1, 1], higher is better
- `l2`: L2 (Euclidean) distance, range [0, ∞], lower is better
- `dot`: Dot product, range [-∞, ∞], higher is better

**Example**:
```python
results = collection.search(
    query_vector=embedding,
    mode="semantic",
    metric="cosine",
    limit=10
)
```

### Lexical Search

Uses BM25 algorithm for keyword matching.

**Required**: `query_text`

**Tokenization**: Automatic (whitespace + lowercase + punctuation removal)

**Example**:
```python
results = collection.search(
    query_text="machine learning neural networks",
    mode="lexical",
    limit=10
)
```

### Hybrid Search

Combines semantic and lexical using Reciprocal Rank Fusion (RRF).

**Required**: Both `query_text` and `query_vector`

**Alpha Parameter**:
- `0.0`: Pure semantic search
- `0.5`: Balanced (default, recommended)
- `1.0`: Pure lexical search

**Example**:
```python
results = collection.search(
    query_text="machine learning neural networks",
    query_vector=embedding,
    mode="hybrid",
    alpha=0.5,
    limit=10
)
```

---

## Error Handling

Common exceptions:

**ValueError**:
- Invalid parameters (wrong dimension, missing required fields)
- Invalid search mode or metric

**KeyError**:
- Collection or document doesn't exist

**TypeError**:
- Wrong type for parameters (e.g., string instead of list for vector)

**Example**:
```python
try:
    collection = db.get_collection("documents")
    results = collection.search(query_vector=vec, limit=10)
except KeyError:
    print("Collection doesn't exist")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

---

## Performance Tips

### 1. Batch Operations
Add documents in batches for better performance:
```python
# Good: Batch add
collection.add(large_list_of_docs)

# Less efficient: Individual adds
for doc in large_list_of_docs:
    collection.add([doc])
```

### 2. Pre-compute Embeddings
Generate embeddings offline and add with vectors:
```python
# Compute once
embeddings = model.encode(texts, batch_size=128)

# Add to collection
docs = [
    {"id": f"d{i}", "text": text, "vector": vec.tolist()}
    for i, (text, vec) in enumerate(zip(texts, embeddings))
]
collection.add(docs)
```

### 3. Use Appropriate Search Mode
- Small exact matches: `lexical`
- Concept matching: `semantic`
- General purpose: `hybrid` (recommended)

### 4. Filter Early
Use metadata filters to reduce search space:
```python
results = collection.search(
    query_vector=vec,
    filter={"category": "relevant_only"},  # Pre-filter
    limit=10
)
```

### 5. Choose Right Metric
- General use: `cosine` (default, normalized)
- Pre-normalized vectors: `dot` (fastest)
- Actual distance matters: `l2`

---

## See Also

- [Search Modes Guide](../guides/search-modes.md) - When to use each mode
- [Understanding Scores](../guides/scoring.md) - How scores are calculated
- [Performance Optimizations](../guides/performance.md) - Optimization details

---

[← Back to Reference](../README.md#reference)
