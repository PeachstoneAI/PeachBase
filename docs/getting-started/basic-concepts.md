# Basic Concepts

Understanding the core concepts of PeachBase.

---

## Architecture Overview

```
┌─────────────┐
│   Database  │  (Storage location)
└──────┬──────┘
       │
       ├─ Collection 1  (Vector dimension: 384)
       │  ├─ Document 1 (text + vector + metadata)
       │  ├─ Document 2
       │  └─ Document N
       │
       └─ Collection 2  (Vector dimension: 768)
          ├─ Document 1
          └─ ...
```

---

## Core Components

### Database

A container for collections. Can be local or S3-based.

```python
# Local database
db = peachbase.connect("./my_db")

# S3 database
db = peachbase.connect("s3://bucket/my_db")
```

### Collection

A table of documents with fixed vector dimension.

```python
collection = db.create_collection(
    name="articles",
    dimension=384  # Must match embedding model dimension
)
```

**Key properties**:
- Fixed vector dimension (all documents must match)
- Unique document IDs
- Supports text, vectors, and metadata
- Built-in search indices (BM25 + vector)

### Document

A single record with text, vector embedding, and metadata.

```python
document = {
    "id": "doc123",              # Required: unique identifier
    "text": "Full text...",      # Required: document text
    "vector": [0.1, 0.2, ...],  # Required: embedding vector
    "metadata": {                # Optional: arbitrary metadata
        "author": "John Doe",
        "date": "2024-01-01",
        "tags": ["ai", "ml"]
    }
}
```

---

## Search Concepts

### Embeddings (Vectors)

Dense numerical representations of text:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

text = "Machine learning is a branch of AI"
vector = model.encode(text).tolist()  # [0.123, -0.456, ...]
```

**Key points**:
- Semantic meaning encoded as numbers
- Similar meanings = similar vectors
- Dimension must match collection
- Pre-compute offline for production

### Search Modes

PeachBase supports three search strategies:

#### 1. Semantic Search
Compares query vector to document vectors.

**How it works**: Vector similarity (cosine, L2, dot product)

**Best for**: Concept matching, synonyms, meaning

**Example**:
```python
# Query: "AI algorithms"
# Matches: "machine learning methods", "neural networks"
```

#### 2. Lexical Search
Compares query keywords to document text.

**How it works**: BM25 algorithm (term frequency + rarity)

**Best for**: Exact keywords, technical terms, names

**Example**:
```python
# Query: "transformer architecture"
# Matches documents containing exactly "transformer" and "architecture"
```

#### 3. Hybrid Search
Combines semantic and lexical using rank fusion.

**How it works**: Reciprocal Rank Fusion (RRF)

**Best for**: Production systems, general purpose

**Example**:
```python
# Query: "transformer models"
# Combines:
# - Semantic: matches "attention mechanisms", "BERT"
# - Lexical: matches exact "transformer" keyword
```

See [Search Modes Guide](../guides/search-modes.md) for details.

---

## Scoring

Each search mode returns documents with scores:

### Semantic Scores

**Cosine similarity** (default): -1 to 1 (higher = more similar)
```python
# 0.95 = very similar
# 0.60 = somewhat similar
# 0.20 = not similar
```

**L2 distance**: 0 to ∞ (lower = more similar)
```python
# 0.1 = very close
# 1.0 = distant
# 5.0 = very far
```

### Lexical Scores

**BM25**: 0 to ∞ (higher = more relevant)
```python
# 10+ = excellent match (many rare terms)
# 5-10 = good match
# 1-5 = weak match
# 0 = no terms found
```

### Hybrid Scores

**RRF**: 0 to ~0.02 (higher = better combined rank)
```python
# 0.016 = top in both modes
# 0.010 = good in both
# 0.005 = present in one mode
```

**Important**: Only compare scores within the same query, not across different queries!

See [Scoring Guide](../guides/scoring.md) for complete details.

---

## Metadata

Arbitrary JSON data attached to documents:

```python
metadata = {
    "category": "technology",
    "author": "Jane Smith",
    "date": "2024-01-15",
    "tags": ["ai", "ml", "nlp"],
    "rating": 4.5,
    "published": True
}
```

### Filtering

Filter search results by metadata:

```python
# Simple filter
results = collection.search(
    query_vector=vec,
    filter={"category": "technology"}
)

# Complex filter (MongoDB-like)
results = collection.search(
    query_vector=vec,
    filter={
        "category": "technology",
        "rating": {"$gte": 4.0},
        "tags": {"$in": ["ai", "ml"]},
        "$or": [
            {"author": "Jane Smith"},
            {"published": True}
        ]
    }
)
```

**Supported operators**:
- Exact match: `{"key": "value"}`
- Greater/less: `{"key": {"$gte": 5}}`, `{"key": {"$lte": 10}}`
- In list: `{"key": {"$in": [1, 2, 3]}}`
- Logical: `{"$or": [...]}`, `{"$and": [...]}`

---

## Persistence

### Save to Disk

```python
collection.save()  # Saves to database path
```

Creates `.pdb` files with:
- Vectors (memory-mapped for fast loading)
- Text content
- Metadata
- Search indices

### Load from Disk

```python
db = peachbase.connect("./my_db")
collection = db.get_collection("articles")  # Auto-loads
```

Loading is fast (memory-mapped):
- 1K docs: ~10ms
- 10K docs: ~50ms
- 50K docs: ~200ms

---

## Performance Characteristics

### Search Latency (10K documents, 384-dim)

| Mode | Latency | QPS | Best For |
|------|---------|-----|----------|
| Semantic (cosine) | 3ms | 300 | Concept matching |
| Lexical (BM25) | 12ms | 85 | Keyword matching |
| Hybrid (RRF) | 16ms | 63 | Production (recommended) |

### Insertion Throughput

- **~2M documents/second** (with OpenMP)
- Batching recommended for best performance
- Pre-compute embeddings offline

### Collection Size Limits

- **Recommended**: <50K documents per collection
- **Maximum**: 100K+ (but search gets slower)
- **For larger**: Shard across multiple collections

### Memory Usage

Approximately:
```
Memory = (num_docs × dimension × 4 bytes) + overhead

Example (10K docs, 384-dim):
= (10,000 × 384 × 4) + ~50MB
= ~65MB
```

---

## SIMD Acceleration

PeachBase uses SIMD (Single Instruction, Multiple Data) for vector operations:

**Performance**: 362x faster than pure Python

**Support**:
- ✅ AVX2 (most modern CPUs)
- ✅ AVX-512 (latest Intel/AMD)
- ✅ Fallback for older CPUs

**Auto-detected** at runtime - no configuration needed!

---

## Multi-Core (OpenMP)

For collections >1K documents, PeachBase uses multiple CPU cores:

**Standard build** (default):
- Uses all CPU cores
- 3-4x faster for large collections
- Requires libgomp (~1.2MB)

**Lambda build** (opt-in):
- Single-threaded
- No dependencies
- Smaller package (43KB)

Build with OpenMP:
```bash
python -m build  # OpenMP enabled by default
```

Build for Lambda:
```bash
PEACHBASE_DISABLE_OPENMP=1 python -m build
```

See [Building Guide](../guides/building.md) for details.

---

## Best Practices

### Document Structure

✅ **Good**:
```python
{
    "id": "unique_id",
    "text": "Concise, meaningful text (300-800 chars)",
    "vector": [...],  # From quality embedding model
    "metadata": {     # Structured, searchable fields
        "category": "technology",
        "date": "2024-01-15"
    }
}
```

❌ **Avoid**:
```python
{
    "id": None,  # Missing ID
    "text": "...",  # Extremely long text (>5000 chars)
    "vector": [...],  # Wrong dimension
    "metadata": "unstructured string"  # Not a dict
}
```

### Chunking Text

For long documents, split into chunks:

```python
def chunk_text(text, chunk_size=500, overlap=50):
    # Split by sentences
    sentences = text.split('. ')

    chunks = []
    current = []
    length = 0

    for sent in sentences:
        if length + len(sent) > chunk_size and current:
            chunks.append('. '.join(current))
            # Keep last sentence for overlap
            current = [current[-1], sent]
            length = len(current[-1]) + len(sent)
        else:
            current.append(sent)
            length += len(sent)

    if current:
        chunks.append('. '.join(current))

    return chunks
```

**Recommendations**:
- Chunk size: 300-800 characters
- Overlap: 50-100 characters
- Split on sentence boundaries

### Embedding Models

**Choose based on**:

**Speed** → `all-MiniLM-L6-v2` (384-dim, fast)
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**Quality** → `all-mpnet-base-v2` (768-dim, accurate)
```python
model = SentenceTransformer('all-mpnet-base-v2')
```

**Multilingual** → `paraphrase-multilingual-MiniLM-L12-v2`
```python
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

---

## Common Patterns

### RAG Pipeline
```
User Query
    ↓
Embed Query → Search Collection → Get Top K Results
    ↓
Build Context from Results
    ↓
Send to LLM → Generate Answer
```

### Semantic Search App
```
Documents → Chunk → Embed → Store in PeachBase
                                    ↓
User Query → Embed → Search → Display Results
```

### Deduplication
```
New Document → Embed → Search for Similar
                              ↓
                        If similar exists → Skip
                        If unique → Add
```

---

## Next Steps

- **[Search Modes Guide](../guides/search-modes.md)** - Deep dive into search strategies
- **[Scoring Guide](../guides/scoring.md)** - Understand score calculation
- **[Examples](../examples/)** - See practical implementations
- **[API Reference](../reference/api.md)** - Complete API documentation

---

[← Back to Quick Start](quick-start.md) | [Next: Search Modes →](../guides/search-modes.md)
