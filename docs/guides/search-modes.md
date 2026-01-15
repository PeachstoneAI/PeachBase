# Search Modes Guide

Complete guide to PeachBase's three search modes.

---

## Overview

PeachBase supports three search strategies:

1. **Semantic** - Vector similarity (meaning-based)
2. **Lexical** - BM25 keyword matching (term-based)
3. **Hybrid** - Reciprocal Rank Fusion of both (recommended)

---

## Semantic Search

**Best for**: Concept matching, synonyms, multilingual search

### How It Works

Compares query embedding vector to document embedding vectors using similarity metrics.

```python
results = collection.search(
    query_vector=[0.1, 0.2, ...],
    mode="semantic",
    metric="cosine",  # or "l2", "dot"
    limit=10
)
```

### Metrics

#### Cosine Similarity (Default)
Measures angle between vectors (ignores magnitude).

**Range**: -1 to 1 (higher = more similar)

**Use when**: General semantic search (most common)

**Example**:
```python
# Query: "machine learning"
# Top results:
# 1. "artificial intelligence algorithms" (0.89)
# 2. "neural network training" (0.85)
# 3. "data science methods" (0.78)
```

#### L2 Distance (Euclidean)
Measures straight-line distance between vectors.

**Range**: 0 to ∞ (lower = more similar)

**Use when**: Actual distance matters

#### Dot Product
Measures alignment of vectors (considers magnitude).

**Range**: -∞ to ∞ (higher = more similar)

**Use when**: Vectors are normalized, fastest computation

### When to Use Semantic

✅ **Good for**:
- Concept matching across different wording
- Handling synonyms ("car" matches "automobile")
- Multilingual search (if using multilingual embeddings)
- Abstract queries ("explain quantum computing")

❌ **Not ideal for**:
- Exact keyword matching ("model number XYZ-123")
- Boolean queries ("Python AND Django")
- Named entities that need exact matches

---

## Lexical Search

**Best for**: Keyword matching, technical terms, names

### How It Works

Uses BM25 algorithm to rank documents based on term frequency and rarity.

```python
results = collection.search(
    query_text="machine learning algorithms",
    mode="lexical",
    limit=10
)
```

### BM25 Algorithm

Scores documents based on:
1. **Term frequency**: How often query terms appear
2. **IDF**: Rare terms score higher than common terms
3. **Length normalization**: Prevents long documents dominating

**Range**: 0 to ∞ (higher = more relevant)

### When to Use Lexical

✅ **Good for**:
- Exact keyword matching
- Technical terminology ("transformer architecture")
- Named entities ("OpenAI", "GPT-4")
- Boolean-style queries
- When exact wording matters

❌ **Not ideal for**:
- Synonym matching
- Handling paraphrases
- Cross-lingual search
- Abstract concept queries

---

## Hybrid Search (Recommended)

**Best for**: Production systems, general purpose

### How It Works

Combines semantic and lexical using Reciprocal Rank Fusion (RRF).

```python
results = collection.search(
    query_text="machine learning algorithms",
    query_vector=[0.1, 0.2, ...],
    mode="hybrid",
    alpha=0.5,  # Balance between lexical/semantic
    limit=10
)
```

### Reciprocal Rank Fusion (RRF)

Merges results based on ranks, not scores:

```
RRF_score = α × (1/(k + rank_lexical)) + (1-α) × (1/(k + rank_semantic))
```

Where:
- `α` = weight (0=semantic only, 1=lexical only)
- `k` = 60 (constant from research)

### Tuning Alpha

**α = 0.0** (Pure Semantic):
- Focus on meaning
- Ignore exact keywords

**α = 0.3-0.4** (Favor Semantic):
- Concept-driven
- Keyword hints
- Good for Q&A systems

**α = 0.5** (Balanced - Default):
- Best of both worlds
- **Recommended starting point**

**α = 0.6-0.7** (Favor Lexical):
- Keyword-driven
- Semantic hints
- Good for technical search

**α = 1.0** (Pure Lexical):
- Only exact keywords
- No semantic understanding

### When to Use Hybrid

✅ **Good for**:
- **Production systems** (best default)
- Diverse query types
- When you want both keyword precision and semantic recall
- RAG applications
- General-purpose search

❌ **Not ideal for**:
- When you specifically need only semantic OR only lexical
- When you can't generate embeddings

### Example Comparison

**Query**: "transformer models in NLP"

**Semantic results**:
1. "Attention mechanisms in neural networks" (concept match)
2. "BERT and GPT architectures" (semantic similarity)
3. "Deep learning for language" (related concept)

**Lexical results**:
1. "Transformer architecture overview" (exact keyword)
2. "NLP using transformer models" (exact keywords)
3. "Implementing transformers" (partial match)

**Hybrid results** (best of both):
1. "Transformer architecture for NLP" (both exact keywords + concept)
2. "BERT transformer model" (keyword + semantic)
3. "Attention-based NLP models" (semantic + related term)

---

## Comparison Table

| Feature | Semantic | Lexical | Hybrid |
|---------|----------|---------|--------|
| **Speed** (10K docs) | 3ms (⚡⚡⚡) | 12ms (⚡⚡) | 16ms (⚡⚡) |
| **Throughput** | 300 QPS | 85 QPS | 63 QPS |
| **Concept matching** | ✅ Excellent | ❌ Poor | ✅ Excellent |
| **Keyword precision** | ❌ Poor | ✅ Excellent | ✅ Excellent |
| **Synonyms** | ✅ Yes | ❌ No | ✅ Yes |
| **Exact terms** | ❌ No | ✅ Yes | ✅ Yes |
| **Setup complexity** | Medium | Low | Medium |
| **Best for** | Concepts | Keywords | Production |

---

## Decision Guide

```
Start here:
    ↓
Do you need exact keyword matching?
    ├─ Yes → Can you also generate embeddings?
    │         ├─ Yes → Use HYBRID ⭐
    │         └─ No → Use LEXICAL
    │
    └─ No → Do you care about meaning/concepts?
              ├─ Yes → Do you have query text too?
              │         ├─ Yes → Use HYBRID ⭐
              │         └─ No → Use SEMANTIC
              │
              └─ No → Use LEXICAL
```

**Rule of thumb**: When in doubt, use **HYBRID** with default settings.

---

## Code Examples

### Semantic Search
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

query = "artificial intelligence in healthcare"
query_vector = model.encode(query).tolist()

results = collection.search(
    query_vector=query_vector,
    mode="semantic",
    metric="cosine",
    limit=10
)
```

### Lexical Search
```python
query = "transformer neural network architecture"

results = collection.search(
    query_text=query,
    mode="lexical",
    limit=10
)
```

### Hybrid Search
```python
query = "transformer neural network architecture"
query_vector = model.encode(query).tolist()

results = collection.search(
    query_text=query,
    query_vector=query_vector,
    mode="hybrid",
    alpha=0.5,  # Balanced
    limit=10
)
```

### With Metadata Filter
```python
results = collection.search(
    query_text=query,
    query_vector=query_vector,
    mode="hybrid",
    filter={"category": "ai", "year": {"$gte": 2020}},
    limit=10
)
```

---

## Performance Tips

### Optimize Semantic Search
- Use `metric="dot"` for fastest computation (if vectors normalized)
- Use `metric="cosine"` for best quality (default)
- Pre-compute embeddings offline
- Batch embed queries when possible

### Optimize Lexical Search
- Keep documents reasonably sized (300-800 chars)
- Use meaningful text (not just keywords)
- Clean text (remove excessive whitespace, etc.)

### Optimize Hybrid Search
- Start with `alpha=0.5` and tune based on results
- For technical docs: increase alpha (favor lexical)
- For conversational queries: decrease alpha (favor semantic)
- Use metadata filters to narrow search space

---

## Advanced Usage

### Custom Alpha Per Query Type

```python
def search_documents(query, query_type="general"):
    query_vector = model.encode(query).tolist()

    # Tune alpha based on query type
    alpha_map = {
        "technical": 0.7,  # Favor keywords
        "conceptual": 0.3,  # Favor semantics
        "general": 0.5,    # Balanced
    }

    return collection.search(
        query_text=query,
        query_vector=query_vector,
        mode="hybrid",
        alpha=alpha_map.get(query_type, 0.5),
        limit=10
    )
```

### Re-ranking

```python
# First pass: Get more results
initial_results = collection.search(
    query_text=query,
    query_vector=query_vector,
    mode="hybrid",
    limit=50  # More than needed
)

# Second pass: Re-rank with LLM or custom logic
reranked = rerank_with_llm(query, initial_results)

# Return top results
final_results = reranked[:10]
```

---

## See Also

- [Understanding Scores](scoring.md) - How scores are calculated
- [Performance Tuning](performance.md) - Optimize for speed
- [API Reference](../reference/api.md) - Complete API docs

---

[← Back to Guides](README.md) | [Next: Understanding Scores →](scoring.md)
