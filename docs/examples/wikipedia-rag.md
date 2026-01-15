# Wikipedia RAG Example

End-to-end Retrieval-Augmented Generation (RAG) system using Wikipedia articles and PeachBase.

---

## Overview

This example demonstrates a complete RAG pipeline:
1. Download Wikipedia articles
2. Chunk text into manageable pieces
3. Generate embeddings using sentence-transformers
4. Store in PeachBase
5. Perform semantic, lexical, and hybrid search
6. Build a Q&A system

**Example Files**:
- [`examples/wikipedia_rag.py`](../../examples/wikipedia_rag.py) - Basic RAG (5 articles, ~100 chunks)
- [`examples/wikipedia_rag_large.py`](../../examples/wikipedia_rag_large.py) - Large scale (50+ articles, 1000+ chunks)
- [`examples/wikipedia_rag_large_hf.py`](../../examples/wikipedia_rag_large_hf.py) - Large scale with HuggingFace models

---

## Prerequisites

Install required dependencies:

```bash
# Install PeachBase
pip install peachbase

# Install RAG dependencies
pip install sentence-transformers wikipedia-api

# Optional: For faster processing
pip install torch  # GPU support for embeddings
```

---

## RAG Pipeline Architecture

```
Wikipedia Articles
       ↓
Text Chunking (500 chars, 50 overlap)
       ↓
Embedding Generation (sentence-transformers)
       ↓
PeachBase Storage (semantic + lexical indices)
       ↓
Hybrid Search (RRF fusion)
       ↓
Context Retrieval
       ↓
LLM Answer Generation (GPT-4, Claude, etc.)
```

---

## Step-by-Step Guide

### Step 1: Download Wikipedia Articles

```python
import wikipediaapi

def download_wikipedia_articles(topics: List[str], max_chars: int = 50000):
    """Download Wikipedia articles for given topics."""
    wiki = wikipediaapi.Wikipedia(
        user_agent='PeachBase-Example/1.0',
        language='en'
    )

    articles = {}
    for topic in topics:
        page = wiki.page(topic)
        if page.exists():
            content = page.text[:max_chars]
            articles[topic] = content

    return articles

# Example: Download AI-related articles
topics = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Computer vision"
]

articles = download_wikipedia_articles(topics)
```

**Output:**
```
📥 Downloading Wikipedia articles...
  - Fetching: Artificial intelligence
    ✓ Got 45,234 chars
  - Fetching: Machine learning
    ✓ Got 38,921 chars
  ...
```

### Step 2: Chunk Text

Break articles into smaller, overlapping chunks for better retrieval:

```python
import re

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks."""
    # Split into sentences first for better boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))

            # Start new chunk with overlap
            overlap_words = ' '.join(current_chunk).split()[-overlap:]
            current_chunk = overlap_words
            current_length = sum(len(w) for w in overlap_words)

        current_chunk.append(sentence)
        current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Chunk all articles
all_chunks = []
for title, content in articles.items():
    chunks = chunk_text(content, chunk_size=500, overlap=50)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "source": title,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk
        })

print(f"Created {len(all_chunks)} chunks from {len(articles)} articles")
```

**Output:**
```
✂️  Chunking articles...
  - Artificial intelligence: 89 chunks
  - Machine learning: 76 chunks
  ...
Created 423 chunks from 5 articles
```

**Why chunk?**
- **Better relevance**: Smaller chunks = more precise retrieval
- **Context limits**: LLMs have token limits (~4K-128K)
- **Performance**: Faster search over smaller text units
- **Granularity**: Return only relevant sections, not entire articles

### Step 3: Generate Embeddings

Use sentence-transformers to create vector embeddings:

```python
from sentence_transformers import SentenceTransformer

# Load embedding model (384-dimensional)
print("📥 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all chunks
print("🔢 Generating embeddings...")
texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

print(f"Generated {len(embeddings)} embeddings, dimension: {embeddings[0].shape[0]}")
```

**Output:**
```
📥 Loading embedding model...
  ✓ Loaded all-MiniLM-L6-v2 (384 dimensions)

🔢 Generating embeddings...
Batches: 100%|████████████| 14/14 [00:03<00:00,  4.2 batches/s]
Generated 423 embeddings, dimension: 384
```

**Model Options:**

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose |
| all-mpnet-base-v2 | 768 | Medium | Better | Higher quality |
| all-MiniLM-L12-v2 | 384 | Fast | Good | Balanced |

### Step 4: Store in PeachBase

Create collection and add documents:

```python
import peachbase

# Connect to database
db = peachbase.connect("./wikipedia_db")

# Create collection (384 dimensions for all-MiniLM-L6-v2)
collection = db.create_collection(
    name="wikipedia",
    dimension=384,
    overwrite=True
)

# Prepare documents for PeachBase
documents = []
for i, chunk in enumerate(all_chunks):
    documents.append({
        "id": f"chunk_{i}",
        "text": chunk["text"],
        "vector": embeddings[i].tolist(),
        "metadata": {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"]
        }
    })

# Add to collection
collection.add(documents)
collection.save()

print(f"✓ Stored {collection.size} chunks in PeachBase")
```

**Output:**
```
💾 Storing in PeachBase...
  - Creating collection 'wikipedia' (384 dimensions)
  - Adding 423 documents...
  - Building BM25 index...
  ✓ Collection saved!
```

### Step 5: Perform Searches

#### Semantic Search (Vector Similarity)

```python
def semantic_search(collection, query: str, model):
    """Search by meaning/concepts."""
    # Generate query embedding
    query_vector = model.encode(query).tolist()

    # Search
    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        metric="cosine",
        limit=5
    )

    return results.to_list()

# Example query
query = "How do neural networks learn?"
results = semantic_search(collection, query, model)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['metadata']['source']} (Score: {result['score']:.4f})")
    print(f"   {result['text'][:150]}...\n")
```

**Output:**
```
🔍 Searching: 'How do neural networks learn?'
   Mode: semantic
   ======================================================================

   Result 1 (Score: 0.8234)
   Source: Deep learning
   Chunk: 12/89
   Text: Neural networks learn through backpropagation, adjusting weights
         based on the error between predicted and actual outputs...

   Result 2 (Score: 0.7891)
   Source: Machine learning
   Chunk: 23/76
   Text: Training involves iteratively updating model parameters to minimize
         a loss function, often using gradient descent...
```

#### Lexical Search (BM25 Keywords)

```python
def lexical_search(collection, query: str):
    """Search by keywords/terms."""
    results = collection.search(
        query_text=query,
        mode="lexical",
        limit=5
    )

    return results.to_list()

# Example: Find exact term mentions
query = "neural network architecture"
results = lexical_search(collection, query)
```

**When to use:**
- Exact terminology (e.g., "gradient descent", "ReLU activation")
- Named entities (e.g., "Geoffrey Hinton", "ImageNet")
- Abbreviations (e.g., "CNN", "RNN", "LSTM")

#### Hybrid Search (Best of Both)

```python
def hybrid_search(collection, query: str, model, alpha: float = 0.5):
    """Combine semantic and lexical search."""
    query_vector = model.encode(query).tolist()

    results = collection.search(
        query_text=query,
        query_vector=query_vector,
        mode="hybrid",
        alpha=alpha,  # 0=semantic only, 1=lexical only, 0.5=balanced
        limit=5
    )

    return results.to_list()

# Example
query = "What is backpropagation?"
results = hybrid_search(collection, query, model, alpha=0.4)  # Favor semantic
```

**Alpha Tuning:**
- `alpha=0.0`: Pure semantic (meaning-based)
- `alpha=0.3`: Semantic-favored (conceptual queries)
- `alpha=0.5`: Balanced (general purpose)
- `alpha=0.7`: Lexical-favored (technical terms)
- `alpha=1.0`: Pure lexical (exact keywords)

### Step 6: Build Q&A System

Retrieve context and generate answers:

```python
def answer_question(collection, question: str, model, top_k: int = 3):
    """Answer question using retrieved context."""
    # Retrieve relevant context using hybrid search
    query_vector = model.encode(question).tolist()
    results = collection.search(
        query_text=question,
        query_vector=query_vector,
        mode="hybrid",
        limit=top_k,
        alpha=0.4  # Favor semantic for Q&A
    )

    # Build context from top results
    context_parts = []
    sources = set()

    for result in results.to_list():
        context_parts.append(result['text'])
        sources.add(result['metadata']['source'])

    context = "\n\n".join(context_parts)

    # In production: Send to LLM
    # answer = llm.generate(f"Context: {context}\n\nQuestion: {question}")

    print(f"📚 Retrieved {len(results)} passages from: {', '.join(sources)}")
    print(f"\n📝 Context:\n{context[:300]}...")

    return context

# Example
question = "What are the main types of machine learning?"
context = answer_question(collection, question, model, top_k=3)
```

**Output:**
```
❓ Question: What are the main types of machine learning?
   ======================================================================

   📚 Retrieved 3 relevant passages from:
      - Machine learning
      - Artificial intelligence

   📝 Context (for LLM to process):
   ──────────────────────────────────────────────────────────────────────
   Machine learning is generally divided into three main categories:
   supervised learning, unsupervised learning, and reinforcement learning.
   Supervised learning involves training on labeled data...
   ──────────────────────────────────────────────────────────────────────

   💡 In production: Send the question + context to an LLM for answer generation
```

**Integration with LLMs:**

```python
# OpenAI GPT-4
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Answer based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
)
answer = response.choices[0].message.content

# Anthropic Claude
import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[{
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    }]
)
answer = message.content[0].text
```

---

## Running the Examples

### Basic Example (5 articles)

```bash
python examples/wikipedia_rag.py
```

**Runtime:** ~30 seconds
**Output:**
- Creates `wikipedia_db/` directory
- 423 chunks from 5 articles
- Demonstrates all 3 search modes
- Example Q&A

### Large Scale Example (50+ articles)

```bash
python examples/wikipedia_rag_large.py
```

**Runtime:** ~3-5 minutes
**Output:**
- 1000+ chunks from 50+ articles
- Full RAG pipeline
- Performance benchmarks

### HuggingFace Models Example

```bash
python examples/wikipedia_rag_large_hf.py
```

**Features:**
- Uses different embedding models
- Compares model quality
- Performance analysis

---

## Performance Characteristics

### Embedding Generation

| Model | Chunks/sec | 1000 chunks | Memory |
|-------|-----------|-------------|---------|
| all-MiniLM-L6-v2 | ~200 | ~5 sec | 400 MB |
| all-mpnet-base-v2 | ~100 | ~10 sec | 800 MB |
| With GPU | ~1000 | ~1 sec | 2 GB |

### PeachBase Search

| Collection Size | Semantic | Lexical | Hybrid |
|----------------|----------|---------|--------|
| 100 chunks | 2 ms | 3 ms | 5 ms |
| 1,000 chunks | 8 ms | 10 ms | 15 ms |
| 10,000 chunks | 45 ms | 50 ms | 80 ms |

*Measured on 384-dim vectors, limit=10*

---

## Best Practices

### Chunking Strategy

**Good chunk sizes:**
- **Articles/Blogs**: 500-800 characters
- **Documentation**: 300-500 characters
- **Books**: 800-1200 characters
- **Code**: By function/class (variable size)

**Overlap:**
- **General**: 50-100 characters (10-20% of chunk size)
- **Technical docs**: 100-150 characters (more context needed)
- **No overlap**: When chunks are self-contained (Q&A pairs, etc.)

### Embedding Models

**For English:**
- General: `all-MiniLM-L6-v2` (384-dim, fast)
- Quality: `all-mpnet-base-v2` (768-dim, better)
- Speed: `paraphrase-MiniLM-L3-v2` (384-dim, fastest)

**For Multilingual:**
- `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- `distiluse-base-multilingual-cased-v2` (512-dim)

### Search Mode Selection

**Use Semantic when:**
- User asks conceptual questions
- Paraphrases and synonyms matter
- Cross-language search (with multilingual models)

**Use Lexical when:**
- Exact terminology required
- Technical documentation
- Code search
- Named entities

**Use Hybrid when:**
- General-purpose Q&A (most common)
- Balance between concepts and exact terms
- Mixed query types

### Production Tips

1. **Batch processing**: Add documents in batches of 100-1000
2. **Caching**: Load model once, reuse for all queries
3. **Pre-compute**: Generate all embeddings offline
4. **Indexing**: Call `.save()` after adding all documents
5. **Monitoring**: Track search latency and quality metrics

---

## Advanced Features

### Metadata Filtering

```python
# Search only specific sources
results = collection.search(
    query_vector=query_vector,
    filter={"source": "Machine learning"},
    limit=10
)

# Search with date range (if you add timestamps)
results = collection.search(
    query_vector=query_vector,
    filter={"timestamp": {"$gte": "2024-01-01"}},
    limit=10
)
```

### Multi-Collection RAG

```python
# Create multiple collections for different topics
ml_collection = db.create_collection("machine_learning", dimension=384)
cv_collection = db.create_collection("computer_vision", dimension=384)

# Search across all collections
def search_all_collections(query):
    all_results = []
    for collection in [ml_collection, cv_collection]:
        results = collection.search(query_vector=query_vector, limit=5)
        all_results.extend(results.to_list())

    # Sort by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:10]
```

### Re-ranking

```python
# First pass: Fast retrieval (get top 100)
initial_results = collection.search(query_vector=query_vector, limit=100)

# Second pass: Re-rank with cross-encoder
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [[query, r["text"]] for r in initial_results.to_list()]
scores = reranker.predict(pairs)

# Sort by re-ranked scores
reranked = sorted(zip(initial_results.to_list(), scores),
                  key=lambda x: x[1], reverse=True)

final_results = [r[0] for r in reranked[:10]]
```

---

## Troubleshooting

**"No module named 'wikipediaapi'"**
```bash
pip install wikipedia-api
```

**"RuntimeError: No sentence-transformers model found"**
```bash
pip install sentence-transformers transformers torch
```

**Embeddings dimension mismatch**
- Solution: Check model output dimension matches collection dimension
- `all-MiniLM-L6-v2` → 384 dimensions
- `all-mpnet-base-v2` → 768 dimensions

**Slow embedding generation**
- Solution: Use GPU if available
```python
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

**Out of memory**
- Solution: Process in smaller batches
```python
batch_size = 32
for i in range(0, len(texts), batch_size):
    batch_embeddings = model.encode(texts[i:i+batch_size])
```

---

## Next Steps

- **Large Scale**: See [large-scale example](large-scale.md) for 10K+ documents
- **Deployment**: Read [deployment guide](../guides/deployment.md) for production
- **API Reference**: Check [API docs](../reference/api.md) for complete API

---

**Questions?** Open an issue: https://github.com/PeachstoneAI/peachbase/issues
