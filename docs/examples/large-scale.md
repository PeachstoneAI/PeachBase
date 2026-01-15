# Large-Scale Example (10K+ Documents)

Guide for working with large document collections (10,000-100,000 documents) in PeachBase.

---

## Overview

This guide covers:
- Handling 10K-100K documents efficiently
- Performance optimization techniques
- Memory management
- Batch processing strategies
- Production deployment patterns

**Example Files**:
- [`examples/wikipedia_rag_large.py`](../../examples/wikipedia_rag_large.py) - 50+ articles, 1000+ chunks
- [`examples/wikipedia_rag_large_hf.py`](../../examples/wikipedia_rag_large_hf.py) - With HuggingFace models
- [`examples/performance_benchmark.py`](../../examples/performance_benchmark.py) - Performance testing

---

## Scale Considerations

### Collection Size Limits

| Documents | Vectors (384-dim) | Memory | Search Time | Recommended |
|-----------|-------------------|---------|-------------|-------------|
| 1,000 | 1.5 MB | 10 MB | 2-5 ms | ✅ Ideal |
| 10,000 | 15 MB | 50 MB | 20-50 ms | ✅ Good |
| 50,000 | 75 MB | 200 MB | 100-200 ms | ⚠️ Acceptable |
| 100,000 | 150 MB | 400 MB | 300-500 ms | ⚠️ Max recommended |
| 500,000+ | 750+ MB | 2+ GB | 1-5 sec | ❌ Split collections |

*Search times for limit=10, with SIMD acceleration*

### When to Split Collections

**Split when:**
- Collection > 100K documents
- Search latency > 500ms
- Memory usage > 1GB
- Different document types/domains

**Splitting strategies:**
- By topic/category
- By date range
- By source/origin
- By language

---

## Performance Optimization

### 1. SIMD Acceleration

PeachBase uses AVX2/AVX-512 SIMD instructions for fast vector operations.

**Enable:**
```bash
# Build with OpenMP for multi-core
python -m build

# Or install pre-built wheel
pip install peachbase
```

**Verify:**
```python
from peachbase import _simd

# Check CPU features
features = _simd.detect_cpu_features()
print(f"SIMD level: {features}")  # 0=none, 1=AVX2, 2=AVX-512

# Check OpenMP
info = _simd.get_openmp_info()
print(f"OpenMP enabled: {info['compiled_with_openmp']}")
print(f"Max threads: {info.get('max_threads', 1)}")
```

**Performance gain:**
- AVX2: ~100-200x faster than pure Python
- AVX-512: ~300-400x faster than pure Python
- OpenMP (4 cores): Additional 3-4x speedup

### 2. Batch Processing

Process documents in batches for better performance:

```python
import peachbase
from sentence_transformers import SentenceTransformer

db = peachbase.connect("./large_db")
collection = db.create_collection("documents", dimension=384, overwrite=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Process in batches
batch_size = 1000
all_texts = [...]  # Your 10K+ documents

for i in range(0, len(all_texts), batch_size):
    print(f"Processing batch {i//batch_size + 1}...")

    # Batch encode embeddings (GPU accelerated if available)
    batch_texts = all_texts[i:i+batch_size]
    batch_embeddings = model.encode(batch_texts, batch_size=32, show_progress_bar=True)

    # Prepare documents
    documents = [
        {
            "id": f"doc_{i+j}",
            "text": text,
            "vector": embedding.tolist(),
            "metadata": {"batch": i//batch_size}
        }
        for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings))
    ]

    # Add to collection
    collection.add(documents)

    # Save periodically (every 10 batches or at end)
    if (i // batch_size + 1) % 10 == 0:
        collection.save()
        print(f"  ✓ Saved {collection.size} documents")

# Final save
collection.save()
print(f"✓ Complete! {collection.size} documents indexed")
```

**Benefits:**
- Efficient GPU utilization for embeddings
- Progress tracking
- Incremental saves (recovery from failures)
- Lower memory footprint

### 3. Memory Management

#### Monitor Memory Usage

```python
import psutil
import os

def print_memory_usage(label=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[{label}] Memory usage: {mem_mb:.1f} MB")

print_memory_usage("Start")

# Load collection
db = peachbase.connect("./large_db")
collection = db.open_collection("documents")
print_memory_usage("After load")

# Perform search
results = collection.search(query_vector=query, limit=10)
print_memory_usage("After search")
```

#### Memory-Efficient Loading

```python
# PeachBase uses memory-mapped files for efficient loading
db = peachbase.connect("./large_db")
collection = db.open_collection("documents")

# Vectors are memory-mapped (not fully loaded into RAM)
# Fast access without loading entire collection
print(f"Loaded {collection.size} documents (memory-mapped)")
```

### 4. Multi-Threading (OpenMP)

Control thread usage for better performance:

```bash
# Set OpenMP threads
export OMP_NUM_THREADS=4

# Run your application
python your_app.py
```

In Python:

```python
import os

# Set before importing peachbase
os.environ['OMP_NUM_THREADS'] = '4'

import peachbase
```

**Thread recommendations:**
- Single query: Use 4-8 threads
- Multiple concurrent queries: Use 2-4 threads per query
- Server deployment: Match CPU core count

---

## Large-Scale Pipeline Example

Complete pipeline for processing 10K+ Wikipedia articles:

```python
"""
Large-scale Wikipedia RAG with 10,000+ chunks.

Runtime: ~5-10 minutes
Memory: ~2 GB peak
Output: Production-ready search index
"""

import peachbase
import wikipediaapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import time

def download_many_articles(category: str, limit: int = 100) -> Dict[str, str]:
    """Download many articles from a Wikipedia category."""
    wiki = wikipediaapi.Wikipedia(user_agent='PeachBase/1.0', language='en')

    # Get category
    cat = wiki.page(f"Category:{category}")

    articles = {}
    for page_name in list(cat.categorymembers.keys())[:limit]:
        page = wiki.page(page_name)
        if page.exists() and page.namespace == 0:  # Main namespace only
            articles[page.title] = page.text[:50000]
            if len(articles) >= limit:
                break

    return articles

def chunk_and_embed_batch(
    texts: List[str],
    model: SentenceTransformer,
    chunk_size: int = 500
) -> List[Dict]:
    """Chunk texts and generate embeddings in batch."""
    # Chunk all texts
    all_chunks = []
    for text_id, text in enumerate(texts):
        chunks = chunk_text(text, chunk_size=chunk_size)
        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append({
                "text_id": text_id,
                "chunk_id": chunk_id,
                "text": chunk
            })

    # Batch encode all chunks
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        chunk_texts,
        batch_size=64,  # Larger batch for GPU
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Add embeddings to chunks
    for chunk, embedding in zip(all_chunks, embeddings):
        chunk["embedding"] = embedding

    return all_chunks

def main():
    """Main execution for large-scale indexing."""
    print("=" * 70)
    print("Large-Scale Wikipedia Indexing")
    print("=" * 70)

    # Configuration
    COLLECTION_NAME = "wikipedia_large"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DIMENSION = 384
    TARGET_ARTICLES = 100
    BATCH_SIZE = 10

    start_time = time.time()

    # Step 1: Load embedding model
    print(f"\n📥 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✓ Model loaded (dimension: {DIMENSION})")

    # Step 2: Download articles
    print(f"\n📥 Downloading {TARGET_ARTICLES} Wikipedia articles...")
    articles = download_many_articles("Artificial_intelligence", limit=TARGET_ARTICLES)
    print(f"  ✓ Downloaded {len(articles)} articles")

    # Step 3: Connect to PeachBase
    print(f"\n💾 Initializing PeachBase...")
    db = peachbase.connect("./wikipedia_large_db")
    collection = db.create_collection(
        name=COLLECTION_NAME,
        dimension=DIMENSION,
        overwrite=True
    )
    print(f"  ✓ Collection created: {COLLECTION_NAME}")

    # Step 4: Process in batches
    print(f"\n🔄 Processing articles in batches of {BATCH_SIZE}...")
    article_items = list(articles.items())
    total_chunks = 0

    for batch_num, i in enumerate(range(0, len(article_items), BATCH_SIZE)):
        batch = article_items[i:i+BATCH_SIZE]
        print(f"\n  Batch {batch_num + 1}/{(len(article_items) + BATCH_SIZE - 1) // BATCH_SIZE}")

        # Chunk and embed batch
        titles, texts = zip(*batch)
        chunks_with_embeddings = chunk_and_embed_batch(list(texts), model)

        # Prepare documents for PeachBase
        documents = []
        for chunk_data in chunks_with_embeddings:
            article_title = titles[chunk_data["text_id"]]
            doc_id = f"article_{i + chunk_data['text_id']}_chunk_{chunk_data['chunk_id']}"

            documents.append({
                "id": doc_id,
                "text": chunk_data["text"],
                "vector": chunk_data["embedding"].tolist(),
                "metadata": {
                    "source": article_title,
                    "chunk_index": chunk_data["chunk_id"],
                    "batch": batch_num
                }
            })

        # Add to collection
        collection.add(documents)
        total_chunks += len(documents)

        print(f"    ✓ Added {len(documents)} chunks (total: {total_chunks})")

        # Save every 5 batches
        if (batch_num + 1) % 5 == 0:
            collection.save()
            print(f"    💾 Checkpoint saved ({collection.size} documents)")

    # Step 5: Final save
    print(f"\n💾 Saving final collection...")
    collection.save()

    elapsed = time.time() - start_time
    print(f"\n✓ Complete!")
    print(f"  - Articles processed: {len(articles)}")
    print(f"  - Total chunks: {total_chunks}")
    print(f"  - Collection size: {collection.size}")
    print(f"  - Time elapsed: {elapsed:.1f} seconds")
    print(f"  - Throughput: {total_chunks / elapsed:.1f} chunks/second")

    # Step 6: Performance test
    print(f"\n🔍 Testing search performance...")
    test_queries = [
        "How do neural networks learn?",
        "What is natural language processing?",
        "Explain machine learning algorithms"
    ]

    for query in test_queries:
        query_vector = model.encode(query).tolist()

        search_start = time.time()
        results = collection.search(
            query_vector=query_vector,
            query_text=query,
            mode="hybrid",
            limit=10
        )
        search_time = (time.time() - search_start) * 1000

        print(f"  - Query: '{query[:40]}...'")
        print(f"    Results: {len(results.to_list())}, Time: {search_time:.1f}ms")

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
======================================================================
Large-Scale Wikipedia Indexing
======================================================================

📥 Loading embedding model: all-MiniLM-L6-v2
  ✓ Model loaded (dimension: 384)

📥 Downloading 100 Wikipedia articles...
  ✓ Downloaded 100 articles

💾 Initializing PeachBase...
  ✓ Collection created: wikipedia_large

🔄 Processing articles in batches of 10...

  Batch 1/10
Batches: 100%|████████████| 8/8 [00:02<00:00,  3.2 batches/s]
    ✓ Added 234 chunks (total: 234)

  ...

  Batch 10/10
Batches: 100%|████████████| 7/7 [00:02<00:00,  3.1 batches/s]
    ✓ Added 198 chunks (total: 2,134)
    💾 Checkpoint saved (2,134 documents)

💾 Saving final collection...

✓ Complete!
  - Articles processed: 100
  - Total chunks: 2,134
  - Collection size: 2,134
  - Time elapsed: 287.3 seconds
  - Throughput: 7.4 chunks/second

🔍 Testing search performance...
  - Query: 'How do neural networks learn?...'
    Results: 10, Time: 12.3ms
  - Query: 'What is natural language processing?...'
    Results: 10, Time: 11.8ms
  - Query: 'Explain machine learning algorithms...'
    Results: 10, Time: 13.1ms
```

---

## Production Deployment

### Server Setup

```python
# Flask API for large-scale search
from flask import Flask, request, jsonify
import peachbase
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Initialize once at startup
db = peachbase.connect("./wikipedia_large_db")
collection = db.open_collection("wikipedia_large")
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    mode = data.get("mode", "hybrid")
    limit = data.get("limit", 10)

    # Generate embedding
    query_vector = model.encode(query).tolist()

    # Search
    results = collection.search(
        query_text=query,
        query_vector=query_vector,
        mode=mode,
        limit=limit
    )

    return jsonify({
        "results": [
            {
                "id": r["id"],
                "text": r["text"][:200],
                "score": r["score"],
                "source": r["metadata"]["source"]
            }
            for r in results.to_list()
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test search endpoint
ab -n 1000 -c 10 -p query.json -T application/json \
   http://localhost:8000/search
```

**Expected results (10K collection):**
```
Requests per second:    50.23 [#/sec]
Time per request:       19.91 [ms] (mean)
Percentage of requests served within:
  50%     18ms
  95%     35ms
  99%     52ms
```

---

## Optimization Checklist

### Indexing Phase
- [ ] Use batch processing (1000+ docs per batch)
- [ ] Enable GPU for embedding generation
- [ ] Save checkpoints every 5-10 batches
- [ ] Monitor memory usage
- [ ] Profile slow operations

### Query Phase
- [ ] Enable SIMD acceleration (verify with `detect_cpu_features()`)
- [ ] Use OpenMP multi-threading
- [ ] Cache embedding model
- [ ] Implement result caching for common queries
- [ ] Monitor search latency

### Deployment
- [ ] Build with OpenMP (`python -m build`)
- [ ] Set optimal thread count (`OMP_NUM_THREADS`)
- [ ] Use memory-mapped loading
- [ ] Configure proper logging
- [ ] Set up monitoring (latency, throughput)

---

## Troubleshooting

### Slow Indexing

**Symptom:** < 1 chunk/second
**Solutions:**
- Enable GPU: `model = SentenceTransformer(..., device='cuda')`
- Increase batch size: `batch_size=64`
- Use faster model: `all-MiniLM-L3-v2`

### High Memory Usage

**Symptom:** > 4GB RAM for 10K docs
**Solutions:**
- Process smaller batches
- Clear embeddings after adding: `del embeddings`
- Use memory-mapped loading (automatic in PeachBase)

### Slow Search

**Symptom:** > 100ms for 10K docs
**Solutions:**
- Verify SIMD enabled: `_simd.detect_cpu_features() > 0`
- Build with OpenMP: `python -m build`
- Reduce search limit: `limit=10` instead of `limit=100`
- Split into multiple collections

---

## Next Steps

- **Deployment**: See [deployment guide](../guides/deployment.md)
- **Performance**: Read [performance optimizations](../guides/performance.md)
- **API**: Check [API reference](../reference/api.md)

---

**Questions?** Open an issue: https://github.com/PeachstoneAI/peachbase/issues
