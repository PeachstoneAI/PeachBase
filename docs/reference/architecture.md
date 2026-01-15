# PeachBase Architecture

Internal architecture and implementation details of PeachBase.

---

## Design Philosophy

PeachBase is designed with three core principles:

1. **Lambda-First**: Optimized for serverless deployments with fast cold starts and minimal dependencies
2. **Performance**: SIMD-accelerated operations and efficient algorithms
3. **Simplicity**: Clean API and straightforward implementation

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        PeachBase API                          │
│                    (Python Interface)                       │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼────────┐ ┌─────▼──────────┐
│   Database     │ │   Collection   │
│   - Management │ │   - Documents  │
│   - Collections│ │   - Search     │
└───────┬────────┘ └─────┬──────────┘
        │                 │
        │     ┌───────────┴───────────────┐
        │     │                           │
┌───────▼─────▼─────┐         ┌──────────▼─────────┐
│  Storage Layer    │         │  Search Layer      │
│  - Binary Format  │         │  - Semantic        │
│  - Memory Mapping │         │  - Lexical (BM25)  │
│  - S3 Integration │         │  - Hybrid (RRF)    │
└───────┬───────────┘         └──────────┬─────────┘
        │                                 │
        │     ┌───────────────────────────┘
        │     │
┌───────▼─────▼──────────────────────────────────┐
│           C Extensions (SIMD)                  │
│  - Vector Operations (AVX2/AVX-512)           │
│  - BM25 Scoring                               │
│  - CPU Feature Detection                      │
└───────────────────────────────────────────────┘
```

---

## Core Components

### 1. Database Layer

**Location**: `src/peachbase/database.py`

**Responsibilities**:
- Database connection management
- Collection lifecycle (create, get, delete)
- Directory structure management
- Collection metadata persistence

**Key Design**:
```python
class Database:
    def __init__(self, path: str):
        self.path = path  # Local or S3 path
        self.collections = {}  # In-memory collection registry
```

**File Structure**:
```
database_path/
├── collections.json          # Collection metadata
└── collection_name/
    ├── vectors.pdb          # Binary vector data
    ├── documents.json       # Document metadata
    └── bm25_index.pkl       # BM25 inverted index
```

### 2. Collection Layer

**Location**: `src/peachbase/collection.py`

**Responsibilities**:
- Document CRUD operations
- Search orchestration
- Index management
- Pre-flattened vector caching

**Key Data Structures**:
```python
class Collection:
    _documents: List[Dict]              # Document storage
    _vectors: List[List[float]]         # Vector storage
    _flattened_vectors: Optional[array] # Cached flattened vectors
    _bm25_index: BM25Index             # Lexical search index
```

**Pre-flattened Vectors**:
```python
# Critical optimization: Cache flattened array
if self._flattened_vectors is None:
    flat = array('f')
    for vec in self._vectors:
        flat.extend(vec)
    self._flattened_vectors = flat
```

This eliminates 55ms overhead per query for large collections.

### 3. Storage Layer

**Location**: `src/peachbase/storage/`

#### Binary Format (`.pdb`)

Custom format optimized for memory-mapping and S3:

```
┌─────────────────────────────────────────────┐
│              Header (256 bytes)             │
│  - Magic: "PCHDB001"                       │
│  - Version, doc count, dimension           │
│  - Section offsets                         │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│         Vector Data (contiguous)            │
│  [doc0_vec] [doc1_vec] ... [docN_vec]     │
│  Each: dimension × 4 bytes (float32)       │
│  SIMD-aligned for fast access              │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│              Text Data                      │
│  [id_len][id_bytes][text_len][text_bytes]  │
│  Variable length, sequential                │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│            Metadata (JSON)                  │
│  [doc_id, metadata_json] × n_docs          │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│           BM25 Index (Pickle)               │
│  - Vocabulary: {term: term_id}             │
│  - IDF scores: {term_id: idf}              │
│  - Inverted index: {term_id: {doc: freq}}  │
│  - Document lengths                         │
└─────────────────────────────────────────────┘
```

**Benefits**:
- Memory-mappable for instant loading
- S3 byte-range friendly (can fetch sections)
- SIMD-aligned vectors
- Compact on-disk representation

#### Memory Mapping

**Location**: `src/peachbase/storage/mmap.py`

**Strategy**:
```python
import mmap

# Map vector section only
with open(pdb_file, 'rb') as f:
    mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    # Direct memory access to vectors
    vector_data = mm[vector_offset:vector_offset+vector_size]
```

**Advantages**:
- Zero-copy loading (OS handles paging)
- Instant "loading" (just map address space)
- Efficient memory usage (OS caches pages)

#### S3 Integration

**Location**: `src/peachbase/utils/s3.py`

**Smart Loading**:
```python
# 1. Fetch header (256 bytes)
header = s3.get_object(
    Bucket=bucket,
    Key=key,
    Range='bytes=0-255'
)

# 2. Parse offsets
vector_offset, vector_size = parse_header(header)

# 3. Fetch critical sections
if need_vectors:
    vectors = s3.get_object(
        Bucket=bucket,
        Key=key,
        Range=f'bytes={vector_offset}-{vector_offset+vector_size}'
    )
```

**Lambda Optimization**:
- Cache in `/tmp` between invocations
- Use S3 Transfer Acceleration if available
- Lazy loading of non-critical sections

---

## Search Architecture

### 1. Semantic Search

**Location**: `src/peachbase/search/semantic.py`

**Algorithm**: Brute-force with SIMD acceleration

**Flow**:
```
Query Vector → Metadata Filter → SIMD Batch Compute → Top-K Selection → Results
```

**Implementation**:
```python
def search_semantic(
    query_vector: List[float],
    vectors: array,
    metric: str = "cosine",
    limit: int = 10
) -> List[Tuple[int, float]]:
    # 1. Call C extension for batch computation
    if metric == "cosine":
        scores = _simd.batch_cosine_similarity(
            query_array,
            vectors_array,
            n_docs,
            dimension
        )
    elif metric == "l2":
        scores = _simd.batch_l2_distance(
            query_array,
            vectors_array,
            n_docs,
            dimension
        )
    elif metric == "dot":
        scores = _simd.batch_dot_product(
            query_array,
            vectors_array,
            n_docs,
            dimension
        )

    # 2. Heap-based top-k selection
    results = heapq.nlargest(
        limit,
        enumerate(scores),
        key=lambda x: x[1]
    )

    return results
```

**Why Brute-Force?**:
- Simple and predictable
- Fast for <50K documents with SIMD
- No index building overhead
- Exact results (not approximate)
- Lambda-friendly (minimal memory)

### 2. Lexical Search (BM25)

**Location**: `src/peachbase/search/bm25.py`

**Algorithm**: BM25 with inverted index

**Flow**:
```
Query Text → Tokenize → Term Lookup → BM25 Score → Top-K → Results
```

**Key Data Structure** (Critical Optimization):
```python
class BM25Index:
    vocabulary: Dict[str, int]              # term → term_id
    idf_scores: Dict[int, float]            # term_id → IDF
    inverted_index: Dict[int, Dict[int, int]]  # term_id → {doc_id: freq}
    doc_lengths: List[int]                  # Document lengths
    avg_doc_length: float
```

**O(1) Lookup Optimization**:
```python
# Before: O(n) linear search
for doc_idx, freq in posting_list:  # List[Tuple[int, int]]
    if doc_idx == target:
        tf = freq
        break

# After: O(1) dict lookup
tf = inverted_index[term_id].get(doc_idx, 0)  # Dict[int, int]
```

**Result**: 31x speedup!

**BM25 Formula**:
```python
def bm25_score(term_freq, doc_length, idf, k1=1.5, b=0.75):
    norm_length = doc_length / avg_doc_length
    tf_component = (term_freq * (k1 + 1)) / (
        term_freq + k1 * (1 - b + b * norm_length)
    )
    return idf * tf_component
```

**Tokenization**:
```python
def tokenize(text: str) -> List[str]:
    # Simple but effective
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    return text.split()
```

No heavy NLP libraries required!

### 3. Hybrid Search (RRF)

**Location**: `src/peachbase/search/hybrid.py`

**Algorithm**: Reciprocal Rank Fusion

**Flow**:
```
Query Text + Vector → Parallel Search → Rank Fusion → Merged Results
              ↓                ↓
        Lexical (BM25)   Semantic (SIMD)
              ↓                ↓
          Ranked List     Ranked List
              ↓                ↓
              └────────┬───────┘
                       ↓
                   RRF Merge
                       ↓
                 Final Ranking
```

**Parallel Execution** (Critical Optimization):
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    # Run both searches concurrently
    lexical_future = executor.submit(bm25_search, query_text)
    semantic_future = executor.submit(semantic_search, query_vector)

    # Wait for both
    lexical_results = lexical_future.result()
    semantic_results = semantic_future.result()
```

**Result**: 73% faster than sequential!

**RRF Formula**:
```python
def reciprocal_rank_fusion(
    lexical_results: List[Tuple[int, float]],
    semantic_results: List[Tuple[int, float]],
    alpha: float = 0.5,
    k: int = 60
) -> List[Tuple[int, float]]:
    scores = {}

    # Lexical contribution
    for rank, (doc_id, _) in enumerate(lexical_results):
        scores[doc_id] = alpha * (1.0 / (k + rank + 1))

    # Semantic contribution
    for rank, (doc_id, _) in enumerate(semantic_results):
        scores[doc_id] = scores.get(doc_id, 0) + \
                         (1 - alpha) * (1.0 / (k + rank + 1))

    # Sort by combined score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Why RRF?**:
- Rank-based (not score-based) - more robust
- No need to normalize different score ranges
- Proven effective in information retrieval
- Simple to implement and tune

---

## C Extensions (SIMD)

**Location**: `csrc/peachbase_simd.c`

### CPU Feature Detection

**Runtime Detection**:
```c
void peachbase_init_cpu_features(void) {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512dq")) {
        g_cpu_features = PEACHBASE_CPU_AVX512;
    } else if (__builtin_cpu_supports("avx2")) {
        g_cpu_features = PEACHBASE_CPU_AVX2;
    } else {
        g_cpu_features = PEACHBASE_CPU_NONE;
    }
}
```

**Dispatch**:
```c
float cosine_similarity(const float* vec1, const float* vec2, size_t dim) {
    switch (g_cpu_features) {
        case PEACHBASE_CPU_AVX512:
            return cosine_similarity_avx512(vec1, vec2, dim);
        case PEACHBASE_CPU_AVX2:
            return cosine_similarity_avx2(vec1, vec2, dim);
        default:
            return cosine_similarity_scalar(vec1, vec2, dim);
    }
}
```

### AVX2 Implementation

**Cosine Similarity**:
```c
static float cosine_similarity_avx2(
    const float* vec1,
    const float* vec2,
    size_t dim
) {
    __m256 sum_dot = _mm256_setzero_ps();
    __m256 sum_norm1 = _mm256_setzero_ps();
    __m256 sum_norm2 = _mm256_setzero_ps();

    // Process 8 floats at a time
    size_t i;
    for (i = 0; i + 8 <= dim; i += 8) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);

        // Fused multiply-add
        sum_dot = _mm256_fmadd_ps(v1, v2, sum_dot);
        sum_norm1 = _mm256_fmadd_ps(v1, v1, sum_norm1);
        sum_norm2 = _mm256_fmadd_ps(v2, v2, sum_norm2);
    }

    // Horizontal sum using intrinsics (optimized)
    float dot = hsum_avx2(sum_dot);
    float norm1 = hsum_avx2(sum_norm1);
    float norm2 = hsum_avx2(sum_norm2);

    // Handle remainder (scalar)
    for (; i < dim; i++) {
        dot += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    return dot / (sqrtf(norm1) * sqrtf(norm2));
}
```

**Optimized Horizontal Sum**:
```c
static inline float hsum_avx2(__m256 v) {
    // Extract 128-bit halves
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);

    // Add halves
    __m128 sum = _mm_add_ps(lo, hi);

    // Horizontal add (2 passes)
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}
```

**Performance**: 298-310x faster than pure Python!

### Batch Operations

**Critical Optimization**:
```c
void peachbase_batch_cosine_similarity(
    const float* query,
    const float* vectors,
    size_t n_vectors,
    size_t dim,
    float* results
) {
    // Single C call, no Python loop overhead
    for (size_t i = 0; i < n_vectors; i++) {
        results[i] = cosine_similarity_avx2(
            query,
            &vectors[i * dim],
            dim
        );
    }
}
```

**Why It Matters**:
- Eliminates Python overhead (function calls, type checking)
- Better memory locality (sequential access)
- Enables compiler optimizations

---

## Performance Optimizations

### 1. Pre-flattened Vectors

**Problem**: Converting `List[List[float]]` to flat array on every search

**Solution**: Cache the flattened representation
```python
class Collection:
    def _ensure_flattened_vectors(self):
        if self._flattened_vectors is None:
            flat = array('f')
            for vec in self._vectors:
                flat.extend(vec)
            self._flattened_vectors = flat
```

**Impact**: Eliminated 55ms overhead per query

### 2. Dict-based BM25 Index

**Problem**: O(n) linear search in posting lists

**Solution**: Dict for O(1) lookup
```python
# Before
inverted_index: Dict[int, List[Tuple[int, int]]]

# After
inverted_index: Dict[int, Dict[int, int]]
```

**Impact**: 31x faster lexical search

### 3. Parallel Hybrid Search

**Problem**: Sequential execution wasting CPU time

**Solution**: ThreadPoolExecutor for parallelism
```python
with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(search_fn, [lexical_task, semantic_task])
```

**Impact**: 73% faster hybrid search

### 4. Heap-based Top-K

**Problem**: Full sort when we only need top-k

**Solution**: Use heapq
```python
# Before: O(n log n)
results.sort(key=lambda x: x[1], reverse=True)
return results[:limit]

# After: O(n log k)
return heapq.nlargest(limit, results, key=lambda x: x[1])
```

**Impact**: 10-20% faster for large result sets

### 5. OpenMP Multi-threading

**Problem**: Single-threaded SIMD not using all cores

**Solution**: OpenMP pragmas in C
```c
#pragma omp parallel for
for (size_t i = 0; i < n_vectors; i++) {
    results[i] = cosine_similarity_avx2(...);
}
```

**Impact**: 3-4x speedup for collections >1K

---

## Design Decisions

### Why No Heavy Dependencies?

**Decision**: Only boto3 as runtime dependency

**Rationale**:
- Lambda package size limits
- Fast cold starts
- Easier deployment
- Fewer security vulnerabilities

**Implementation**:
- Custom tokenizer instead of NLTK/spaCy
- Custom binary format instead of parquet
- Manual SIMD instead of numpy

### Why Brute-Force Search?

**Decision**: No ANN algorithms (HNSW, IVF)

**Rationale**:
- Simplicity and predictability
- Fast enough with SIMD for <50K docs
- No index building overhead
- Exact results, not approximate
- Lambda-friendly (low memory)

**When to Reconsider**:
- Collections >50K documents
- Need for sub-10ms latency at scale
- Willing to trade accuracy for speed

### Why Custom Binary Format?

**Decision**: Custom `.pdb` instead of existing formats

**Rationale**:
- Memory-mappable structure
- S3 byte-range friendly
- SIMD-aligned vectors
- No external dependencies
- Optimized for our access patterns

**Alternatives Considered**:
- ❌ Parquet: Too heavy, requires pyarrow
- ❌ HDF5: Complex, large dependency
- ❌ pickle: Not memory-mappable, not S3-friendly

### Why Both Semantic AND Lexical?

**Decision**: Support both search paradigms + hybrid

**Rationale**:
- Different use cases need different approaches
- Semantic misses exact keywords
- Lexical misses semantic similarity
- Hybrid gets best of both worlds
- Rare in lightweight solutions

**Result**: More versatile than pure vector DBs

---

## Future Enhancements

### Potential Improvements

**1. HNSW Index for Large Collections**
- Approximate nearest neighbors
- Sub-linear search time
- Memory overhead
- Index building cost

**2. ARM NEON SIMD Support**
- Support Apple Silicon (M1/M2)
- Support AWS Graviton
- Similar performance to x86 SIMD

**3. Pre-normalized Vectors**
- Store normalized vectors separately
- Skip normalization in cosine search
- 20-30% faster cosine similarity

**4. Query Caching**
- Cache frequent queries
- LRU eviction policy
- Useful for production APIs

**5. Incremental Index Updates**
- Update BM25 index without full rebuild
- Faster document additions
- More complex implementation

---

## See Also

- [API Reference](api.md) - Public API documentation
- [Performance Benchmarks](performance.md) - Detailed performance data
- [Performance Optimizations](../guides/performance.md) - Optimization journey

---

[← Back to Reference](../README.md#reference)
