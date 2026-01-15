# PeachBase Search Scoring Explained

## Overview

PeachBase supports three search modes, each with its own scoring mechanism:

1. **Semantic Search** - Vector similarity metrics
2. **Lexical Search** - BM25 algorithm
3. **Hybrid Search** - Reciprocal Rank Fusion (RRF)

---

## 1. Semantic Search Scoring

Semantic search compares query and document embeddings using distance/similarity metrics.

### Available Metrics

#### Cosine Similarity (Default)
**Range**: -1 to 1 (higher is better)

**Formula**:
```
cosine(q, d) = (q · d) / (||q|| × ||d||)
```

Where:
- `q · d` = dot product of query and document vectors
- `||q||` = L2 norm (magnitude) of query vector
- `||d||` = L2 norm of document vector

**C Implementation** (SIMD optimized):
```c
static float cosine_similarity_avx2(const float* vec1, const float* vec2, size_t dim) {
    __m256 sum_dot = _mm256_setzero_ps();
    __m256 sum_norm1 = _mm256_setzero_ps();
    __m256 sum_norm2 = _mm256_setzero_ps();

    for (i = 0; i + 8 <= dim; i += 8) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);

        sum_dot = _mm256_fmadd_ps(v1, v2, sum_dot);      // q · d
        sum_norm1 = _mm256_fmadd_ps(v1, v1, sum_norm1);  // ||q||²
        sum_norm2 = _mm256_fmadd_ps(v2, v2, sum_norm2);  // ||d||²
    }

    float dot = hsum_avx2(sum_dot);
    float norm1 = sqrtf(hsum_avx2(sum_norm1));
    float norm2 = sqrtf(hsum_avx2(sum_norm2));

    return dot / (norm1 * norm2);
}
```

**Example**:
```python
# Query vector (3-dim for illustration)
query = [0.5, 0.3, 0.8]

# Document vector
doc = [0.6, 0.4, 0.7]

# Calculation:
# dot product = 0.5*0.6 + 0.3*0.4 + 0.8*0.7 = 0.3 + 0.12 + 0.56 = 0.98
# ||query|| = sqrt(0.5² + 0.3² + 0.8²) = sqrt(0.98) = 0.99
# ||doc|| = sqrt(0.6² + 0.4² + 0.7²) = sqrt(1.01) = 1.00

# cosine = 0.98 / (0.99 * 1.00) = 0.9899

# Score: 0.9899 (very similar!)
```

**Interpretation**:
- `1.0`: Vectors point in same direction (identical or scaled)
- `0.0`: Vectors are orthogonal (no similarity)
- `-1.0`: Vectors point in opposite directions

**When to use**: Default choice. Good for most semantic search tasks. Ignores magnitude, only considers direction.

---

#### L2 Distance (Euclidean Distance)
**Range**: 0 to ∞ (lower is better)

**Formula**:
```
L2(q, d) = sqrt(Σ(qᵢ - dᵢ)²)
```

**C Implementation**:
```c
static float l2_distance_avx2(const float* vec1, const float* vec2, size_t dim) {
    __m256 sum = _mm256_setzero_ps();

    for (i = 0; i + 8 <= dim; i += 8) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);
        __m256 diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_fmadd_ps(diff, diff, sum);  // (qᵢ - dᵢ)²
    }

    float result = hsum_avx2(sum);
    return sqrtf(result);
}
```

**Example**:
```python
query = [0.5, 0.3, 0.8]
doc = [0.6, 0.4, 0.7]

# Calculation:
# (0.5-0.6)² + (0.3-0.4)² + (0.8-0.7)² = 0.01 + 0.01 + 0.01 = 0.03
# sqrt(0.03) = 0.173

# Score: 0.173 (very close!)
```

**Interpretation**:
- `0.0`: Vectors are identical
- Small values: Vectors are close
- Large values: Vectors are far apart

**When to use**: When actual distance matters (not just direction). Sensitive to vector magnitude.

---

#### Dot Product
**Range**: -∞ to ∞ (higher is better)

**Formula**:
```
dot(q, d) = Σ(qᵢ × dᵢ)
```

**C Implementation**:
```c
static float dot_product_avx2(const float* vec1, const float* vec2, size_t dim) {
    __m256 sum = _mm256_setzero_ps();

    for (i = 0; i + 8 <= dim; i += 8) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);
        sum = _mm256_fmadd_ps(v1, v2, sum);  // qᵢ × dᵢ
    }

    return hsum_avx2(sum);
}
```

**Example**:
```python
query = [0.5, 0.3, 0.8]
doc = [0.6, 0.4, 0.7]

# Calculation:
# 0.5*0.6 + 0.3*0.4 + 0.8*0.7 = 0.3 + 0.12 + 0.56 = 0.98

# Score: 0.98
```

**Interpretation**:
- Higher values: More similar (considering both direction and magnitude)
- Lower/negative values: Less similar or opposite

**When to use**: When vectors are normalized or when you want magnitude to affect similarity. Fastest to compute (no sqrt).

---

## 2. Lexical Search Scoring (BM25)

BM25 (Best Match 25) is a probabilistic ranking function for text search.

### Formula

For a query Q with terms {q₁, q₂, ..., qₙ} and document d:

```
BM25(d, Q) = Σ IDF(qᵢ) × (tf × (k₁ + 1)) / (tf + k₁ × (1 - b + b × (|d| / avgdl)))
              qᵢ∈Q∩d
```

Where:
- **IDF(qᵢ)** = Inverse Document Frequency of term qᵢ
- **tf** = Term frequency (how many times qᵢ appears in d)
- **|d|** = Length of document d (in tokens)
- **avgdl** = Average document length in the collection
- **k₁** = Parameter controlling term frequency saturation (default: 1.5)
- **b** = Parameter controlling length normalization (default: 0.75)

### Component Calculations

#### IDF (Inverse Document Frequency)
```
IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1.0)
```

Where:
- **N** = Total number of documents
- **df** = Document frequency (number of documents containing the term)

**Purpose**: Rare terms get higher scores than common terms.

**Example**:
```python
# Collection with 1000 documents
N = 1000

# Term "machine" appears in 200 documents
df_machine = 200
IDF_machine = log((1000 - 200 + 0.5) / (200 + 0.5) + 1.0)
            = log(800.5 / 200.5 + 1.0)
            = log(5.0)
            = 1.609

# Term "learning" appears in 50 documents (rarer)
df_learning = 50
IDF_learning = log((1000 - 50 + 0.5) / (50 + 0.5) + 1.0)
             = log(950.5 / 50.5 + 1.0)
             = log(19.82)
             = 2.986

# "learning" gets higher IDF because it's rarer
```

#### Term Frequency Component
```
TF_component = (tf × (k₁ + 1)) / (tf + k₁ × normalized_length)

Where:
normalized_length = 1 - b + b × (|d| / avgdl)
```

**Purpose**: Balance between rewarding term frequency and document length.

**Python Implementation**:
```python
def _score_document(self, doc_idx, query_term_ids, query_idfs):
    score = 0.0
    doc_len = self.doc_lengths[doc_idx]
    normalized_len = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)

    for tid, idf in zip(query_term_ids, query_idfs):
        tf = self.inverted_index[tid].get(doc_idx, 0)

        if tf > 0:
            # BM25 formula
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * normalized_len
            score += idf * (numerator / denominator)

    return score
```

### Complete Example

```python
# Query: "machine learning"
# Document: "machine learning is a subset of artificial intelligence..."

# Setup:
k1 = 1.5
b = 0.75
avgdl = 100  # Average doc length
doc_len = 120  # This doc length

# Term frequencies in document:
tf_machine = 2
tf_learning = 1

# IDF scores (from collection):
idf_machine = 1.609
idf_learning = 2.986

# Normalized length:
normalized_len = 1.0 - 0.75 + 0.75 * (120 / 100)
               = 0.25 + 0.9
               = 1.15

# Score for "machine":
numerator = 2 * (1.5 + 1.0) = 5.0
denominator = 2 + 1.5 * 1.15 = 3.725
score_machine = 1.609 * (5.0 / 3.725) = 2.159

# Score for "learning":
numerator = 1 * (1.5 + 1.0) = 2.5
denominator = 1 + 1.5 * 1.15 = 2.725
score_learning = 2.986 * (2.5 / 2.725) = 2.738

# Total BM25 score:
BM25 = 2.159 + 2.738 = 4.897
```

### Range and Interpretation

**Range**: 0 to ∞ (higher is better)

**Typical values**:
- `0`: No query terms in document
- `1-5`: Weak match (few terms or common words)
- `5-10`: Good match (multiple terms or rare words)
- `>10`: Excellent match (many relevant terms)

**Interpretation depends on**:
- Query length (more terms = higher possible scores)
- Term rarity (rare terms = higher scores)
- Term frequency in document
- Document length relative to average

---

## 3. Hybrid Search Scoring (RRF)

Hybrid search combines lexical and semantic results using **Reciprocal Rank Fusion (RRF)**.

### RRF Formula

```
RRF_score(d) = α × RRF_lexical(d) + (1-α) × RRF_semantic(d)

Where:
RRF_lexical(d) = 1 / (k + rank_lexical(d))
RRF_semantic(d) = 1 / (k + rank_semantic(d))
```

**Parameters**:
- **α (alpha)**: Weight between lexical and semantic (0 to 1)
  - `α = 0.0`: Pure semantic search
  - `α = 0.5`: Balanced (default)
  - `α = 1.0`: Pure lexical search
- **k**: Constant to reduce impact of high ranks (default: 60, from research literature)

### Why RRF?

RRF is **rank-based**, not score-based, which solves several problems:

1. **No normalization needed**: Lexical and semantic scores are on different scales
2. **Robust**: Not sensitive to outlier scores
3. **Research-backed**: k=60 is proven effective across domains
4. **Simple**: Easy to understand and tune

### Complete Example

```python
# Query: "neural networks"

# Lexical results (BM25):
# 1. doc_A: 8.5
# 2. doc_B: 7.2
# 3. doc_C: 6.8
# 4. doc_D: 5.1

# Semantic results (cosine):
# 1. doc_C: 0.92
# 2. doc_B: 0.88
# 3. doc_E: 0.85
# 4. doc_A: 0.82

# Build rank dictionaries:
lexical_ranks = {
    'doc_A': 0,  # rank 1 (0-indexed)
    'doc_B': 1,  # rank 2
    'doc_C': 2,  # rank 3
    'doc_D': 3   # rank 4
}

semantic_ranks = {
    'doc_C': 0,  # rank 1
    'doc_B': 1,  # rank 2
    'doc_E': 2,  # rank 3
    'doc_A': 3   # rank 4
}

# Compute RRF scores (k=60, α=0.5):

# doc_A:
RRF_lex_A = 1 / (60 + 0) = 0.01667
RRF_sem_A = 1 / (60 + 3) = 0.01587
RRF_A = 0.5 * 0.01667 + 0.5 * 0.01587 = 0.01627

# doc_B:
RRF_lex_B = 1 / (60 + 1) = 0.01639
RRF_sem_B = 1 / (60 + 1) = 0.01639
RRF_B = 0.5 * 0.01639 + 0.5 * 0.01639 = 0.01639  # ← Highest!

# doc_C:
RRF_lex_C = 1 / (60 + 2) = 0.01613
RRF_sem_C = 1 / (60 + 0) = 0.01667
RRF_C = 0.5 * 0.01613 + 0.5 * 0.01667 = 0.01640  # ← Very close!

# doc_D (only in lexical):
RRF_lex_D = 1 / (60 + 3) = 0.01587
RRF_sem_D = 0  # Not in semantic results
RRF_D = 0.5 * 0.01587 + 0.5 * 0 = 0.00794

# doc_E (only in semantic):
RRF_lex_E = 0  # Not in lexical results
RRF_sem_E = 1 / (60 + 2) = 0.01613
RRF_E = 0.5 * 0 + 0.5 * 0.01613 = 0.00806

# Final ranking:
# 1. doc_C: 0.01640  (top semantic, good lexical)
# 2. doc_B: 0.01639  (good in both)
# 3. doc_A: 0.01627  (top lexical, okay semantic)
# 4. doc_E: 0.00806  (only semantic)
# 5. doc_D: 0.00794  (only lexical)
```

### Tuning Alpha

**α = 0.0** (Pure Semantic):
```python
# Focus on meaning, ignore keywords
results = collection.search(
    query_text="neural networks",
    query_vector=query_vec,
    mode="hybrid",
    alpha=0.0  # Pure semantic
)
```

**α = 0.5** (Balanced, Default):
```python
# Best of both worlds
results = collection.search(
    query_text="neural networks",
    query_vector=query_vec,
    mode="hybrid",
    alpha=0.5  # Balanced
)
```

**α = 1.0** (Pure Lexical):
```python
# Focus on exact keywords
results = collection.search(
    query_text="neural networks",
    query_vector=query_vec,
    mode="hybrid",
    alpha=1.0  # Pure lexical
)
```

**Recommended values**:
- `α = 0.3-0.4`: Favor semantic (concept matching)
- `α = 0.5`: Balanced (general purpose)
- `α = 0.6-0.7`: Favor lexical (keyword matching)

---

## Summary Table

| Search Mode | Score Range | Higher/Lower Better | Based On | Best For |
|-------------|-------------|---------------------|----------|----------|
| **Semantic (Cosine)** | -1 to 1 | Higher | Vector direction | Concept matching |
| **Semantic (L2)** | 0 to ∞ | Lower | Vector distance | Exact similarity |
| **Semantic (Dot)** | -∞ to ∞ | Higher | Vector alignment | Normalized vectors |
| **Lexical (BM25)** | 0 to ∞ | Higher | Term frequency & rarity | Keyword matching |
| **Hybrid (RRF)** | 0 to ~0.02 | Higher | Rank fusion | General purpose |

---

## Practical Tips

### Choosing a Search Mode

**Use Semantic (Cosine)** when:
- ✅ Meaning matters more than exact words
- ✅ Synonyms should match
- ✅ Multilingual search
- ✅ Abstract concepts

**Use Lexical (BM25)** when:
- ✅ Exact keywords matter
- ✅ Technical terms or names
- ✅ Boolean-style queries
- ✅ Legal/medical documents

**Use Hybrid (RRF)** when:
- ✅ **Best default choice**
- ✅ Want both meaning and keywords
- ✅ Diverse query types
- ✅ Production systems

### Interpreting Scores

**High scores don't always mean "better"** - they're relative to:
- Collection size
- Query complexity
- Term distribution

**Compare scores within a query**, not across queries:
```python
# This is meaningful:
# Query: "machine learning"
# Result 1: score 0.85 (more relevant)
# Result 2: score 0.72 (less relevant)

# This is NOT meaningful:
# Query 1: "ML" → Result: score 0.85
# Query 2: "artificial intelligence" → Result: score 0.62
# Can't say Query 1 result is "better" than Query 2 result
```

---

Made with 🍑 by the PeachBase team
