# PeachBase Roadmap

This document outlines planned features and improvements for PeachBase, including implementation plans for TODO items found in the codebase.

---

## Current Version: 0.1.0

**Status**: Initial release ✅
**Date**: 2026-01-15

---

## Immediate TODOs (From Codebase)

These are TODO comments found in the source code that need implementation:

### 1. S3 Listing Feature
**File**: `src/peachbase/database.py:116`
**Status**: 🔴 Not Implemented
**Priority**: High
**Estimated Effort**: 2-3 days

```python
# TODO: Implement S3 listing
```

**Description**: Currently, `list_collections()` only works with local filesystem. Need to implement S3 bucket listing to discover available collections.

**Implementation Plan**:

#### Phase 1: Research & Design (Day 1)
- [ ] Review boto3 S3 list_objects_v2 API
- [ ] Design collection naming convention for S3 (e.g., `{collection_name}.pdb`)
- [ ] Handle pagination for large S3 buckets
- [ ] Define error handling for missing buckets, permissions errors
- [ ] Document S3 URI format: `s3://bucket-name/path/to/collections/`

#### Phase 2: Implementation (Day 2)
- [ ] Add S3 client initialization in `Database.__init__()`
- [ ] Implement `_list_s3_collections()` method:
  ```python
  def _list_s3_collections(self) -> list[str]:
      """List collections stored in S3 bucket."""
      s3 = boto3.client('s3')
      bucket, prefix = self._parse_s3_uri(self.uri)

      paginator = s3.get_paginator('list_objects_v2')
      collections = []

      for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
          for obj in page.get('Contents', []):
              key = obj['Key']
              if key.endswith('.pdb'):
                  # Extract collection name from key
                  name = key[len(prefix):].rstrip('.pdb')
                  collections.append(name)

      return collections
  ```
- [ ] Update `list_collections()` to call `_list_s3_collections()` when using S3 URI
- [ ] Add caching mechanism to avoid repeated S3 API calls

#### Phase 3: Testing (Day 3)
- [ ] Unit tests with moto (S3 mocking library)
- [ ] Test empty buckets
- [ ] Test pagination (>1000 objects)
- [ ] Test permission errors
- [ ] Integration test with real S3 bucket
- [ ] Update documentation with S3 listing examples

#### Dependencies:
- boto3 (already required)
- moto (for testing, add to dev dependencies)

#### Success Criteria:
- [ ] `db.list_collections()` works with S3 URIs
- [ ] Handles pagination for large buckets
- [ ] Proper error messages for permission issues
- [ ] Performance: < 1 second for buckets with < 100 collections
- [ ] All tests pass

---

### 2. S3 Deletion Feature
**File**: `src/peachbase/database.py:138`
**Status**: 🔴 Not Implemented
**Priority**: Medium
**Estimated Effort**: 1-2 days

```python
# TODO: Implement S3 deletion
```

**Description**: Currently, `drop_collection()` only works with local filesystem. Need to implement S3 object deletion.

**Implementation Plan**:

#### Phase 1: Design (Day 1 Morning)
- [ ] Review boto3 S3 delete_object API
- [ ] Design deletion strategy:
  - Delete `.pdb` file
  - Delete associated metadata files (if any)
  - Handle already-deleted objects (idempotent)
- [ ] Define error handling for permission errors
- [ ] Consider soft delete vs hard delete (mark deleted vs actually remove)

#### Phase 2: Implementation (Day 1 Afternoon)
- [ ] Implement `_drop_s3_collection()` method:
  ```python
  def _drop_s3_collection(self, name: str) -> None:
      """Delete collection from S3 bucket."""
      s3 = boto3.client('s3')
      bucket, prefix = self._parse_s3_uri(self.uri)
      key = f"{prefix}{name}.pdb"

      try:
          s3.delete_object(Bucket=bucket, Key=key)
          # Also delete from local cache if exists
          self._collections.pop(name, None)
      except s3.exceptions.NoSuchKey:
          # Already deleted or never existed (idempotent)
          pass
      except Exception as e:
          raise RuntimeError(f"Failed to delete {name} from S3: {e}")
  ```
- [ ] Update `drop_collection()` to call `_drop_s3_collection()` when using S3 URI
- [ ] Add confirmation prompt for production use (optional)
- [ ] Clear local cache entry

#### Phase 3: Testing (Day 2)
- [ ] Unit tests with moto
- [ ] Test successful deletion
- [ ] Test deleting non-existent collection (should be silent/idempotent)
- [ ] Test permission errors
- [ ] Test concurrent deletion (race conditions)
- [ ] Integration test with real S3 bucket
- [ ] Update documentation

#### Safety Considerations:
- [ ] Add `--force` flag for CLI if we add one
- [ ] Log deletion operations
- [ ] Consider versioning in S3 bucket for recovery

#### Success Criteria:
- [ ] `db.drop_collection()` works with S3 URIs
- [ ] Idempotent (safe to call multiple times)
- [ ] Proper error messages
- [ ] All tests pass

---

### 3. Complete Query Builder Implementation
**File**: `src/peachbase/collection.py:243`
**Status**: 🔴 Not Implemented
**Priority**: Low (current implementation works)
**Estimated Effort**: 3-5 days

```python
# TODO: Implement actual search logic in later phases
```

**Context**: The Query class exists but is currently just a wrapper. The comment suggests this was meant to be a more advanced query builder API.

**Implementation Plan**:

#### Phase 1: Requirements & Design (Day 1-2)
- [ ] Define query builder API:
  ```python
  # Fluent interface
  results = (collection
      .query()
      .where("category", "==", "tech")
      .where("year", ">=", 2023)
      .semantic(query_vector)
      .limit(10)
      .execute())

  # Or builder pattern
  query = Query(collection)
  query.filter(category="tech", year={"$gte": 2023})
  query.semantic_search(query_vector)
  results = query.limit(10).execute()
  ```
- [ ] Design advanced features:
  - Query composition (multiple filters, multiple queries)
  - Query optimization (reorder operations for performance)
  - Query caching
  - Pagination support (offset + limit)
  - Aggregations (count, stats)
- [ ] Review similar APIs (LanceDB, Qdrant, Weaviate)
- [ ] Get community feedback on design

#### Phase 2: Core Implementation (Day 3-4)
- [ ] Implement `Query` class methods:
  - `.where()` - Add filter
  - `.semantic()` - Semantic search
  - `.lexical()` - Lexical search
  - `.hybrid()` - Hybrid search
  - `.limit()` - Result limit
  - `.offset()` - Skip results (pagination)
  - `.execute()` - Run query
- [ ] Implement query optimization:
  - Apply filters before search (reduce search space)
  - Combine multiple filters efficiently
  - Cache query plans
- [ ] Maintain backward compatibility with direct `collection.search()` calls

#### Phase 3: Advanced Features (Day 5)
- [ ] Pagination support:
  ```python
  # Page 1
  results = collection.query().limit(10).offset(0).execute()

  # Page 2
  results = collection.query().limit(10).offset(10).execute()
  ```
- [ ] Aggregations:
  ```python
  # Count by category
  counts = collection.query().aggregate("category").count()

  # Stats on scores
  stats = collection.query().aggregate().stats("score")
  ```
- [ ] Query explain (show execution plan):
  ```python
  query.explain()  # Show query plan without executing
  ```

#### Phase 4: Testing & Documentation (Ongoing)
- [ ] Unit tests for each query operation
- [ ] Integration tests for complex queries
- [ ] Performance benchmarks
- [ ] Update documentation
- [ ] Add examples

#### Success Criteria:
- [ ] Fluent query interface working
- [ ] All operations properly composed
- [ ] Performance equal to or better than direct search()
- [ ] Backward compatible with existing API
- [ ] Comprehensive documentation

---

## Short-Term Roadmap (v0.2.0 - Q1 2026)

### Performance Improvements
**Priority**: High
**Target Release**: v0.2.0

- [ ] **Query Result Caching**
  - Cache recent query results
  - LRU eviction policy
  - Configurable cache size
  - Invalidation on collection updates

- [ ] **Batch Operations API**
  - `collection.add_batch()` - Optimized batch inserts
  - `collection.delete_batch()` - Bulk deletes
  - `collection.update_batch()` - Batch updates
  - Progress callbacks for large batches

- [ ] **ARM/NEON SIMD Support**
  - Implement NEON intrinsics for ARM processors
  - Automatic CPU detection (AVX2 vs AVX-512 vs NEON)
  - Performance parity with x86_64 on ARM
  - Target: Apple Silicon, AWS Graviton

### Feature Additions
**Priority**: Medium
**Target Release**: v0.2.0

- [ ] **Additional Distance Metrics**
  - Manhattan distance (L1)
  - Hamming distance (binary vectors)
  - Jaccard similarity
  - Configurable per collection

- [ ] **Multi-Vector Documents**
  - Store multiple vectors per document
  - Search across all vectors
  - Use case: multi-modal embeddings (text + image)
  - Aggregation strategies (max, mean, weighted)

---

## Mid-Term Roadmap (v0.3.0 - Q2 2026)

### Scalability
**Priority**: High for production deployments
**Target Release**: v0.3.0

- [ ] **Approximate Nearest Neighbor (ANN)**
  - HNSW index implementation
  - IVF (Inverted File) index
  - Configurable accuracy vs speed tradeoff
  - Target: 100K-10M vectors
  - Benchmark: < 10ms for 1M vectors

- [ ] **Incremental Indexing**
  - Add documents without full reindex
  - Background index updates
  - Minimal search disruption
  - Use case: Real-time updates

- [ ] **Distributed Collections**
  - Sharding across multiple files
  - Parallel search across shards
  - Automatic shard selection
  - Target: 10M+ vectors

### Developer Experience
**Priority**: Medium
**Target Release**: v0.3.0

- [ ] **Async API Support**
  - async/await search operations
  - Batch async operations
  - Compatible with FastAPI, aiohttp
  - Use case: High-concurrency servers

- [ ] **CLI Tool**
  - `peachbase create` - Create collection
  - `peachbase add` - Add documents from file
  - `peachbase search` - Interactive search
  - `peachbase info` - Collection stats
  - `peachbase migrate` - Upgrade format

---

## Long-Term Roadmap (v1.0.0 - Q3-Q4 2026)

### Production Features
**Priority**: High for enterprise
**Target Release**: v1.0.0

- [ ] **Monitoring & Observability**
  - Built-in metrics (latency, throughput)
  - OpenTelemetry integration
  - Prometheus exporter
  - Health check endpoint

- [ ] **Access Control**
  - API key authentication
  - Role-based access control (RBAC)
  - Collection-level permissions
  - Audit logging

- [ ] **High Availability**
  - Replication across S3 regions
  - Failover support
  - Consistency guarantees
  - Automatic recovery

### Advanced Features
**Priority**: Medium-Low
**Target Release**: v1.0.0+

- [ ] **Graph Search**
  - Document relationships
  - Graph traversal queries
  - Community detection
  - Use case: Knowledge graphs

- [ ] **Time-Series Support**
  - Time-based filtering
  - Time-decay scoring
  - Sliding window queries
  - Use case: Recent news, trends

- [ ] **Multi-Language Support**
  - Bindings for JavaScript/TypeScript
  - Bindings for Rust
  - Bindings for Go
  - Use case: Broader ecosystem

---

## Experimental Ideas (Research Phase)

### Under Consideration
**No timeline - research phase**

- [ ] **Automatic Embedding**
  - Built-in embedding models
  - Automatic text vectorization
  - Model selection API
  - Trade-off: Increases package size

- [ ] **Learned Indices**
  - ML-based index structures
  - Adapt to query patterns
  - Potential for better performance
  - Research: Feasibility on Lambda

- [ ] **Federated Search**
  - Search across multiple databases
  - Distributed query planning
  - Result merging
  - Use case: Multi-region deployments

---

## Contributing

Want to help implement these features? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Priority Features for Contributors:**
1. S3 listing and deletion (well-defined, high impact)
2. Additional distance metrics (good first issue)
3. ARM/NEON SIMD support (challenging, high impact)
4. Query builder API (design feedback welcome)

---

## Feedback

Have ideas for the roadmap?
- Open an [issue](https://github.com/PeachstoneAI/peachbase/issues) with label `feature request`
- Start a [discussion](https://github.com/PeachstoneAI/peachbase/discussions)
- Comment on existing issues

---

**Last Updated**: 2026-01-15
**Version**: 0.1.0

---

Made with 🍑 by PeachstoneAI
