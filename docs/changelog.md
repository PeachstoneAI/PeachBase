# Changelog

All notable changes to PeachBase will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.2] - 2026-01-15

### 🎉 Initial Release

First public release of PeachBase - a lightweight vector database optimized for AWS Lambda.

### Added

#### Core Features
- **Three Search Modes**: Semantic (vector), Lexical (BM25), and Hybrid (RRF fusion)
- **SIMD Acceleration**: AVX2/AVX-512 optimized vector operations (100-400x faster than pure Python)
- **OpenMP Support**: Multi-threaded search for 3-4x additional speedup
- **Memory-Mapped Storage**: Fast loading with custom `.pdb` binary format
- **Metadata Filtering**: MongoDB-like query syntax for filtering results

#### API
- `peachbase.connect()` - Connect to local or S3 database
- `Database.create_collection()` - Create vector collection
- `Database.open_collection()` - Open existing collection
- `Database.list_collections()` - List all collections
- `Database.drop_collection()` - Delete collection
- `Collection.add()` - Add documents with vectors
- `Collection.get()` - Get document by ID
- `Collection.delete()` - Delete document by ID
- `Collection.search()` - Search with semantic/lexical/hybrid modes
- `Collection.save()` - Persist to disk
- `Collection.load()` - Load from disk (classmethod)

#### Search Capabilities
- **Semantic Search**: Cosine similarity, L2 distance, dot product
- **Lexical Search**: BM25 algorithm with configurable parameters
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) combining semantic and lexical
- **Metadata Filters**: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`

#### Performance
- Cold start: < 100ms for 1K documents
- Search latency: 2-50ms depending on collection size
- SIMD acceleration: 100-400x faster than pure Python
- Memory footprint: Minimal (only boto3 dependency)

#### Documentation
- Comprehensive getting started guide
- API reference documentation
- Deployment guides (Lambda, Docker, servers)
- Performance optimization guide
- Example implementations (basic usage, Wikipedia RAG, large-scale)
- Architecture documentation

#### Examples
- `examples/basic_usage.py` - Basic CRUD and search operations
- `examples/hybrid_search.py` - Comparing search modes
- `examples/wikipedia_rag.py` - End-to-end RAG pipeline
- `examples/wikipedia_rag_large.py` - Large-scale indexing (1000+ chunks)
- `examples/wikipedia_rag_large_hf.py` - Using HuggingFace models
- `examples/lambda_deployment.py` - AWS Lambda function example
- `examples/performance_benchmark.py` - Performance testing

#### Testing
- 114 unit tests with >85% coverage
- Tests for all search modes
- SIMD extension tests
- Storage persistence tests
- Filter and metadata tests

### Technical Details

#### C Extensions
- `_simd` module: AVX2/AVX-512 vector operations
  - `cosine_similarity()` - Cosine similarity between vectors
  - `l2_distance()` - L2 (Euclidean) distance
  - `dot_product()` - Dot product similarity
  - `batch_cosine_similarity()` - Batch operations for search
  - `batch_l2_distance()` - Batch L2 distance
  - `batch_dot_product()` - Batch dot product
  - `detect_cpu_features()` - Runtime CPU feature detection
  - `get_openmp_info()` - OpenMP configuration info

- `_bm25` module: Optimized BM25 scoring
  - Inverted index construction
  - IDF calculation
  - BM25 scoring with configurable k1 and b parameters

#### Binary Format
- Custom `.pdb` file format
- Memory-mapped vector data (SIMD-aligned)
- Efficient text and metadata storage
- Built-in BM25 index
- Fast loading without deserialization

#### Build System
- `pyproject.toml` - Modern PEP 517/518 packaging
- `setup.py` - C extension compilation with platform detection
- Environment variable `PEACHBASE_DISABLE_OPENMP` for Lambda builds
- Automatic CPU feature detection at runtime

### Known Limitations

- **Collection Size**: Optimized for <100K vectors (brute-force search)
- **Dimensions**: Tested with 384-1536 dimensional vectors
- **S3 Features**: S3 listing and deletion not yet implemented (local filesystem only)
- **Query Builder**: Advanced query API not yet implemented
- **Platform**: Best performance on x86_64 with AVX2/AVX-512

### Dependencies

#### Runtime
- `boto3 >= 1.34.0` - AWS S3 integration

#### Development
- `pytest >= 7.0` - Testing framework
- `pytest-cov >= 4.0` - Code coverage
- `ruff >= 0.3.0` - Linting and formatting
- `mypy >= 1.8` - Type checking

### Compatibility

- **Python**: 3.11, 3.12, 3.13
- **Platforms**: Linux (x86_64), macOS (x86_64, arm64), Windows (x86_64)
- **AWS Lambda**: Python 3.11/3.12 runtime
- **CPU**: AVX2 or AVX-512 recommended (falls back to scalar operations)

---

## [Unreleased]

### Planned Features

See [roadmap](roadmap.md) for upcoming features:
- Approximate Nearest Neighbor (HNSW/IVF) for 100K+ vectors
- Complete S3 integration (listing, deletion)
- Query builder API
- Batch operations API
- Additional distance metrics (Manhattan, Hamming)
- Query result caching
- ARM/NEON SIMD support
- Multi-vector documents
- Async API support

---

## Release Notes

### Version 0.1.0 Highlights

PeachBase is production-ready for:
- **AWS Lambda deployments** with fast cold starts
- **RAG (Retrieval-Augmented Generation)** systems
- **Semantic search** applications
- **Hybrid search** combining keywords and meaning
- **Document collections** up to 100K vectors

**Not recommended for:**
- Collections > 100K vectors (use approximate methods)
- Real-time updates with millions of queries/second
- Distributed multi-node deployments (single-node only)

### Migration Guide

This is the initial release, no migrations needed.

### Breaking Changes

None (initial release).

---

## How to Upgrade

```bash
# Upgrade from PyPI (when published)
pip install --upgrade peachbase

# Or from source
git pull
python -m build
pip install dist/peachbase-*.whl --force-reinstall
```

---

## Contributors

- Philipp - Initial development

---

## Links

- **GitHub**: https://github.com/PeachstoneAI/PeachBase
- **Issues**: https://github.com/PeachstoneAI/PeachBase/issues
- **Documentation**: [docs/README.md](docs/README.md)

---

Made with 🍑 by PeachstoneAI
