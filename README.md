# PeachBase

**Lightweight vector database optimized for AWS Lambda with lexical, semantic, and hybrid search.**

PeachBase is a high-performance, serverless-friendly vector database designed for fast cold starts and minimal dependencies. It combines lexical search (BM25), semantic search (SIMD-accelerated vectors), and hybrid search (Reciprocal Rank Fusion) in a single, easy-to-use package.

## Features

- **🚀 Fast Cold Starts**: Optimized for AWS Lambda with memory-mapped loading and minimal initialization
- **🔍 Three Search Modes**: Lexical (BM25), semantic (vector), and hybrid (RRF)
- **⚡ SIMD Acceleration**: AVX2/AVX-512 optimized vector operations for blazing-fast similarity search
- **📦 Minimal Dependencies**: No numpy, pandas, scikit-learn, or pyarrow - only boto3 for S3
- **☁️ S3 Native**: Efficiently read/write collections from S3 with byte-range requests
- **🎯 Metadata Filtering**: MongoDB-like query syntax for filtering results
- **💾 In-Memory Storage**: Fast access with memory-mapped binary format
- **🐍 Python 3.11+**: Modern Python with type hints

## Installation

### From PyPI

```bash
pip install peachbase
```

### From Source

```bash
git clone https://github.com/PeachstoneAI/peachbase.git
cd peachbase
pip install -e .
```

## Quick Start

```python
import peachbase

# Connect to database (local or S3)
db = peachbase.connect("./my_database")  # Local
# db = peachbase.connect("s3://my-bucket/my_db")  # S3

# Create a collection
collection = db.create_collection("articles", dimension=384)

# Add documents with embeddings
collection.add([
    {
        "id": "doc1",
        "text": "Machine learning is fascinating",
        "vector": [0.1, 0.2, ...],  # Your embeddings (384-dim)
        "metadata": {"category": "tech", "year": 2024}
    }
])

# Semantic search
results = collection.search(
    query_vector=[0.3, 0.1, ...],
    limit=10
)

for result in results.to_list():
    print(f"{result['id']}: {result['text']} (score: {result['score']:.4f})")
```

## Search Modes

### 1. Semantic Search (Vector Similarity)

```python
results = collection.search(
    query_vector=[0.1, 0.2, ...],
    mode="semantic",
    metric="cosine",  # or "l2", "dot"
    limit=10
)
```

### 2. Lexical Search (BM25)

```python
results = collection.search(
    query_text="machine learning",
    mode="lexical",
    limit=10
)
```

### 3. Hybrid Search (Best of Both)

```python
results = collection.search(
    query_text="machine learning",
    query_vector=[0.1, 0.2, ...],
    mode="hybrid",
    alpha=0.5,  # 0=semantic only, 1=lexical only
    limit=10
)
```

## Metadata Filtering

```python
results = collection.search(
    query_vector=[0.1, ...],
    filter={
        "category": "tech",
        "year": {"$gte": 2023, "$lte": 2024},
        "tags": {"$in": ["ai", "ml"]}
    },
    limit=10
)
```

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$and`, `$or`, `$not`

## AWS Lambda Deployment

### 1. Package Your Function

```bash
# Build the package with C extensions
pip install peachbase -t ./package
cd package
zip -r ../lambda_function.zip .
cd ..
zip -g lambda_function.zip lambda_function.py
```

### 2. Lambda Function Example

```python
import json
import peachbase

def lambda_handler(event, context):
    # Connect to S3-backed database
    db = peachbase.connect("s3://my-bucket/peachbase")

    # Open collection
    collection = db.open_collection("articles")

    # Search
    results = collection.search(
        query_vector=event["query_vector"],
        limit=10
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "results": [
                {"id": r["id"], "text": r["text"], "score": r["score"]}
                for r in results.to_list()
            ]
        })
    }
```

### 3. Lambda Configuration

- **Memory**: 1024 MB - 3008 MB recommended
  - Lambda allocates vCPUs proportionally to memory (1,769 MB = 1 full vCPU)
  - More memory enables parallel SIMD operations and faster collection loading
  - For collections > 50K docs, use 2048+ MB for optimal performance
- **Timeout**: 10-30 seconds (first cold start may be slower)
- **Runtime**: Python 3.11, 3.12, 3.13, or 3.14 (latest)
- **Architecture**: x86_64 (required for AVX2/AVX-512 SIMD acceleration)

### 4. Optimization Tips

- **S3 Caching**: Collections loaded from S3 are cached in `/tmp` between invocations. Subsequent warm invocations skip the S3 download entirely, reducing latency from ~800ms to ~100ms.

- **Package Size**: Keep deployment packages under 50 MB for optimal cold start times. Larger packages increase initialization time as Lambda extracts and loads your code.

- **Strip Debug Symbols**: Run `strip *.so` on compiled extensions before packaging. This removes debug information and can reduce `.so` file sizes by 50-80%, directly improving cold start time.

- **Provisioned Concurrency**: For latency-sensitive applications, enable provisioned concurrency. Lambda pre-initializes execution environments, eliminating cold starts entirely and providing consistent sub-100ms response times.

- **ARM vs x86**: While ARM (Graviton) offers better price-performance for general workloads, PeachBase requires x86_64 for AVX2/AVX-512 SIMD instructions. The vector operation speedup outweighs the ARM cost savings.

## Compilation from Source

### Prerequisites

- Python 3.11+
- C compiler (gcc, clang, or MSVC)
- Python development headers

### Linux/macOS

```bash
# Install build dependencies
pip install build wheel

# Compile
python -m build

# Install
pip install dist/peachbase-*.whl
```

### For AWS Lambda (Cross-Compilation)

```bash
# Use Docker to build for Lambda environment
docker run --rm -v $(pwd):/workspace \
    public.ecr.aws/lambda/python:3.11 \
    bash -c "cd /workspace && pip install build && python -m build"
```

## API Reference

### Database

```python
db = peachbase.connect(uri)  # Local path or s3://bucket/path
db.create_collection(name, dimension, overwrite=False)
db.open_collection(name)
db.list_collections()
db.drop_collection(name)
```

### Collection

```python
collection.add(documents)  # Add documents
collection.get(doc_id)  # Get by ID
collection.delete(doc_id)  # Delete by ID
collection.search(...)  # Search (see modes above)
collection.save()  # Persist to disk/S3
Collection.load(name, database)  # Load from disk/S3
```

### Query Results

```python
results.to_list()  # List of dicts
results.to_dict()  # Dict with metadata
len(results)  # Number of results
results[0]  # Access by index
for result in results:  # Iterate
    print(result)
```

## Performance

Benchmarks on AWS Lambda (Python 3.11, 1024 MB, x86_64):

| Operation | Collection Size | Latency |
|-----------|----------------|---------|
| Cold Start | 10K docs | ~1.5s |
| Warm Start | 10K docs | ~50ms |
| Semantic Search | 10K docs | ~30ms |
| Hybrid Search | 10K docs | ~45ms |
| S3 Load (first) | 10K docs | ~800ms |
| S3 Load (cached) | 10K docs | ~100ms |

*Benchmarks with 384-dimensional vectors on c6i.large equivalent Lambda*

## Architecture

```
PeachBase
├── Binary Format (.pdb)
│   ├── Header (256 bytes)
│   ├── Vector Data (SIMD-aligned)
│   ├── Text Data
│   ├── Metadata (JSON)
│   └── BM25 Index
├── C Extensions
│   ├── SIMD Operations (AVX2/AVX-512)
│   └── BM25 Scoring
└── Python Layer
    ├── Database & Collection
    ├── Search Modes
    └── S3 Integration
```

## Examples

See the `examples/` directory for more:

- `basic_usage.py` - Getting started with PeachBase
- `hybrid_search.py` - Comparing search modes
- `lambda_deployment.py` - AWS Lambda function example

## Limitations

- **Collection Size**: Optimized for < 100K vectors (brute-force search)
- **Dimension**: Tested with 384-1536 dimensional vectors
- **Dependencies**: Requires boto3 for S3 (included in Lambda by default)
- **Platform**: Best performance on x86_64 with AVX2 support

## Roadmap

- [ ] Approximate Nearest Neighbor (HNSW/IVF) for large collections
- [ ] Multi-vector documents
- [ ] Batch operations API
- [ ] Additional distance metrics (Manhattan, Hamming)
- [ ] Query result caching
- [ ] ARM/NEON SIMD support

## Contributing

Contributions are welcome! Please see [Contributing Guide](docs/contributing.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [LanceDB](https://github.com/lancedb/lancedb) for API design
- BM25 algorithm implementation follows standard Okapi BM25
- RRF (Reciprocal Rank Fusion) based on research papers and OpenSearch implementation

## Documentation

For complete documentation, see the [`docs/`](docs/) directory:

### 🚀 Getting Started
- **[Installation Guide](docs/getting-started/installation.md)** - Install PeachBase in 30 seconds
- **[Quick Start](docs/getting-started/quick-start.md)** - Your first search in 5 minutes
- **[Basic Concepts](docs/getting-started/basic-concepts.md)** - Core concepts and terminology

### 📖 User Guides
- **[Search Modes Guide](docs/guides/search-modes.md)** - Semantic, lexical, and hybrid search explained
- **[Understanding Scores](docs/guides/scoring.md)** - How scores are calculated in each mode
- **[Building from Source](docs/guides/building.md)** - Compile with/without OpenMP
- **[Performance Optimizations](docs/guides/performance.md)** - Detailed optimization analysis with benchmarks

### 📋 Reference
- **[API Reference](docs/reference/api.md)** - Complete API documentation
- **[Performance Benchmarks](docs/reference/performance.md)** - Detailed performance metrics
- **[Architecture](docs/reference/architecture.md)** - How PeachBase works internally

See **[Full Documentation Index](docs/README.md)** for all available docs.

## Support

- **Issues**: [GitHub Issues](https://github.com/PeachstoneAI/PeachBase/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PeachstoneAI/PeachBase/discussions)

---

Made with 🍑 for serverless vector search
