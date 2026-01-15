# PeachBase Documentation

Complete documentation for PeachBase - a lightweight, high-performance vector database optimized for AWS Lambda and production deployments.

---

## 📚 Documentation Index

### 🚀 Getting Started
Perfect for first-time users:

- **[Installation Guide](getting-started/installation.md)** - Install PeachBase in 30 seconds
- **[Quick Start](getting-started/quick-start.md)** - Your first search in 5 minutes
- **[Basic Concepts](getting-started/basic-concepts.md)** - Core concepts and terminology

### 📖 User Guides
In-depth guides for common tasks:

- **[Search Modes Guide](guides/search-modes.md)** - Semantic, lexical, and hybrid search explained
- **[Understanding Scores](guides/scoring.md)** - How scores are calculated in each mode
- **[Building from Source](guides/building.md)** - Compile with/without OpenMP
- **[Performance Optimizations](guides/performance.md)** - Detailed optimization analysis with benchmarks
- **[Deployment Guide](guides/deployment.md)** - Deploy to Lambda, Docker, or servers

### 💡 Examples
Practical examples and tutorials:

- **[Examples Overview](examples/)** - All available examples
- **[Basic Usage](examples/basic-usage.md)** - Simple CRUD operations
- **[Wikipedia RAG](examples/wikipedia-rag.md)** - End-to-end RAG pipeline
- **[Large Scale (10K+ docs)](examples/large-scale.md)** - Production-scale examples

### 📋 Reference
Technical reference documentation:

- **[API Reference](reference/api.md)** - Complete API documentation
- **[Performance Benchmarks](reference/performance.md)** - Detailed performance metrics
- **[Architecture](reference/architecture.md)** - How PeachBase works internally

---

## Quick Links

### Common Tasks

**Install PeachBase**
```bash
pip install dist/peachbase-*.whl
```
→ [Installation Guide](getting-started/installation.md)

**Run First Example**
```bash
python examples/quick_test.py
```
→ [Quick Start](getting-started/quick-start.md)

**Build with OpenMP** (multi-core, faster)
```bash
python -m build
```
→ [Building Guide](guides/building.md)

**Build for Lambda** (minimal dependencies)
```bash
PEACHBASE_DISABLE_OPENMP=1 python -m build
```
→ [Deployment Guide](guides/deployment.md)

**Large-Scale Example** (10K+ documents)
```bash
python examples/wikipedia_rag_large_hf.py
```
→ [Large Scale Guide](examples/large-scale.md)

---

## Documentation by Use Case

### 🎯 I want to...

**...get started quickly**
1. [Installation Guide](getting-started/installation.md)
2. [Quick Start](getting-started/quick-start.md)
3. Run `python examples/quick_test.py`

**...understand search modes**
1. [Search Modes Guide](guides/search-modes.md)
2. [Understanding Scores](guides/scoring.md)
3. Run `python examples/hybrid_search.py`

**...build a RAG system**
1. [Wikipedia RAG Example](examples/wikipedia-rag.md)
2. [Large Scale Guide](examples/large-scale.md)
3. Run `python examples/wikipedia_rag_large_hf.py`

**...optimize performance**
1. [Performance Optimizations](guides/performance.md)
2. [Building with OpenMP](guides/building.md)
3. [Performance Benchmarks](reference/performance.md)

**...deploy to production**
1. [Deployment Guide](guides/deployment.md)
2. [Building Guide](guides/building.md)
3. [API Reference](reference/api.md)

---

## Key Features

✅ **Three Search Modes**: Semantic (vector), Lexical (BM25), Hybrid (RRF fusion)
✅ **SIMD Accelerated**: 362x faster than pure Python (AVX2/AVX-512)
✅ **Multi-Core**: OpenMP support for 3-4x speedup on large collections
✅ **Lightweight**: 43KB package (or +1.2MB with OpenMP)
✅ **No Heavy Dependencies**: No numpy, pandas, or scikit-learn
✅ **Lambda Ready**: Optimized for serverless deployments
✅ **Fast Searches**: 3-16ms latency at 10K scale
✅ **Production Ready**: Handles 100K+ documents efficiently

---

## Getting Help

- **Quick Test**: Run `python examples/quick_test.py` to verify installation
- **Examples**: See `examples/` directory for working code
- **Issues**: Report at [GitHub Issues](https://github.com/PeachstoneAI/peachbase/issues)
- **API Docs**: See [API Reference](reference/api.md)

---

## Version

**Current Version**: 0.1.0
**Python**: 3.11+
**License**: MIT

---

Made with 🍑 by the PeachBase team
