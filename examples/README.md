# PeachBase Examples

This directory contains examples demonstrating PeachBase's capabilities.

## Quick Test

Verify your PeachBase installation with a simple test:

```bash
python quick_test.py
```

This tests:
- ✅ Package import
- ✅ C extension loading (SIMD and BM25)
- ✅ Database and collection creation
- ✅ All three search modes (semantic, lexical, hybrid)
- ✅ Metadata filtering
- ✅ Save and load functionality

## Basic Usage

Learn the fundamentals of PeachBase:

```bash
python basic_usage.py
```

Shows:
- Creating a database and collection
- Adding documents with embeddings
- Semantic search with cosine similarity
- Lexical search with BM25
- Metadata filtering
- Saving and loading collections

## Hybrid Search

Compare semantic, lexical, and hybrid search:

```bash
python hybrid_search.py
```

Demonstrates:
- How different search modes rank results
- Balancing lexical and semantic with alpha parameter
- When to use each mode
- Understanding RRF (Reciprocal Rank Fusion)

## Wikipedia RAG (End-to-End)

Complete RAG pipeline with Wikipedia articles:

### Prerequisites

```bash
pip install sentence-transformers wikipedia-api
```

### Run

```bash
python wikipedia_rag.py
```

This example:
1. **Downloads** 5 Wikipedia articles (AI, ML, Python, NLP, Vector DB)
2. **Chunks** text into manageable pieces (~500 chars with overlap)
3. **Generates embeddings** using sentence-transformers (384-dim)
4. **Stores** in PeachBase collection
5. **Demonstrates** all search modes with real data
6. **Simulates Q&A** by retrieving relevant context

Expected output:
- ~100-150 document chunks from 5 articles
- Real embeddings (not mocks) using MiniLM model
- Semantic, lexical, and hybrid search comparisons
- Q&A demonstrations with context retrieval

**Note**: First run will download the embedding model (~80MB), subsequent runs are faster.

### What You'll Learn

- **Text Chunking**: Splitting long documents with overlap
- **Real Embeddings**: Using sentence-transformers
- **RAG Pipeline**: Complete retrieval flow
- **Search Comparison**: When each mode excels
- **Context Retrieval**: Building Q&A systems

## Wikipedia RAG Large-Scale (10,000+ Documents) ⭐

End-to-end RAG example at scale with 10,000+ document chunks.

**Two versions available**:

### Version 1: HF Datasets (Recommended) 🚀

Uses Hugging Face `datasets` library for fast Wikipedia access:

#### Prerequisites

```bash
pip install sentence-transformers datasets tqdm
```

#### Run

```bash
# Default: 10K chunks (completes in ~2 minutes)
python wikipedia_rag_large_hf.py

# Scale to 50K chunks
python wikipedia_rag_large_hf.py --target 50000 --articles 5000

# Try Simple Wikipedia (easier text)
python wikipedia_rag_large_hf.py --language 20220301.simple --target 5000

# Use larger embedding model
python wikipedia_rag_large_hf.py --model all-mpnet-base-v2
```

**Advantages**:
- ✅ **Much faster**: No API rate limiting
- ✅ **Instant access**: Streams from full Wikipedia dump
- ✅ **Scalable**: Can easily reach 100K+ documents
- ✅ **Reliable**: No network issues or caching complexity
- ✅ **Multiple languages**: Simple English, German, French, etc.

**Time**: ~2 minutes for 10K chunks (first run downloads dataset)

### Version 2: Wikipedia API

Uses Wikipedia API with category crawling:

#### Prerequisites

```bash
pip install sentence-transformers wikipedia-api tqdm
```

#### Run

```bash
# Default: ~10K chunks from AI-related articles
python wikipedia_rag_large.py

# Custom target (e.g., 20K chunks)
python wikipedia_rag_large.py --target 20000

# Custom category
python wikipedia_rag_large.py --category "Machine_learning" --target 15000

# Resume from cached downloads
python wikipedia_rag_large.py --resume
```

**Advantages**:
- ✅ **Category-based**: Automatically crawls related articles
- ✅ **Fresh content**: Uses latest Wikipedia
- ✅ **Cached**: Downloads saved locally

**Time**: 2-5 minutes (depending on cache and target size)

### What Both Versions Do

1. **Loads Wikipedia articles**: Either from HF datasets (streaming) or Wikipedia API (crawling)
2. **Chunks efficiently**: Creates 10,000+ manageable chunks (~400 chars each with 50-char overlap)
3. **Generates embeddings**: Batch processing with sentence-transformers and progress bars
4. **Builds collection**: Adds all documents to PeachBase with multi-core acceleration
5. **Benchmarks search**: Tests performance on 15 queries across all modes (semantic, lexical, hybrid)
6. **Demonstrates**: Shows real search results with relevance scores and source articles

### Output Statistics

```
📊 Final Statistics:
   • Articles processed: 487
   • Document chunks: 10,127
   • Vector dimension: 384
   • Total time: 3.2 minutes

⚡ Search Performance:
   • Semantic: 3.45ms (290 QPS)
   • Lexical: 12.18ms (82 QPS)
   • Hybrid: 16.34ms (61 QPS)

💡 OpenMP Multi-Core:
   ✓ Enabled (16 threads)
   ✓ 3-4x faster for large collections
```

### Features

- **Smart category crawling**: Discovers related articles automatically
- **Caching**: Downloaded articles cached locally - instant re-runs
- **Progress bars**: Real-time progress with `tqdm`
- **Batch embeddings**: Efficient GPU/CPU utilization
- **Performance metrics**: Detailed timing and throughput statistics
- **Multi-core**: Utilizes all CPU cores with OpenMP
- **Resumable**: Can resume from cached data with `--resume`

### Options

```bash
--target N          # Target number of chunks (default: 10000)
--category NAME     # Wikipedia category to crawl (default: Artificial_intelligence)
--max-articles N    # Max articles to download (default: 500)
--chunk-size N      # Characters per chunk (default: 400)
--resume            # Resume from cached articles
```

### Use Cases

Perfect for:
- **Testing at scale**: See how PeachBase performs with real data
- **RAG prototyping**: Build production-like RAG systems
- **Benchmarking**: Compare search strategies on large collections
- **Domain expertise**: Build specialized knowledge bases from Wikipedia categories

### Performance Highlights

With 10K+ documents, you'll see:
- **Multi-core acceleration**: OpenMP utilizing all CPU cores
- **Fast insertion**: ~2M documents/sec
- **Quick searches**: 3-16ms depending on mode
- **High throughput**: 60-290 QPS
- **Memory efficient**: Pre-flattened vectors, optimized storage

**This example showcases PeachBase's production-ready performance!**

## Performance Benchmark

Comprehensive performance testing and benchmarking:

```bash
python performance_benchmark.py --size small   # 1K docs, quick test
python performance_benchmark.py --size medium  # 10K docs, balanced
python performance_benchmark.py --size large   # 50K docs, stress test
```

Tests and measures:
- **Document insertion** throughput (docs/sec)
- **Search latency** for all three modes
- **SIMD acceleration** impact (300x+ speedup!)
- **Scalability** across different collection sizes
- **Cold start vs warm start** performance
- **Memory usage** patterns

Sample results (1K docs, 384-dim vectors):
```
📦 Collection Creation:
   • Document insertion: ~2M docs/sec
   • Save to disk: ~100ms

🔍 Search Performance (per query):
   • Semantic (SIMD): 11ms (87 QPS)
   • Lexical (BM25): 35ms (29 QPS)
   • Hybrid (RRF): 47ms (21 QPS)

⚡ SIMD Acceleration:
   • CPU: AVX2
   • Speedup: 308x faster than Python
```

**Options:**
- `--size small/medium/large`: Test with different collection sizes
- `--dimension N`: Test with different vector dimensions
- `--queries N`: Number of test queries

## AWS Lambda Deployment

Learn how to deploy PeachBase to AWS Lambda:

```bash
python lambda_deployment.py
```

Shows:
- Lambda function structure
- S3-backed collections
- Request/response handling
- Local testing of Lambda handler

For full Lambda deployment guide, see the main [README.md](../README.md#aws-lambda-deployment).

## Performance Tips

### For Development
- Use `quick_test.py` for rapid testing
- Mock embeddings are fast for prototyping
- Start with small collections (<1000 docs)

### For Production
- Use real embedding models (sentence-transformers)
- Chunk size 300-800 chars works well
- Overlap of 50-100 chars for context continuity
- Hybrid search (alpha=0.5) is usually best

### For Lambda
- Pre-build embeddings offline
- Store collections in S3
- Use memory caching in /tmp
- 512-1024 MB memory is sufficient
- Keep collections < 50K documents for fast cold starts

## Troubleshooting

### Import Error

```
ImportError: No module named 'peachbase'
```

**Solution**: Install PeachBase first:
```bash
pip install peachbase
# or from source:
pip install -e ..
```

### C Extension Error

```
ImportError: cannot import name '_simd'
```

**Solution**: C extensions not compiled. Rebuild:
```bash
cd ..
python -m build
pip install dist/peachbase-*.whl --force-reinstall
```

### Wikipedia Download Error

```
Error downloading Wikipedia articles
```

**Solution**: Check internet connection and install dependencies:
```bash
pip install wikipedia-api
```

### Embedding Model Error

```
OSError: Can't load tokenizer for 'all-MiniLM-L6-v2'
```

**Solution**: Install sentence-transformers and ensure internet access:
```bash
pip install sentence-transformers
```

First run downloads the model (~80MB).

## Next Steps

1. **Experiment** with the examples
2. **Modify** chunk sizes and search parameters
3. **Try** your own documents
4. **Deploy** to Lambda
5. **Integrate** with LLMs for full RAG

For more information, see the main [README.md](../README.md).
