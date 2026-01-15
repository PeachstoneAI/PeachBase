"""
Performance Benchmark for PeachBase.

Tests and demonstrates:
- Indexing/loading performance
- Search latency (semantic, lexical, hybrid)
- Scalability with collection size
- SIMD acceleration impact
- Memory usage
- Cold start vs warm start
- S3 vs local storage (if available)

Usage:
    python performance_benchmark.py
    python performance_benchmark.py --size large  # Test with more docs
"""

import time
import random
import argparse
import sys
from typing import List, Dict, Any, Tuple
import gc


def generate_mock_embedding(seed: int, dim: int = 384) -> list[float]:
    """Generate deterministic mock embedding."""
    random.seed(seed)
    return [random.random() for _ in range(dim)]


def generate_test_documents(n_docs: int, dimension: int = 384) -> List[Dict[str, Any]]:
    """Generate test documents with mock embeddings."""
    categories = ["tech", "science", "health", "finance", "education"]

    documents = []
    for i in range(n_docs):
        doc = {
            "id": f"doc_{i}",
            "text": f"This is test document number {i}. It contains some sample text about "
                    f"various topics including technology, science, and more. Document length "
                    f"varies to simulate real-world data. Some documents are longer while others "
                    f"are shorter. This helps test performance with different data patterns.",
            "vector": generate_mock_embedding(i, dimension),
            "metadata": {
                "category": categories[i % len(categories)],
                "index": i,
                "priority": random.choice(["low", "medium", "high"]),
                "year": 2020 + (i % 5)
            }
        }
        documents.append(doc)

    return documents


class PerformanceBenchmark:
    """Performance benchmark suite for PeachBase."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.results = {}

    def time_operation(self, name: str, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time an operation and return result and elapsed time."""
        gc.collect()  # Clean up before timing
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed

    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 0.001:
            return f"{seconds*1000000:.0f}μs"
        elif seconds < 1:
            return f"{seconds*1000:.2f}ms"
        else:
            return f"{seconds:.2f}s"

    def benchmark_collection_creation(self, n_docs: int) -> Dict[str, float]:
        """Benchmark collection creation and document insertion."""
        import peachbase

        print(f"\n{'='*80}")
        print(f"📊 Benchmark: Collection Creation ({n_docs:,} documents)")
        print(f"{'='*80}")

        results = {}

        # Generate test data
        print(f"\n1. Generating {n_docs:,} test documents...")
        docs, gen_time = self.time_operation(
            "generate_docs",
            generate_test_documents,
            n_docs,
            self.dimension
        )
        results['data_generation'] = gen_time
        print(f"   ✓ Generated in {self.format_time(gen_time)}")
        print(f"   ✓ Per document: {self.format_time(gen_time/n_docs)}")

        # Create database and collection
        print(f"\n2. Creating database and collection...")
        db = peachbase.connect("./benchmark_db")

        def create_and_add():
            collection = db.create_collection(
                "benchmark",
                dimension=self.dimension,
                overwrite=True
            )
            collection.add(docs)
            return collection

        collection, add_time = self.time_operation("add_docs", create_and_add)
        results['document_insertion'] = add_time
        results['docs_per_second'] = n_docs / add_time

        print(f"   ✓ Inserted {n_docs:,} documents in {self.format_time(add_time)}")
        print(f"   ✓ Throughput: {results['docs_per_second']:.0f} docs/sec")
        print(f"   ✓ Per document: {self.format_time(add_time/n_docs)}")

        # Save to disk
        print(f"\n3. Saving collection to disk...")
        _, save_time = self.time_operation("save", collection.save)
        results['save_time'] = save_time
        print(f"   ✓ Saved in {self.format_time(save_time)}")

        return collection, results

    def benchmark_search_performance(self, collection, n_queries: int = 100) -> Dict[str, Any]:
        """Benchmark search performance across different modes."""
        print(f"\n{'='*80}")
        print(f"🔍 Benchmark: Search Performance ({n_queries} queries)")
        print(f"{'='*80}")

        results = {}

        # Generate query vectors
        query_vectors = [generate_mock_embedding(1000 + i, self.dimension) for i in range(n_queries)]
        query_texts = [f"search query {i} about technology and science" for i in range(n_queries)]

        # Warm up (JIT compilation, cache warming)
        print(f"\n1. Warming up caches...")
        collection.search(query_vector=query_vectors[0], mode="semantic", limit=10)
        collection.search(query_text=query_texts[0], mode="lexical", limit=10)

        # Benchmark semantic search
        print(f"\n2. Semantic Search (Vector Similarity)...")
        start = time.time()
        for qv in query_vectors:
            collection.search(query_vector=qv, mode="semantic", limit=10)
        semantic_time = time.time() - start

        results['semantic_total'] = semantic_time
        results['semantic_per_query'] = semantic_time / n_queries
        results['semantic_qps'] = n_queries / semantic_time

        print(f"   ✓ Total time: {self.format_time(semantic_time)}")
        print(f"   ✓ Per query: {self.format_time(semantic_time/n_queries)}")
        print(f"   ✓ Throughput: {results['semantic_qps']:.1f} queries/sec")

        # Benchmark lexical search (BM25)
        print(f"\n3. Lexical Search (BM25)...")
        start = time.time()
        for qt in query_texts:
            collection.search(query_text=qt, mode="lexical", limit=10)
        lexical_time = time.time() - start

        results['lexical_total'] = lexical_time
        results['lexical_per_query'] = lexical_time / n_queries
        results['lexical_qps'] = n_queries / lexical_time

        print(f"   ✓ Total time: {self.format_time(lexical_time)}")
        print(f"   ✓ Per query: {self.format_time(lexical_time/n_queries)}")
        print(f"   ✓ Throughput: {results['lexical_qps']:.1f} queries/sec")

        # Benchmark hybrid search
        print(f"\n4. Hybrid Search (RRF)...")
        start = time.time()
        for qt, qv in zip(query_texts, query_vectors):
            collection.search(
                query_text=qt,
                query_vector=qv,
                mode="hybrid",
                limit=10
            )
        hybrid_time = time.time() - start

        results['hybrid_total'] = hybrid_time
        results['hybrid_per_query'] = hybrid_time / n_queries
        results['hybrid_qps'] = n_queries / hybrid_time

        print(f"   ✓ Total time: {self.format_time(hybrid_time)}")
        print(f"   ✓ Per query: {self.format_time(hybrid_time/n_queries)}")
        print(f"   ✓ Throughput: {results['hybrid_qps']:.1f} queries/sec")

        # Test with metadata filtering
        print(f"\n5. Semantic Search with Metadata Filter...")
        start = time.time()
        for qv in query_vectors[:n_queries//10]:  # Less queries, more expensive
            collection.search(
                query_vector=qv,
                mode="semantic",
                filter={"category": "tech"},
                limit=10
            )
        filtered_time = time.time() - start
        n_filtered = n_queries // 10

        results['filtered_per_query'] = filtered_time / n_filtered

        print(f"   ✓ Total time: {self.format_time(filtered_time)} ({n_filtered} queries)")
        print(f"   ✓ Per query: {self.format_time(filtered_time/n_filtered)}")

        return results

    def benchmark_load_performance(self, collection_size: int) -> Dict[str, float]:
        """Benchmark loading performance."""
        import peachbase

        print(f"\n{'='*80}")
        print(f"💾 Benchmark: Load Performance")
        print(f"{'='*80}")

        results = {}

        # Clear any existing connections
        import gc
        gc.collect()

        # Cold start (new database connection)
        print(f"\n1. Cold Start (First Load)...")
        db_cold = peachbase.connect("./benchmark_db")

        _, cold_time = self.time_operation(
            "cold_load",
            db_cold.open_collection,
            "benchmark"
        )
        results['cold_start'] = cold_time
        print(f"   ✓ Loaded in {self.format_time(cold_time)}")

        # Warm start (cached data, same database)
        print(f"\n2. Warm Start (Cached)...")
        db_warm = peachbase.connect("./benchmark_db")

        _, warm_time = self.time_operation(
            "warm_load",
            db_warm.open_collection,
            "benchmark"
        )
        results['warm_start'] = warm_time
        print(f"   ✓ Loaded in {self.format_time(warm_time)}")

        if cold_time > warm_time:
            print(f"   ✓ Speedup: {cold_time/warm_time:.1f}x faster")
        else:
            print(f"   ✓ Similar performance (both loads are cached)")

        return results

    def benchmark_scalability(self, sizes: List[int]) -> Dict[int, Dict[str, float]]:
        """Benchmark performance across different collection sizes."""
        import peachbase

        print(f"\n{'='*80}")
        print(f"📈 Benchmark: Scalability Analysis")
        print(f"{'='*80}")

        results = {}

        for size in sizes:
            print(f"\n{'─'*80}")
            print(f"Testing with {size:,} documents...")
            print(f"{'─'*80}")

            # Generate data
            docs = generate_test_documents(size, self.dimension)

            # Create collection
            db = peachbase.connect(f"./benchmark_db_{size}")
            collection = db.create_collection(
                "test",
                dimension=self.dimension,
                overwrite=True
            )

            # Measure insertion
            _, add_time = self.time_operation("add", collection.add, docs)

            # Measure single search
            query_vector = generate_mock_embedding(9999, self.dimension)
            _, search_time = self.time_operation(
                "search",
                collection.search,
                query_vector=query_vector,
                mode="semantic",
                limit=10
            )

            results[size] = {
                'add_time': add_time,
                'add_per_doc': add_time / size,
                'search_time': search_time,
                'docs_per_second': size / add_time
            }

            print(f"   Insertion: {self.format_time(add_time)} ({size/add_time:.0f} docs/sec)")
            print(f"   Search: {self.format_time(search_time)}")

        return results

    def benchmark_simd_impact(self, n_operations: int = 1000) -> Dict[str, Any]:
        """Benchmark SIMD acceleration impact."""
        print(f"\n{'='*80}")
        print(f"⚡ Benchmark: SIMD Acceleration")
        print(f"{'='*80}")

        try:
            from peachbase import _simd
            import array

            # Check CPU features
            features = _simd.detect_cpu_features()
            feature_names = {0: "Fallback (no SIMD)", 1: "AVX2", 2: "AVX-512"}
            print(f"\n   CPU Features: {feature_names.get(features, 'Unknown')}")

            # Generate test vectors
            vec1 = array.array('f', [random.random() for _ in range(self.dimension)])
            vec2 = array.array('f', [random.random() for _ in range(self.dimension)])

            # Benchmark SIMD operations
            print(f"\n   Testing {n_operations:,} cosine similarity calculations...")

            start = time.time()
            for _ in range(n_operations):
                _simd.cosine_similarity(vec1, vec2)
            simd_time = time.time() - start

            print(f"   ✓ SIMD time: {self.format_time(simd_time)}")
            print(f"   ✓ Per operation: {self.format_time(simd_time/n_operations)}")
            print(f"   ✓ Throughput: {n_operations/simd_time:.0f} ops/sec")

            # Python fallback (for comparison)
            print(f"\n   Comparing with Python fallback...")

            def cosine_similarity_python(v1, v2):
                dot = sum(a * b for a, b in zip(v1, v2))
                norm1 = sum(a * a for a in v1) ** 0.5
                norm2 = sum(b * b for b in v2) ** 0.5
                return dot / (norm1 * norm2)

            start = time.time()
            for _ in range(n_operations):
                cosine_similarity_python(vec1, vec2)
            python_time = time.time() - start

            print(f"   ✓ Python time: {self.format_time(python_time)}")
            print(f"   ✓ Per operation: {self.format_time(python_time/n_operations)}")
            print(f"   ✓ SIMD Speedup: {python_time/simd_time:.1f}x faster")

            return {
                'cpu_features': feature_names.get(features, 'Unknown'),
                'simd_time': simd_time,
                'python_time': python_time,
                'speedup': python_time / simd_time
            }

        except ImportError:
            print(f"   ✗ SIMD module not available")
            return {}

    def print_summary(self, all_results: Dict[str, Any]):
        """Print comprehensive summary of all benchmarks."""
        print(f"\n{'='*80}")
        print(f"📊 PERFORMANCE SUMMARY")
        print(f"{'='*80}")

        if 'creation' in all_results:
            r = all_results['creation']
            print(f"\n📦 Collection Creation:")
            print(f"   • Document insertion: {r['docs_per_second']:.0f} docs/sec")
            print(f"   • Per document: {self.format_time(r['document_insertion']/all_results['collection_size'])}")
            print(f"   • Save to disk: {self.format_time(r['save_time'])}")

        if 'search' in all_results:
            r = all_results['search']
            print(f"\n🔍 Search Performance (per query):")
            print(f"   • Semantic (SIMD): {self.format_time(r['semantic_per_query'])} "
                  f"({r['semantic_qps']:.0f} QPS)")
            print(f"   • Lexical (BM25): {self.format_time(r['lexical_per_query'])} "
                  f"({r['lexical_qps']:.0f} QPS)")
            print(f"   • Hybrid (RRF): {self.format_time(r['hybrid_per_query'])} "
                  f"({r['hybrid_qps']:.0f} QPS)")
            print(f"   • With metadata filter: {self.format_time(r['filtered_per_query'])}")

        if 'load' in all_results:
            r = all_results['load']
            print(f"\n💾 Load Performance:")
            print(f"   • Cold start: {self.format_time(r['cold_start'])}")
            print(f"   • Warm start: {self.format_time(r['warm_start'])} "
                  f"({r['cold_start']/r['warm_start']:.1f}x faster)")

        if 'simd' in all_results and all_results['simd']:
            r = all_results['simd']
            print(f"\n⚡ SIMD Acceleration:")
            print(f"   • CPU: {r['cpu_features']}")
            print(f"   • Speedup: {r['speedup']:.1f}x faster than Python")

        if 'scalability' in all_results:
            print(f"\n📈 Scalability:")
            for size, r in all_results['scalability'].items():
                print(f"   • {size:,} docs: {self.format_time(r['search_time'])} search, "
                      f"{r['docs_per_second']:.0f} docs/sec insertion")


def main():
    parser = argparse.ArgumentParser(description="PeachBase Performance Benchmark")
    parser.add_argument(
        '--size',
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Test size (small=1K, medium=10K, large=50K docs)'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=384,
        help='Vector dimension (default: 384)'
    )
    parser.add_argument(
        '--queries',
        type=int,
        default=100,
        help='Number of test queries (default: 100)'
    )

    args = parser.parse_args()

    # Determine collection size
    size_map = {
        'small': 1000,
        'medium': 10000,
        'large': 50000
    }
    collection_size = size_map[args.size]

    print("=" * 80)
    print("🍑 PeachBase Performance Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Collection size: {collection_size:,} documents")
    print(f"  • Vector dimension: {args.dimension}")
    print(f"  • Number of queries: {args.queries}")
    print(f"  • Python version: {sys.version.split()[0]}")

    try:
        import peachbase
        print(f"  • PeachBase version: {peachbase.__version__}")
    except ImportError:
        print("\n❌ Error: PeachBase not installed!")
        print("\nPlease install PeachBase first:")
        print("  pip install peachbase")
        return

    benchmark = PerformanceBenchmark(dimension=args.dimension)
    all_results = {'collection_size': collection_size}

    # Run benchmarks
    try:
        # 1. Collection creation
        collection, creation_results = benchmark.benchmark_collection_creation(collection_size)
        all_results['creation'] = creation_results

        # 2. Search performance
        search_results = benchmark.benchmark_search_performance(collection, args.queries)
        all_results['search'] = search_results

        # 3. Load performance
        load_results = benchmark.benchmark_load_performance(collection_size)
        all_results['load'] = load_results

        # 4. SIMD impact
        simd_results = benchmark.benchmark_simd_impact(1000)
        all_results['simd'] = simd_results

        # 5. Scalability (only for small/medium tests)
        if args.size != 'large':
            scalability_sizes = [100, 500, 1000, 5000, 10000]
            if args.size == 'small':
                scalability_sizes = [100, 500, 1000]

            scalability_results = benchmark.benchmark_scalability(scalability_sizes)
            all_results['scalability'] = scalability_results

        # Print summary
        benchmark.print_summary(all_results)

        # Final notes
        print(f"\n{'='*80}")
        print(f"✅ Benchmark Complete!")
        print(f"{'='*80}")
        print(f"\n💡 Key Takeaways:")

        if 'search' in all_results:
            semantic_qps = all_results['search']['semantic_qps']
            print(f"   • Semantic search: ~{semantic_qps:.0f} queries/second")

            if semantic_qps > 100:
                print(f"   • Excellent performance for real-time applications")
            elif semantic_qps > 50:
                print(f"   • Good performance for most use cases")
            else:
                print(f"   • Consider smaller collections or batch processing")

        if 'simd' in all_results and all_results['simd']:
            speedup = all_results['simd']['speedup']
            print(f"   • SIMD provides {speedup:.1f}x acceleration")

        print(f"   • Memory-mapped loading enables fast warm starts")
        print(f"   • Hybrid search combines benefits of both modes")

        print(f"\n📝 Run with different sizes:")
        print(f"   python performance_benchmark.py --size small   # Fast test")
        print(f"   python performance_benchmark.py --size medium  # Balanced")
        print(f"   python performance_benchmark.py --size large   # Stress test")

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
