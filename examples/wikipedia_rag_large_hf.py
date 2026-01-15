"""
Large-scale Wikipedia RAG using Hugging Face Datasets - 10,000+ Documents

This example uses the `datasets` library for fast, efficient Wikipedia access:
- Instant access to full Wikipedia dump
- No API rate limiting
- Can easily scale to 100K+ documents
- Clean, structured data

Requirements:
    pip install peachbase sentence-transformers datasets tqdm

Usage:
    # Quick start (10K chunks in ~2 minutes)
    python wikipedia_rag_large_hf.py

    # Custom size
    python wikipedia_rag_large_hf.py --target 50000

    # Different language
    python wikipedia_rag_large_hf.py --language simple --target 5000
"""

import argparse
import re
import time
from typing import List, Dict, Any
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_wikipedia_articles(
    num_articles: int = 1000,
    language: str = "20220301.en",
    min_text_length: int = 1000
) -> List[Dict[str, str]]:
    """Load Wikipedia articles using Hugging Face datasets.

    Args:
        num_articles: Number of articles to load
        language: Wikipedia language/date (e.g., "20220301.en" for English 2022-03-01)
        min_text_length: Minimum text length to include

    Returns:
        List of articles with title and text
    """
    print(f"\n📚 Loading Wikipedia dataset ({language})...")
    print(f"   Target articles: {num_articles}")
    print(f"   This may take a moment on first run (downloads dataset)...")

    # Load Wikipedia dataset
    # Available configs: "20220301.en", "20220301.simple", etc.
    dataset = load_dataset(
        "wikipedia",
        language,
        split="train",
        streaming=True  # Stream to avoid loading entire dataset
    )

    articles = []
    processed = 0

    print(f"   Streaming articles...")
    for article in tqdm(dataset, desc="   Loading", total=num_articles):
        # Filter out short articles
        text = article['text']
        if len(text) < min_text_length:
            continue

        articles.append({
            'title': article['title'],
            'text': text
        })

        processed += 1
        if processed >= num_articles:
            break

    print(f"   ✓ Loaded {len(articles)} articles")

    return articles


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))

            # Keep last sentences for overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def prepare_documents(
    articles: List[Dict[str, str]],
    target_chunks: int = 10000,
    chunk_size: int = 400
) -> List[Dict[str, Any]]:
    """Chunk articles to reach target number of documents.

    Args:
        articles: List of articles with title and text
        target_chunks: Target number of chunks
        chunk_size: Size of each chunk

    Returns:
        List of document dicts
    """
    print(f"\n✂️  Chunking articles (target: {target_chunks:,} chunks)...")

    documents = []
    doc_id = 0

    for article in tqdm(articles, desc="   Processing"):
        chunks = chunk_text(article['text'], chunk_size=chunk_size, overlap=50)

        for i, chunk_text in enumerate(chunks):
            # Skip very short chunks
            if len(chunk_text) < 100:
                continue

            documents.append({
                "id": f"doc_{doc_id}",
                "text": chunk_text,
                "metadata": {
                    "source": article['title'],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_length": len(chunk_text)
                }
            })
            doc_id += 1

            # Stop if we've reached target
            if len(documents) >= target_chunks:
                break

        if len(documents) >= target_chunks:
            break

    print(f"   ✓ Created {len(documents):,} document chunks")
    print(f"   ✓ From {len(set(d['metadata']['source'] for d in documents))} unique articles")

    # Calculate statistics
    avg_chunk_len = sum(d['metadata']['chunk_length'] for d in documents) / len(documents)
    total_chars = sum(d['metadata']['chunk_length'] for d in documents)

    print(f"   ✓ Average chunk length: {avg_chunk_len:.0f} characters")
    print(f"   ✓ Total text: {total_chars:,} characters ({total_chars/1024/1024:.1f} MB)")

    return documents


def generate_embeddings_batched(
    documents: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128
) -> tuple:
    """Generate embeddings in batches with progress bar.

    Args:
        documents: List of documents
        model_name: Sentence transformer model name
        batch_size: Batch size for embedding generation

    Returns:
        Tuple of (documents_with_embeddings, model)
    """
    print(f"\n🧠 Generating embeddings ({model_name})...")
    print(f"   Documents: {len(documents):,}")
    print(f"   Batch size: {batch_size}")

    # Load model
    print(f"   Loading model...")
    model = SentenceTransformer(model_name)
    dimension = model.get_sentence_embedding_dimension()
    print(f"   ✓ Model loaded (dimension: {dimension})")

    # Extract texts
    texts = [doc["text"] for doc in documents]

    # Generate embeddings with progress bar
    print(f"   Generating embeddings...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    elapsed = time.time() - start

    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings):
        doc["vector"] = embedding.tolist()

    print(f"   ✓ Generated {len(embeddings):,} embeddings in {elapsed:.2f}s")
    print(f"   ✓ Throughput: {len(embeddings)/elapsed:.0f} embeddings/sec")

    return documents, model


def create_peachbase_collection(
    documents: List[Dict[str, Any]],
    db_path: str = "./wikipedia_large_db"
):
    """Create PeachBase collection with timing."""
    import peachbase

    print(f"\n💾 Creating PeachBase collection...")
    print(f"   Database: {db_path}")
    print(f"   Documents: {len(documents):,}")

    # Connect
    db = peachbase.connect(db_path)

    # Get dimension
    dimension = len(documents[0]["vector"])

    # Create collection
    start = time.time()
    collection = db.create_collection(
        name="wikipedia_large",
        dimension=dimension,
        overwrite=True
    )

    # Add documents
    print(f"   Adding documents...")
    collection.add(documents)
    add_time = time.time() - start

    # Save to disk
    print(f"   Saving to disk...")
    start = time.time()
    collection.save()
    save_time = time.time() - start

    print(f"   ✓ Added {len(documents):,} docs in {add_time:.2f}s ({len(documents)/add_time:.0f} docs/sec)")
    print(f"   ✓ Saved to disk in {save_time:.2f}s")

    return collection


def benchmark_search(collection, model: SentenceTransformer, queries: List[str]):
    """Benchmark search performance with multiple queries."""
    print(f"\n⚡ Benchmarking Search Performance")
    print("=" * 80)

    results_summary = {}

    for mode in ["semantic", "lexical", "hybrid"]:
        print(f"\n🔍 {mode.capitalize()} Search:")

        times = []
        for query in tqdm(queries, desc=f"   Testing"):
            # Generate query vector if needed
            if mode in ["semantic", "hybrid"]:
                query_vector = model.encode(query).tolist()
            else:
                query_vector = None

            # Measure search time
            start = time.time()
            results = collection.search(
                query_text=query if mode in ["lexical", "hybrid"] else None,
                query_vector=query_vector,
                mode=mode,
                limit=10
            )
            elapsed = (time.time() - start) * 1000  # Convert to ms
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        qps = 1000 / avg_time  # Queries per second

        print(f"   ✓ Average: {avg_time:.2f}ms per query")
        print(f"   ✓ Range: {min_time:.2f}ms - {max_time:.2f}ms")
        print(f"   ✓ Throughput: {qps:.1f} QPS")

        results_summary[mode] = {
            "avg_ms": avg_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "qps": qps
        }

    return results_summary


def demo_search(collection, model: SentenceTransformer):
    """Demonstrate search capabilities."""
    print(f"\n🎯 Search Demonstration")
    print("=" * 80)

    # Example queries
    queries = [
        ("What is machine learning?", "semantic"),
        ("neural network architecture", "lexical"),
        ("How do transformers work in AI?", "hybrid"),
    ]

    for query, mode in queries:
        print(f"\n🔍 Query: '{query}'")
        print(f"   Mode: {mode}")
        print("   " + "-" * 70)

        # Generate query vector if needed
        if mode in ["semantic", "hybrid"]:
            query_vector = model.encode(query).tolist()
        else:
            query_vector = None

        # Search
        start = time.time()
        results = collection.search(
            query_text=query if mode in ["lexical", "hybrid"] else None,
            query_vector=query_vector,
            mode=mode,
            limit=3
        )
        search_time = (time.time() - start) * 1000

        # Display results
        for i, result in enumerate(results.to_list(), 1):
            print(f"\n   {i}. {result['metadata']['source']} (score: {result['score']:.4f})")
            print(f"      {result['text'][:200]}...")

        print(f"\n   Search time: {search_time:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description='Large-scale Wikipedia RAG with HF Datasets')
    parser.add_argument('--target', type=int, default=10000,
                       help='Target number of document chunks (default: 10000)')
    parser.add_argument('--articles', type=int, default=1000,
                       help='Number of articles to load (default: 1000)')
    parser.add_argument('--chunk-size', type=int, default=400,
                       help='Chunk size in characters (default: 400)')
    parser.add_argument('--language', type=str, default='20220301.en',
                       help='Wikipedia language/date (default: 20220301.en)')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence transformer model (default: all-MiniLM-L6-v2)')
    args = parser.parse_args()

    print("=" * 80)
    print("🍑 PeachBase - Large-Scale Wikipedia RAG (HF Datasets)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Target chunks: {args.target:,}")
    print(f"  • Wikipedia: {args.language}")
    print(f"  • Articles to load: {args.articles:,}")
    print(f"  • Chunk size: {args.chunk_size} chars")
    print(f"  • Embedding model: {args.model}")
    print()

    overall_start = time.time()

    # Step 1: Load Wikipedia articles
    articles = load_wikipedia_articles(
        num_articles=args.articles,
        language=args.language
    )

    if not articles:
        print("\n❌ No articles loaded. Exiting.")
        return

    # Step 2: Chunk into documents
    documents = prepare_documents(
        articles,
        target_chunks=args.target,
        chunk_size=args.chunk_size
    )

    # Step 3: Generate embeddings
    documents, model = generate_embeddings_batched(
        documents,
        model_name=args.model
    )

    # Step 4: Create PeachBase collection
    collection = create_peachbase_collection(documents)

    # Step 5: Benchmark search performance
    benchmark_queries = [
        "What is deep learning?",
        "How do neural networks work?",
        "Explain natural language processing",
        "What is computer vision?",
        "How does reinforcement learning work?",
        "What are transformers in AI?",
        "Explain gradient descent",
        "What is supervised learning?",
        "How do convolutional neural networks work?",
        "What is the attention mechanism?",
        "Explain backpropagation",
        "What is overfitting in machine learning?",
        "How do recurrent neural networks work?",
        "What is transfer learning?",
        "Explain ensemble methods",
    ]

    perf_results = benchmark_search(collection, model, benchmark_queries)

    # Step 6: Demo search
    demo_search(collection, model)

    # Final summary
    total_time = time.time() - overall_start

    print("\n" + "=" * 80)
    print("✅ COMPLETE - Large-Scale Wikipedia RAG")
    print("=" * 80)

    print(f"\n📊 Final Statistics:")
    print(f"   • Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   • Articles loaded: {len(articles):,}")
    print(f"   • Document chunks: {len(documents):,}")
    print(f"   • Collection size: {collection.size:,}")
    print(f"   • Vector dimension: {len(documents[0]['vector'])}")

    print(f"\n⚡ Search Performance:")
    for mode, metrics in perf_results.items():
        print(f"   • {mode.capitalize():8s}: {metrics['avg_ms']:6.2f}ms ({metrics['qps']:6.1f} QPS)")

    print(f"\n💡 OpenMP Multi-Core:")
    from peachbase import _simd
    omp_info = _simd.get_openmp_info()
    if omp_info['compiled_with_openmp']:
        print(f"   ✓ Enabled ({omp_info['max_threads']} threads)")
        print(f"   ✓ 3-4x faster for large collections")
    else:
        print(f"   ✗ Disabled (single-threaded)")

    print(f"\n📝 Database saved to: ./wikipedia_large_db/")
    print(f"\n💡 Why HF Datasets is Great:")
    print(f"   ✓ No API rate limiting")
    print(f"   ✓ Instant access to millions of articles")
    print(f"   ✓ Streaming for memory efficiency")
    print(f"   ✓ Clean, structured data")
    print(f"   ✓ Multiple languages available")

    print(f"\n🚀 Next steps:")
    print(f"   • Scale to 50K+ documents (--target 50000)")
    print(f"   • Try simple Wikipedia (--language 20220301.simple)")
    print(f"   • Use larger models (--model all-mpnet-base-v2)")
    print(f"   • Connect to an LLM for answer generation")
    print(f"   • Deploy to production")
    print()


if __name__ == "__main__":
    main()
