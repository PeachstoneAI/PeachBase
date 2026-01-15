"""
Large-scale Wikipedia RAG Example - 10,000+ Documents

This example demonstrates PeachBase at scale:
1. Crawls Wikipedia categories to collect many articles
2. Chunks them into 10,000+ document pieces
3. Generates embeddings efficiently in batches
4. Performs fast semantic, lexical, and hybrid search
5. Shows performance metrics and multi-core utilization

Requirements:
    pip install peachbase sentence-transformers wikipedia-api tqdm

Usage:
    # Quick start (1-2 minutes, ~500 articles -> ~10K chunks)
    python wikipedia_rag_large.py

    # Custom target
    python wikipedia_rag_large.py --target 20000

    # Resume from saved data
    python wikipedia_rag_large.py --resume
"""

import argparse
import json
import os
import re
import time
from typing import List, Dict, Any, Set
import wikipediaapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def crawl_wikipedia_category(
    category_name: str,
    max_articles: int = 500,
    max_depth: int = 2,
    user_agent: str = 'PeachBase-LargeExample/1.0'
) -> List[str]:
    """Crawl Wikipedia category to collect article titles.

    Args:
        category_name: Starting category name
        max_articles: Maximum number of articles to collect
        max_depth: How deep to traverse subcategories
        user_agent: User agent string

    Returns:
        List of article titles
    """
    print(f"\n📚 Crawling Wikipedia category: {category_name}")
    print(f"   Max articles: {max_articles}, Max depth: {max_depth}")

    wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language='en')

    articles: Set[str] = set()
    visited_categories: Set[str] = set()

    def crawl_category(cat_name: str, depth: int = 0):
        if depth > max_depth or len(articles) >= max_articles:
            return

        if cat_name in visited_categories:
            return
        visited_categories.add(cat_name)

        cat = wiki.page(f"Category:{cat_name}")
        if not cat.exists():
            return

        # Get articles in this category
        for page_title in cat.categorymembers.keys():
            if len(articles) >= max_articles:
                break

            page = wiki.page(page_title)

            # Only add actual articles (not categories or other pages)
            if page.exists() and page.ns == 0:  # ns=0 means article namespace
                articles.add(page.title)
                if len(articles) % 50 == 0:
                    print(f"   Collected {len(articles)} articles...")

        # Recursively crawl subcategories
        if depth < max_depth:
            for member_title in cat.categorymembers.keys():
                if len(articles) >= max_articles:
                    break
                if member_title.startswith("Category:"):
                    subcat_name = member_title.replace("Category:", "")
                    crawl_category(subcat_name, depth + 1)

    crawl_category(category_name)

    article_list = list(articles)[:max_articles]
    print(f"   ✓ Collected {len(article_list)} article titles")

    return article_list


def download_articles_batch(
    titles: List[str],
    max_chars_per_article: int = 20000,
    cache_file: str = "wikipedia_articles_cache.json"
) -> Dict[str, str]:
    """Download Wikipedia articles with progress bar and caching.

    Args:
        titles: List of article titles
        max_chars_per_article: Maximum characters per article
        cache_file: File to cache downloaded articles

    Returns:
        Dict mapping title to content
    """
    print(f"\n📥 Downloading {len(titles)} Wikipedia articles...")

    # Load cache if exists
    articles = {}
    if os.path.exists(cache_file):
        print(f"   Loading cache from {cache_file}...")
        with open(cache_file, 'r') as f:
            articles = json.load(f)
        print(f"   ✓ Loaded {len(articles)} cached articles")

    # Download missing articles
    wiki = wikipediaapi.Wikipedia(user_agent='PeachBase-LargeExample/1.0', language='en')

    new_downloads = 0
    for title in tqdm(titles, desc="   Downloading"):
        if title in articles:
            continue

        page = wiki.page(title)
        if page.exists():
            content = page.text[:max_chars_per_article]
            articles[title] = content
            new_downloads += 1

            # Save cache periodically
            if new_downloads % 50 == 0:
                with open(cache_file, 'w') as f:
                    json.dump(articles, f)

        # Rate limiting - be nice to Wikipedia
        if new_downloads % 10 == 0:
            time.sleep(0.5)

    # Save final cache
    if new_downloads > 0:
        with open(cache_file, 'w') as f:
            json.dump(articles, f)
        print(f"   ✓ Downloaded {new_downloads} new articles")

    return {title: articles[title] for title in titles if title in articles}


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries."""
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
    articles: Dict[str, str],
    target_chunks: int = 10000,
    chunk_size: int = 400
) -> List[Dict[str, Any]]:
    """Chunk articles to reach target number of documents.

    Args:
        articles: Dict of article_title -> content
        target_chunks: Target number of chunks to create
        chunk_size: Size of each chunk

    Returns:
        List of document dicts
    """
    print(f"\n✂️  Chunking articles (target: {target_chunks} chunks)...")

    documents = []
    doc_id = 0

    for title, content in tqdm(articles.items(), desc="   Processing"):
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=50)

        for i, chunk_text in enumerate(chunks):
            # Skip very short chunks
            if len(chunk_text) < 100:
                continue

            documents.append({
                "id": f"doc_{doc_id}",
                "text": chunk_text,
                "metadata": {
                    "source": title,
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

    print(f"   ✓ Created {len(documents)} document chunks")
    print(f"   ✓ From {len(set(d['metadata']['source'] for d in documents))} unique articles")

    return documents


def generate_embeddings_batched(
    documents: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128
) -> List[Dict[str, Any]]:
    """Generate embeddings in batches with progress bar.

    Args:
        documents: List of documents
        model_name: Sentence transformer model name
        batch_size: Batch size for embedding generation

    Returns:
        Documents with embeddings
    """
    print(f"\n🧠 Generating embeddings ({model_name})...")
    print(f"   Documents: {len(documents)}")
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

    print(f"   ✓ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
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
    print(f"   Documents: {len(documents)}")

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

    print(f"   ✓ Added {len(documents)} docs in {add_time:.2f}s ({len(documents)/add_time:.0f} docs/sec)")
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
        qps = 1000 / avg_time  # Queries per second

        print(f"   ✓ Average: {avg_time:.2f}ms per query")
        print(f"   ✓ Throughput: {qps:.1f} QPS")

        results_summary[mode] = {
            "avg_ms": avg_time,
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
        results = collection.search(
            query_text=query if mode in ["lexical", "hybrid"] else None,
            query_vector=query_vector,
            mode=mode,
            limit=3
        )

        # Display results
        for i, result in enumerate(results.to_list(), 1):
            print(f"\n   {i}. {result['metadata']['source']} (score: {result['score']:.4f})")
            print(f"      {result['text'][:150]}...")


def main():
    parser = argparse.ArgumentParser(description='Large-scale Wikipedia RAG with PeachBase')
    parser.add_argument('--target', type=int, default=10000,
                       help='Target number of document chunks (default: 10000)')
    parser.add_argument('--category', type=str, default='Artificial_intelligence',
                       help='Wikipedia category to crawl (default: Artificial_intelligence)')
    parser.add_argument('--max-articles', type=int, default=500,
                       help='Maximum articles to download (default: 500)')
    parser.add_argument('--chunk-size', type=int, default=400,
                       help='Chunk size in characters (default: 400)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from cached articles')
    args = parser.parse_args()

    print("=" * 80)
    print("🍑 PeachBase - Large-Scale Wikipedia RAG Example")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Target chunks: {args.target:,}")
    print(f"  • Category: {args.category}")
    print(f"  • Max articles: {args.max_articles}")
    print(f"  • Chunk size: {args.chunk_size} chars")
    print()

    overall_start = time.time()

    # Step 1: Collect article titles
    if not args.resume:
        titles = crawl_wikipedia_category(
            args.category,
            max_articles=args.max_articles,
            max_depth=2
        )
    else:
        print(f"\n⏩ Resuming from cached articles...")
        titles = []

    # Step 2: Download articles
    articles = download_articles_batch(
        titles,
        cache_file="wikipedia_articles_large_cache.json"
    )

    if not articles:
        print("\n❌ No articles downloaded. Exiting.")
        return

    print(f"\n📊 Downloaded articles statistics:")
    total_chars = sum(len(content) for content in articles.values())
    print(f"   • Articles: {len(articles)}")
    print(f"   • Total characters: {total_chars:,}")
    print(f"   • Average per article: {total_chars // len(articles):,} chars")

    # Step 3: Chunk into documents
    documents = prepare_documents(
        articles,
        target_chunks=args.target,
        chunk_size=args.chunk_size
    )

    # Step 4: Generate embeddings
    documents, model = generate_embeddings_batched(documents)

    # Step 5: Create PeachBase collection
    collection = create_peachbase_collection(documents)

    # Step 6: Benchmark search performance
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
    ]

    perf_results = benchmark_search(collection, model, benchmark_queries)

    # Step 7: Demo search
    demo_search(collection, model)

    # Final summary
    total_time = time.time() - overall_start

    print("\n" + "=" * 80)
    print("✅ COMPLETE - Large-Scale Wikipedia RAG")
    print("=" * 80)

    print(f"\n📊 Final Statistics:")
    print(f"   • Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   • Articles processed: {len(articles)}")
    print(f"   • Document chunks: {len(documents):,}")
    print(f"   • Collection size: {collection.size:,}")
    print(f"   • Vector dimension: {len(documents[0]['vector'])}")

    print(f"\n⚡ Search Performance:")
    for mode, metrics in perf_results.items():
        print(f"   • {mode.capitalize()}: {metrics['avg_ms']:.2f}ms ({metrics['qps']:.1f} QPS)")

    print(f"\n💡 OpenMP Multi-Core:")
    from peachbase import _simd
    omp_info = _simd.get_openmp_info()
    if omp_info['compiled_with_openmp']:
        print(f"   ✓ Enabled ({omp_info['max_threads']} threads)")
        print(f"   ✓ 3-4x faster for large collections")
    else:
        print(f"   ✗ Disabled (single-threaded)")

    print(f"\n📝 Database saved to: ./wikipedia_large_db/")
    print(f"\n💡 Next steps:")
    print(f"   • Connect to an LLM for answer generation")
    print(f"   • Deploy to production")
    print(f"   • Scale to even more documents")
    print(f"   • Add hybrid search tuning")
    print()


if __name__ == "__main__":
    main()
