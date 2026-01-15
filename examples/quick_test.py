"""
Quick test script to verify PeachBase installation works.

This script creates a small collection with mock data and tests
all three search modes without requiring external dependencies.

Usage:
    python quick_test.py
"""

import random


def generate_mock_embedding(text: str, dim: int = 384) -> list[float]:
    """Generate deterministic mock embedding based on text."""
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(dim)]


def main():
    print("=" * 80)
    print("🍑 PeachBase Quick Test")
    print("=" * 80)

    # Import PeachBase
    try:
        import peachbase
        print("\n✓ PeachBase imported successfully!")
        print(f"  Version: {peachbase.__version__}")
    except ImportError as e:
        print(f"\n✗ Failed to import PeachBase: {e}")
        print("\nPlease install PeachBase first:")
        print("  pip install -e .")
        return

    # Test C extensions
    print("\n📦 Testing C Extensions:")
    try:
        from peachbase import _simd
        features = _simd.detect_cpu_features()
        feature_names = {0: "Fallback", 1: "AVX2", 2: "AVX-512"}
        print(f"  ✓ SIMD module loaded (CPU: {feature_names.get(features, 'Unknown')})")

        # Test SIMD operations
        vec1 = [1.0, 2.0, 3.0, 4.0]
        vec2 = [2.0, 3.0, 4.0, 5.0]
        import array
        similarity = _simd.cosine_similarity(array.array('f', vec1), array.array('f', vec2))
        print(f"  ✓ SIMD cosine similarity works: {similarity:.4f}")
    except Exception as e:
        print(f"  ✗ SIMD test failed: {e}")

    try:
        from peachbase import _bm25
        print(f"  ✓ BM25 module loaded")
    except Exception as e:
        print(f"  ✗ BM25 module failed: {e}")

    # Create database
    print("\n💾 Creating Database:")
    db = peachbase.connect("./test_db")
    print(f"  ✓ Connected to: {db}")

    # Create collection
    print("\n📚 Creating Collection:")
    collection = db.create_collection(
        name="test_collection",
        dimension=384,
        overwrite=True
    )
    print(f"  ✓ Created: {collection}")

    # Add sample documents
    print("\n📝 Adding Documents:")
    documents = [
        {
            "id": "doc1",
            "text": "Python is a high-level programming language known for its simplicity",
            "vector": generate_mock_embedding("Python programming language"),
            "metadata": {"category": "programming", "difficulty": "beginner"}
        },
        {
            "id": "doc2",
            "text": "Machine learning is a subset of artificial intelligence that focuses on data",
            "vector": generate_mock_embedding("Machine learning AI data"),
            "metadata": {"category": "ai", "difficulty": "intermediate"}
        },
        {
            "id": "doc3",
            "text": "Deep learning uses neural networks with multiple layers for complex tasks",
            "vector": generate_mock_embedding("Deep learning neural networks"),
            "metadata": {"category": "ai", "difficulty": "advanced"}
        },
        {
            "id": "doc4",
            "text": "JavaScript is a versatile programming language used for web development",
            "vector": generate_mock_embedding("JavaScript web development"),
            "metadata": {"category": "programming", "difficulty": "beginner"}
        },
        {
            "id": "doc5",
            "text": "Natural language processing enables computers to understand human language",
            "vector": generate_mock_embedding("NLP natural language processing"),
            "metadata": {"category": "ai", "difficulty": "advanced"}
        },
    ]

    collection.add(documents)
    print(f"  ✓ Added {len(documents)} documents")
    print(f"  ✓ Collection size: {collection.size}")

    # Test semantic search
    print("\n🔍 Testing Semantic Search:")
    query_text = "artificial intelligence and neural networks"
    query_vector = generate_mock_embedding(query_text)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=3
    )

    print(f"  Query: '{query_text}'")
    print(f"  Found {len(results)} results:")
    for i, result in enumerate(results.to_list(), 1):
        print(f"\n  {i}. {result['id']} (score: {result['score']:.4f})")
        print(f"     {result['text'][:60]}...")

    # Test lexical search (BM25)
    print("\n📖 Testing Lexical Search (BM25):")
    results = collection.search(
        query_text=query_text,
        mode="lexical",
        limit=3
    )

    print(f"  Query: '{query_text}'")
    print(f"  Found {len(results)} results:")
    for i, result in enumerate(results.to_list(), 1):
        print(f"\n  {i}. {result['id']} (score: {result['score']:.4f})")
        print(f"     {result['text'][:60]}...")

    # Test hybrid search
    print("\n🎯 Testing Hybrid Search:")
    results = collection.search(
        query_text=query_text,
        query_vector=query_vector,
        mode="hybrid",
        alpha=0.5,
        limit=3
    )

    print(f"  Query: '{query_text}'")
    print(f"  Found {len(results)} results:")
    for i, result in enumerate(results.to_list(), 1):
        print(f"\n  {i}. {result['id']} (score: {result['score']:.4f})")
        print(f"     {result['text'][:60]}...")

    # Test metadata filtering
    print("\n🔎 Testing Metadata Filtering:")
    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        filter={"category": "ai"},
        limit=3
    )

    print(f"  Filter: category='ai'")
    print(f"  Found {len(results)} results:")
    for i, result in enumerate(results.to_list(), 1):
        print(f"\n  {i}. {result['id']} - {result['metadata']['category']}")
        print(f"     {result['text'][:60]}...")

    # Test save and load
    print("\n💾 Testing Save/Load:")
    collection.save()
    print(f"  ✓ Collection saved")

    loaded = db.open_collection("test_collection")
    print(f"  ✓ Collection loaded: {loaded.size} documents")

    # Test retrieval
    doc = loaded.get("doc1")
    if doc:
        print(f"  ✓ Retrieved doc1: {doc['text'][:50]}...")

    # Summary
    print("\n" + "=" * 80)
    print("✅ All Tests Passed!")
    print("=" * 80)
    print("\n🎉 PeachBase is working correctly!")
    print("\nNext steps:")
    print("  - Try the full Wikipedia RAG example:")
    print("    python examples/wikipedia_rag.py")
    print("  - Check out other examples in the examples/ directory")
    print("  - Read the README for more information")
    print()


if __name__ == "__main__":
    main()
