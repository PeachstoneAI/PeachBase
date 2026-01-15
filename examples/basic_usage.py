"""
Basic usage example for PeachBase.

Shows how to create a collection, add documents, and perform semantic search.
"""

import peachbase

# Example embeddings (in practice, you would use a model like sentence-transformers)
# These are random 384-dimensional vectors for demonstration
def generate_mock_embedding(text: str) -> list[float]:
    """Generate a mock embedding. In practice, use a real embedding model."""
    import random
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(384)]


def main():
    # Connect to local database
    db = peachbase.connect("./my_database")
    print(f"Connected to database: {db}")

    # Create a collection
    collection = db.create_collection(
        name="articles",
        dimension=384,
        overwrite=True  # Overwrite if exists
    )
    print(f"Created collection: {collection}")

    # Add documents with embeddings
    documents = [
        {
            "id": "doc1",
            "text": "Machine learning is a subset of artificial intelligence",
            "vector": generate_mock_embedding("Machine learning is a subset of artificial intelligence"),
            "metadata": {"category": "tech", "year": 2023}
        },
        {
            "id": "doc2",
            "text": "Python is a popular programming language for data science",
            "vector": generate_mock_embedding("Python is a popular programming language for data science"),
            "metadata": {"category": "tech", "year": 2023}
        },
        {
            "id": "doc3",
            "text": "Climate change is a pressing global issue",
            "vector": generate_mock_embedding("Climate change is a pressing global issue"),
            "metadata": {"category": "environment", "year": 2024}
        },
        {
            "id": "doc4",
            "text": "Deep learning models have revolutionized computer vision",
            "vector": generate_mock_embedding("Deep learning models have revolutionized computer vision"),
            "metadata": {"category": "tech", "year": 2024}
        },
    ]

    collection.add(documents)
    print(f"Added {len(documents)} documents to collection")

    # Perform semantic search
    query_text = "artificial intelligence and deep learning"
    query_vector = generate_mock_embedding(query_text)

    print(f"\n--- Semantic Search ---")
    print(f"Query: {query_text}")

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        metric="cosine",
        limit=3
    )

    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results.to_list(), 1):
        print(f"\n{i}. Document ID: {result['id']}")
        print(f"   Text: {result['text']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Metadata: {result['metadata']}")

    # Search with metadata filter
    print(f"\n--- Semantic Search with Filter (category='tech') ---")

    filtered_results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        filter={"category": "tech"},
        limit=3
    )

    print(f"\nTop {len(filtered_results)} filtered results:")
    for i, result in enumerate(filtered_results.to_list(), 1):
        print(f"\n{i}. Document ID: {result['id']}")
        print(f"   Text: {result['text']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Category: {result['metadata']['category']}")

    # Lexical (BM25) search
    print(f"\n--- Lexical Search (BM25) ---")
    print(f"Query: {query_text}")

    lexical_results = collection.search(
        query_text=query_text,
        mode="lexical",
        limit=3
    )

    print(f"\nTop {len(lexical_results)} results:")
    for i, result in enumerate(lexical_results.to_list(), 1):
        print(f"\n{i}. Document ID: {result['id']}")
        print(f"   Text: {result['text']}")
        print(f"   BM25 Score: {result['score']:.4f}")

    # Save collection
    print(f"\n--- Saving Collection ---")
    collection.save()
    print(f"Collection saved to: {db.get_collection_path('articles')}")

    # Load collection
    print(f"\n--- Loading Collection ---")
    loaded_collection = db.open_collection("articles")
    print(f"Loaded collection: {loaded_collection}")
    print(f"Documents in loaded collection: {loaded_collection.size}")


if __name__ == "__main__":
    main()
