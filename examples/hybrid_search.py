"""
Hybrid search example for PeachBase.

Demonstrates combining lexical (BM25) and semantic (vector) search
using Reciprocal Rank Fusion (RRF).
"""

import peachbase


def generate_mock_embedding(text: str) -> list[float]:
    """Generate a mock embedding. In practice, use a real embedding model."""
    import random
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(384)]


def main():
    # Connect to database
    db = peachbase.connect("./my_database")

    # Create collection
    collection = db.create_collection(
        name="research_papers",
        dimension=384,
        overwrite=True
    )

    # Sample research paper abstracts
    papers = [
        {
            "id": "paper1",
            "text": "This paper introduces a novel deep learning architecture for image classification. "
                    "Our approach achieves state-of-the-art results on ImageNet benchmark.",
            "metadata": {"year": 2023, "field": "computer vision"}
        },
        {
            "id": "paper2",
            "text": "We present a new transformer-based model for natural language understanding. "
                    "The model outperforms BERT on several downstream tasks.",
            "metadata": {"year": 2023, "field": "nlp"}
        },
        {
            "id": "paper3",
            "text": "This work explores reinforcement learning for robotics control. "
                    "We demonstrate improved sample efficiency using our proposed method.",
            "metadata": {"year": 2024, "field": "robotics"}
        },
        {
            "id": "paper4",
            "text": "We investigate deep neural networks for medical image segmentation. "
                    "Our approach shows significant improvements in tumor detection accuracy.",
            "metadata": {"year": 2024, "field": "medical ai"}
        },
        {
            "id": "paper5",
            "text": "This paper proposes a novel attention mechanism for transformer models. "
                    "The mechanism reduces computational complexity while maintaining performance.",
            "metadata": {"year": 2024, "field": "nlp"}
        },
    ]

    # Add embeddings
    for paper in papers:
        paper["vector"] = generate_mock_embedding(paper["text"])

    collection.add(papers)
    print(f"Added {len(papers)} research papers\n")

    # Query
    query_text = "deep learning for computer vision and image analysis"
    query_vector = generate_mock_embedding(query_text)

    print(f"Query: {query_text}\n")
    print("=" * 80)

    # Compare different search modes

    # 1. Semantic-only search
    print("\n1. SEMANTIC SEARCH (vector similarity only)")
    print("-" * 80)
    semantic_results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=3
    )

    for i, result in enumerate(semantic_results.to_list(), 1):
        print(f"{i}. {result['id']}: Score={result['score']:.4f}")
        print(f"   {result['text'][:100]}...")

    # 2. Lexical-only search (BM25)
    print("\n\n2. LEXICAL SEARCH (BM25, keyword matching)")
    print("-" * 80)
    lexical_results = collection.search(
        query_text=query_text,
        mode="lexical",
        limit=3
    )

    for i, result in enumerate(lexical_results.to_list(), 1):
        print(f"{i}. {result['id']}: Score={result['score']:.4f}")
        print(f"   {result['text'][:100]}...")

    # 3. Hybrid search with balanced weighting
    print("\n\n3. HYBRID SEARCH (RRF, balanced: alpha=0.5)")
    print("-" * 80)
    print("Combines both lexical and semantic signals for better results")

    hybrid_results = collection.search(
        query_text=query_text,
        query_vector=query_vector,
        mode="hybrid",
        alpha=0.5,  # 0.5 = equal weight to lexical and semantic
        limit=3
    )

    for i, result in enumerate(hybrid_results.to_list(), 1):
        print(f"{i}. {result['id']}: Score={result['score']:.4f}")
        print(f"   {result['text'][:100]}...")

    # 4. Hybrid search favoring semantic
    print("\n\n4. HYBRID SEARCH (semantic-heavy: alpha=0.3)")
    print("-" * 80)
    print("More weight on semantic similarity (alpha closer to 0)")

    hybrid_semantic = collection.search(
        query_text=query_text,
        query_vector=query_vector,
        mode="hybrid",
        alpha=0.3,  # Favor semantic (vector) search
        limit=3
    )

    for i, result in enumerate(hybrid_semantic.to_list(), 1):
        print(f"{i}. {result['id']}: Score={result['score']:.4f}")
        print(f"   {result['text'][:100]}...")

    # 5. Hybrid search favoring lexical
    print("\n\n5. HYBRID SEARCH (lexical-heavy: alpha=0.7)")
    print("-" * 80)
    print("More weight on keyword matching (alpha closer to 1)")

    hybrid_lexical = collection.search(
        query_text=query_text,
        query_vector=query_vector,
        mode="hybrid",
        alpha=0.7,  # Favor lexical (BM25) search
        limit=3
    )

    for i, result in enumerate(hybrid_lexical.to_list(), 1):
        print(f"{i}. {result['id']}: Score={result['score']:.4f}")
        print(f"   {result['text'][:100]}...")

    print("\n" + "=" * 80)
    print("\nHybrid search combines the strengths of both approaches:")
    print("- Lexical (BM25): Good at exact keyword matching")
    print("- Semantic (vector): Good at understanding meaning and context")
    print("- Hybrid (RRF): Best of both worlds, more robust results")
    print(f"\nAlpha parameter: 0 = semantic only, 1 = lexical only, 0.5 = balanced")


if __name__ == "__main__":
    main()
