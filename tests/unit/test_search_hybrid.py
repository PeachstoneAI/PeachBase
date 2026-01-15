"""Unit tests for hybrid search (RRF)."""
import pytest
import peachbase


def test_hybrid_search_basic(temp_db_path, sample_documents, query_vector):
    """Test basic hybrid search."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_text="machine learning",
        query_vector=query_vector,
        mode="hybrid",
        limit=3
    )

    results_list = results.to_list()
    assert len(results_list) <= 3
    assert all("score" in r for r in results_list)


def test_hybrid_search_requires_both_queries(temp_db_path, sample_documents, query_vector):
    """Test that hybrid search requires both query_text and query_vector."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    # Missing query_vector
    with pytest.raises((ValueError, TypeError)):
        collection.search(
            query_text="machine learning",
            mode="hybrid",
            limit=5
        )

    # Missing query_text
    with pytest.raises((ValueError, TypeError)):
        collection.search(
            query_vector=query_vector,
            mode="hybrid",
            limit=5
        )


def test_hybrid_search_alpha_parameter(temp_db_path, sample_documents, query_vector):
    """Test hybrid search with different alpha values."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    # Alpha = 0.0 (pure semantic)
    results_semantic = collection.search(
        query_text="machine learning",
        query_vector=query_vector,
        mode="hybrid",
        alpha=0.0,
        limit=5
    )

    # Alpha = 1.0 (pure lexical)
    results_lexical = collection.search(
        query_text="machine learning",
        query_vector=query_vector,
        mode="hybrid",
        alpha=1.0,
        limit=5
    )

    # Alpha = 0.5 (balanced)
    results_balanced = collection.search(
        query_text="machine learning",
        query_vector=query_vector,
        mode="hybrid",
        alpha=0.5,
        limit=5
    )

    # All should return results
    assert len(results_semantic.to_list()) > 0
    assert len(results_lexical.to_list()) > 0
    assert len(results_balanced.to_list()) > 0


def test_hybrid_search_default_alpha(temp_db_path, sample_documents, query_vector):
    """Test that default alpha is 0.5."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    # Without alpha (should default to 0.5)
    results_default = collection.search(
        query_text="machine learning",
        query_vector=query_vector,
        mode="hybrid",
        limit=5
    )

    # With alpha=0.5 explicitly
    results_explicit = collection.search(
        query_text="machine learning",
        query_vector=query_vector,
        mode="hybrid",
        alpha=0.5,
        limit=5
    )

    # Should produce same results
    default_ids = [r["id"] for r in results_default.to_list()]
    explicit_ids = [r["id"] for r in results_explicit.to_list()]

    assert default_ids == explicit_ids


def test_hybrid_search_combines_semantic_and_lexical(temp_db_path, query_vector):
    """Test that hybrid search combines both search modes."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Add documents where one is good semantically, another lexically
    docs = [
        {
            "id": "semantic_match",
            "text": "Different wording but similar meaning to query",
            "vector": query_vector,  # Perfect semantic match!
            "metadata": {}
        },
        {
            "id": "lexical_match",
            "text": "machine learning machine learning machine learning",
            "vector": [0.5] * 384,  # Poor semantic match
            "metadata": {}
        },
        {
            "id": "unrelated",
            "text": "Completely unrelated document about cooking",
            "vector": [0.9] * 384,
            "metadata": {}
        }
    ]
    collection.add(docs)

    results = collection.search(
        query_text="machine learning",
        query_vector=query_vector,
        mode="hybrid",
        alpha=0.5,
        limit=3
    )

    results_list = results.to_list()
    doc_ids = [r["id"] for r in results_list]

    # Should include both semantic and lexical matches
    assert "semantic_match" in doc_ids
    assert "lexical_match" in doc_ids


def test_hybrid_search_limit(temp_db_path, sample_documents, query_vector):
    """Test that limit parameter is respected."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    for limit in [1, 2, 3, 10]:
        results = collection.search(
            query_text="learning",
            query_vector=query_vector,
            mode="hybrid",
            limit=limit
        )
        results_list = results.to_list()
        assert len(results_list) == min(limit, len(sample_documents))


def test_hybrid_search_results_sorted(temp_db_path, sample_documents, query_vector):
    """Test that results are sorted by RRF score."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_text="learning",
        query_vector=query_vector,
        mode="hybrid",
        limit=5
    )

    results_list = results.to_list()
    scores = [r["score"] for r in results_list]

    # RRF scores should be descending (higher is better)
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_empty_collection(temp_db_path, query_vector):
    """Test hybrid search on empty collection."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    results = collection.search(
        query_text="machine learning",
        query_vector=query_vector,
        mode="hybrid",
        limit=5
    )

    assert len(results.to_list()) == 0


def test_hybrid_search_wrong_dimension(temp_db_path, sample_documents):
    """Test that query vector with wrong dimension causes error."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    wrong_dim_vector = [0.1] * 100

    # Note: Implementation doesn't validate dimensions, causes IndexError
    with pytest.raises(IndexError):
        collection.search(
            query_text="machine learning",
            query_vector=wrong_dim_vector,
            mode="hybrid",
            limit=5
        )


def test_hybrid_search_alpha_no_validation(temp_db_path, sample_documents, query_vector):
    """Test that alpha values outside [0,1] are not validated."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    # Note: Alpha is not validated - values outside [0,1] are allowed
    # Alpha < 0 is allowed
    results = collection.search(
        query_text="learning",
        query_vector=query_vector,
        mode="hybrid",
        alpha=-0.1,
        limit=5
    )
    assert results is not None

    # Alpha > 1 is allowed
    results = collection.search(
        query_text="learning",
        query_vector=query_vector,
        mode="hybrid",
        alpha=1.5,
        limit=5
    )
    assert results is not None


def test_hybrid_search_result_contains_document_data(temp_db_path, sample_documents, query_vector):
    """Test that results contain complete document data."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_text="learning",
        query_vector=query_vector,
        mode="hybrid",
        limit=1
    )

    result = results.to_list()[0]

    # Check all expected fields are present
    assert "id" in result
    assert "text" in result
    assert "vector" in result
    assert "metadata" in result
    assert "score" in result


def test_hybrid_search_with_metric_parameter(temp_db_path, sample_documents, query_vector):
    """Test hybrid search with different semantic metrics."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    for metric in ["cosine", "l2", "dot"]:
        results = collection.search(
            query_text="learning",
            query_vector=query_vector,
            mode="hybrid",
            metric=metric,
            limit=5
        )

        assert len(results.to_list()) > 0


def test_hybrid_search_large_collection(temp_db_path, query_vector):
    """Test hybrid search on larger collection."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Add 1000 documents
    docs = [
        {
            "id": f"doc{i}",
            "text": f"Document {i} about machine learning" if i % 3 == 0 else f"Document {i}",
            "vector": [i * 0.001 + j * 0.0001 for j in range(384)],
            "metadata": {"index": i}
        }
        for i in range(1000)
    ]
    collection.add(docs)

    results = collection.search(
        query_text="machine learning",
        query_vector=query_vector,
        mode="hybrid",
        limit=10
    )

    results_list = results.to_list()
    assert len(results_list) == 10

    # Should find relevant results
    assert all(r["score"] > 0 for r in results_list)
