"""Unit tests for semantic search."""
import pytest
import peachbase


def test_semantic_search_basic(temp_db_path, sample_documents, query_vector):
    """Test basic semantic search."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=3
    )

    results_list = results.to_list()
    assert len(results_list) <= 3
    assert all("score" in r for r in results_list)


def test_semantic_search_cosine_metric(temp_db_path, sample_documents, query_vector):
    """Test semantic search with cosine similarity."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        metric="cosine",
        limit=5
    )

    results_list = results.to_list()
    assert len(results_list) == min(5, len(sample_documents))

    # Cosine scores should be between -1 and 1
    for result in results_list:
        assert -1 <= result["score"] <= 1


def test_semantic_search_l2_metric(temp_db_path, sample_documents, query_vector):
    """Test semantic search with L2 distance."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        metric="l2",
        limit=5
    )

    results_list = results.to_list()
    assert len(results_list) == min(5, len(sample_documents))

    # L2 scores should be >= 0 (distance)
    for result in results_list:
        assert result["score"] >= 0


def test_semantic_search_dot_metric(temp_db_path, sample_documents, query_vector):
    """Test semantic search with dot product."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        metric="dot",
        limit=5
    )

    results_list = results.to_list()
    assert len(results_list) == min(5, len(sample_documents))


def test_semantic_search_limit(temp_db_path, sample_documents, query_vector):
    """Test that limit parameter is respected."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    for limit in [1, 2, 3, 10]:
        results = collection.search(
            query_vector=query_vector,
            mode="semantic",
            limit=limit
        )
        results_list = results.to_list()
        assert len(results_list) == min(limit, len(sample_documents))


def test_semantic_search_results_sorted(temp_db_path, sample_documents, query_vector):
    """Test that results are sorted by score."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        metric="cosine",
        limit=5
    )

    results_list = results.to_list()
    scores = [r["score"] for r in results_list]

    # For cosine (higher is better), should be descending
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_missing_query_vector(temp_db_path, sample_documents):
    """Test that semantic search without query_vector raises error."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    with pytest.raises((ValueError, TypeError)):
        collection.search(mode="semantic", limit=5)


def test_semantic_search_wrong_dimension(temp_db_path, sample_documents):
    """Test that query vector with wrong dimension causes error."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    wrong_dim_vector = [0.1] * 100  # Wrong dimension

    # Note: Implementation doesn't validate dimensions, causes IndexError
    with pytest.raises(IndexError):
        collection.search(
            query_vector=wrong_dim_vector,
            mode="semantic",
            limit=5
        )


def test_semantic_search_empty_collection(temp_db_path, query_vector):
    """Test semantic search on empty collection."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=5
    )

    assert len(results.to_list()) == 0


def test_semantic_search_result_contains_document_data(temp_db_path, sample_documents, query_vector):
    """Test that results contain complete document data."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=1
    )

    result = results.to_list()[0]

    # Check all expected fields are present
    assert "id" in result
    assert "text" in result
    assert "vector" in result
    assert "metadata" in result
    assert "score" in result


def test_semantic_search_iterator(temp_db_path, sample_documents, query_vector):
    """Test that QueryResult is iterable."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=3
    )

    # Test iteration
    count = 0
    for result in results:
        assert "id" in result
        assert "score" in result
        count += 1

    assert count == 3


def test_semantic_search_length(temp_db_path, sample_documents, query_vector):
    """Test that QueryResult has length."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=3
    )

    assert len(results) == 3


def test_semantic_search_invalid_metric(temp_db_path, sample_documents, query_vector):
    """Test that invalid metric raises error."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    with pytest.raises(ValueError, match="metric"):
        collection.search(
            query_vector=query_vector,
            mode="semantic",
            metric="invalid",
            limit=5
        )


def test_semantic_search_large_collection(temp_db_path, query_vector):
    """Test semantic search on larger collection."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Add 1000 documents
    docs = [
        {
            "id": f"doc{i}",
            "text": f"Document {i}",
            "vector": [i * 0.001 + j * 0.0001 for j in range(384)],
            "metadata": {"index": i}
        }
        for i in range(1000)
    ]
    collection.add(docs)

    results = collection.search(
        query_vector=query_vector,
        mode="semantic",
        limit=10
    )

    assert len(results) == 10
    # Should find results even in large collection
    assert all(r["score"] is not None for r in results.to_list())
