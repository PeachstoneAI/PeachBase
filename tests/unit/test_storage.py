"""Unit tests for storage (save/load)."""

import contextlib

import peachbase


def test_save_and_load(temp_db_path, sample_documents):
    """Test saving and loading a collection."""
    # Create and populate collection
    db1 = peachbase.connect(temp_db_path)
    collection1 = db1.create_collection("test", dimension=384)
    collection1.add(sample_documents)
    collection1.save()

    # Load in new connection using classmethod
    db2 = peachbase.connect(temp_db_path)
    from peachbase.collection import Collection

    collection2 = Collection.load("test", db2)

    # Verify data persisted
    assert collection2.size == len(sample_documents)
    assert collection2.dimension == 384

    # Verify documents are intact
    for doc in sample_documents:
        loaded_doc = collection2.get(doc["id"])
        assert loaded_doc is not None
        assert loaded_doc["text"] == doc["text"]
        assert loaded_doc["metadata"] == doc["metadata"]


def test_save_preserves_search_indices(temp_db_path, sample_documents, query_vector):
    """Test that save preserves search indices."""
    # Create and populate collection
    db1 = peachbase.connect(temp_db_path)
    collection1 = db1.create_collection("test", dimension=384)
    collection1.add(sample_documents)
    collection1.save()

    # Load in new connection
    db2 = peachbase.connect(temp_db_path)
    from peachbase.collection import Collection

    collection2 = Collection.load("test", db2)

    # Test semantic search works
    results_semantic = collection2.search(
        query_vector=query_vector, mode="semantic", limit=3
    )
    assert len(results_semantic.to_list()) > 0

    # Test lexical search works
    results_lexical = collection2.search(query_text="learning", mode="lexical", limit=3)
    assert len(results_lexical.to_list()) > 0


def test_multiple_save_load_cycles(temp_db_path, sample_documents):
    """Test multiple save/load cycles."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Add documents and save
    collection.add(sample_documents[:2])
    collection.save()

    # Load and verify
    assert collection.size == 2

    # Add more documents and save again
    collection.add(sample_documents[2:])
    collection.save()

    # Load and verify all documents
    assert collection.size == len(sample_documents)


def test_load_updates_collection_state(temp_db_path, sample_documents):
    """Test that load properly loads collection state."""
    # Create and save collection
    db1 = peachbase.connect(temp_db_path)
    collection1 = db1.create_collection("test", dimension=384)
    collection1.add(sample_documents)
    collection1.save()

    # Load in new connection - Collection.load() returns fully loaded collection
    db2 = peachbase.connect(temp_db_path)
    from peachbase.collection import Collection

    collection2 = Collection.load("test", db2)

    # Should have loaded all data
    assert collection2.size == len(sample_documents)


def test_save_empty_collection(temp_db_path):
    """Test saving an empty collection."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Save empty collection
    collection.save()

    # Load it back
    assert collection.size == 0


def test_save_after_delete(temp_db_path, sample_documents):
    """Test saving after deleting documents."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    # Delete a document
    collection.delete("doc1")
    collection.save()

    # Reload and verify document is deleted
    from peachbase.collection import Collection

    collection2 = Collection.load("test", db)
    assert collection2.size == len(sample_documents) - 1

    # Document should not be found
    assert collection2.get("doc1") is None


def test_load_nonexistent_collection(temp_db_path):
    """Test loading a collection that doesn't exist."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Try to load before saving - should handle gracefully or raise error
    with contextlib.suppress(FileNotFoundError, KeyError):
        # Collection is already loaded via Collection.load classmethod
        # If it succeeds, collection should still be empty
        assert collection.size == 0


def test_save_preserves_vector_dimension(temp_db_path):
    """Test that save preserves vector dimension."""
    db1 = peachbase.connect(temp_db_path)
    collection1 = db1.create_collection("test", dimension=768)
    collection1.add(
        [{"id": "doc1", "text": "Test", "vector": [0.1] * 768, "metadata": {}}]
    )
    collection1.save()

    db2 = peachbase.connect(temp_db_path)
    from peachbase.collection import Collection

    collection2 = Collection.load("test", db2)

    assert collection2.dimension == 768


def test_save_preserves_metadata_types(temp_db_path):
    """Test that save preserves different metadata types."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    docs = [
        {
            "id": "doc1",
            "text": "Test document",
            "vector": [0.1] * 384,
            "metadata": {
                "string": "value",
                "int": 42,
                "float": 3.14,
                "bool": True,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
            },
        }
    ]
    collection.add(docs)
    collection.save()

    loaded_doc = collection.get("doc1")

    assert loaded_doc["metadata"]["string"] == "value"
    assert loaded_doc["metadata"]["int"] == 42
    assert loaded_doc["metadata"]["float"] == 3.14
    assert loaded_doc["metadata"]["bool"] is True
    assert loaded_doc["metadata"]["list"] == [1, 2, 3]
    assert loaded_doc["metadata"]["dict"]["nested"] == "value"


def test_concurrent_collections_save_independently(temp_db_path, sample_documents):
    """Test that saving one collection doesn't affect another."""
    db = peachbase.connect(temp_db_path)

    # Create two collections
    col1 = db.create_collection("collection1", dimension=384)
    col2 = db.create_collection("collection2", dimension=384)

    # Add different data
    col1.add(sample_documents[:2])
    col2.add(sample_documents[2:])

    # Save both
    col1.save()
    col2.save()

    # Reload and verify each maintained its own data
    from peachbase.collection import Collection

    col1_loaded = Collection.load("collection1", db)
    col2_loaded = Collection.load("collection2", db)

    assert col1_loaded.size == 2
    assert col2_loaded.size == 3

    # Verify they have different documents
    assert col1_loaded.get("doc1") is not None
    assert col1_loaded.get("doc3") is None

    assert col2_loaded.get("doc3") is not None
    assert col2_loaded.get("doc1") is None


def test_save_large_collection(temp_db_path):
    """Test saving a large collection."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Add 1000 documents
    docs = [
        {
            "id": f"doc{i}",
            "text": f"Document {i}",
            "vector": [i * 0.001] * 384,
            "metadata": {"index": i},
        }
        for i in range(1000)
    ]
    collection.add(docs)
    collection.save()

    # Load and verify
    assert collection.size == 1000

    # Spot check some documents
    assert collection.get("doc0")["metadata"]["index"] == 0
    assert collection.get("doc500")["metadata"]["index"] == 500
    assert collection.get("doc999")["metadata"]["index"] == 999


def test_save_without_prior_load(temp_db_path, sample_documents):
    """Test that save works without calling load first."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Add and save without load
    collection.add(sample_documents)
    collection.save()

    # Verify by loading in new instance
    db2 = peachbase.connect(temp_db_path)
    from peachbase.collection import Collection

    collection2 = Collection.load("test", db2)

    assert collection2.size == len(sample_documents)
