"""Unit tests for Collection class."""
import pytest
import peachbase


def test_add_documents(temp_db_path, sample_documents):
    """Test adding documents to a collection."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    collection.add(sample_documents)

    assert collection.size == len(sample_documents)


def test_add_empty_list(temp_db_path):
    """Test adding an empty list of documents."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    collection.add([])

    assert collection.size == 0


def test_add_duplicate_id_raises_error(temp_db_path):
    """Test that adding a document with duplicate ID raises ValueError."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    doc1 = {
        "id": "doc1",
        "text": "Original text",
        "vector": [0.1] * 384,
        "metadata": {"version": 1}
    }

    doc2 = {
        "id": "doc1",
        "text": "Updated text",
        "vector": [0.2] * 384,
        "metadata": {"version": 2}
    }

    collection.add([doc1])
    assert collection.size == 1

    # Adding duplicate ID should raise ValueError
    with pytest.raises(ValueError, match="already exists"):
        collection.add([doc2])

    # Original document should still be there
    retrieved = collection.get("doc1")
    assert retrieved is not None
    assert retrieved["text"] == "Original text"


def test_add_wrong_dimension_raises_error(temp_db_path):
    """Test that adding a vector with wrong dimension raises an error."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    doc = {
        "id": "doc1",
        "text": "Test document",
        "vector": [0.1] * 100,  # Wrong dimension!
        "metadata": {}
    }

    with pytest.raises(ValueError, match="dimension"):
        collection.add([doc])


def test_add_missing_required_fields(temp_db_path):
    """Test that adding a document missing required fields raises an error."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Missing 'text' is allowed (text is optional)
    collection.add([{
        "id": "doc1",
        "vector": [0.1] * 384,
        "metadata": {}
    }])
    assert collection.size == 1

    # Missing 'vector' should raise ValueError
    with pytest.raises(ValueError, match="vector"):
        collection.add([{
            "id": "doc2",
            "text": "Test",
            "metadata": {}
        }])

    # Missing 'id' should raise ValueError
    with pytest.raises(ValueError, match="id"):
        collection.add([{
            "text": "Test",
            "vector": [0.1] * 384,
            "metadata": {}
        }])


def test_get_document(temp_db_path, sample_documents):
    """Test getting a document by ID."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    doc = collection.get("doc1")

    assert doc["id"] == "doc1"
    assert doc["text"] == sample_documents[0]["text"]
    assert doc["metadata"] == sample_documents[0]["metadata"]


def test_get_nonexistent_document_returns_none(temp_db_path):
    """Test that getting a nonexistent document returns None."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    result = collection.get("nonexistent")
    assert result is None


def test_delete_document(temp_db_path, sample_documents):
    """Test deleting a document."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)
    collection.add(sample_documents)

    initial_size = collection.size
    collection.delete("doc1")

    assert collection.size == initial_size - 1

    # After delete, document should not be found
    result = collection.get("doc1")
    assert result is None


def test_delete_nonexistent_document_silent(temp_db_path):
    """Test that deleting a nonexistent document doesn't raise error."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Should not raise an error
    collection.delete("nonexistent")


def test_collection_size_property(temp_db_path, sample_documents):
    """Test the size property tracks document count correctly."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    assert collection.size == 0

    collection.add(sample_documents[:2])
    assert collection.size == 2

    collection.add(sample_documents[2:])
    assert collection.size == 5

    collection.delete("doc1")
    assert collection.size == 4


def test_collection_dimension_property(temp_db_path):
    """Test the dimension property."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=768)

    assert collection.dimension == 768


def test_collection_name_property(temp_db_path):
    """Test the name property."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("my_collection", dimension=384)

    assert collection.name == "my_collection"


def test_metadata_optional(temp_db_path):
    """Test that metadata is optional."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    doc = {
        "id": "doc1",
        "text": "Test document",
        "vector": [0.1] * 384
    }

    collection.add([doc])

    retrieved = collection.get("doc1")
    assert "metadata" in retrieved
    assert retrieved["metadata"] == {} or retrieved["metadata"] is None


def test_large_batch_add(temp_db_path):
    """Test adding a large batch of documents."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test", dimension=384)

    # Generate 1000 documents
    docs = [
        {
            "id": f"doc{i}",
            "text": f"Document number {i}",
            "vector": [i * 0.001] * 384,
            "metadata": {"index": i}
        }
        for i in range(1000)
    ]

    collection.add(docs)

    assert collection.size == 1000

    # Verify random samples
    assert collection.get("doc0")["metadata"]["index"] == 0
    assert collection.get("doc500")["metadata"]["index"] == 500
    assert collection.get("doc999")["metadata"]["index"] == 999
