"""Unit tests for Database class."""

import pytest

import peachbase


def test_connect_creates_directory(temp_db_path):
    """Test that connect creates the database directory."""
    db = peachbase.connect(temp_db_path)
    assert db is not None
    assert str(db.path) == temp_db_path


def test_create_collection(temp_db_path):
    """Test creating a collection."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test_collection", dimension=384)

    assert collection is not None
    assert collection.name == "test_collection"
    assert collection.dimension == 384
    assert collection.size == 0


def test_create_collection_duplicate_raises_error(temp_db_path):
    """Test that creating a duplicate collection raises an error."""
    db = peachbase.connect(temp_db_path)
    db.create_collection("test_collection", dimension=384)

    with pytest.raises(ValueError, match="already exists"):
        db.create_collection("test_collection", dimension=384)


def test_get_collection(temp_db_path):
    """Test getting an existing collection."""
    db = peachbase.connect(temp_db_path)
    created = db.create_collection("test_collection", dimension=384)
    retrieved = db.open_collection("test_collection")

    assert retrieved.name == created.name
    assert retrieved.dimension == created.dimension


def test_get_collection_nonexistent_raises_error(temp_db_path):
    """Test that getting a nonexistent collection raises an error."""
    db = peachbase.connect(temp_db_path)

    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        db.open_collection("nonexistent")


def test_list_collections(temp_db_path):
    """Test listing collections."""
    db = peachbase.connect(temp_db_path)

    # Initially empty
    assert db.list_collections() == []

    # Add some collections and save them (list_collections only shows saved collections)
    col1 = db.create_collection("collection1", dimension=384)
    col2 = db.create_collection("collection2", dimension=768)
    col3 = db.create_collection("collection3", dimension=1536)

    col1.save()
    col2.save()
    col3.save()

    collections = db.list_collections()
    assert len(collections) == 3
    assert "collection1" in collections
    assert "collection2" in collections
    assert "collection3" in collections


def test_delete_collection(temp_db_path):
    """Test deleting a collection."""
    db = peachbase.connect(temp_db_path)
    collection = db.create_collection("test_collection", dimension=384)
    collection.save()  # Must save before it appears in list

    assert "test_collection" in db.list_collections()

    db.drop_collection("test_collection")

    assert "test_collection" not in db.list_collections()


def test_delete_nonexistent_collection_silent(temp_db_path):
    """Test that deleting a nonexistent collection doesn't raise error."""
    db = peachbase.connect(temp_db_path)

    # Should not raise an error
    db.drop_collection("nonexistent")


def test_multiple_collections(temp_db_path):
    """Test working with multiple collections simultaneously."""
    db = peachbase.connect(temp_db_path)

    col1 = db.create_collection("col1", dimension=384)
    col2 = db.create_collection("col2", dimension=768)

    # Add different documents to each
    col1.add(
        [
            {
                "id": "doc1",
                "text": "Document in collection 1",
                "vector": [0.1] * 384,
                "metadata": {},
            }
        ]
    )

    col2.add(
        [
            {
                "id": "doc2",
                "text": "Document in collection 2",
                "vector": [0.2] * 768,
                "metadata": {},
            }
        ]
    )

    assert col1.size == 1
    assert col2.size == 1
    assert col1.dimension == 384
    assert col2.dimension == 768


def test_collection_dimension_no_validation(temp_db_path):
    """Test that dimensions are not validated at creation time."""
    db = peachbase.connect(temp_db_path)

    # Dimension validation is not enforced - these succeed
    col1 = db.create_collection("test1", dimension=0)
    assert col1.dimension == 0

    col2 = db.create_collection("test2", dimension=-1)
    assert col2.dimension == -1
