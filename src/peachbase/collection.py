"""Collection class for managing documents and search operations."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from peachbase.database import Database
    from peachbase.query import Query


class Collection:
    """Collection represents a table of documents with vectors and metadata.

    A collection stores documents with their text, embeddings (vectors), and metadata.
    It supports lexical (BM25), semantic (vector), and hybrid search modes.

    Args:
        name: Name of the collection
        dimension: Vector dimension for embeddings
        database: Parent database instance
        metadata: Optional metadata for the collection

    Examples:
        >>> collection = db.create_collection("docs", dimension=384)
        >>> collection.add([
        ...     {"id": "1", "text": "Hello", "vector": [...], "metadata": {...}}
        ... ])
        >>> results = collection.search(query_vector=[...], limit=10)
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        database: Database,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a collection.

        Args:
            name: Collection name
            dimension: Vector embedding dimension
            database: Parent database
            metadata: Optional metadata
        """
        self.name = name
        self.dimension = dimension
        self.database = database
        self.metadata = metadata or {}

        # Storage for documents
        self._documents: list[dict[str, Any]] = []
        self._vectors: list[list[float]] = []
        self._doc_index: dict[str, int] = {}

        # Pre-flattened vectors for fast SIMD access (avoids flattening on every search)
        self._flat_vectors: array | None = None
        self._flat_vectors_dirty = True

        # Search indices (will be built when needed)
        self._bm25_index: Any | None = None
        self._is_dirty = True  # Flag to track if indices need rebuilding

    @property
    def size(self) -> int:
        """Get number of documents in collection."""
        return len(self._documents)

    def add(
        self,
        documents: list[dict[str, Any]],
        vectors: list[list[float]] | None = None,
    ) -> None:
        """Add documents to the collection.

        Args:
            documents: List of dicts with 'id', 'text', optional 'vector'/'metadata'
            vectors: Optional separate list of vectors if not included in documents

        Examples:
            >>> collection.add([
            ...     {
            ...         "id": "doc1",
            ...         "text": "Machine learning is fascinating",
            ...         "vector": [0.1, 0.2, ...],  # 384-dim vector
            ...         "metadata": {"category": "tech"}
            ...     }
            ... ])
        """
        for i, doc in enumerate(documents):
            doc_id = doc.get("id")
            if doc_id is None:
                raise ValueError(f"Document at index {i} missing required 'id' field")

            if doc_id in self._doc_index:
                raise ValueError(f"Document with id '{doc_id}' already exists")

            # Get vector from document or separate vectors list
            if "vector" in doc:
                vector = doc["vector"]
            elif vectors is not None and i < len(vectors):
                vector = vectors[i]
            else:
                raise ValueError(f"No vector provided for document '{doc_id}'")

            # Validate vector dimension
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch for doc '{doc_id}': "
                    f"expected {self.dimension}, got {len(vector)}"
                )

            # Store document
            doc_data = {
                "id": doc_id,
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            }
            self._documents.append(doc_data)
            self._vectors.append(vector)
            self._doc_index[doc_id] = len(self._documents) - 1

        self._is_dirty = True
        self._flat_vectors_dirty = True

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document dict or None if not found
        """
        idx = self._doc_index.get(doc_id)
        if idx is None:
            return None

        doc = self._documents[idx].copy()
        doc["vector"] = self._vectors[idx]
        return doc

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted, False if not found
        """
        idx = self._doc_index.get(doc_id)
        if idx is None:
            return False

        # Remove from storage
        del self._documents[idx]
        del self._vectors[idx]
        del self._doc_index[doc_id]

        # Rebuild index mapping
        self._doc_index = {doc["id"]: i for i, doc in enumerate(self._documents)}

        self._is_dirty = True
        self._flat_vectors_dirty = True
        return True

    def _ensure_flat_vectors(self) -> array:
        """Ensure flat vectors array is built and up-to-date.

        Returns:
            Flattened array of all vectors for fast SIMD access
        """
        if self._flat_vectors is None or self._flat_vectors_dirty:
            # Build flattened array once
            flat_data = []
            for vec in self._vectors:
                flat_data.extend(vec)
            self._flat_vectors = array("f", flat_data)
            self._flat_vectors_dirty = False

        return self._flat_vectors

    def search(
        self,
        query_text: str | None = None,
        query_vector: list[float] | None = None,
        mode: Literal["lexical", "semantic", "hybrid"] = "semantic",
        metric: Literal["cosine", "l2", "dot"] = "cosine",
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        alpha: float = 0.5,
    ) -> Query:
        """Search the collection.

        Args:
            query_text: Text query for lexical/hybrid search
            query_vector: Vector query for semantic/hybrid search
            mode: Search mode - "lexical", "semantic", or "hybrid"
            metric: Distance metric for semantic search
            limit: Maximum number of results
            filter: Metadata filter (MongoDB-like syntax)
            alpha: Weight for hybrid search (0=semantic only, 1=lexical only)

        Returns:
            Query object with results

        Examples:
            >>> # Semantic search
            >>> results = collection.search(query_vector=[0.1, ...], limit=10)
            >>>
            >>> # Lexical search
            >>> results = collection.search(
            ...     query_text="machine learning",
            ...     mode="lexical",
            ...     limit=10
            ... )
            >>>
            >>> # Hybrid search
            >>> results = collection.search(
            ...     query_text="machine learning",
            ...     query_vector=[0.1, ...],
            ...     mode="hybrid",
            ...     alpha=0.5
            ... )
            >>>
            >>> # With metadata filter
            >>> results = collection.search(
            ...     query_vector=[0.1, ...],
            ...     filter={"category": "tech", "year": {"$gte": 2023}}
            ... )
        """
        from peachbase.query import Query

        query = Query(
            collection=self,
            query_text=query_text,
            query_vector=query_vector,
            mode=mode,
            metric=metric,
            limit=limit,
            filter=filter,
            alpha=alpha,
        )

        # Execute search based on mode
        # TODO: Implement actual search logic in later phases
        return query

    def save(self) -> None:
        """Save collection to disk or S3."""
        from peachbase.storage.writer import upload_collection_to_s3, write_collection

        path = self.database.get_collection_path(self.name)

        if self.database.is_s3:
            # Upload to S3
            upload_collection_to_s3(
                self, self.database.bucket, f"{self.database.prefix}/{self.name}.pdb"
            )
        else:
            # Save to local disk
            write_collection(self, path)

    @classmethod
    def load(cls, name: str, database: Database) -> Collection:
        """Load collection from disk or S3.

        Args:
            name: Collection name
            database: Parent database

        Returns:
            Loaded collection instance
        """
        from peachbase.search.bm25 import BM25Index
        from peachbase.storage.reader import load_collection_from_s3, read_collection

        path = database.get_collection_path(name)

        # Load data
        if database.is_s3:
            collection_data, bm25_index_data = load_collection_from_s3(
                database.bucket, f"{database.prefix}/{name}.pdb"
            )
        else:
            collection_data, bm25_index_data = read_collection(path)

        # Create collection instance
        collection = cls(
            name=name,
            dimension=collection_data["dimension"],
            database=database,
            metadata={},
        )

        # Restore documents and vectors
        collection._documents = collection_data["documents"]
        collection._vectors = collection_data["vectors"]
        collection._doc_index = {
            doc["id"]: i for i, doc in enumerate(collection._documents)
        }

        # Restore BM25 index if available
        if bm25_index_data:
            collection._bm25_index = BM25Index.from_dict(
                bm25_index_data, collection._documents
            )
        else:
            collection._bm25_index = None

        collection._is_dirty = False

        return collection

    def _rebuild_indices(self) -> None:
        """Rebuild search indices if needed."""
        if not self._is_dirty:
            return

        # Build BM25 index from documents
        if self._documents:
            from peachbase.search.bm25 import build_bm25_index

            self._bm25_index = build_bm25_index(self._documents)

        self._is_dirty = False

    def __repr__(self) -> str:
        """String representation of the collection."""
        return (
            f"Collection(name='{self.name}', dimension={self.dimension}, "
            f"size={self.size})"
        )

    def __len__(self) -> int:
        """Get number of documents."""
        return self.size
