"""Database class for managing PeachBase collections."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peachbase.collection import Collection


class Database:
    """Database class for managing collections in PeachBase.

    A database can be stored locally or on S3. Collections are organized
    within the database directory/bucket.

    Args:
        uri: Database location (local path or s3://bucket/path)

    Examples:
        >>> db = Database("./my_database")  # Local
        >>> db = Database("s3://my-bucket/databases/my_db")  # S3
    """

    def __init__(self, uri: str) -> None:
        """Initialize a database connection.

        Args:
            uri: Database location (local path or S3 URI)
        """
        self.uri = uri
        self.is_s3 = uri.startswith("s3://")

        if self.is_s3:
            # Parse S3 URI
            parts = uri[5:].split("/", 1)
            self.bucket = parts[0]
            self.prefix = parts[1] if len(parts) > 1 else ""
        else:
            # Local filesystem
            self.path = Path(uri)
            self.bucket = None
            self.prefix = None

            # Create directory if it doesn't exist
            if not self.path.exists():
                self.path.mkdir(parents=True, exist_ok=True)

        self._collections: dict[str, Any] = {}
        self._metadata_file = "peachbase_metadata.json"

    def create_collection(
        self,
        name: str,
        dimension: int,
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Collection:
        """Create a new collection in the database.

        Args:
            name: Name of the collection
            dimension: Vector dimension for embeddings
            overwrite: If True, overwrite existing collection
            metadata: Optional metadata for the collection

        Returns:
            Collection instance

        Raises:
            ValueError: If collection exists and overwrite=False
        """
        from peachbase.collection import Collection

        if name in self._collections and not overwrite:
            raise ValueError(
                f"Collection '{name}' already exists. Use overwrite=True to replace."
            )

        collection = Collection(
            name=name,
            dimension=dimension,
            database=self,
            metadata=metadata or {},
        )

        self._collections[name] = collection
        return collection

    def open_collection(self, name: str) -> Collection:
        """Open an existing collection.

        Args:
            name: Name of the collection to open

        Returns:
            Collection instance

        Raises:
            ValueError: If collection doesn't exist
        """
        from peachbase.collection import Collection

        # Check if already loaded in memory
        if name in self._collections:
            return self._collections[name]

        # Try to load from disk/S3
        collection = Collection.load(name=name, database=self)
        self._collections[name] = collection
        return collection

    def list_collections(self) -> list[str]:
        """List all collections in the database.

        Returns:
            List of collection names
        """
        if self.is_s3:
            # List .pdb files from S3
            from peachbase.utils.s3 import list_s3_collections

            return list_s3_collections(self.bucket, self.prefix)
        else:
            # List .pdb files in local directory
            if not self.path.exists():
                return []

            collections = []
            for file in self.path.glob("*.pdb"):
                collections.append(file.stem)
            return collections

    def drop_collection(self, name: str) -> None:
        """Delete a collection from the database.

        Args:
            name: Name of the collection to delete
        """
        if name in self._collections:
            del self._collections[name]

        if self.is_s3:
            # Delete .pdb file from S3
            from peachbase.utils.s3 import delete_s3_object

            prefix = f"{self.prefix}/" if self.prefix else ""
            key = f"{prefix}{name}.pdb"
            delete_s3_object(self.bucket, key)
        else:
            # Delete .pdb file
            collection_file = self.path / f"{name}.pdb"
            if collection_file.exists():
                collection_file.unlink()

    def get_collection_path(self, name: str) -> str:
        """Get the full path/URI for a collection.

        Args:
            name: Collection name

        Returns:
            Full path or S3 URI to collection file
        """
        if self.is_s3:
            prefix = f"{self.prefix}/" if self.prefix else ""
            return f"s3://{self.bucket}/{prefix}{name}.pdb"
        else:
            return str(self.path / f"{name}.pdb")

    def __repr__(self) -> str:
        """String representation of the database."""
        storage_type = "S3" if self.is_s3 else "local"
        n_collections = len(self._collections)
        return f"Database(uri='{self.uri}', type={storage_type}, n={n_collections})"


def connect(uri: str) -> Database:
    """Connect to a PeachBase database.

    Convenience function for creating a Database instance.

    Args:
        uri: Database location (local path or s3://bucket/path)

    Returns:
        Database instance

    Examples:
        >>> import peachbase
        >>> db = peachbase.connect("./my_database")
        >>> db = peachbase.connect("s3://my-bucket/databases/my_db")
    """
    return Database(uri)
