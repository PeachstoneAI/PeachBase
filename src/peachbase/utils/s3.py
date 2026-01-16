"""S3 utilities for PeachBase with lazy loading.

Provides efficient downloading, uploading, and listing of collections to/from S3,
optimized for AWS Lambda environments. boto3 is only imported when S3 operations
are actually used, keeping the package lightweight for local-only usage.
"""

import sys
from pathlib import Path
from typing import Any


class _S3Client:
    """Lazy-loading wrapper for boto3 S3 client.

    boto3 is only imported when methods are actually called, not at module import time.
    This keeps the package lightweight for users who don't need S3 functionality.
    """

    def __init__(self):
        self._client = None
        self._boto3 = None
        self._ClientError = None

    def _ensure_boto3(self):
        """Import boto3 if not already imported."""
        if self._boto3 is None:
            try:
                import boto3
                from botocore.exceptions import ClientError

                self._boto3 = boto3
                self._ClientError = ClientError
            except ImportError as e:
                raise ImportError(
                    "boto3 is required for S3 operations. "
                    "Install it with: pip install boto3"
                ) from e

    def _get_client(self):
        """Get or create the S3 client."""
        if self._client is None:
            self._ensure_boto3()
            self._client = self._boto3.client("s3")
        return self._client

    def download_file(self, bucket: str, key: str, local_path: str) -> None:
        """Download a file from S3."""
        client = self._get_client()
        try:
            client.download_file(bucket, key, local_path)
        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise RuntimeError(
                f"Failed to download s3://{bucket}/{key}: {error_code}"
            ) from e

    def get_object(self, bucket: str, key: str, **kwargs):
        """Get an object from S3."""
        client = self._get_client()
        try:
            return client.get_object(Bucket=bucket, Key=key, **kwargs)
        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise RuntimeError(
                f"Failed to get s3://{bucket}/{key}: {error_code}"
            ) from e

    def upload_file(
        self, local_path: str, bucket: str, key: str, extra_args: dict | None = None
    ) -> None:
        """Upload a file to S3."""
        client = self._get_client()
        try:
            client.upload_file(local_path, bucket, key, ExtraArgs=extra_args or {})
        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise RuntimeError(
                f"Failed to upload to s3://{bucket}/{key}: {error_code}"
            ) from e

    def head_object(self, bucket: str, key: str):
        """Get metadata for an S3 object."""
        client = self._get_client()
        try:
            return client.head_object(Bucket=bucket, Key=key)
        except self._ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                raise FileNotFoundError(f"s3://{bucket}/{key} not found") from e
            raise

    def list_objects_v2(self, bucket: str, **kwargs):
        """List objects in an S3 bucket."""
        client = self._get_client()
        try:
            return client.list_objects_v2(Bucket=bucket, **kwargs)
        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise RuntimeError(f"Failed to list s3://{bucket}: {error_code}") from e

    def delete_object(self, bucket: str, key: str) -> None:
        """Delete an object from S3."""
        client = self._get_client()
        try:
            client.delete_object(Bucket=bucket, Key=key)
        except self._ClientError as e:
            # Don't raise error if object doesn't exist (idempotent)
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code != "NoSuchKey":
                raise RuntimeError(
                    f"Failed to delete s3://{bucket}/{key}: {error_code}"
                ) from e

    def get_paginator(self, operation_name: str):
        """Get a paginator for an S3 operation."""
        client = self._get_client()
        return client.get_paginator(operation_name)


# Global lazy-loading S3 client
_s3_client = _S3Client()


def download_from_s3(
    bucket: str, key: str, local_path: str, byte_range: str | None = None
) -> None:
    """Download a file from S3 to local disk.

    Args:
        bucket: S3 bucket name
        key: S3 key (path)
        local_path: Local file path to save to
        byte_range: Optional byte range (e.g., "bytes=0-255" for header only)

    Raises:
        RuntimeError: If download fails
    """
    # Ensure local directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    if byte_range:
        # Download specific byte range
        response = _s3_client.get_object(bucket, key, Range=byte_range)
        with open(local_path, "wb") as f:
            f.write(response["Body"].read())
    else:
        # Download entire file
        _s3_client.download_file(bucket, key, local_path)


def upload_to_s3(
    local_path: str, bucket: str, key: str, extra_args: dict[str, Any] | None = None
) -> None:
    """Upload a local file to S3.

    Args:
        local_path: Local file path
        bucket: S3 bucket name
        key: S3 key (path)
        extra_args: Optional extra arguments (e.g., ServerSideEncryption)

    Raises:
        RuntimeError: If upload fails
    """
    _s3_client.upload_file(local_path, bucket, key, extra_args)


def check_s3_object_exists(bucket: str, key: str) -> bool:
    """Check if an S3 object exists.

    Args:
        bucket: S3 bucket name
        key: S3 key (path)

    Returns:
        True if object exists, False otherwise
    """
    try:
        _s3_client.head_object(bucket, key)
        return True
    except FileNotFoundError:
        return False


def get_s3_object_size(bucket: str, key: str) -> int:
    """Get the size of an S3 object in bytes.

    Args:
        bucket: S3 bucket name
        key: S3 key (path)

    Returns:
        Object size in bytes

    Raises:
        RuntimeError: If object doesn't exist or operation fails
    """
    response = _s3_client.head_object(bucket, key)
    return response["ContentLength"]


def list_s3_collections(bucket: str, prefix: str = "") -> list[str]:
    """List all PeachBase collections (.pdb files) in an S3 bucket/prefix.

    Args:
        bucket: S3 bucket name
        prefix: Optional prefix to filter objects (e.g., "databases/my_db/")

    Returns:
        List of collection names (without .pdb extension)

    Examples:
        >>> collections = list_s3_collections("my-bucket", "databases/prod/")
        >>> print(collections)
        ['users', 'products', 'orders']
    """
    collections = []

    # Ensure prefix ends with / if not empty
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    # Use paginator to handle large numbers of objects
    paginator = _s3_client.get_paginator("list_objects_v2")

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            # Check if any objects were returned
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]

                # Only include .pdb files
                if key.endswith(".pdb"):
                    # Extract collection name (remove prefix and .pdb extension)
                    collection_name = key[len(prefix) :].rstrip(".pdb")

                    # Skip if name contains subdirectories (only top-level collections)
                    if "/" not in collection_name:
                        collections.append(collection_name)

    except Exception as e:
        raise RuntimeError(
            f"Failed to list collections in s3://{bucket}/{prefix}: {e}"
        ) from e

    return sorted(collections)


def delete_s3_object(bucket: str, key: str) -> None:
    """Delete an object from S3.

    This operation is idempotent - it won't raise an error if the object
    doesn't exist.

    Args:
        bucket: S3 bucket name
        key: S3 key (path)

    Raises:
        RuntimeError: If deletion fails (except for NoSuchKey errors)
    """
    _s3_client.delete_object(bucket, key)


def download_s3_with_cache(
    bucket: str, key: str, cache_dir: str = "/tmp/peachbase_cache"
) -> str:
    """Download from S3 with local caching (useful for Lambda).

    Downloads the file to a cache directory. If it already exists locally,
    returns the cached path without re-downloading.

    Args:
        bucket: S3 bucket name
        key: S3 key (path)
        cache_dir: Local cache directory (default: /tmp/peachbase_cache for Lambda)

    Returns:
        Path to local cached file
    """
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate local file name from S3 key
    local_filename = key.replace("/", "_")
    local_path = cache_path / local_filename

    # Check if already cached
    if local_path.exists():
        # Verify file size matches S3 object
        try:
            s3_size = get_s3_object_size(bucket, key)
            local_size = local_path.stat().st_size
            if s3_size == local_size:
                # Cache hit - return cached path
                return str(local_path)
        except Exception:
            # If verification fails, re-download
            pass

    # Download from S3
    download_from_s3(bucket, key, str(local_path))
    return str(local_path)


def is_boto3_available() -> bool:
    """Check if boto3 is available without importing it.

    Returns:
        True if boto3 can be imported, False otherwise
    """
    return (
        "boto3" in sys.modules
        or _s3_client._boto3 is not None
        or (
            # Try to find boto3 without importing
            __import__("importlib.util").util.find_spec("boto3") is not None
        )
    )
