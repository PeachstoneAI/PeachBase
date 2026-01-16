"""Unit tests for S3 operations with lazy loading."""

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Set fake AWS credentials for moto
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def clean_boto3_import():
    """Fixture to clean boto3 imports and restore after test."""
    # Save original state
    boto3_modules = {
        mod: sys.modules[mod]
        for mod in list(sys.modules.keys())
        if "boto3" in mod or "botocore" in mod
    }

    # Remove boto3 from sys.modules
    for mod in list(boto3_modules.keys()):
        if mod in sys.modules:
            del sys.modules[mod]

    # Reset S3 client
    from peachbase.utils import s3

    original_client = s3._s3_client._client
    original_boto3 = s3._s3_client._boto3
    original_error = s3._s3_client._ClientError

    s3._s3_client._client = None
    s3._s3_client._boto3 = None
    s3._s3_client._ClientError = None

    yield

    # Restore original state
    for mod, module in boto3_modules.items():
        sys.modules[mod] = module

    s3._s3_client._client = original_client
    s3._s3_client._boto3 = original_boto3
    s3._s3_client._ClientError = original_error


def test_s3_lazy_loading(clean_boto3_import):
    """Test that boto3 is not imported until S3 operations are used."""
    # Import peachbase - boto3 should NOT be imported yet
    from peachbase.utils import s3

    assert "boto3" not in sys.modules, (
        "boto3 should not be imported on peachbase import"
    )
    assert s3._s3_client._boto3 is None, "S3 client should not have boto3 loaded yet"


def test_s3_lazy_loading_on_use():
    """Test that boto3 is imported when S3 operations are actually called."""
    # Skip if boto3 is not installed
    pytest.importorskip("boto3")
    pytest.importorskip("moto")

    from moto import mock_aws

    from peachbase.utils import s3

    # Reset S3 client
    s3._s3_client._client = None
    s3._s3_client._boto3 = None

    with mock_aws():
        # Now call an S3 operation - boto3 should be imported
        with contextlib.suppress(Exception):
            s3.check_s3_object_exists("test-bucket", "test-key")

        assert s3._s3_client._boto3 is not None, (
            "boto3 should be loaded after S3 operation"
        )


@pytest.mark.skipif(
    not pytest.importorskip("boto3", reason="boto3 not installed")
    or not pytest.importorskip("moto", reason="moto not installed"),
    reason="Requires boto3 and moto",
)
class TestS3Operations:
    """Test S3 operations with mocked AWS."""

    def test_list_s3_collections_empty(self):
        """Test listing collections from empty S3 bucket."""
        import boto3
        from moto import mock_aws

        from peachbase.utils.s3 import list_s3_collections

        with mock_aws():
            # Create empty bucket
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test-bucket")

            # List collections
            collections = list_s3_collections("test-bucket", "")
            assert collections == []

    def test_list_s3_collections_with_files(self):
        """Test listing collections from S3 bucket with .pdb files."""
        import boto3
        from moto import mock_aws

        from peachbase.utils.s3 import list_s3_collections

        with mock_aws():
            # Create bucket with .pdb files
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test-bucket")

            # Upload test files
            s3.put_object(
                Bucket="test-bucket", Key="collection1.pdb", Body=b"test data 1"
            )
            s3.put_object(
                Bucket="test-bucket", Key="collection2.pdb", Body=b"test data 2"
            )
            s3.put_object(
                Bucket="test-bucket", Key="not_a_collection.txt", Body=b"other"
            )

            # List collections
            collections = list_s3_collections("test-bucket", "")
            assert sorted(collections) == ["collection1", "collection2"]

    def test_list_s3_collections_with_prefix(self):
        """Test listing collections with prefix."""
        import boto3
        from moto import mock_aws

        from peachbase.utils.s3 import list_s3_collections

        with mock_aws():
            # Create bucket with files in different prefixes
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test-bucket")

            # Upload files with different prefixes
            s3.put_object(Bucket="test-bucket", Key="db1/col1.pdb", Body=b"data1")
            s3.put_object(Bucket="test-bucket", Key="db1/col2.pdb", Body=b"data2")
            s3.put_object(Bucket="test-bucket", Key="db2/col3.pdb", Body=b"data3")
            s3.put_object(
                Bucket="test-bucket", Key="col4.pdb", Body=b"data4"
            )  # root level

            # List with prefix
            collections_db1 = list_s3_collections("test-bucket", "db1")
            assert sorted(collections_db1) == ["col1", "col2"]

            collections_db2 = list_s3_collections("test-bucket", "db2")
            assert collections_db2 == ["col3"]

            # List root level
            collections_root = list_s3_collections("test-bucket", "")
            assert collections_root == ["col4"]

    def test_list_s3_collections_pagination(self):
        """Test that pagination works for large number of objects."""
        import boto3
        from moto import mock_aws

        from peachbase.utils.s3 import list_s3_collections

        with mock_aws():
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test-bucket")

            # Create many collections
            for i in range(1500):  # More than S3's default page size (1000)
                s3.put_object(Bucket="test-bucket", Key=f"col{i:04d}.pdb", Body=b"data")

            # List all collections
            collections = list_s3_collections("test-bucket", "")
            assert len(collections) == 1500
            assert "col0000" in collections
            assert "col1499" in collections

    def test_delete_s3_object(self):
        """Test deleting an S3 object."""
        import boto3
        from moto import mock_aws

        from peachbase.utils.s3 import check_s3_object_exists, delete_s3_object

        with mock_aws():
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test-bucket")
            s3.put_object(Bucket="test-bucket", Key="test.pdb", Body=b"data")

            # Verify exists
            assert check_s3_object_exists("test-bucket", "test.pdb")

            # Delete
            delete_s3_object("test-bucket", "test.pdb")

            # Verify deleted
            assert not check_s3_object_exists("test-bucket", "test.pdb")

    def test_delete_s3_object_idempotent(self):
        """Test that deleting non-existent object doesn't raise error."""
        import boto3
        from moto import mock_aws

        from peachbase.utils.s3 import delete_s3_object

        with mock_aws():
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test-bucket")

            # Delete non-existent object - should not raise error
            delete_s3_object("test-bucket", "nonexistent.pdb")

    def test_database_list_collections_s3(self):
        """Test Database.list_collections() with S3."""
        import boto3
        from moto import mock_aws

        import peachbase

        with mock_aws():
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test-bucket")

            # Upload collections
            s3.put_object(Bucket="test-bucket", Key="mydb/users.pdb", Body=b"data1")
            s3.put_object(Bucket="test-bucket", Key="mydb/products.pdb", Body=b"data2")

            # Create database
            db = peachbase.connect("s3://test-bucket/mydb")

            # List collections
            collections = db.list_collections()
            assert sorted(collections) == ["products", "users"]

    def test_database_drop_collection_s3(self):
        """Test Database.drop_collection() with S3."""
        import boto3
        from moto import mock_aws

        import peachbase
        from peachbase.utils.s3 import check_s3_object_exists

        with mock_aws():
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test-bucket")

            # Upload collection
            s3.put_object(Bucket="test-bucket", Key="mydb/users.pdb", Body=b"data")

            # Create database
            db = peachbase.connect("s3://test-bucket/mydb")

            # Verify collection exists
            assert check_s3_object_exists("test-bucket", "mydb/users.pdb")
            assert "users" in db.list_collections()

            # Drop collection
            db.drop_collection("users")

            # Verify deleted
            assert not check_s3_object_exists("test-bucket", "mydb/users.pdb")
            assert "users" not in db.list_collections()

    def test_s3_download_with_cache(self):
        """Test S3 download with caching."""
        import boto3
        from moto import mock_aws

        from peachbase.utils.s3 import download_s3_with_cache

        with mock_aws():
            s3 = boto3.client("s3")
            s3.create_bucket(Bucket="test-bucket")
            s3.put_object(Bucket="test-bucket", Key="test.pdb", Body=b"test data")

            # Create temp cache dir
            cache_dir = tempfile.mkdtemp(prefix="test_cache_")
            try:
                # First download
                path1 = download_s3_with_cache("test-bucket", "test.pdb", cache_dir)
                assert Path(path1).exists()
                assert Path(path1).read_bytes() == b"test data"

                # Second download (should use cache)
                path2 = download_s3_with_cache("test-bucket", "test.pdb", cache_dir)
                assert path1 == path2  # Same path

            finally:
                shutil.rmtree(cache_dir, ignore_errors=True)


def test_no_boto3_error():
    """Test that helpful error is raised when boto3 not available."""
    # This test can only run if boto3 is not installed, which is rare
    # Just verify the error message structure
    from peachbase.utils.s3 import _S3Client

    client = _S3Client()
    client._boto3 = None  # Force it to None

    # Mock the import to fail
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("No module named 'boto3'")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = mock_import

    try:
        with pytest.raises(ImportError, match="boto3 is required for S3 operations"):
            client._ensure_boto3()
    finally:
        builtins.__import__ = original_import


def test_is_boto3_available():
    """Test is_boto3_available function."""
    from peachbase.utils.s3 import is_boto3_available

    # The function should return True if boto3 can be found
    # (whether it's installed or not)
    result = is_boto3_available()
    assert isinstance(result, bool)
