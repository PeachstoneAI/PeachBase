"""Writer for serializing PeachBase collections to binary format."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from peachbase.storage.format import (
    FORMAT_VERSION,
    HEADER_SIZE,
    MAGIC_NUMBER,
    BM25IndexSection,
    MetadataSection,
    PeachBaseHeader,
    TextSection,
    VectorSection,
)

if TYPE_CHECKING:
    from peachbase.collection import Collection


def write_collection(collection: "Collection", path: str) -> None:
    """Write a collection to disk in PeachBase binary format.

    Args:
        collection: Collection to write
        path: Output file path (.pdb file)
    """
    # Build BM25 index if needed
    collection._rebuild_indices()

    # Prepare sections
    vector_section = VectorSection(vectors=collection._vectors)

    # Build text and metadata mappings
    texts = {doc["id"]: doc["text"] for doc in collection._documents}
    metadata = {doc["id"]: doc["metadata"] for doc in collection._documents}

    text_section = TextSection(texts=texts)
    metadata_section = MetadataSection(metadata=metadata)

    # Build BM25 index section
    if collection._bm25_index is not None:
        # Convert BM25Index object to dict
        bm25_data = collection._bm25_index.to_dict()
        bm25_section = BM25IndexSection(
            vocabulary=bm25_data.get("vocabulary", {}),
            idf_scores=bm25_data.get("idf_scores", []),
            doc_lengths=bm25_data.get("doc_lengths", []),
            avg_doc_len=bm25_data.get("avg_doc_len", 0.0),
            inverted_index=bm25_data.get("inverted_index", {}),
        )
    else:
        # Empty BM25 index
        bm25_section = BM25IndexSection(
            vocabulary={},
            idf_scores=[],
            doc_lengths=[0] * collection.size,
            avg_doc_len=0.0,
            inverted_index={},
        )

    # Serialize sections to bytes
    vector_data = vector_section.to_bytes()
    text_data = text_section.to_bytes()
    metadata_data = metadata_section.to_bytes()
    bm25_data = bm25_section.to_bytes()

    # Calculate offsets
    vector_offset = HEADER_SIZE
    text_offset = vector_offset + len(vector_data)
    metadata_offset = text_offset + len(text_data)
    bm25_index_offset = metadata_offset + len(metadata_data)

    # Create header
    header = PeachBaseHeader(
        magic=MAGIC_NUMBER,
        version=FORMAT_VERSION,
        n_documents=collection.size,
        dimension=collection.dimension,
        flags=0,  # No special flags yet
        vector_offset=vector_offset,
        text_offset=text_offset,
        metadata_offset=metadata_offset,
        bm25_index_offset=bm25_index_offset,
    )

    header_data = header.to_bytes()

    # Write to file
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(header_data)
        f.write(vector_data)
        f.write(text_data)
        f.write(metadata_data)
        f.write(bm25_data)


def upload_collection_to_s3(
    collection: "Collection", bucket: str, key: str, **kwargs: Any
) -> None:
    """Upload a collection to S3.

    Args:
        collection: Collection to upload
        bucket: S3 bucket name
        key: S3 key (path)
        **kwargs: Additional arguments for S3 upload (e.g., ServerSideEncryption)
    """
    import tempfile

    import boto3

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        write_collection(collection, tmp_path)

    try:
        # Upload to S3
        s3_client = boto3.client("s3")
        s3_client.upload_file(tmp_path, bucket, key, ExtraArgs=kwargs)
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)
