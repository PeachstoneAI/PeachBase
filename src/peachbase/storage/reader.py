"""Reader for loading PeachBase collections from binary format."""

import mmap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from peachbase.storage.format import (
    HEADER_SIZE,
    BM25IndexSection,
    MetadataSection,
    PeachBaseHeader,
    TextSection,
    VectorSection,
)

if TYPE_CHECKING:
    pass


def read_collection(
    path: str, use_mmap: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a collection from disk.

    Args:
        path: Path to .pdb file
        use_mmap: Whether to use memory mapping (faster)

    Returns:
        Tuple of (collection_data, bm25_index) dicts
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Collection file not found: {path}")

    with open(file_path, "rb") as f:
        # Read header
        header_bytes = f.read(HEADER_SIZE)
        header = PeachBaseHeader.from_bytes(header_bytes)

        if use_mmap:
            # Use memory mapping for fast access
            return _read_collection_mmap(f, header)
        else:
            # Read entire file into memory
            return _read_collection_buffered(f, header)


def _read_collection_mmap(
    f, header: PeachBaseHeader
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read collection using memory mapping."""
    # Memory-map the file
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        # Read vector section
        vector_start = header.vector_offset
        vector_end = header.text_offset
        vector_data = mm[vector_start:vector_end]
        vector_section = VectorSection.from_bytes(
            vector_data, header.n_documents, header.dimension
        )

        # Read text section
        text_start = header.text_offset
        text_end = header.metadata_offset
        text_data = mm[text_start:text_end]
        text_section = TextSection.from_bytes(text_data)

        # Read metadata section
        metadata_start = header.metadata_offset
        metadata_end = header.bm25_index_offset
        metadata_data = mm[metadata_start:metadata_end]
        metadata_section = MetadataSection.from_bytes(metadata_data)

        # Read BM25 index section
        bm25_start = header.bm25_index_offset
        bm25_data = mm[bm25_start:]
        if len(bm25_data) > 0:
            bm25_section = BM25IndexSection.from_bytes(bm25_data)
            bm25_index = {
                "vocabulary": bm25_section.vocabulary,
                "idf_scores": bm25_section.idf_scores,
                "doc_lengths": bm25_section.doc_lengths,
                "avg_doc_len": bm25_section.avg_doc_len,
                "inverted_index": bm25_section.inverted_index,
            }
        else:
            bm25_index = None

    finally:
        mm.close()

    # Build documents list
    documents = []
    for _i, (doc_id, text) in enumerate(text_section.texts.items()):
        documents.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": metadata_section.metadata.get(doc_id, {}),
            }
        )

    collection_data = {
        "documents": documents,
        "vectors": vector_section.vectors,
        "dimension": header.dimension,
    }

    return collection_data, bm25_index


def _read_collection_buffered(
    f, header: PeachBaseHeader
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read collection by loading entire file into memory."""
    # Seek to vector section
    f.seek(header.vector_offset)
    vector_size = header.text_offset - header.vector_offset
    vector_data = f.read(vector_size)
    vector_section = VectorSection.from_bytes(
        vector_data, header.n_documents, header.dimension
    )

    # Read text section
    text_size = header.metadata_offset - header.text_offset
    text_data = f.read(text_size)
    text_section = TextSection.from_bytes(text_data)

    # Read metadata section
    metadata_size = header.bm25_index_offset - header.metadata_offset
    metadata_data = f.read(metadata_size)
    metadata_section = MetadataSection.from_bytes(metadata_data)

    # Read BM25 index section (rest of file)
    bm25_data = f.read()
    if len(bm25_data) > 0:
        bm25_section = BM25IndexSection.from_bytes(bm25_data)
        bm25_index = {
            "vocabulary": bm25_section.vocabulary,
            "idf_scores": bm25_section.idf_scores,
            "doc_lengths": bm25_section.doc_lengths,
            "avg_doc_len": bm25_section.avg_doc_len,
            "inverted_index": bm25_section.inverted_index,
        }
    else:
        bm25_index = None

    # Build documents list
    documents = []
    for _i, (doc_id, text) in enumerate(text_section.texts.items()):
        documents.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": metadata_section.metadata.get(doc_id, {}),
            }
        )

    collection_data = {
        "documents": documents,
        "vectors": vector_section.vectors,
        "dimension": header.dimension,
    }

    return collection_data, bm25_index


def load_collection_from_s3(
    bucket: str, key: str, use_mmap: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a collection from S3.

    Args:
        bucket: S3 bucket name
        key: S3 key (path)
        use_mmap: Whether to use memory mapping after downloading

    Returns:
        Tuple of (collection_data, bm25_index) dicts
    """
    import tempfile

    from peachbase.utils.s3 import download_from_s3

    # Download to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        download_from_s3(bucket, key, tmp_path)
        return read_collection(tmp_path, use_mmap=use_mmap)
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)
