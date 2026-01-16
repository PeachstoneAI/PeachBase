"""Binary storage format specification for PeachBase collections.

Format (.pdb file):
- Header (256 bytes): Magic number, metadata, offsets
- Vector Data Section: Contiguous float32 array
- Text Data Section: Document texts
- Metadata Section: Document metadata (JSON)
- BM25 Index Section: Vocabulary, IDF scores, inverted index
"""

import json
import struct
from dataclasses import dataclass
from typing import Any

# Magic number for PeachBase files
MAGIC_NUMBER = b"PCHDB001"
HEADER_SIZE = 256
FORMAT_VERSION = 1


@dataclass
class PeachBaseHeader:
    """Header structure for .pdb files."""

    magic: bytes  # 8 bytes
    version: int  # 4 bytes
    n_documents: int  # 8 bytes
    dimension: int  # 4 bytes
    flags: int  # 4 bytes
    vector_offset: int  # 8 bytes
    text_offset: int  # 8 bytes
    metadata_offset: int  # 8 bytes
    bm25_index_offset: int  # 8 bytes

    # Total: 60 bytes + reserved space up to 256 bytes

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        header = bytearray(HEADER_SIZE)

        # Pack header fields
        struct.pack_into("8s", header, 0, self.magic)
        struct.pack_into("<I", header, 8, self.version)
        struct.pack_into("<Q", header, 12, self.n_documents)
        struct.pack_into("<I", header, 20, self.dimension)
        struct.pack_into("<I", header, 24, self.flags)
        struct.pack_into("<Q", header, 28, self.vector_offset)
        struct.pack_into("<Q", header, 36, self.text_offset)
        struct.pack_into("<Q", header, 44, self.metadata_offset)
        struct.pack_into("<Q", header, 52, self.bm25_index_offset)

        return bytes(header)

    @classmethod
    def from_bytes(cls, data: bytes) -> "PeachBaseHeader":
        """Deserialize header from bytes."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header data too short: {len(data)} < {HEADER_SIZE}")

        magic = struct.unpack_from("8s", data, 0)[0]
        if magic != MAGIC_NUMBER:
            raise ValueError(f"Invalid magic number: {magic} != {MAGIC_NUMBER}")

        version = struct.unpack_from("<I", data, 8)[0]
        n_documents = struct.unpack_from("<Q", data, 12)[0]
        dimension = struct.unpack_from("<I", data, 20)[0]
        flags = struct.unpack_from("<I", data, 24)[0]
        vector_offset = struct.unpack_from("<Q", data, 28)[0]
        text_offset = struct.unpack_from("<Q", data, 36)[0]
        metadata_offset = struct.unpack_from("<Q", data, 44)[0]
        bm25_index_offset = struct.unpack_from("<Q", data, 52)[0]

        return cls(
            magic=magic,
            version=version,
            n_documents=n_documents,
            dimension=dimension,
            flags=flags,
            vector_offset=vector_offset,
            text_offset=text_offset,
            metadata_offset=metadata_offset,
            bm25_index_offset=bm25_index_offset,
        )


@dataclass
class VectorSection:
    """Vector data section - contiguous float32 array."""

    vectors: list[list[float]]  # [n_docs x dim]

    def to_bytes(self) -> bytes:
        """Serialize vectors to bytes (row-major float32 array)."""
        if not self.vectors:
            return b""

        # Flatten to single array and convert to bytes
        flat_data = []
        for vec in self.vectors:
            for val in vec:
                flat_data.append(struct.pack("<f", val))

        return b"".join(flat_data)

    @classmethod
    def from_bytes(cls, data: bytes, n_docs: int, dim: int) -> "VectorSection":
        """Deserialize vectors from bytes."""
        expected_size = n_docs * dim * 4  # 4 bytes per float32
        if len(data) < expected_size:
            raise ValueError(f"Vector data too short: {len(data)} < {expected_size}")

        vectors = []
        offset = 0
        for _ in range(n_docs):
            vec = []
            for _ in range(dim):
                val = struct.unpack_from("<f", data, offset)[0]
                vec.append(val)
                offset += 4
            vectors.append(vec)

        return cls(vectors=vectors)


@dataclass
class TextSection:
    """Text data section - document texts."""

    texts: dict[str, str]  # doc_id -> text

    def to_bytes(self) -> bytes:
        """Serialize texts to bytes."""
        parts = []

        for doc_id, text in self.texts.items():
            doc_id_bytes = doc_id.encode("utf-8")
            text_bytes = text.encode("utf-8")

            # Format: [doc_id_len (4), doc_id_bytes, text_len (4), text_bytes]
            parts.append(struct.pack("<I", len(doc_id_bytes)))
            parts.append(doc_id_bytes)
            parts.append(struct.pack("<I", len(text_bytes)))
            parts.append(text_bytes)

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "TextSection":
        """Deserialize texts from bytes."""
        texts = {}
        offset = 0

        while offset < len(data):
            if offset + 4 > len(data):
                break

            # Read doc_id
            doc_id_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            doc_id = data[offset : offset + doc_id_len].decode("utf-8")
            offset += doc_id_len

            # Read text
            text_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            text = data[offset : offset + text_len].decode("utf-8")
            offset += text_len

            texts[doc_id] = text

        return cls(texts=texts)


@dataclass
class MetadataSection:
    """Metadata section - document metadata as JSON."""

    metadata: dict[str, dict[str, Any]]  # doc_id -> metadata dict

    def to_bytes(self) -> bytes:
        """Serialize metadata to bytes."""
        parts = []

        for doc_id, meta in self.metadata.items():
            doc_id_bytes = doc_id.encode("utf-8")
            meta_json = json.dumps(meta).encode("utf-8")

            # Format: [doc_id_len (4), doc_id_bytes, json_len (4), json_bytes]
            parts.append(struct.pack("<I", len(doc_id_bytes)))
            parts.append(doc_id_bytes)
            parts.append(struct.pack("<I", len(meta_json)))
            parts.append(meta_json)

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MetadataSection":
        """Deserialize metadata from bytes."""
        metadata = {}
        offset = 0

        while offset < len(data):
            if offset + 4 > len(data):
                break

            # Read doc_id
            doc_id_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            doc_id = data[offset : offset + doc_id_len].decode("utf-8")
            offset += doc_id_len

            # Read metadata JSON
            json_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            meta_json = data[offset : offset + json_len].decode("utf-8")
            offset += json_len

            metadata[doc_id] = json.loads(meta_json)

        return cls(metadata=metadata)


@dataclass
class BM25IndexSection:
    """BM25 index section - vocabulary and inverted index."""

    vocabulary: dict[str, int]  # term -> term_id
    idf_scores: list[float]  # IDF scores indexed by term_id
    doc_lengths: list[int]  # Document lengths (token counts)
    avg_doc_len: float  # Average document length
    inverted_index: dict[
        int, list[tuple[int, int]]
    ]  # term_id -> [(doc_idx, term_freq), ...]

    def to_bytes(self) -> bytes:
        """Serialize BM25 index to bytes."""
        # Serialize vocabulary
        vocab_json = json.dumps(self.vocabulary).encode("utf-8")

        # Serialize IDF scores
        idf_data = struct.pack(f"<{len(self.idf_scores)}f", *self.idf_scores)

        # Serialize doc lengths
        doc_len_data = struct.pack(f"<{len(self.doc_lengths)}I", *self.doc_lengths)

        # Serialize average doc length
        avg_len_data = struct.pack("<f", self.avg_doc_len)

        # Serialize inverted index (now dict: term_id -> {doc_idx: term_freq})
        inv_index_parts = []
        for term_id, postings_dict in sorted(self.inverted_index.items()):
            # Format: [term_id (4), n_postings (4), [(doc_idx (4), term_freq (4)), ...]]
            inv_index_parts.append(struct.pack("<II", term_id, len(postings_dict)))
            for doc_idx, term_freq in sorted(postings_dict.items()):
                inv_index_parts.append(struct.pack("<II", doc_idx, term_freq))
        inv_index_data = b"".join(inv_index_parts)

        # Combine all parts with lengths
        parts = [
            struct.pack("<I", len(vocab_json)),
            vocab_json,
            struct.pack("<I", len(self.idf_scores)),
            idf_data,
            struct.pack("<I", len(self.doc_lengths)),
            doc_len_data,
            avg_len_data,
            struct.pack("<I", len(inv_index_data)),
            inv_index_data,
        ]

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BM25IndexSection":
        """Deserialize BM25 index from bytes."""
        offset = 0

        # Read vocabulary
        vocab_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        vocab_json = data[offset : offset + vocab_len].decode("utf-8")
        vocabulary = json.loads(vocab_json)
        offset += vocab_len

        # Read IDF scores
        n_idf = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        idf_scores = list(struct.unpack_from(f"<{n_idf}f", data, offset))
        offset += n_idf * 4

        # Read doc lengths
        n_docs = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        doc_lengths = list(struct.unpack_from(f"<{n_docs}I", data, offset))
        offset += n_docs * 4

        # Read average doc length
        avg_doc_len = struct.unpack_from("<f", data, offset)[0]
        offset += 4

        # Read inverted index
        inv_index_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        inv_index_end = offset + inv_index_len

        inverted_index = {}
        while offset < inv_index_end:
            term_id = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            n_postings = struct.unpack_from("<I", data, offset)[0]
            offset += 4

            # Read as dict: {doc_idx: term_freq}
            postings_dict = {}
            for _ in range(n_postings):
                doc_idx = struct.unpack_from("<I", data, offset)[0]
                offset += 4
                term_freq = struct.unpack_from("<I", data, offset)[0]
                offset += 4
                postings_dict[doc_idx] = term_freq

            inverted_index[term_id] = postings_dict

        return cls(
            vocabulary=vocabulary,
            idf_scores=idf_scores,
            doc_lengths=doc_lengths,
            avg_doc_len=avg_doc_len,
            inverted_index=inverted_index,
        )
