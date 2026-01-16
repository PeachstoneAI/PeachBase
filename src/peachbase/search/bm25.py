"""BM25 lexical search implementation for PeachBase.

Implements BM25 algorithm with inverted index and uses C extension for scoring.
"""

import heapq
import math
from collections import defaultdict
from typing import Any

from peachbase.text.tokenizer import Tokenizer


class BM25Index:
    """BM25 inverted index for lexical search.

    Args:
        documents: List of documents with 'id' and 'text' fields
        k1: BM25 parameter (default: 1.5)
        b: BM25 parameter (default: 0.75)
        tokenizer: Tokenizer instance (default: Tokenizer())
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or Tokenizer()

        # Build index
        self.documents = documents
        self.doc_index = {doc["id"]: i for i, doc in enumerate(documents)}
        self._build_index()

    def _build_index(self) -> None:
        """Build BM25 index from documents."""
        # Tokenize all documents
        self.doc_tokens: list[list[str]] = []
        self.doc_lengths: list[int] = []

        for doc in self.documents:
            tokens = self.tokenizer.tokenize(doc.get("text", ""))
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

        # Calculate average document length
        self.avg_doc_len = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        )

        # Build vocabulary and inverted index
        self.vocabulary: dict[str, int] = {}  # term -> term_id
        self.inverted_index: dict[int, dict[int, int]] = defaultdict(
            dict
        )  # term_id -> {doc_idx: term_freq}

        term_id = 0
        for doc_idx, tokens in enumerate(self.doc_tokens):
            # Count term frequencies in this document
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            # Add to inverted index
            for term, freq in term_freqs.items():
                if term not in self.vocabulary:
                    self.vocabulary[term] = term_id
                    term_id += 1

                tid = self.vocabulary[term]
                self.inverted_index[tid][doc_idx] = freq

        # Calculate IDF scores
        n_docs = len(self.documents)
        self.idf_scores: list[float] = [0.0] * len(self.vocabulary)

        for _term, tid in self.vocabulary.items():
            df = len(
                self.inverted_index[tid]
            )  # Document frequency (number of docs containing term)
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1.0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_scores[tid] = idf

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Search documents using BM25.

        Args:
            query: Query string
            limit: Maximum number of results

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Get query term IDs and IDFs
        query_term_ids = []
        query_idfs = []
        for token in query_tokens:
            if token in self.vocabulary:
                tid = self.vocabulary[token]
                query_term_ids.append(tid)
                query_idfs.append(self.idf_scores[tid])

        if not query_term_ids:
            return []

        # Get candidate documents (union of all postings)
        candidate_docs = set()
        for tid in query_term_ids:
            candidate_docs.update(self.inverted_index[tid].keys())

        if not candidate_docs:
            return []

        # Score documents
        scores = []
        for doc_idx in candidate_docs:
            score = self._score_document(doc_idx, query_term_ids, query_idfs)
            if score > 0:
                doc_id = self.documents[doc_idx]["id"]
                scores.append((doc_id, score))

        # Use heapq for efficient top-k selection instead of full sort
        return heapq.nlargest(limit, scores, key=lambda x: x[1])

    def _score_document(
        self, doc_idx: int, query_term_ids: list[int], query_idfs: list[float]
    ) -> float:
        """Score a single document for the query.

        Args:
            doc_idx: Document index
            query_term_ids: List of query term IDs
            query_idfs: List of IDF scores for query terms

        Returns:
            BM25 score
        """
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        normalized_len = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)

        # Get term frequencies for this document using O(1) dict lookup
        for tid, idf in zip(query_term_ids, query_idfs, strict=True):
            # Fast O(1) lookup instead of O(n) linear search
            tf = self.inverted_index[tid].get(doc_idx, 0)

            if tf > 0:
                # BM25 formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * normalized_len
                score += idf * (numerator / denominator)

        return score

    def to_dict(self) -> dict[str, Any]:
        """Export index to dictionary for serialization."""
        return {
            "vocabulary": self.vocabulary,
            "idf_scores": self.idf_scores,
            "doc_lengths": self.doc_lengths,
            "avg_doc_len": self.avg_doc_len,
            "inverted_index": dict(self.inverted_index),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        documents: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Load index from dictionary.

        Args:
            data: Index data dictionary
            documents: Document list
            k1: BM25 k1 parameter
            b: BM25 b parameter

        Returns:
            BM25Index instance
        """
        index = cls.__new__(cls)
        index.k1 = k1
        index.b = b
        index.tokenizer = Tokenizer()
        index.documents = documents
        index.doc_index = {doc["id"]: i for i, doc in enumerate(documents)}

        index.vocabulary = data["vocabulary"]
        index.idf_scores = data["idf_scores"]
        index.doc_lengths = data["doc_lengths"]
        index.avg_doc_len = data["avg_doc_len"]
        # Convert nested dict structure: term_id -> {doc_idx: freq}
        index.inverted_index = defaultdict(
            dict, {int(k): v for k, v in data["inverted_index"].items()}
        )

        # Note: doc_tokens is not stored, regenerate if needed
        index.doc_tokens = []

        return index


def build_bm25_index(
    documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75
) -> BM25Index:
    """Build BM25 index from documents.

    Convenience function.

    Args:
        documents: List of document dicts with 'id' and 'text'
        k1: BM25 parameter
        b: BM25 parameter

    Returns:
        BM25Index instance
    """
    return BM25Index(documents, k1=k1, b=b)
